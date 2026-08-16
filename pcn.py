# pcn_baseline

import os, glob, random, numpy as np
from pathlib import Path
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------- Hyper-parameters ----------
REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("TOPO_DATA_ROOT", REPO_ROOT / "data" / "air_plane_"))
CLEAN_DIR = os.getenv("TOPO_CLEAN_DIR", str(DATA_ROOT / "clean_with_holes"))
DROPOUT_DIR = os.getenv("TOPO_DROPOUT_DIR", str(DATA_ROOT / "dropout_local_0_with_holes"))
SAVE_DIR = os.getenv("TOPO_PCN_SAVE_DIR", str(REPO_ROOT / "runs" / "plane_pcn_baseline"))
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS        = 200
BATCH_SIZE    = 10
INPUT_NPTS    = 1024
COARSE_NPTS   = 1024
GRID_SIZE     = 3
GRID_SCALE    = 0.02
ALPHA_FINE    = 0.1
REG_WEIGHT    = 1e-3
LR            = 1e-4
DEVICE        = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

HOLE_THRESH = 0.1;  MAX_HOLE_PTS = 512
F_THRESH    = 0.01
PATIENCE    = 8;    SEED = 42

# ---------- Early stopping ----------
class EarlyStop:
    def __init__(self, patience=PATIENCE):
        self.best = None; self.cnt = 0; self.pat = patience; self.stop = False
    def __call__(self, val):
        if self.best is None or val < self.best:
            self.best, self.cnt = val, 0
        else:
            self.cnt += 1
            self.stop = self.cnt >= self.pat

# ---------- Dataset ----------
class PlaneDS(Dataset):
    """Loads dropout / clean .txt files + light augmentation."""
    def __init__(self, drop_paths, clean_paths, n_in, train=True):
        self.drops, self.cleans = drop_paths, clean_paths
        self.n_in, self.train = n_in, train
    def _load(self, fp, n):
        pts = np.loadtxt(fp, dtype=np.float32, skiprows=2)
        pts = pts[np.random.choice(len(pts), n, replace=len(pts) < n)]
        pts -= pts.mean(0);  pts /= np.linalg.norm(pts, 2, 1).max() + 1e-9
        if self.train:
            pts *= np.random.uniform(0.9, 1.1)
            pts += np.random.uniform(-0.05, 0.05, 3).astype(np.float32)
            pts += np.random.normal(0, 0.002, pts.shape).astype(np.float32)
        return pts
    def __len__(self): return len(self.drops)
    def __getitem__(self, i):
        x = torch.from_numpy(self._load(self.drops[i],  self.n_in))
        y = torch.from_numpy(self._load(self.cleans[i], self.n_in * 4))
        return x, y

# ---------- Small helpers ----------
def mlp(ch):
    layers=[]
    for i in range(len(ch)-1):
        layers += [nn.Linear(ch[i],ch[i+1]), nn.ReLU(True),
                   nn.LayerNorm(ch[i+1]), nn.Dropout(0.2)]
    return nn.Sequential(*layers)

def chamfer(a,b):
    d = torch.cdist(a.float(), b.float())**2
    return d.min(-1)[0].mean() + d.min(-2)[0].mean()

def fscore(a,b,thr=F_THRESH):
    d1 = torch.cdist(a,b).min(-1)[0]; d2 = torch.cdist(b,a).min(-1)[0]
    p  = (d1<thr).float().mean();      r  = (d2<thr).float().mean()
    return 2*p*r/(p+r+1e-8)

# ---------- PCN model ----------
class PCN(nn.Module):
    """Two-stage PointNet encoder + folding decoder."""
    def __init__(self, nc=COARSE_NPTS, g=GRID_SIZE, gs=GRID_SCALE):
        super().__init__();  self.nc, self.g = nc, g
        # encoder
        self.enc1 = mlp([3,128,256])
        self.enc2 = mlp([512,512,1024])  # 256 local + 256 global
        # coarse decoder
        self.dec_c = mlp([1024,1024,1024,nc*3])
        # folding MLP
        self.fold = nn.Sequential(
            nn.Linear(1029,512), nn.ReLU(True),
            nn.Linear(512,512),  nn.ReLU(True),
            nn.Linear(512,3)
        )
        # 2-D grid (g^2,2)
        gx,gy = torch.meshgrid(torch.linspace(-gs,gs,g), torch.linspace(-gs,gs,g), indexing='ij')
        self.register_buffer('grid', torch.stack([gx,gy],-1).view(-1,2))
    def forward(self,x):
        B,N,_ = x.shape
        f     = self.enc1(x)                       # (B,N,256)
        g     = f.max(1,keepdim=True)[0].expand(-1,N,-1)
        feat  = torch.cat([f,g],-1)                # (B,N,512)
        feat  = self.enc2(feat).max(1)[0]          # (B,1024)
        # coarse output + clamp
        coarse = self.dec_c(feat).view(B,self.nc,3).clamp(-1,1)
        # folding
        grid   = self.grid[None,None].expand(B,self.nc,-1,-1)          # (B,nc,g^2,2)
        center = coarse[:,:,None,:].expand(-1,-1,self.g**2,-1)         # (B,nc,g^2,3)
        gfeat  = feat[:,None,None,:].expand(-1,self.nc,self.g**2,-1)    # (B,nc,g^2,1024)
        foldin = torch.cat([grid,center,gfeat],-1)                      # (B,nc,g^2,1029)
        fine   = (self.fold(foldin)+center).view(B,-1,3).clamp(-1,1)    # (B, nc*g^2, 3)
        return coarse, fine

# ---------- Evaluation ----------
@torch.no_grad()
def evaluate(model, loader):
    model.eval(); cd_tot=hole_tot=pts_tot=fs_tot=0.0
    for x,y in loader:
        x,y = x.to(DEVICE), y.to(DEVICE)
        _,pred = model(x); B=x.size(0)
        cd_tot += chamfer(pred,y).item()*B
        # hole-CD
        m = (torch.cdist(y,x).min(-1)[0] > HOLE_THRESH)
        for b in range(B):
            idx = torch.nonzero(m[b],as_tuple=False)[:,0]
            if len(idx):
                idx = idx[torch.randperm(len(idx))[:MAX_HOLE_PTS]]
                hole_tot += chamfer(pred[b:b+1], y[b][idx][None]).item()*len(idx)
                pts_tot  += len(idx)
        fs_tot += fscore(pred,y).item()*B
    n=len(loader.dataset)
    return cd_tot/n, (hole_tot/pts_tot if pts_tot else 0.0), fs_tot/n

# ---------- Visualise only best model ----------
def save_vis(model, ds, tag="best"):
    x,_ = ds[0]; inp=x.numpy()
    model.eval()
    with torch.no_grad():
        _,pred = model(x[None].to(DEVICE)); pred=pred.cpu()[0].numpy()
    fig=plt.figure(); ax=fig.add_subplot(111,projection='3d')
    ax.scatter(inp[:,0],inp[:,1],inp[:,2],s=5,c='red',label='Input')
    ax.scatter(pred[:,0],pred[:,1],pred[:,2],s=2,c='green',label='Pred')
    ax.legend(); ax.set_title('PCN Baseline ('+tag+')')
    plt.savefig(os.path.join(SAVE_DIR,f'vis_{tag}.png')); plt.close(fig)

# ---------- Training loop ----------
def main():
    # data
    drops  = sorted(glob.glob(os.path.join(DROPOUT_DIR,'sample_*.txt')))
    cleans = sorted(glob.glob(os.path.join(CLEAN_DIR,  'sample_*.txt')))
    tr_d,val_d,tr_c,val_c = train_test_split(drops,cleans,test_size=0.2,random_state=SEED)
    tr_loader = DataLoader(PlaneDS(tr_d,tr_c,INPUT_NPTS,True),  BATCH_SIZE,shuffle=True, num_workers=4,pin_memory=True)
    val_loader= DataLoader(PlaneDS(val_d,val_c,INPUT_NPTS,False),BATCH_SIZE,shuffle=False,num_workers=2,pin_memory=True)

    model = PCN().to(DEVICE)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    scaler= torch.cuda.amp.GradScaler()
    stop  = EarlyStop(); best_cd=float('inf')

    for ep in range(1,EPOCHS+1):
        model.train(); pbar=tqdm(tr_loader,desc=f'Epoch {ep}/{EPOCHS}')
        for x,y in pbar:
            x,y=x.to(DEVICE),y.to(DEVICE); opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                coarse,fine = model(x)
                loss_c  = chamfer(coarse,y)
                loss_f  = chamfer(fine,  y)
                reg     = (coarse**2).mean() + (fine**2).mean()
                loss    = loss_c + ALPHA_FINE*loss_f + REG_WEIGHT*reg
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update()
            pbar.set_postfix(loss=loss.item())
        sched.step()

        # validation
        cd_val,hole_val,fs_val = evaluate(model,val_loader)
        print(f'Val CD={cd_val:.4f}  HoleCD={hole_val:.4f}  F={fs_val:.4f}')
        if cd_val < best_cd:
            best_cd=cd_val
            torch.save(model.state_dict(),os.path.join(SAVE_DIR,'best_model_pcn.pth'))
            print('↳ Best model saved')
        stop(cd_val);  print('-'*60)
        if stop.stop:
            print(f'Early-stopping at epoch {ep}'); break

    # final vis on best model
    model.load_state_dict(torch.load(os.path.join(SAVE_DIR,'best_model_pcn.pth')))
    save_vis(model, val_loader.dataset)

if __name__ == '__main__':
    main()
