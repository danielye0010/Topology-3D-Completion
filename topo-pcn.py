"""
Topo-PCN with a 3-D topology feature vector.

Training is driven by differentiable geometric reconstruction losses.
Persistent-homology bottleneck distance is reported as a topology-aware
monitoring/evaluation metric; it is not a differentiable training loss.
"""

# ---------- imports ----------
import os, glob, random, numpy as np
from pathlib import Path
from math import pi
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import gudhi, matplotlib; matplotlib.use("Agg")

# ---------- hyper-params ----------
REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("TOPO_DATA_ROOT", REPO_ROOT / "data" / "air_plane_"))
CLEAN_DIR = os.getenv("TOPO_CLEAN_DIR", str(DATA_ROOT / "clean_with_holes"))
DROPOUT_DIR = os.getenv("TOPO_DROPOUT_DIR", str(DATA_ROOT / "dropout_local_0_with_holes"))
SAVE_DIR = os.getenv("TOPO_PCN_TOPO_SAVE_DIR", str(REPO_ROOT / "runs" / "plane_topo_cf"))
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS, BATCH, N_IN = 200, 8, 1024
COARSE, GRID = 1024, 3
FINE = COARSE * GRID**2
LR, ALPHA = 1e-4, 0.1
HOLE_TH, SAMPLE, F_TH = 0.1, 1024, 0.01
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

# ---------- dataset ----------
class PlaneDS(Dataset):
    """Returns (incomplete_pts, gt_pts, topo_vec)."""
    def __init__(self, drop, clean, npts, train=True):
        self.drop, self.clean, self.n, self.train = drop, clean, npts, train

    def _load(self, path, n):
        with open(path, "r") as f:
            f.readline()
            topo = np.fromstring(f.readline(), sep=" ", dtype=np.float32)
            if topo.size != 3: topo = np.zeros(3, dtype=np.float32)
        pts = np.loadtxt(path, skiprows=2, dtype=np.float32)
        idx = np.random.choice(len(pts), n, replace=len(pts) < n)
        pts = pts[idx]; pts -= pts.mean(0); pts /= pts.max()
        if self.train:
            ang = np.random.uniform(0, 2 * pi); c, s = np.cos(ang), np.sin(ang)
            R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], np.float32)
            pts = pts @ R.T
            pts *= np.random.uniform(0.9, 1.1)
            pts += np.random.uniform(-.05, .05, 3)
            pts += np.random.normal(0, .002, pts.shape)
        return pts.astype(np.float32), topo

    def __len__(self): return len(self.drop)

    def __getitem__(self, i):
        xin, t = self._load(self.drop[i], N_IN)
        yin, _ = self._load(self.clean[i], FINE)
        return torch.from_numpy(xin), torch.from_numpy(yin), torch.from_numpy(t)

# ---------- model ----------
def mlp(ch):
    layers=[]
    for a, b in zip(ch[:-1], ch[1:]):
        layers += [nn.Linear(a, b), nn.ReLU(True), nn.LayerNorm(b), nn.Dropout(0.2)]
    return nn.Sequential(*layers)

class FoldingDecoder(nn.Module):
    def __init__(self, g, feat, nc):
        super().__init__(); self.g2, self.nc = g**2, nc
        self.mlp = mlp([feat + 5, 512, 512, 3])
        grid = torch.stack(torch.meshgrid(torch.linspace(-0.05, 0.05, g),
                                          torch.linspace(-0.05, 0.05, g), indexing='ij'), -1).view(-1, 2)
        self.register_buffer("grid", grid)

    def forward(self, coarse, code):
        B = coarse.size(0)
        grid = self.grid[None, None].repeat(B, self.nc, 1, 1).view(B, -1, 2)
        center = coarse.unsqueeze(2).repeat(1, 1, self.g2, 1).view(B, -1, 3)
        code_exp = code.unsqueeze(1).repeat(1, center.size(1), 1)
        return center + self.mlp(torch.cat([grid, center, code_exp], -1))

class TopoPCN(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = mlp([3, 128, 256])
        self.enc2 = mlp([512, 512, 1024])
        self.topo_fc = nn.Linear(3, 32)
        self.coarse_mlp = mlp([1056, 1024, 1024, COARSE * 3])
        self.fold = FoldingDecoder(GRID, 1056, COARSE)

    def forward(self, x, topo):
        f1 = self.enc1(x)
        g = f1.max(1, keepdim=True)[0]
        code = self.enc2(torch.cat([f1, g.repeat(1, x.size(1), 1)], 2)).max(1)[0]
        code = torch.cat([code, self.topo_fc(topo)], 1)
        coarse = self.coarse_mlp(code).view(-1, COARSE, 3)
        fine = self.fold(coarse, code)
        return coarse, fine

# ---------- metrics ----------
def chamfer(a, b):
    d = torch.cdist(a, b) ** 2
    return d.min(-1)[0].mean() + d.min(-2)[0].mean()

def diag_h1(pts):
    if pts.shape[0] > SAMPLE:
        pts = pts[np.random.choice(len(pts), SAMPLE, False)]
    st = gudhi.RipsComplex(points=pts, max_edge_length=1.0).create_simplex_tree(max_dimension=1)
    st.persistence()
    return np.array(st.persistence_intervals_in_dimension(1))

def bottleneck(d1, d2):
    return gudhi.bottleneck_distance(d1, d2) if len(d1) and len(d2) else 1.0

def fscore(pred, gt):
    d1 = torch.cdist(pred, gt); d2 = d1.transpose(1, 2)
    p = (d1.min(-1)[0] < F_TH).float().mean(1)
    r = (d2.min(-1)[0] < F_TH).float().mean(1)
    return (2 * p * r / (p + r + 1e-8)).mean().item()

@torch.no_grad()
def evaluate(model, loader):
    model.eval(); cd_c = cd_f = hole = h1 = fs = pts_h = n = 0
    for x, y, t in loader:
        x, y, t = x.to(DEVICE), y.to(DEVICE), t.to(DEVICE)
        c, f = model(x, t); B = x.size(0); n += B
        cd_c += chamfer(c, y).item() * B
        cd_f += chamfer(f, y).item() * B
        d = torch.cdist(y, x).min(-1)[0]; mask = d > HOLE_TH
        for b in range(B):
            idx = mask[b].nonzero(as_tuple=False).squeeze()
            if idx.numel():
                hole += chamfer(f[b:b+1], y[b][idx].unsqueeze(0)).item() * idx.numel()
                pts_h += idx.numel()
        h1 += sum(bottleneck(diag_h1(f[b].cpu().numpy()), diag_h1(y[b].cpu().numpy())) for b in range(B))
        idxp = torch.randperm(FINE)[:2048]; idxg = torch.randperm(FINE)[:2048]
        fs += fscore(f[:, idxp], y[:, idxg]) * B
    return dict(cd_c=cd_c / n, cd_f=cd_f / n,
                hole_cd=hole / max(1, pts_h), h1=h1 / n, fscore=fs / n)

# ---------- train ----------
def main():
    drop = sorted(glob.glob(os.path.join(DROPOUT_DIR, 'sample_*.txt')))
    clean = sorted(glob.glob(os.path.join(CLEAN_DIR, 'sample_*.txt')))
    tr_d, vd, tr_c, vc = train_test_split(drop, clean, test_size=0.2, random_state=SEED)
    tr_loader = DataLoader(PlaneDS(tr_d, tr_c, N_IN, True), BATCH, True, num_workers=4)
    val_loader = DataLoader(PlaneDS(vd, vc, N_IN, False), BATCH, False, num_workers=2)
    model = TopoPCN().to(DEVICE); opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=EPOCHS)
    scaler = torch.cuda.amp.GradScaler(); best = 1e9
    for ep in range(EPOCHS):
        model.train()
        for x, y, t in tqdm(tr_loader, desc=f"Ep {ep+1}/{EPOCHS}"):
            x, y, t = x.to(DEVICE), y.to(DEVICE), t.to(DEVICE); opt.zero_grad()
            with torch.cuda.amp.autocast():
                c, f = model(x, t)
                loss_c = chamfer(c, y)
                loss_f = chamfer(f, y)
                loss = loss_c + ALPHA * loss_f
            scaler.scale(loss).backward()
            scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.)
            scaler.step(opt); scaler.update()
        sched.step()
        if (ep + 1) % 10 == 0:
            m = evaluate(model, val_loader)
            print("Validation @ ep", ep + 1, m)
            if m['cd_f'] < best:
                best = m['cd_f']; torch.save(model.state_dict(), os.path.join(SAVE_DIR, 'best.pth'))
                print("Saved best @ ep", ep + 1, m)

if __name__ == "__main__":
    main()
