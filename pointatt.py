r"""
PointAttN comparison model.

* Backbone: PointAttN-style self/cross-attention on points
* Training objective: Chamfer distance
* Topology: H1 bottleneck distance reported as a monitoring/evaluation metric
"""

import glob
import os
import random
from math import pi
from pathlib import Path
from typing import List, Tuple

import gudhi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


# ---------- hyper-params ----------
REPO_ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path(os.getenv("TOPO_DATA_ROOT", REPO_ROOT / "data" / "air_plane_"))
CLEAN_DIR = os.getenv("TOPO_CLEAN_DIR", str(DATA_ROOT / "clean_with_holes"))
DROPOUT_DIR = os.getenv(
    "TOPO_DROPOUT_DIR", str(DATA_ROOT / "dropout_local_0_with_holes")
)
SAVE_DIR = os.getenv(
    "TOPO_POINTATT_SAVE_DIR", str(REPO_ROOT / "runs" / "strict_pointattn")
)
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS = 200
BATCH_SIZE = 10
INPUT_NPTS = 1024
OUTPUT_NPTS = 4096
BASE_LR = 1e-4
WEIGHT_DECAY = 1e-3
CLIP_NORM = 1.0
EVAL_FREQ = 5
PATIENCE = 10
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

HOLE_THRESH = 0.1
TOPO_SAMPLE = 1024
MAX_HOLE_PTS = 512
FSCORE_TH = 0.01

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ---------- pair discovery ----------
def collect_pairs(drop_dir, clean_dir):
    drops = {
        Path(path).name: path
        for path in glob.glob(os.path.join(drop_dir, "sample_*.txt"))
    }
    cleans = {
        Path(path).name: path
        for path in glob.glob(os.path.join(clean_dir, "sample_*.txt"))
    }

    if not drops or not cleans:
        raise FileNotFoundError(
            "No paired point-cloud files found. "
            "Set TOPO_DROPOUT_DIR / TOPO_CLEAN_DIR or extract data.zip."
        )

    missing_clean = sorted(set(drops) - set(cleans))
    missing_drop = sorted(set(cleans) - set(drops))
    if missing_clean or missing_drop:
        raise ValueError(
            "Dropout/clean filenames must match one-to-one. "
            f"Missing clean: {missing_clean[:5]}; missing dropout: {missing_drop[:5]}"
        )

    return [(drops[name], cleans[name]) for name in sorted(drops)]


# ---------- dataset ----------
class PlaneDS(Dataset):
    """Load paired incomplete/clean point clouds in one shared coordinate frame."""

    def __init__(
        self,
        pairs: List[Tuple[str, str]],
        npts: int,
        train: bool = True,
    ):
        self.pairs = pairs
        self.npts = npts
        self.train = train

    @staticmethod
    def _load_points(path: str, n: int) -> np.ndarray:
        pts = np.loadtxt(path, dtype=np.float32, skiprows=2)
        idx = np.random.choice(len(pts), n, replace=len(pts) < n)
        return pts[idx].astype(np.float32)

    @staticmethod
    def _normalize_pair(x: np.ndarray, y: np.ndarray):
        center = x.mean(axis=0, keepdims=True)
        x = x - center
        y = y - center
        scale = np.linalg.norm(x, axis=1).max() + 1e-9
        return x / scale, y / scale

    @staticmethod
    def _augment_pair(x: np.ndarray, y: np.ndarray):
        ang = np.random.uniform(0, 2 * pi)
        c, s = np.cos(ang), np.sin(ang)
        rotation = np.array(
            [[c, -s, 0], [s, c, 0], [0, 0, 1]],
            dtype=np.float32,
        )
        x = x @ rotation.T
        y = y @ rotation.T

        scale = np.float32(np.random.uniform(0.9, 1.1))
        shift = np.random.uniform(-0.05, 0.05, 3).astype(np.float32)
        x = x * scale + shift
        y = y * scale + shift

        x = x + np.random.normal(0, 0.002, x.shape).astype(np.float32)
        return x, y

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor, torch.Tensor]:
        drop_fp, clean_fp = self.pairs[i]
        x = self._load_points(drop_fp, self.npts)
        y = self._load_points(clean_fp, OUTPUT_NPTS)

        x, y = self._normalize_pair(x, y)
        if self.train:
            x, y = self._augment_pair(x, y)

        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(
            y.astype(np.float32)
        )


# ---------- helpers ----------
def mlp(ch: List[int]) -> nn.Sequential:
    layers = []
    for i in range(len(ch) - 1):
        layers += [
            nn.Linear(ch[i], ch[i + 1]),
            nn.ReLU(True),
            nn.LayerNorm(ch[i + 1]),
            nn.Dropout(0.2),
        ]
    return nn.Sequential(*layers)


# ---------- attention blocks ----------
class SFA(nn.Module):
    """Self-Feature-Attention."""

    def __init__(self, dim: int, heads: int = 4, ffn_mult: int = 2):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm1 = nn.LayerNorm(dim)
        self.ffn = mlp([dim, dim * ffn_mult, dim])
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        res = x
        x, _ = self.attn(x, x, x)
        x = self.norm1(x + res)
        res = x
        x = self.ffn(x)
        return self.norm2(x + res)


class GDP(nn.Module):
    """Global-Detail Propagation with synchronized FPS downsampling."""

    def __init__(self, dim: int, down_ratio: int = 2, heads: int = 4):
        super().__init__()
        self.down = down_ratio
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    @staticmethod
    def fps(coords: torch.Tensor, k: int) -> torch.Tensor:
        """Farthest-point sampling indices."""
        B, N, _ = coords.shape
        idx = torch.zeros(B, k, dtype=torch.long, device=coords.device)
        dist = torch.full((B, N), 1e10, device=coords.device)
        far = torch.randint(0, N, (B,), device=coords.device)
        batch = torch.arange(B, device=coords.device)

        for i in range(k):
            idx[:, i] = far
            centroid = coords[batch, far].unsqueeze(1)
            d = ((coords - centroid) ** 2).sum(-1)
            dist = torch.minimum(dist, d)
            far = dist.max(-1)[1]
        return idx

    def forward(self, feats: torch.Tensor, coords: torch.Tensor):
        B, N, D = feats.shape
        if coords.size(1) != N:
            raise ValueError(
                f"Feature/coordinate point counts must match, got {N} and "
                f"{coords.size(1)}"
            )

        k = max(1, N // self.down)
        idx = self.fps(coords, k)
        key_val = feats.gather(1, idx.unsqueeze(-1).expand(-1, -1, D))
        key_coords = coords.gather(
            1, idx.unsqueeze(-1).expand(-1, -1, coords.size(-1))
        )
        att, _ = self.attn(key_val, feats, feats)
        return self.norm(att + key_val), key_coords


# ---------- PointAttN ----------
class PointAttN(nn.Module):
    """PointAttN-style completion model producing 4096 points directly."""

    def __init__(self, out_pts: int = OUTPUT_NPTS, feat_dim: int = 256):
        super().__init__()
        self.out_pts = out_pts

        self.input_proj = mlp([3, feat_dim])
        self.gdp1 = GDP(feat_dim, 4)
        self.sfa1 = SFA(feat_dim)
        self.gdp2 = GDP(feat_dim, 2)
        self.sfa2 = SFA(feat_dim)
        self.gdp3 = GDP(feat_dim, 2)
        self.sfa3 = SFA(feat_dim)

        self.pool = nn.AdaptiveMaxPool1d(1)
        self.decoder = mlp([feat_dim, feat_dim * 2, 3 * out_pts])

    def forward(self, pts: torch.Tensor) -> torch.Tensor:
        feats = self.input_proj(pts)
        coords = pts

        feats, coords = self.gdp1(feats, coords)
        feats = self.sfa1(feats)
        feats, coords = self.gdp2(feats, coords)
        feats = self.sfa2(feats)
        feats, coords = self.gdp3(feats, coords)
        feats = self.sfa3(feats)

        code = self.pool(feats.transpose(1, 2)).squeeze(-1)
        return self.decoder(code).view(-1, self.out_pts, 3)


# ---------- geometry / topology metrics ----------
def chamfer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Squared L2 bidirectional Chamfer Distance."""
    d = torch.cdist(a, b, p=2) ** 2
    return d.min(-1)[0].mean() + d.min(-2)[0].mean()


def diag_h1(pts_np: np.ndarray) -> np.ndarray:
    if pts_np.shape[0] > TOPO_SAMPLE:
        pts_np = pts_np[
            np.random.choice(pts_np.shape[0], TOPO_SAMPLE, replace=False)
        ]
    st = gudhi.RipsComplex(points=pts_np, max_edge_length=1.0).create_simplex_tree(
        max_dimension=1
    )
    st.persistence()
    return np.array(st.persistence_intervals_in_dimension(1))


def bottleneck(d1: np.ndarray, d2: np.ndarray) -> float:
    if len(d1) == 0 or len(d2) == 0:
        return 1.0
    return gudhi.bottleneck_distance(d1, d2)


def fscore(
    pred: torch.Tensor,
    gt: torch.Tensor,
    thr: float = FSCORE_TH,
) -> float:
    """F-score @ threshold."""
    d1 = torch.cdist(pred, gt, p=2)
    d2 = d1.transpose(1, 2)
    precision = (d1.min(-1)[0] < thr).float().mean(1)
    recall = (d2.min(-1)[0] < thr).float().mean(1)
    return (
        2 * precision * recall / (precision + recall + 1e-8)
    ).mean().item()


# ---------- evaluation ----------
@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader):
    """Return sample-averaged CD, HoleCD, H1 bottleneck, and F-score."""
    model.eval()
    tot_cd = tot_hole = tot_h1 = tot_hole_pts = 0.0
    tot_fs = 0.0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)
        pred = model(x)
        B = x.size(0)

        tot_cd += chamfer(pred, y).item() * B

        d_gt_in = torch.cdist(y, x).min(-1)[0]
        masks = d_gt_in > HOLE_THRESH
        for b in range(B):
            idxs = torch.nonzero(masks[b], as_tuple=False)[:, 0]
            nh = idxs.numel()
            if nh:
                if nh > MAX_HOLE_PTS:
                    perm = torch.randperm(nh, device=idxs.device)[:MAX_HOLE_PTS]
                    idxs = idxs[perm]
                    nh = MAX_HOLE_PTS
                tot_hole += (
                    chamfer(
                        pred[b].unsqueeze(0),
                        y[b][idxs].unsqueeze(0),
                    ).item()
                    * nh
                )
                tot_hole_pts += nh

            h1_pred = diag_h1(pred[b].cpu().numpy())
            h1_gt = diag_h1(y[b].cpu().numpy())
            tot_h1 += bottleneck(h1_pred, h1_gt)

        n_pred = min(2048, pred.size(1))
        n_gt = min(2048, y.size(1))
        idx_pred = torch.randperm(pred.size(1), device=pred.device)[:n_pred]
        idx_gt = torch.randperm(y.size(1), device=y.device)[:n_gt]
        tot_fs += fscore(pred[:, idx_pred], y[:, idx_gt]) * B

    n = len(loader.dataset)
    cd = tot_cd / n
    hole = tot_hole / tot_hole_pts if tot_hole_pts else 0.0
    h1 = tot_h1 / n
    fs = tot_fs / n
    return cd, hole, h1, fs


# ---------- visualisation ----------
@torch.no_grad()
def visualize(model: nn.Module, ds: Dataset, idx: int = 0):
    """Scatter-plot input / prediction / hole-GT."""
    x, y = ds[idx]
    inp = x.numpy()
    pred = model(x.unsqueeze(0).to(DEVICE)).cpu().squeeze(0).numpy()
    d = np.linalg.norm(y.numpy()[:, None] - inp[None], axis=-1)
    holes = y.numpy()[d.min(1) > HOLE_THRESH]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(inp[:, 0], inp[:, 1], inp[:, 2], s=5, c="red", label="Input")
    ax.scatter(pred[:, 0], pred[:, 1], pred[:, 2], s=2, c="green", label="Pred")
    ax.scatter(holes[:, 0], holes[:, 1], holes[:, 2], s=8, c="blue", label="Hole GT")
    ax.set_title("PointAttN Completion")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, "topo_completion.png"))
    plt.close(fig)


# ---------- early stopping ----------
class EarlyStopping:
    def __init__(self, patience: int = PATIENCE):
        self.patience = patience
        self.best = None
        self.cnt = 0
        self.early = False

    def __call__(self, metric: float):
        if self.best is None or metric < self.best:
            self.best = metric
            self.cnt = 0
        else:
            self.cnt += 1
            if self.cnt >= self.patience:
                self.early = True


# ---------- training ----------
def main():
    pairs = collect_pairs(DROPOUT_DIR, CLEAN_DIR)
    train_pairs, val_pairs = train_test_split(
        pairs, test_size=0.2, random_state=SEED
    )

    tr_loader = DataLoader(
        PlaneDS(train_pairs, INPUT_NPTS, True),
        BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        PlaneDS(val_pairs, INPUT_NPTS, False),
        BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    model = PointAttN().to(DEVICE)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=BASE_LR,
        weight_decay=WEIGHT_DECAY,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        opt, T_0=EPOCHS
    )
    scaler = torch.cuda.amp.GradScaler()
    stopper = EarlyStopping()
    best_cd = float("inf")

    for ep in range(1, EPOCHS + 1):
        model.train()
        for x, y in tqdm(tr_loader, desc=f"Epoch {ep}/{EPOCHS}"):
            x, y = x.to(DEVICE), y.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                pred = model(x)
                loss = chamfer(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), CLIP_NORM)
            scaler.step(opt)
            scaler.update()
        sched.step()

        if ep % EVAL_FREQ == 0 or ep == EPOCHS:
            cd, hcd, h1, fs = evaluate(model, val_loader)
            print(
                f"Val CD={cd:.4f} HoleCD={hcd:.4f} "
                f"H1={h1:.4f} F={fs:.4f}"
            )
            if cd < best_cd:
                best_cd = cd
                torch.save(
                    model.state_dict(),
                    os.path.join(SAVE_DIR, "best_model_pointattn.pth"),
                )
                print("↳ Best model saved")
            stopper(cd)
            if stopper.early:
                print(f"Early stopping at epoch {ep}")
                break

    model.load_state_dict(
        torch.load(
            os.path.join(SAVE_DIR, "best_model_pointattn.pth"),
            map_location=DEVICE,
        )
    )
    visualize(model, val_loader.dataset)


if __name__ == "__main__":
    main()
