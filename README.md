# Topology-Aware Point Cloud Completion

This repository explores topology-aware 3D point-cloud completion for structural defects such as holes. It includes three training pipelines:

- **PCN baseline** — geometry-only Point Completion Network-style baseline.
- **Topo-PCN** — augments the PCN latent representation with a 3D topology feature vector derived from persistent-homology annotations.
- **PointAttN comparison** — attention-based completion baseline evaluated with the same geometric and topology-aware metrics.

## What is topology-aware here?

Topo-PCN uses the 3D topology vector as an actual network input: the vector is projected through a learned layer and concatenated with the geometric latent code before decoding.

Training is driven by differentiable geometric reconstruction objectives such as Chamfer distance. Persistent-homology **H1 bottleneck distance** is computed with GUDHI as a structural evaluation/monitoring metric. Because that computation is performed outside the PyTorch autograd graph, it is **not used as a differentiable topology loss** in the current implementation.

This separation keeps the project technically clear: topology enters Topo-PCN through explicit features, while persistent-homology distance measures how well completed geometry preserves structural characteristics.

## Dataset format

The scripts expect paired clean and corrupted point-cloud files. A convenient repository-relative layout is:

```text
data/
└── air_plane_/
    ├── clean_with_holes/
    │   ├── sample_000.txt
    │   ├── sample_001.txt
    │   └── ...
    └── dropout_local_0_with_holes/
        ├── sample_000.txt
        ├── sample_001.txt
        └── ...
```

Each `.txt` file uses:

1. a header line,
2. a line containing three topology features (Topo-PCN uses these; other models skip the line),
3. point coordinates `x y z`, one point per line.

The repository also contains the original `data.zip` artifact. It is intentionally left unchanged; you can extract or reorganize it into the layout above as needed.

## Installation

```bash
pip install -r requirements.txt
```

PyTorch installation can vary by CUDA/toolchain, so for GPU environments use the appropriate PyTorch build for your system.

## Path configuration

The scripts no longer contain machine-specific absolute paths. By default they read:

```text
data/air_plane_/clean_with_holes/
data/air_plane_/dropout_local_0_with_holes/
```

You can override paths without editing source code:

```bash
export TOPO_CLEAN_DIR=/path/to/clean_with_holes
export TOPO_DROPOUT_DIR=/path/to/dropout_local_0_with_holes
```

Or set a shared dataset root:

```bash
export TOPO_DATA_ROOT=/path/to/air_plane_
```

Model outputs are written under `runs/` by default.

## Running

```bash
python pcn.py
python topo-pcn.py
python pointatt.py
```

### `pcn.py`
Geometry-only PCN-style baseline with coarse/fine Chamfer objectives, hole-region evaluation, F-score, and early stopping.

### `topo-pcn.py`
PCN-style model augmented with a learned projection of the 3D topology feature vector. Validation reports coarse/fine Chamfer distance, hole-region error, H1 bottleneck distance, and F-score.

### `pointatt.py`
Attention-based comparison model trained with Chamfer distance and evaluated with the same geometry/topology metrics.

## Method scope

The project investigates whether explicit topology descriptors improve completion when geometric defects alter connectivity or hole structure. The current implementation should be interpreted as **topology-feature-augmented learning with topology-aware evaluation**, rather than as a differentiable persistent-homology-loss framework.

## Acknowledgement

This project was developed as part of COM S 6720: Advanced Topics in Artificial Intelligence at Iowa State University by:

- Daniel Ye
- Shakiba Khourashahi
- Ilia Jahanshahi
