# Topology-Aware Point Cloud Completion

A 3D point-cloud completion project that augments geometric reconstruction with **persistent-homology topology descriptors** to better represent structural defects such as holes and missing regions.

The repository compares three completion pipelines under the same benchmark and evaluation setup:

- **PCN baseline** — geometry-only Point Completion Network-style reconstruction.
- **Topo-PCN** — injects a learned projection of a 3D topology descriptor into the PCN latent representation before decoding.
- **PointAttN comparison** — attention-based completion model evaluated with the same geometric and topology-aware metrics.

## Topo-PCN

Topo-PCN extends a PCN-style encoder-decoder with explicit topological information derived from persistent-homology annotations. The topology vector is projected through a learned layer and fused with the geometric latent representation, giving the decoder access to both local/global geometry and structural information.

The pipeline combines:

- PointNet-style geometric encoding
- coarse-to-fine point-cloud decoding
- learned topology-feature fusion
- Chamfer-distance reconstruction objectives
- hole-region error analysis
- persistent-homology H1 bottleneck evaluation
- F-score and early stopping

This setup makes it possible to study whether explicit structural descriptors help completion models recover geometry when corruption changes hole/connectivity structure rather than only removing random points.

## Dataset format

The scripts expect paired clean and corrupted point-cloud files:

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

Each `.txt` file contains:

1. a header line,
2. three topology features,
3. point coordinates `x y z`, one point per line.

The repository also includes the original `data.zip` artifact.

## Installation

```bash
pip install -r requirements.txt
```

For GPU environments, install the PyTorch build appropriate for the local CUDA/toolchain.

## Path configuration

By default the scripts read:

```text
data/air_plane_/clean_with_holes/
data/air_plane_/dropout_local_0_with_holes/
```

Paths can be overridden without editing source code:

```bash
export TOPO_CLEAN_DIR=/path/to/clean_with_holes
export TOPO_DROPOUT_DIR=/path/to/dropout_local_0_with_holes
```

or with a shared root:

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
Geometry-only PCN-style baseline with coarse/fine Chamfer reconstruction, hole-region evaluation, F-score, and early stopping.

### `topo-pcn.py`
Topology-augmented PCN model with learned fusion of the 3D topology descriptor. Validation reports geometric reconstruction metrics together with H1 bottleneck distance.

### `pointatt.py`
Attention-based comparison model evaluated under the same geometry/topology protocol.

## Topology-aware evaluation

Persistent-homology H1 bottleneck distance is computed with GUDHI to quantify structural similarity between prediction and target. This complements pointwise/geometric metrics by tracking whether the reconstructed point cloud preserves higher-level hole structure.

## Implementation note

The GUDHI bottleneck calculation is performed outside the PyTorch autograd graph, so in the current implementation it serves as a topology-aware evaluation/monitoring metric rather than a differentiable gradient term. Topological information still enters Topo-PCN directly through the learned topology-feature branch.

## Acknowledgement

Developed as part of COM S 6720: Advanced Topics in Artificial Intelligence at Iowa State University by:

- Daniel Ye
- Shakiba Khourashahi
- Ilia Jahanshahi
