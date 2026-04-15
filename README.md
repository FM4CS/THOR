
# THOR: Transformer based foundation model for Heterogeneous Observation and Resolution
## A Versatile Foundation Model for Earth Observation Climate and Society Applications 

[![Website](https://img.shields.io/badge/Website-THOR-0F62FE)](https://thor-model.notion.site/THOR-Foundation-Model-Showcase-2ee64c7f3cb78087bf77feb6350bdcc6)
[![arXiv](https://img.shields.io/badge/arXiv-2601.16011-b31b1b?logo=arxiv)](https://arxiv.org/abs/2601.16011)
[![Models](https://img.shields.io/badge/Models-HuggingFace-FFD21E?logo=huggingface)](https://huggingface.co/FM4CS)
[![TerraTorch Extension](https://img.shields.io/badge/TerraTorch-Extension-EE4B2B?logo=github)](https://github.com/FM4CS/thor_terratorch_ext)
[![Dataset](https://img.shields.io/badge/Dataset-HuggingFace-FFD21E?logo=huggingface)](https://huggingface.co/datasets/FM4CS/THOR-Pretrain)

THOR is a compute-adaptive geospatial foundation model developed by Norwegian Computing Center (NR), UiT The Arctic University of Norway and ESA Φ-lab.

## Key Features

- **Multi-sensor support**: Sentinel-1 (SAR), Sentinel-2 (MSI), Sentinel-3 OLCI & SLSTR
- **Flexible resolution**: 10 m to 1000 m native resolutions
- **Compute-adaptive**: Flexible patch sizes and ground covers (1000 m to +100,000 m)
- **Data-efficient**: State-of-the-art performance in data-limited regimes
- **Model type**: Vision Transformer (FlexiViT)

## Model Description

THOR unifies data from Copernicus Sentinel-1, -2, and -3 (OLCI & SLSTR) satellites, processing their native 10 m to 1000 m resolutions in a single model. THOR is pre-trained with a novel randomized patch and input image size strategy, allowing deployment at inference with any patch size for dynamic trade-offs between computational cost and feature resolution without retraining.

## Setup

1. Using `uv` to set up the environment:
```bash
uv sync
```

2. Optional dependencies for development:
```bash
uv sync --group dev --extra scripts --extra test
```

## Usage

For downstream applications, we recommend using the [terratorch](https://github.com/terrastackai/terratorch) framework with our [THOR terratorch extension](https://github.com/FM4CS/thor_terratorch_ext).

### Terratorch backbone loading example

```python
# Example usage of THOR ViT backbone with terratorch

# Import our custom thor_terratorch_ext module to register THOR backbones
import thor_terratorch_ext  # noqa: F401

# Load the backbone registry
from terratorch import BACKBONE_REGISTRY

# List available THOR backbones
print([b for b in list(BACKBONE_REGISTRY) if "thor" in b])

# Build a THOR ViT model with specific bands
model = BACKBONE_REGISTRY.build(
    "thor_v1_tiny",
    pretrained=True,
    model_bands=["BLUE", "GREEN", "RED", "VV", "VH"],
    input_params=dict(  # Optional input parameters to customize
        ground_covers=[
            2880
        ],  # Ground cover in meters (typically input image size [px] * input image resolution)
        flexivit_patch_size_seqs=[8],  # Patch size in pixels
    ),
)
```

## Memory-efficient inference with FlexAttention

THOR uses 2D ALiBi (Attention with Linear Biases) for position encoding, which normally requires materializing an N×N bias matrix per attention head. For large token counts this becomes the dominant memory cost.

At inference time THOR can instead use PyTorch's [`flex_attention`](https://pytorch.org/docs/stable/nn.attention.flex_attention.html) with a compact representation that avoids allocating that matrix entirely. Rather than storing the full `(H, N, N)` bias tensor, only the 1D x/y coordinate vectors of shape `(N,)` are kept alongside the head slopes. The ALiBi bias for any query–key pair is then computed on-the-fly inside the attention kernel via a `score_mod` closure.

This reduces the position-encoding memory footprint from O(H·N²) to O(H + N).

### Requirements

- PyTorch ≥ 2.6

### Enabling

```bash
USE_FLEX_ATTENTION=1 python your_script.py
```

Flex attention is **off by default** and only activates during eval (training still uses the standard dense ALiBi path with SDPA). When enabled, the attention kernel is compiled once with `torch.compile` for additional throughput. Compilation can be disabled by setting `COMPILE_FLEX_ATTENTION=0`.

## Model Training

### Pretrain THOR architecture on THOR-Pretrain data from scratch
```bash
uv run main.py train thor/config/pretrain/final/thor-base.yaml
```

## Evaluation

THOR demonstrates highly competitive performance on the PANGAEA benchmark, particularly in data-limited regimes. With only 10% training data, THOR-Base achieves the best average rank across all datasets.

| Model | HLS Burns | MADOS | PASTIS | Sen1Floods11 | FBP | DynEarthNet | CropMap | SN7 | AI4Farms |
|-------|-----------|-------|--------|--------------|-----|-------------|---------|-----|----------|
| CROMA | 76.44 | 32.44 | 32.80 | *87.22* | 37.39 | 36.08 | 36.77 | 42.15 | 38.48 |
| DOFA | 71.98 | 23.77 | 27.68 | 82.84 | 27.82 | **39.15** | 29.91 | 46.10 | 27.74 |
| Prithvi | 77.73 | 21.24 | 33.56 | 86.28 | 29.98 | 32.28 | 27.71 | 36.78 | 35.04 |
| SpectralGPT | **83.35** | 20.29 | 34.53 | 83.12 | 39.51 | 35.33 | 31.06 | 36.31 | 37.35 |
| Terramind-B | 77.39 | **44.06** | **39.96** | 84.43 | *54.00* | *37.35* | 35.65 | 43.21 | 38.59 |
| UNet Baseline | *79.46* | 24.30 | 29.53 | **88.55** | 52.58 | 35.59 | 13.88 | 46.08 | 34.84 |
| ViT Baseline | 75.92 | 10.18 | 38.44 | 81.85 | **56.53** | 35.39 | 27.76 | 36.01 | **39.20** |
| THOR-B | 76.90 | 40.67 | *38.93* | 86.29 | 42.80 | 35.21 | **42.23** | *55.94* | *38.90* |
| THOR-T | 75.98 | *41.65* | 36.26 | 82.70 | 42.81 | 34.03 | *37.82* | **58.52** | 38.56 |

*Results in mIoU on PANGAEA benchmark with 10% training data. **Bold** = best, *italic* = second-best.*

## Attribution

The development of THOR was funded and supported by European Space Agency (ESA) Φ-lab (FM4CS project, contract no. 4000143489/24/I-DT), and the Research Council of Norway (KnowEarth project no. 337481).

## Citation

If you use THOR in your research, please cite the [paper](https://arxiv.org/abs/2601.16011):

```bibtex
@article{forgaard2026thor,
      title={THOR: A Versatile Foundation Model for Earth Observation Climate and Society Applications}, 
      author={Theodor Forgaard and Jarle H. Reksten and Anders U. Waldeland and Valerio Marsocci and Nicolas Longépé and Michael Kampffmeyer and Arnt-Børre Salberg},
      year={2026},
      eprint={2601.16011},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2601.16011}, 
}
```

## License

THOR is released under the MIT License.

This project is based in part on code from the Stanford Machine Learning Group,
also licensed under MIT.