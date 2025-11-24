
# THOR: Transformer based foundation model for Heterogeneous Observation and Resolution
## A Versatile Foundation Model for Earth Observation Climate and Society Applications 

THOR builds on the original USat codebase.


## Setup
Performing the following step in `THOR` directory. 
1. Using `uv` to set up the environment.
```bash
uv sync
```
2. Optional dependencies for development
```bash
uv sync --group dev --extra scripts --extra test
```

## Model Training
### Pretrain THOR architecture on THOR-Pretrain data from scratch
```bash
uv run main.py train thor/config/pretrain/finale/thor-base.yaml
```