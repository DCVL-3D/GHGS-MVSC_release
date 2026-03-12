# SemanticGHGS (Release)

SemanticGHGS is a feed-forward pipeline for human rendering / reconstruction based on Gaussian Splatting-style primitives and a geometry-aware transformer backbone.

> **Status**: Research release (training + evaluation scripts included)

---

## Highlights
- **CUDA 11.8 / PyTorch 2.0.1** tested
- Supports **THuman2.0** preprocessing + rendering-based data preparation
- Training / evaluation scripts for reproduction

---

## Environment

### Tested setup
- OS: Ubuntu 20.04/22.04
- GPU: NVIDIA (tested on RTX 3090 / 4090)
- CUDA: **11.8**
- PyTorch: **2.0.1**
- torchvision: **0.15.2**
- torchaudio: **2.0.2**
- PyTorch3D: **0.7.7** (via conda)

> If you use a different CUDA/PyTorch combo, you may need to rebuild CUDA extensions (e.g., rasterizer).

---

## Installation

### 1) Clone this repository
```bash
git clone https://github.com/jingi0614/SemanticGHGS_release.git
cd SemanticGHGS_release
```

### 2) Install dependencies (recommended)
We provide a one-shot installer:
```bash
bash install.sh
```

This will:
- create a conda env (`SemanticGHGS`)
- install PyTorch(+CUDA), PyTorch3D, iopath
- install `requirements.txt`
- install `./submodules/diff-gaussian-rasterization/`

> If you prefer manual installation, see **Manual Install** below.

### 3) VGGT dependency
This project relies on VGGT. Clone it as follows:
```bash
# from the repo root
git clone <VGGT_REPO_URL> vggt
```

Make sure your code can import VGGT (e.g., by keeping `vggt/` under the project root, or by setting `PYTHONPATH`).

---

## Dataset: THuman2.0

### 1) Download
Download THuman2.0 from the official source and place it under a dataset directory.

Recommended structure:
```text
datasets/
  THuman2.0/
    raw/            # downloaded data
    processed/      # outputs from preprocessing
```

> Notes:
> - You may need to request access depending on the dataset policy.
> - Ensure file permissions and paths are correct.

---

## Preprocessing (Rendering-based)

We provide preprocessing scripts that render / prepare training inputs.

### 1) Configure paths
Edit config(s) in:
- `config/` (e.g., `config_thu.yaml`)

Make sure dataset root paths are correct.

### 2) Run preprocessing
Example:
```bash
python prepare_data/render_data.py \
  --config config/config_thu.yaml
```

Outputs (example):
```text
datasets/THuman2.0/processed/
  images/
  masks/
  cameras/
  ...
```

---

## Training

### 1) Configure training
Check the main config:
- `config/config_thu.yaml`

Set:
- dataset paths
- batch size / num views
- output directory (`experiments/...`)
- checkpoints saving schedule

### 2) Run training
```bash
python train.py --config config/config_thu.yaml
```

Checkpoints will be saved to:
```text
experiments/<exp_name>/ckpt/
```

---

## Evaluation / Inference

### 1) Run inference
```bash
python test.py --config config/config_thu.yaml --ckpt <PATH_TO_CKPT>
```

Outputs:
```text
experiments/<exp_name>/test/
  *.png / *.jpg
  metrics.json
```

---

## Project Structure

```text
SemanticGHGS_release/
  config/                      # configs
  gaussian_renderer/           # renderer modules
  lib/                         # datasets, networks, utils
  prepare_data/                # preprocessing scripts
  submodules/
    diff-gaussian-rasterization/  # CUDA extension
  vggt/                        # external dependency (clone separately)
  install.sh
  requirements.txt
  train.py
  test.py
```

---

## Manual Install (optional)

If you prefer installing step-by-step (instead of `install.sh`), you can follow:
```bash
conda env create -f environment.yml
conda activate SemanticGHGS
pip install -r requirements.txt
pip install ./submodules/diff-gaussian-rasterization/
```

---

## Troubleshooting

### Build / CUDA extension issues
If you see errors like:
- `CUDA_HOME not set`
- `nvcc not found`
- `undefined symbol` / ABI mismatch
- torch-cuda version mismatch

Try:
1) verify CUDA toolkit and driver versions
2) ensure PyTorch CUDA version matches your runtime
3) clean and reinstall extension:
```bash
pip uninstall -y diff-gaussian-rasterization
pip install ./submodules/diff-gaussian-rasterization/ --no-build-isolation
```

### PyTorch3D installation
We recommend installing PyTorch3D via conda (already included in `install.sh`).
If solver fails, ensure channels include `pytorch3d`, `pytorch`, and `nvidia`.

---

## License
This repository is released for research use. Third-party code may have its own licenses under `submodules/` and `vggt/`.

---

## Citation
If you use this codebase, please cite:
```bibtex
@misc{SemanticGHGS,
  title={SemanticGHGS},
  author={...},
  year={2026},
  howpublished={\url{https://github.com/jingi0614/SemanticGHGS_release}}
}
```

---

## Contact
- Jingi Kim (김진기): <YOUR_EMAIL>
