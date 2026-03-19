<h1 align="center">Generalizable Human Gaussian Splatting<br>via Multi-view Semantic Consistency</h1>

Official PyTorch implementation of **"Generalizable Human Gaussian Splatting via Multi-view Semantic Consistency"**  
🎸 *CVPR 2026 Findings* 🎸

<p align="center">
  <!-- Replace with your teaser figures -->
  <!-- <img src="figures/teaser.gif" width="90%"/> -->
</p>

---

## :eyes: Overview
This repository provides the training and evaluation code for **generalizable human Gaussian splatting** from sparse multi-view inputs.

Our method:
- encodes multi-view inputs with a **pre-trained VGGT** backbone,
- predicts per-view depth maps and **unprojects latent embeddings into a shared 3D space**,
- recalibrates spatially adjacent embeddings with **cross-view attention weighted by semantic consistency**,
- regresses Gaussian attributes and renders novel views with a rasterizer-based pipeline.

---

## 📦 Installation

### 1) Clone
```bash
git clone https://github.com/jingi0614/SemanticGHGS_release.git
cd SemanticGHGS_release
```

### 2) Create environment + install dependencies
We provide an installer:
```bash
bash install.sh
```

This will:
- create a conda env (`SemanticGHGS`, Python 3.8)
- install **PyTorch 2.0.1 + CUDA 11.8**, PyTorch3D, iopath
- install Python deps via `requirements.txt`
- build/install `./submodules/diff-gaussian-rasterization/`
---

## 🗂️ Dataset (THuman2.0)

### Download
Download **THuman2.0** from the official source (access may be required).

Recommended structure:
```text
datasets/
  THuman2.0/
    raw/
    processed/
```

---

## 🧪 Training

1) Set dataset paths in the config (example):
- `config/config_thu.yaml`

2) Run training:
```bash
python train.py --config config/config_thu.yaml
```

Checkpoints are saved under:
```text
experiments/<exp_name>/ckpt/
```

---

## 🎬 Evaluation / Inference

```bash
python test.py --config config/config_thu.yaml --ckpt <PATH_TO_CKPT>
```

Outputs (example):
```text
experiments/<exp_name>/test/
  *.jpg / *.png
  result.json
```

---

## License
This repository is released for research use.  
Third-party components under `submodules/` and `vggt/` may have their own licenses—please check them before use.

---

## Citation
If you use this codebase, please cite our paper:
```bibtex
@inproceedings{SemanticGHGS2026,
  title     = {Generalizable Human Gaussian Splatting via Multi-view Semantic Consistency},
  author    = {Jingi Kim and ...},
  booktitle = {CVPR (Findings)},
  year      = {2026}
}
```
