#!/usr/bin/env bash
set -euo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"

conda env create -f environment.yml || true
conda activate SemanticGHGS

pip install -r requirements.txt
pip install ./submodules/diff-gaussian-rasterization/