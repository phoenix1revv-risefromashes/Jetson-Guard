#!/usr/bin/env bash

set -e

python3 -m venv .venv --system-site-packages
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements_jetson.txt

python -m pip install torch==2.8.0 torchvision==0.23.0 \
  --index-url=https://pypi.jetson-ai-lab.io/jp6/cu126

python -m pip install ultralytics --no-deps

echo "Jetson environment install complete."
echo "Remember to set:"
echo "export PYTHONNOUSERSITE=1"
echo "export LD_LIBRARY_PATH=/usr/local/cuda-12.6/extras/CUPTI/lib64:/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
