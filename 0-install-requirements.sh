#!/bin/bash

pip install -r pdb-dataset/requirements.txt
pip install -r vae-wgan-gp-128/requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/torch_stable.html

read -rsp 'Press <Enter> to exit...'