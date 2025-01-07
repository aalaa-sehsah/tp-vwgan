#!/bin/bash

python3 -m pip install -r pdb-dataset/requirements.txt
python3 -m pip install -r TP-VWGAN/requirements.txt
python3 -m pip install torch torchvision --index-url https://download.pytorch.org/whl/torch_stable.html

read -rsp 'Press <Enter> to exit...'