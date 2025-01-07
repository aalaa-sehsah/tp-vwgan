# vae-wgan-gp-128

We train a VAE-WGAN-GP model using generated distance matrices (128x128). 

## Environment

- VS Code
- Python 3.12
- Libraries in `requirements.txt`
- Nvidia Graphics Card (Supports CUDA)

## Files and Their Functions

- `requirements.txt`
  - Contains the required Python libraries.
  - USER: Run `python -m pip install -r requirements.txt` in terminal.
  - USER: Run `pip install torch torchvision --index-url https://download.pytorch.org/whl/torch_stable.html`
- `train.py`
  - Trains VAE-WGAN-GP-128 model.
  - Saves model checkpoints (in `models`) and models progress (in `runs`).
