# tp-vwgan

This repository contains the implementation and supplementary materials related to an unpublished scientific paper. The paper is currently under review. Detailed descriptions of the methods, experiments, and results can be found in the paper upon its publication.

## Environment

- VS Code
- Python 3.12
- Libraries in `requirements.txt` in all folders.
- Nvidia Graphics Card (Supports CUDA)

# Git

To clone the repo with submodules, run this command in the terminal: `git clone --recursive -j8 https://github.com/aalaa-sehsah/tp-vwgan tp-vwgan`

# Steps

Follow the following steps to successfully train the model.

## Step #0: Install Python Requirements

Run `0-install-requirements.sh`

## Step #1: Download PDB Database

Run `1-download-database.sh` or follow `pdb-database/readme.md`

## Step #2: Create PDB Datasets

Run `2-create-datasets` or follow `pdb-dataset/readme.md`

## Step #3: Train TP-VWGAN Model

Run `3-train-models.sh` or follow `TP-VWGAN/readme.md`

# Note

If you need the trained models, contact me at `aalaa.sehsah@gmail.com`
