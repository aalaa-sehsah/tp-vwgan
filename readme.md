# tp-vwgan

## Abstract

Elucidating the tertiary structure of proteins is important for understanding their functions and interactions. While deep neural networks have advanced the prediction of a protein’s native structure from its amino acid sequence, the focus on a single-structure view limits understanding of the dynamic nature of protein molecules. Acquiring a multi-structure view of protein molecules remains a broader challenge in computational structural biology. Alternative representations, such as distance matrices, offer a compact and effective way to explore and generate realistic tertiary protein structures. This paper presents TP-VWGAN, a hybrid model to improve the realism of generating distance matrix representations of tertiary protein structures. The model integrates the probabilistic representation learning of the Variational Autoencoder (VAE) with the realistic data generation strength of the Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP). The main modification of TP-VWGAN is incorporating residual blocks into its VAE architecture to improve its performance. The experimental results show that TP-VWGAN with and without residual blocks outperforms existing methods in generating realistic protein structures, but incorporating residual blocks enhances its ability to capture key structural features. Comparisons also demonstrate that the more accurately a model learns symmetry features in the generated distance matrices, the better it captures key structural features, as demonstrated through benchmarking against existing methods. This work moves us closer to more advanced deep generative models that can explore a broader range of protein structures and be applied to drug design and protein engineering. 

## Model

![model](https://github.com/user-attachments/assets/73f3b083-5358-4cee-9d85-2f6a92184502)


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

If you need the trained models files, contact me at `aalaa.sehsah@gmail.com`
