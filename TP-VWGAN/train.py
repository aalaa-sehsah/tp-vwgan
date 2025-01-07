import random
from pathlib import Path
import h5py
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.utils as vutils
from torch.utils.tensorboard import SummaryWriter

from models import ConvVAE, Critic, gradient_penalty, loss_function
from termcolor import colored


def log_msg(msg: str, level: str = "info", src="?"):
    color = {
        "info": "green",
        "error": "red",
        "warn": "yellow",
    }.get(level, "yellow")
    print(f"[{colored(level.upper(), color)}] ({src}) {msg}")


class VAE_WGAN_GP:
    def __init__(
        self,
        dataset_path: str,
        res: int,
        batch_size: int,
        epochs: int,
        lr: float,
        manualseed: int = 777,
        start: int | None = None,
        end: int | None = None,
        resume_model_path: str | None = None,
    ) -> None:
        self.dataset_path = dataset_path
        self.res = res
        self.batch_size = batch_size
        self.epochs = epochs
        self.lr = lr
        self.manualseed = manualseed
        self.start = start
        self.end = end
        self.resume_model_path = resume_model_path

        self.writer = SummaryWriter()  # For tracking progress

        self.nz = 512  # Size of latent vector

        self.models_path = Path("models")
        self.models_path.mkdir(exist_ok=True)

        # WGAN hyper-parameters
        self.critic_iterations = 5
        self.lambda_gp = 10

        self._set_random_seed()
        self._set_device()

    def _set_random_seed(self) -> None:
        """Set random seed for reproducibility"""
        random.seed(self.manualseed)
        torch.manual_seed(self.manualseed)

    def _set_device(self) -> None:
        """setting device on GPU if available, else CPU"""
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        log_msg(f"Used Device: {self.device}", src=self._set_device.__name__)

    def _load_dataset(self) -> None:
        log_msg(
            f"Load dataset: '{self.dataset_path}'",
            level="load",
            src=self._load_dataset.__name__,
        )

        try:
            with h5py.File(self.dataset_path, "r") as fp:
                self.data = fp["data"][self.start : self.end]
                self.data_length = self.data.shape[0]
        except Exception as _:
            log_msg("Dataset not found!", "error", src=self._load_dataset.__name__)
            exit(1)

        log_msg("Dataset loaded successfully.", src=self._load_dataset.__name__)

    def _create_fixed_noise(self) -> None:
        """
        Create batch of latent vectors that we will use to visualize
        the progression of the generator
        """
        self.fixed_noise = torch.randn(self.res, self.nz, device=self.device)

    def _weights_init(self, m) -> None:
        """Initial weights of VAE and Critic"""
        classname = m.__class__.__name__
        if classname.find("Conv") != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find("BatchNorm") != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    def _build_models(self) -> None:
        # Create the generator and critic
        self.vae = ConvVAE().to(self.device)
        self.critic = Critic().to(self.device)

        # Apply the weights_init function to randomly initialize all weights
        self.vae.encoder.apply(self._weights_init)
        self.vae.decoder.apply(self._weights_init)
        self.critic.apply(self._weights_init)

        # Setup Adam optimizers for both VAE and Critic
        self.optimizer_vae = optim.Adam(
            self.vae.parameters(), lr=self.lr, betas=(0.5, 0.999)
        )
        self.optimizer_critic = optim.Adam(
            self.critic.parameters(), lr=self.lr, betas=(0.5, 0.999)
        )

        self.optimizer_encoder = optim.Adam(
            self.vae.encoder.parameters(), lr=self.lr, betas=(0.5, 0.999)
        )
        self.optimizer_decoder = optim.Adam(
            self.vae.decoder.parameters(), lr=self.lr, betas=(0.5, 0.999)
        )

        log_msg(
            "Models have been created successfully.",
            src=self._build_models.__name__,
        )

        if self.resume_model_path:
            checkpoint = torch.load(self.resume_model_path)
            self.vae.load_state_dict(checkpoint["vae_state_dict"])
            self.critic.load_state_dict(checkpoint["critic_state_dict"])
            self.optimizer_vae.load_state_dict(checkpoint["optimizerVAE_state_dict"])
            self.optimizer_critic.load_state_dict(checkpoint["optimizerC_state_dict"])

            log_msg(
                "Trained models have been loaded successfully.",
                src=self._build_models.__name__,
            )

    def save_model(self, epoch: int) -> None:
        torch.save(
            {
                "vae_state_dict": self.vae.state_dict(),
                "critic_state_dict": self.critic.state_dict(),
                "optimizer_vae_state_dict": self.optimizer_vae.state_dict(),
                "optimizer_critic_state_dict": self.optimizer_critic.state_dict(),
            },
            self.models_path
            / f"{self.res}_{self.batch_size}_{epoch+1}_{self.lr}_{self.manualseed}.pt",
        )

        log_msg(
            f"({epoch=}) Model has been saved successfully.",
            src=self.save_model.__name__,
        )

    def log_to_tensorboard(
        self,
        total_critic_loss,
        adverserial_loss,
        total_vae_loss,
        rec_loss,
        kl_loss,
        current_data,
        epoch,
        number_batches,
        iters,
        i,
    ) -> None:
        self.writer.add_scalars(
            "loss tracker",
            {
                "Total Critic Loss": total_critic_loss,
                "adverserial/generator Loss": adverserial_loss,
                "Total VAE Loss": total_vae_loss,
                "Reconstruction Loss": rec_loss,
                "KLD Loss": kl_loss,
            },
            iters,
        )

        # Check how the generator is doing by saving G's output on fixed_noise
        if (iters % int((self.data_length / self.batch_size) * 0.1) == 0) or (
            (epoch == self.epochs - 1) and (i == number_batches - 1)
        ):
            with torch.no_grad():
                generated_samples = self.vae.decoder(self.fixed_noise).detach().cuda()

            self.writer.add_image(
                "Generated_samples Decode Output",
                vutils.make_grid(
                    (
                        generated_samples.clamp(min=0)
                        + generated_samples.clamp(min=0).permute(0, 1, 3, 2)
                    )
                    / 2,
                    padding=2,
                    normalize=True,
                ),
                iters,
            )

            with torch.no_grad():
                x_rec, _, _ = self.vae(current_data)

            self.writer.add_image(
                "Reconstruction Decoder Output",
                vutils.make_grid(
                    (x_rec.clamp(min=0) + x_rec.clamp(min=0).permute(0, 1, 3, 2)) / 2,
                    padding=2,
                    normalize=True,
                ),
                iters,
            )

    def train(self) -> None:
        self._create_fixed_noise()
        self._load_dataset()

        dataloader = DataLoader(self.data, batch_size=self.batch_size, shuffle=True)
        number_batches = len(dataloader)

        self._build_models()

        iters = 0  # Training Loop

        log_msg("Training started.", src=self.train.__name__)

        for epoch in tqdm(range(self.epochs), desc="Training Epochs"):
            # For each batch in the dataloader
            for i, self.data in tqdm(
                enumerate(dataloader, 0),
                total=int(self.data_length / self.batch_size),
                desc="Steps",
            ):
                real_data = (self.data.unsqueeze(dim=1).type(torch.FloatTensor)).to(
                    self.device
                )

                current_batch_size = real_data.size(0)

                # Train VAE
                rec, mu, logvar = self.vae(real_data)
                rec_loss, kl_loss = loss_function(rec, real_data, mu, logvar)
                total_vae_loss = rec_loss + kl_loss

                self.vae.zero_grad()
                total_vae_loss.backward()
                self.optimizer_vae.step()

                # Train critic with gradient penalty
                # max - [E[critic(real)] - E[critic(gen_fake)]] + lambda_gp * gp
                for _ in range(self.critic_iterations):
                    noise = torch.randn(current_batch_size, self.nz, device=self.device)
                    fake = self.vae.decoder(noise)

                    critic_real = self.critic(real_data).reshape(-1)
                    critic_fake = self.critic(fake).reshape(-1)

                    # Calculate gradient penalty
                    gp = gradient_penalty(
                        critic=self.critic,
                        real=real_data,
                        fake=fake,
                        device=self.device,
                    )

                    critic_loss = (
                        -(torch.mean(critic_real) - torch.mean(critic_fake))
                        + self.lambda_gp * gp
                    )

                    self.critic.zero_grad()
                    critic_loss.backward(retain_graph=True)
                    self.optimizer_critic.step()

                # Train decoder/generator: min -E[critic(gen_fake)]
                eval_fake = self.critic(fake).reshape(-1)
                adverserial_loss = -torch.mean(eval_fake)

                self.vae.decoder.zero_grad()
                adverserial_loss.backward()
                self.optimizer_decoder.step()

                self.log_to_tensorboard(
                    total_critic_loss=critic_loss.item(),
                    adverserial_loss=adverserial_loss.item(),
                    total_vae_loss=total_vae_loss.item(),
                    rec_loss=rec_loss.item(),
                    kl_loss=kl_loss.item(),
                    current_data=real_data,
                    epoch=epoch,
                    number_batches=number_batches,
                    iters=iters,
                    i=i,
                )

                iters += 1

            self.save_model(epoch)


if __name__ == "__main__":
    import sys
    import os

    dir_ = os.path.dirname(sys.argv[0])
    dir_ and os.chdir(dir_)

    data_path = r"../pdb-dataset/train_dataset/128aa.hdf5"

    model = VAE_WGAN_GP(
        dataset_path=data_path,
        res=128,
        batch_size=64,
        epochs=100,
        lr=0.00001,
        manualseed=777,
        resume_model_path=None,  # To resume from checkpoint
    )

    model.train()
