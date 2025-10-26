from typing import Dict
from omegaconf import DictConfig

import torch
import torch.nn as nn


class EventstVAE(nn.Module):
    """A temporal VAE for processing events data.

    Temporal in the sense that we do not use the VAE to reconstruct the DVS matrix at time t, but the DVA metrix at time t + 1 (given its state at time t), i.e., we force our latents to include temporal information.
    """

    # TODO currently, the VAE predicts the events at time t! should predict the events at time t+1, as in the docstring above

    def __init__(self, nets_config: DictConfig):
        super().__init__()
        # movement to the device is handled internally through nets_config
        self.encoder = EventsEncoder(nets_config)
        self.decoder = EventsDecoder(
            nets_config, self.encoder.final_H, self.encoder.final_W, self.conv_outdim
        )

    def forward(self, x):
        z, z_mean, z_logvar = self.encoder(x)
        reconstruction = self.decoder(z)
        # return the latent as well, to pass it to the other nets
        return reconstruction, z, z_mean, z_logvar


class EventsEncoder(nn.Module):
    """The encoder for the events t-VAE."""

    def __init__(self, nets_config: DictConfig):
        super().__init__()
        device = torch.device(nets_config["device"])

        # construct the nets for the event stack
        Kc, s, p, d, Km = (
            nets_config["conv_ker_size"],
            nets_config["conv_stride"],
            nets_config["conv_padding"],
            nets_config["conv_dilation"],
            nets_config["pool_ker_size"],
        )

        layers = []
        in_dim = 1  # the number of input channels
        H, W = 200, 200  # initial height and width of the frames
        for hidden_dim in nets_config["conv_dims"]:
            layers.append(
                nn.Conv2d(
                    in_dim, hidden_dim, kernel_size=Kc, stride=s, padding=p, dilation=d
                )
            )
            # adjust the height and width (ignore the depth because it varies)
            # (from https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)
            H = (H + 2 * p - d * (Kc - 1) - 1 + s) // s
            W = (W + 2 * p - d * (Kc - 1) - 1 + s) // s
            layers.append(nn.SiLU())
            layers.append(nn.MaxPool2d(kernel_size=Km))
            # adjust the height and width (ignore the depth because it varies)
            # (from https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html: the formula below is simpler because we use default values for padding and dilation)
            H, W = H // Km, W // Km
            in_dim = hidden_dim  # update the input channels
        self.events_convnet = nn.Sequential(*layers).to(device)

        self.final_H, self.final_W, self.conv_outdim = (
            H,
            W,
            in_dim,
        )  # useful for the decoder

        # use same sizes for both nets for convenience
        self.events_fcnet_mu = nn.Linear(
            in_dim * H * W, nets_config["events_fc_out"]
        ).to(device)
        self.events_fcnet_logsigma = nn.Linear(
            in_dim * H * W, nets_config["events_fc_out"]
        ).to(device)

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # get output from the event nets
        # unsqueeze to have a (B, in_ch, H, W) tensor
        events_out = self.events_convnet(inputs["event_stack"].unsqueeze(1))

        z_mean = self.events_fcnet_mu(events_out)
        z_logvar = self.events_fcnet_logsigma(events_out)
        z = self._reparametrise(z_mean, z_logvar)

        return z, z_mean, z_logvar

    def _reparametrise(
        self, z_mean: torch.Tensor, z_logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample from the latent space using the reparametrisation trick."""
        batch, dim = z_mean.shape
        eps = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_logvar) * eps


class EventsDecoder(nn.Module):
    """The decoder for the events t-VAE."""

    # TODO

    def __init__(self, nets_config: DictConfig, initial_H, initial_W, initial_indim):
        super().__init__()
        device = torch.device(nets_config["device"])

        # construct the nets for the event stack
        Kc, s, p, d, Km = (
            nets_config["conv_ker_size"],
            nets_config["conv_stride"],
            nets_config["conv_padding"],
            nets_config["conv_dilation"],
            nets_config["pool_ker_size"],
        )

        # invert final FC layer
        self.d_fcnet = nn.Linear(
            nets_config["events_fc_out"], initial_indim * initial_H * initial_W
        ).to(device)

        layers = []
        in_dim = initial_indim  # the number of input channels (from the encoder)
        H, W = (
            initial_H,
            initial_W,
        )  # initial height and width of the frames (from the encoder)
        # ==============================
        # TODO THIS INVERSE PASS THROUGH THE CONVNET NEEDS TO BE CHECKED!
        # ==============================
        for hidden_dim in reversed(nets_config["conv_dims"]):
            layers.append(nn.MaxPool2d(kernel_size=Km))
            # adjust the height and width (ignore the depth because it varies)
            # (from https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxUnpool2d.html)
            H, W = (
                H // Km,
                W // Km,
            )  # TODO i shouldn't simplify this formula => it's not generic! adjust in idea_one as well
            layers.append(nn.SiLU())
            layers.append(
                nn.Conv2d(
                    in_dim, hidden_dim, kernel_size=Kc, stride=s, padding=p, dilation=d
                )
            )
            # adjust the height and width (ignore the depth because it varies)
            # (from https://docs.pytorch.org/docs/stable/generated/torch.nn.ConvTranspose2d.html)
            H = (H + 2 * p - d * (Kc - 1) - 1 + s) // s
            W = (W + 2 * p - d * (Kc - 1) - 1 + s) // s
            in_dim = hidden_dim  # update the input channels
        # invert convolutional layers
        self.d_convnet = nn.Sequential(*layers).to(device)

    def forward(self, z) -> torch.Tensor:
        z = self.d_fcnet(z)
        # unflatten to have a (B, final_in_ch, final_H, final_W) tensor
        z = z.view(-1, self.initial_indim, self.initial_H, self.initial_W)
        events_out = self.d_convnet(z)
        return events_out
