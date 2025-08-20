from typing import Dict
from omegaconf import DictConfig

import torch
import torch.nn as nn


class EventstVAE(nn.Module):
    """A temporal VAE for processing events data.

    Temporal in the sense that we do not use the VAE to reconstruct the DVS matrix at time t, but the DVA metrix at time t + 1 (given its state at time t), i.e., we force our latents to include temporal information.
    """

    def __init__(self, nets_config: DictConfig):
        super().__init__()
        self.device = torch.device(nets_config["device"])

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
        self.events_convnet = nn.Sequential(*layers).to(self.device)

        self.events_fcnet = nn.Linear(in_dim * H * W, nets_config["events_fc_out"])

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # get output from the event nets
        # unsqueeze to have a (B, in_ch, H, W) tensor
        events_out = self.events_convnet(inputs["event_stack"].unsqueeze(1))

        events_out = torch.flatten(events_out, start_dim=1)
        events_out = self.events_fcnet(events_out)

        return events_out
