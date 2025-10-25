from omegaconf import DictConfig

import torch
import torch.nn as nn


class RangemeterGRU(nn.Module):
    """A GRU + MLP network processing the rangemeter data."""

    def __init__(self, nets_config: DictConfig):
        super().__init__()
        self.device = torch.device(nets_config["device"])

        layers = []
        # the latent dimension from the events VAE + the scalar value from the rangemeter
        in_dim = nets_config["events_fc_out"] + 1
        self.rangemeter_gru = nn.GRU(
            in_dim, nets_config["range_gru_hdim"], batch_first=True, device=self.device
        )
        in_dim = nets_config["range_gru_hdim"]
        for hidden_dim in nets_config["range_dims"]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, nets_config["range_out"]))
        self.rangemeter_fcnet = nn.Sequential(*layers).to(self.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # get output from the rangemeter nets
        # unsqueeze to have a (B, seq_len, in_dim) tensor
        _, h_n = self.rangemeter_gru(inputs["rangemeter"].unsqueeze(-1))
        # use the output of the last GRU layer
        rangemeter_out = self.rangemeter_fcnet(h_n[-1])

        return rangemeter_out
