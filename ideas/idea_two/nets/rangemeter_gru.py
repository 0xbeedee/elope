from omegaconf import DictConfig

import torch
import torch.nn as nn


class RangemeterGRU(nn.Module):
    """A GRU + MLP network processing the rangemeter data with optional latent conditioning."""

    def __init__(self, nets_config: DictConfig):
        super().__init__()
        self.device = torch.device(nets_config["device"])

        layers = []
        in_dim = 1
        self.rangemeter_gru = nn.GRU(
            in_dim, nets_config["range_gru_hdim"], batch_first=True, device=self.device
        )
        # FC input: GRU hidden + latent z (if provided)
        in_dim = nets_config["range_gru_hdim"] + nets_config["events_fc_out"]
        for hidden_dim in nets_config["range_dims"]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, nets_config["range_out"]))
        self.rangemeter_fcnet = nn.Sequential(*layers).to(self.device)

    def forward(self, rangemeter: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # rangemeter: (B, seq_len), z: (B, latent_dim)
        # unsqueeze to have a (B, seq_len, 1) tensor for GRU
        _, h_n = self.rangemeter_gru(rangemeter.unsqueeze(-1))
        # concatenate GRU hidden state with latent z
        combined = torch.cat((h_n[-1], z), dim=1)
        # pass through FC layers
        rangemeter_out = self.rangemeter_fcnet(combined)

        return rangemeter_out
