from typing import Optional

import torch
import torch.nn as nn
from omegaconf import DictConfig
from torch.nn.utils.rnn import pack_padded_sequence


class RangemeterGRU(nn.Module):
    """A GRU + MLP network processing the rangemeter data with optional latent conditioning.

    This network can be used in two modes:
    - Without latent (use_latent=False): Processes rangemeter data only
    - With latent (use_latent=True): Concatenates VAE latent with GRU output
    """

    def __init__(self, nets_config: DictConfig, use_latent: bool = False):
        super().__init__()
        self.device = torch.device(nets_config["device"])
        self.use_latent = use_latent

        layers = []
        in_dim = 1
        self.rangemeter_gru = nn.GRU(
            in_dim, nets_config["range_gru_hdim"], batch_first=True, device=self.device
        )
        # FC input: GRU hidden + latent z (if use_latent=True)
        in_dim = nets_config["range_gru_hdim"]
        if use_latent:
            in_dim += nets_config["events_fc_out"]

        for hidden_dim in nets_config["range_dims"]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, nets_config["range_out"]))
        self.rangemeter_fcnet = nn.Sequential(*layers).to(self.device)

    def forward(
        self,
        rangemeter: torch.Tensor,
        z: Optional[torch.Tensor] = None,
        lengths: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # rangemeter: (B, seq_len), z: (B, latent_dim) or None
        # unsqueeze to have a (B, seq_len, 1) tensor for GRU
        rangemeter_3d = rangemeter.unsqueeze(-1)

        if lengths is not None:
            # pack_padded_sequence for variable-length sequences
            packed = pack_padded_sequence(
                rangemeter_3d, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
            _, h_n = self.rangemeter_gru(packed)
        else:
            _, h_n = self.rangemeter_gru(rangemeter_3d)

        gru_out = h_n[-1]
        # concatenate with latent z if provided
        if self.use_latent and z is not None:
            combined = torch.cat((gru_out, z), dim=1)
        else:
            combined = gru_out

        rangemeter_out = self.rangemeter_fcnet(combined)
        return rangemeter_out
