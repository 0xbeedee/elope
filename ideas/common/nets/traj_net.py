import torch
import torch.nn as nn
from omegaconf import DictConfig


class TrajNet(nn.Module):
    """A simple MLP net for trajectory data.

    Note that the network operates with one datapoint at a time, i.e., it attempts to predict
    position and velocity at time t, given the rest of the trajectory information at time t
    (attitude and angular velocity).

    This network can be used in two modes:
    - Without latent: input_dim = 6 (phi, theta, psi, p, q, r only)
    - With latent: input_dim = events_fc_out + 6 (VAE latent + trajectory data)
    """

    def __init__(self, nets_config: DictConfig, input_dim: int):
        super().__init__()
        self.device = torch.device(nets_config["device"])

        layers = []
        in_dim = input_dim
        for hidden_dim in nets_config["traj_dims"]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, nets_config["traj_out"]))
        self.traj_net = nn.Sequential(*layers).to(self.device)

    def forward(self, x):
        return self.traj_net(x)
