from omegaconf import DictConfig

import torch
import torch.nn as nn


class TrajNet(nn.Module):
    """A simple MLP net for trajecetory data.

    Note that the network operates with one datapoint at a time, i.e., it attempts to predict position and velocity at time t, given the rest of the trajectory information at time t (attitude and angular velocity).
    """

    def __init__(self, nets_config: DictConfig):
        super().__init__()
        self.device = torch.device(nets_config["device"])

        layers = []
        # the latent dime from the events VAE + phi, theta, psi, p, q, r
        in_dim = nets_config["events_fc_out"] + 6
        for hidden_dim in nets_config["traj_dims"]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, nets_config["traj_out"]))
        self.traj_net = nn.Sequential(*layers).to(self.device)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        # get output from the traj net
        traj_out = self.traj_net(inputs["trajectory"])
        return traj_out
