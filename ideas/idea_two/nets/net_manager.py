from typing import Dict
from omegaconf import DictConfig

import torch
import torch.nn as nn

from .events_tvae import EventstVAE
from .traj_net import TrajNet
from .rangemeter_gru import RangemeterGRU


class NetManager(nn.Module):
    """A class which manages the various subnets (events, trajectory and rangemeter).

    On top of being a manager, it adds a final MLP, i.e., it is a perfectly adequate torch Module.
    """

    def __init__(self, nets_config: DictConfig):
        super().__init__()
        self.device = torch.device(nets_config["device"])

        self.events_vae = EventstVAE(nets_config["events"])
        self.traj_net = TrajNet(nets_config["traj"])
        self.rangemeter_net = RangemeterGRU(nets_config["range"])

        # construct the final net
        layers = []
        # ignore the events, because the tVAE is trained separately
        in_dim = nets_config["traj_out"] + nets_config["range_out"]
        for hidden_dim in nets_config["traj_dims"]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, nets_config["final_out"]))
        self.final_net = nn.Sequential(*layers).to(self.device)

    def forward(
        self, inputs: Dict[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            _, z, _ = self.events_vae(inputs["event_stack"].unsqueeze(1))
        traj_out = self.traj_net(z)
        _, h_n = self.rangemeter_net(z)
        rangemeter_out = self.rangemeter_net(h_n[-1])

        final_out = self.final_net(torch.concat((traj_out, rangemeter_out), dim=1))
        return final_out
