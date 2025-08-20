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

        self.events_net = EventstVAE(nets_config["events"])
        self.traj_net = TrajNet(nets_config["traj"])
        self.rangemeter_net = RangemeterGRU(nets_config["range"])

        # construct the final net
        layers = []
        in_dim = (
            nets_config["events_fc_out"]
            + nets_config["traj_out"]
            + nets_config["range_out"]
        )
        for hidden_dim in nets_config["traj_dims"]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, nets_config["final_out"]))
        self.final_net = nn.Sequential(*layers).to(self.device)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # get output from the event nets
        # unsqueeze to have a (B, in_ch, D, H, W) tensor
        events_out = self.events_convnet(inputs["event_stack"].unsqueeze(1))
        # flatten the outputs before feeding them to the FC part
        events_out = events_out.view(events_out.size(0), -1)
        events_out = self.events_fcnet(events_out)

        # get output from the traj net
        traj_out = self.traj_net(inputs["trajectory"])

        # get output from the rangemeter nets
        # unsqueeze to have a (B, seq_len, in_dim) tensor
        _, h_n = self.rangemeter_gru(inputs["rangemeter"].unsqueeze(-1))
        # use the output of the last GRU layer
        rangemeter_out = self.rangemeter_fcnet(h_n[-1])

        # get final output by concatenating the previous outputs
        # TODO there might be smarter way to combine these inputs
        final_out = self.final_net(
            torch.concat((events_out, traj_out, rangemeter_out), dim=1)
        )
        return final_out
