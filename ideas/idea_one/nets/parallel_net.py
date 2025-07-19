from typing import Dict
from omegaconf import DictConfig

import torch
import torch.nn as nn

import numpy as np


class ParallelNet(nn.Module):
    """A network processing the data stream in parallel, with a CNN subnet working with events, and a standard MLP working with the rest.

    We separate the two streams because of their inherent difference: the events are clearly spatial, due to the nature of the sensor, while the rest are just real values.
    """

    def __init__(self, nets_config: DictConfig):
        super().__init__()
        self.device = torch.device(nets_config["device"])

        layers = []
        in_dim = 1  # the number of input channels
        for hidden_dim in nets_config["conv_dims"]:
            layers.append(nn.Conv3d(in_dim, hidden_dim, kernel_size=3, padding=1))
            layers.append(nn.SiLU())
            layers.append(nn.MaxPool3d(kernel_size=2))
            in_dim = hidden_dim  # update the input channels
        # TODO the input dimension of this is not quite correct: through pooling, i reduce H and W (so both should be <200)!
        layers.append(nn.AdaptiveAvgPool3d((nets_config["N_out_conv"], 200, 200)))
        self.events_convnet = nn.Sequential(*layers).to(self.device)
        self.events_fcnet = nn.Linear(
            nets_config["N_out_conv"] * 200 * 200, nets_config["events_fc_out"]
        )

        layers.clear()
        in_dim = 1
        for hidden_dim in nets_config["traj_dims"]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, nets_config["traj_out"]))
        self.traj_net = nn.Sequential(*layers).to(self.device)

        # TODO finish the GRU net for the rangemeter
        # self.rangemeter_gru = nn.GRU(1, 16, batch_first=True, device=self.device)
        # self.rangemeter_fc = nn.Sequential(
        #     nn.GRU(1, 16, batch_first=True), nn.ReLU(), nn.Linear(), nn.ReLU()
        # ).to(self.device)

        layers.clear()
        in_dim = nets_config["events_fc_out"] + nets_config["traj_out"]
        for hidden_dim in nets_config["traj_dims"]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, nets_config["final_out"]))
        self.final_net = nn.Sequential(*layers).to(self.device)

    def forward(self, inputs: Dict[str, np.ndarray]) -> torch.Tensor:
        events_out = self.events_convnet(inputs["event_stack"])
        # flatten the outputs before feedin them to the FC part
        events_out = events_out.view(events_out.size(0), -1)
        events_out = self.events_fcnet(events_out)

        traj_out = self.traj_net(inputs["trajectory"])
        # rangemeter_out = self.reals_net(inputs["rangemeter"])

        # TODO there might be smarter way to combine these inputs
        final_out = self.final_net(torch.concat(events_out, traj_out), dim=1)
        return final_out
