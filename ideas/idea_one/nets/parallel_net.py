from typing import Dict
from omegaconf import DictConfig

import torch
import torch.nn as nn


class ParallelNet(nn.Module):
    """A network processing the data stream in parallel, with a CNN subnet working with events, and a standard MLP working with the rest.

    We separate the two streams because of their inherent difference: the events are clearly spatial, due to the nature of the sensor, while the rest are just real values.
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
                nn.Conv3d(
                    in_dim, hidden_dim, kernel_size=Kc, stride=s, padding=p, dilation=d
                )
            )
            # adjust the height and width (ignore the depth because it varies)
            # (from https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv2d.html#torch.nn.Conv2d)
            H = (H + 2 * p - d * (Kc - 1) - 1 + s) // s
            W = (W + 2 * p - d * (Kc - 1) - 1 + s) // s
            layers.append(nn.SiLU())
            layers.append(nn.MaxPool3d(kernel_size=Km))
            # adjust the height and width (ignore the depth because it varies)
            # (from https://docs.pytorch.org/docs/stable/generated/torch.nn.MaxPool2d.html: the formula below is simpler because we use default values for padding and dilation)
            H, W = H // Km, W // Km
            in_dim = hidden_dim  # update the input channels
        # (1, 1, 1) to perform global average pooling
        layers.append(nn.AdaptiveAvgPool3d((1, 1, 1)))
        self.events_convnet = nn.Sequential(*layers).to(self.device)
        self.events_fcnet = nn.Linear(in_dim, nets_config["events_fc_out"])

        # construct the net for the trajectory data
        layers.clear()
        in_dim = 6  # phi, theta, psi, p, q, r (xs and vs are labels)
        for hidden_dim in nets_config["traj_dims"]:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, nets_config["traj_out"]))
        self.traj_net = nn.Sequential(*layers).to(self.device)

        # construct the net for the rangemeter data
        layers.clear()
        in_dim = 1  # each rangemeter reading is a scalar value
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

        # construct the final net
        layers.clear()
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
