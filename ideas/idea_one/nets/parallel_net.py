from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig

from ideas.common.nets.rangemeter_gru import RangemeterGRU
from ideas.common.nets.traj_net import TrajNet


class ParallelNet(nn.Module):
    """A network processing the data stream in parallel, with a CNN subnet working with events, and a standard MLP working with the rest.

    We separate the two streams because of their inherent difference: the events are clearly spatial, due to the nature of the sensor, while the rest are just real values.
    """

    def __init__(self, nets_config: DictConfig):
        super().__init__()
        device = torch.device(nets_config["device"])

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
        self.events_convnet = nn.Sequential(*layers).to(device)
        self.events_fcnet = nn.Linear(in_dim, nets_config["events_fc_out"]).to(device)

        # construct the net for the trajectory data
        self.traj_net = TrajNet(nets_config, input_dim=6)

        # construct the net for the rangemeter data
        self.rangemeter_net = RangemeterGRU(nets_config, use_latent=False)

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
        self.final_net = nn.Sequential(*layers).to(device)

    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # get output from the event nets
        # unsqueeze to have a (B, in_ch, D, H, W) tensor
        events_out = self.events_convnet(inputs["event_stack"].unsqueeze(1))
        # flatten the outputs before feeding them to the FC part
        events_out = events_out.view(events_out.size(0), -1)
        events_out = self.events_fcnet(events_out)

        # get output from the traj net
        traj_out = self.traj_net(inputs["trajectory"])

        # get output from the rangemeter net
        rangemeter_out = self.rangemeter_net(
            inputs["rangemeter"],
            z=None,
            lengths=inputs.get("range_lengths", None),
        )

        # get final output by concatenating the previous outputs
        final_out = self.final_net(
            torch.concat((events_out, traj_out, rangemeter_out), dim=1)
        )
        return final_out
