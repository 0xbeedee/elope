from typing import Tuple

import torch
import torch.nn as nn

import numpy as np


# TODO this is uses a basic conv, but i think i can make it better by using convs as they are used for videos: we have temporal consistency, just like in videos => transformers/RNNs? look into nets used for training no video prediction!
class ParallelNet(nn.Module):
    """A network processing the data stream in parallel, with a CNN subnet working with events, and a standard MLP working with the rest.

    We separate the two streams because of their inherent difference: the events are clearly spatial, due to the nature of the sensor, while the rest are just real values.
    """

    # TODO the device should probably be a CLI arg
    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device

        # TODO set input and output dimensions correctly
        self.events_net = nn.Sequential(
            nn.Conv2d(self.in_dim, 64, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(128, self.h_dim, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((1, 1)),  # global average pooling
        ).to(self.device)

        # TODO set input and output dimensions correctly
        self.reals_net = nn.Sequential(
            nn.Linear(), nn.ReLU(), nn.Linear(), nn.ReLU()
        ).to(self.device)

        # TODO set input and output dimensions correctly
        self.final_net = nn.Sequential(nn.Linear(), nn.ReLU(), nn.Linear()).to(
            self.device
        )

    def forward(self, inputs: Tuple[np.ndarray]) -> torch.Tensor:
        # TODO process the inputs
        # TODO the events need to be turned into actual 2D matrices, containing polarity, filled based on (x, y)
        events_out = self.events_net(inputs["events"])
        reals_out = self.reals_net(inputs["trajectory"])
        # TODO there might be smarter way to combine these inputs
        final_out = self.final_net(torch.concat(events_out, reals_out), dim=1)
        return final_out
