from typing import Dict

import torch
import torch.nn as nn
from omegaconf import DictConfig

from .events_tvae import EventstVAE
from .rangemeter_gru import RangemeterGRU
from .traj_net import TrajNet


class NetManager(nn.Module):
    """A class which manages the various subnets (events, trajectory and rangemeter).

    On top of being a manager, it adds a final MLP, i.e., it is a perfectly adequate torch Module.
    """

    def __init__(self, nets_config: DictConfig):
        super().__init__()
        self.device = torch.device(nets_config["device"])

        self.events_vae = EventstVAE(nets_config)
        self.traj_net = TrajNet(nets_config)
        self.rangemeter_net = RangemeterGRU(nets_config)

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
        event_stack = inputs["event_stack"]
        B = event_stack.shape[0]

        with torch.no_grad():
            # latent repr from the second-to-last valid frame
            if "event_lengths" in inputs:
                # select correct frame per sample
                lengths = inputs["event_lengths"]
                frames = []
                for i in range(B):
                    frame_idx = max(0, lengths[i].item() - 2)
                    frames.append(event_stack[i, frame_idx, :, :])
                t_frames = torch.stack(frames).unsqueeze(1)  # (B, 1, H, W)
            else:
                t_frames = event_stack[:, -2, :, :].unsqueeze(1)

            _, z, _ = self.events_vae.encoder(t_frames)

        traj_out = self.traj_net(torch.cat((z, inputs["trajectory"]), dim=1))

        # Pass range_lengths if available
        range_lengths = inputs.get("range_lengths", None)
        rangemeter_out = self.rangemeter_net(inputs["rangemeter"], z, range_lengths)

        final_out = self.final_net(torch.concat((traj_out, rangemeter_out), dim=1))
        return final_out
