from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
from omegaconf import DictConfig
from tqdm import tqdm

from ideas.idea_one.idea import IdeaOne
from ideas.idea_two.nets import NetManager

# dict for mapping optimiser names to the correct classes
OPTIM_MAP = {
    "adam": torch.optim.Adam,
}


def vae_loss(
    recon_x: torch.Tensor,  # (B, 3, H, W)
    x: torch.Tensor,  # (B, H, W)
    mu: torch.Tensor,  # (B, z.dim)
    logvar: torch.Tensor,  # (B, z.dim)
    beta: float = 1.0,
):
    # use cross_entropy because we have three classes for the events data (-1, 0, 1)
    ce_loss = F.cross_entropy(recon_x, x, reduction="mean")
    kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return ce_loss + beta * kld_loss


# dict for mapping loss names to the correct classes
LOSS_MAP = {
    "vae": vae_loss,
    "ce": torch.nn.CrossEntropyLoss,
    "mse": torch.nn.MSELoss,
}


class IdeaTwo(IdeaOne):
    """Idea number 2: add self-supervised learning to Idea number 1, to hopefully maximise information extraction."""

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        # Override self.net with NetManager (parent sets it to ParallelNet)
        self.net = NetManager(config["nets"])

        self.n_epochs = self.conf["n_epochs"]
        # TODO possibly multiple optimisers?
        self.optimizer = OPTIM_MAP[self.conf["optimiser"]](
            self.net.parameters(), lr=self.conf["optim_lr"]
        )
        self.criteria = [
            LOSS_MAP["vae"],
            self.criterion,
        ]  # the YAML file is used to specify the traj_net and range_net losses

    def _one_epoch_train_(self, tqdm_ctxt: tqdm) -> Tuple[float, int]:
        """Trains the network for a single epoch (i.e., a single pass over all the training data).

        The final underscore indicates that this function modifies its inputs as a side-effect. In this case, this is done to update the `tqdm_ctxt`.
        """
        total_samples = 0
        total_loss = 0.0
        acc_samples, acc_labels = [], []
        last_batch_loss = 0.0
        # ignore the file number during training
        for X_batch, y_batch, _ in tqdm_ctxt:
            self._train_events(X_batch)

            acc_samples.append(X_batch)
            acc_labels.append(y_batch)
            if len(acc_samples) == self.conf["acc_steps"]:
                last_batch_loss = self._acc_batch_train(acc_samples, acc_labels)
                total_loss += last_batch_loss * len(acc_samples)
                # calculate total samples this way because iterable datasets do not have a __len__
                total_samples += len(acc_samples)
                acc_samples.clear()
                acc_labels.clear()
                tqdm_ctxt.set_postfix({"loss": last_batch_loss})

        if acc_samples:
            # handle the remaining samples
            last_batch_loss = self._acc_batch_train(acc_samples, acc_labels)
            total_loss += last_batch_loss * len(acc_samples)
            total_samples += len(acc_samples)
            tqdm_ctxt.set_postfix({"loss": last_batch_loss})

        return total_loss, total_samples

    def _train_events(self, X_dict: Dict[str, torch.Tensor]) -> float:
        """Trains the events tVAE with temporal prediction (t -> t + 1).

        We train the events tVAE before the other networks because the latter use the tVAE latents.
        All (t, t+1) frame pairs are batched together for a single forward/backward pass.
        """
        event_stack = X_dict["event_stack"]
        B, T, H, W = event_stack.shape

        if T < 2:
            return 0.0  # need at least 2 frames for temporal prediction

        # create all frame pairs
        t_frames = event_stack[:, :-1, :, :]  # (B, T-1, H, W)
        tp1_frames = event_stack[:, 1:, :, :]  # (B, T-1, H, W)

        # (B, T-1, H, W) -> (B*(T-1), 1, H, W)
        num_pairs = B * (T - 1)
        t_frames = t_frames.reshape(num_pairs, 1, H, W)
        tp1_frames = tp1_frames.reshape(num_pairs, H, W)
        tp1_frames = (tp1_frames + 1).long()  # from {-1, 0, 1} to {0, 1, 2}

        self.optimizer.zero_grad()
        recon_tp1, _, z_mean, z_logvar = self.net.events_vae(t_frames)
        loss = self.criteria[0](recon_tp1, tp1_frames, z_mean, z_logvar)
        loss.backward()
        self.optimizer.step()

        return loss.item()
