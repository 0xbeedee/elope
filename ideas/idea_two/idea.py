from typing import Dict, List, Tuple
from omegaconf import DictConfig
from tqdm import tqdm

from ideas.idea_one.idea import IdeaOne
from ideas.idea_two.nets import NetManager

import torch
import torch.nn.functional as F

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
        self.net_manager = NetManager(config["nets"])

        self.n_epochs = self.conf["n_epochs"]
        # TODO possibly multiple optimisers?
        self.optimizer = OPTIM_MAP[self.conf["optimiser"]](
            self.p_net.parameters(), lr=self.conf["optim_lr"]
        )
        self.criteria = [
            LOSS_MAP["vae"],
            self.criterion,
        ]  # the YAML file is used to specify the traj_net and range_net losses

    def train_model(self) -> None:
        return super().train_model()

    @torch.no_grad
    def run_model(self) -> Dict[int, Dict[str, List[float]]]:
        return super().run_model()

    def _one_epoch_train_(self, tqdm_ctxt: tqdm) -> Tuple[float, float]:
        """Trains the network for a single epoch (i.e., a single pass over all the training data).

        The final underscore indicates that this function modifies its inputs as a side-effect. In this case, this is done to update the `tqdm_ctxt`.
        """
        total_samples = 0
        acc_samples, acc_labels = [], []
        # ignore the file number during training
        for X_batch, y_batch, _ in tqdm_ctxt:
            self._train_events(X_batch)

            acc_samples.append(X_batch)
            acc_labels.append(y_batch)
            if len(acc_samples) == self.conf["acc_steps"]:
                acc_loss = self._acc_batch_train(acc_samples, acc_labels)
                # calculate total samples this way because iterable datasets do not have a __len__
                total_samples += len(acc_samples)
                acc_samples.clear()
                acc_labels.clear()

        if acc_samples:
            # handle the remaining samples
            res_acc_loss = self._acc_batch_train(acc_samples, acc_labels)
        tqdm_ctxt.set_postfix({"loss": acc_loss, "res_los": res_acc_loss})

        return acc_loss, res_acc_loss, total_samples

    def _acc_batch_train(self, samples: list, labels: list) -> float:
        """Processes a single accumulated batch of data, training the trajectory and the rangemeter networks.

        The events tVAE is trained before both, because both networks use the tVAE latents."""
        total_loss = 0
        self.optimizer.zero_grad()
        for X, y in zip(samples, labels):
            # handles single samples to preserve temporal and spatial semantics
            # (inefficient, but the alternative approaches are not convincing)
            pred = self.net_manager(X)
            loss = self.criterion(pred, y)
            loss.backward()
            total_loss += loss.item()
        self.optimizer.step()
        # mean-reduce the loss
        return total_loss / len(samples)

    def _train_events(self, X_dict: Dict[str, torch.Tensor]) -> float:
        """Trains the events tVAE with temporal prediction (t -> t + 1).

        We train the events tVAE before the other networks because the latter use the tVAE latents.
        """
        event_stack = X_dict["event_stack"]
        B, T, H, W = event_stack.shape
        num_windows = T - 1  # can have at most T-1 sliding windows (with stride=1)

        # TODO sequential training is inefficient (mostly done to preserve temporal smantics, but should probably be parallelised)!
        total_loss = 0
        self.optimizer.zero_grad()
        for i in range(num_windows):
            # t frame
            t_frame = event_stack[:, i : i + 1, :, :]  # (B, 1, H, W)
            # t + 1 frame
            tp1_frame = event_stack[:, i + 1, :, :]  # (B, H, W)
            tp1_frame = (tp1_frame + 1).long()  # from {-1, 0, 1} to {0, 1, 2}

            recon_tp1, _, z_mean, z_logvar = self.net_manager.events_vae(t_frame)

            loss = self.criteria[0](recon_tp1, tp1_frame, z_mean, z_logvar)
            loss.backward()
            total_loss += loss.item()
        self.optimizer.step()
        return total_loss / num_windows  # avg loss per window
