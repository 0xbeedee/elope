from typing import Tuple, Dict, List
from omegaconf import DictConfig

from ideas.idea_one.idea import IdeaOne
from ideas.idea_two.nets import NetManager

import torch
import torch.nn.functional as F
from tqdm import tqdm

# dict for mapping optimiser names to the correct classes
OPTIM_MAP = {
    "adam": torch.optim.Adam,
}


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    # use cross_entropy because we have three classes for the events data (-1, 0, 1)
    ce_loss = F.cross_entropy(recon_x, x, reduction="sum")
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
        self.net_manager = NetManager(config["nets"])

        self.n_epochs = self.conf["n_epochs"]
        # TODO possibly multiple optimisers?
        self.optimizer = OPTIM_MAP[self.conf["optimiser"]](
            self.p_net.parameters(), lr=self.conf["optim_lr"]
        )
        self.criteria = [LOSS_MAP[loss]() for loss in self.conf["loss"]]
        assert len(self.criteria) >= 3, (
            "[!] Need at least three losses (one for the VAE, one for the trajectory net, and one for the rangemeter net)."
        )

    def train_model(self) -> None:
        return super().train_model()

    @torch.no_grad
    def run_model(self) -> Dict[int, Dict[str, List[float]]]:
        return super().run_model()

    def _one_epoch_train_(self, tqdm_ctxt: tqdm) -> Tuple[float, float]:
        """Trains the network ensemble for a single epoch (i.e., a single pass over all the training data).

        The final underscore indicates that this function modifies its inputs as a side-effect. In this case, this is done to update the `tqdm_ctxt`.
        """
        total_samples = 0
        # ignore the file number during training
        for X_batch, y_batch, _ in tqdm_ctxt:
            acc_loss = self._train_alternating(X_batch, y_batch)
            # calculate total samples this way because iterable datasets do not have a __len__
            total_samples += len(y_batch)
        tqdm_ctxt.set_postfix({"loss": acc_loss})

        res_acc_loss = 0  # for API compatibility
        return acc_loss, res_acc_loss, total_samples

    def _train_alternating(
        self, Xs: Dict[str, torch.Tensor], ys: torch.Tensor
    ) -> float:
        """Trains the various network in an alternating fashion.

        We train the events tVAE and the rangemeter tVAE first (in parallel), then we use the latent
        """
        total_loss = 0

        # TODO train nets as on the paper notes => what do i pass to the traj net? why not pass the step-wise latents from events and rangemeter to an RNN, and pass that to the trajnet? like i already to for the rangemeter
        self.optimizer.zero_grad()
        for X, y in zip(Xs, ys):
            # handles single samples to preserve temporal and spatial semantics
            # (inefficient, but the alternative approaches are not convincing)
            pred = self.net_manager(X)
            loss = self.criterion(pred, y)
            loss.backward()
            total_loss += loss.item()
        self.optimizer.step()

        # mean-reduce the loss
        return total_loss / len(Xs)
