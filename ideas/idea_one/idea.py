from typing import Tuple, Dict, List
from omegaconf import DictConfig
from tqdm import tqdm
import sys

from ideas.base import Idea
from ideas.idea_one.nets.parallel_net import ParallelNet
from ideas.idea_one.utils import EventsTrajDataset, preprocess_data_streaming

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# dict for mapping optimiser names to the correct classes
OPTIM_MAP = {
    "adam": torch.optim.Adam,
}

# dict for mapping loss names to the correct classes
LOSS_MAP = {
    "mse": torch.nn.MSELoss,
}


class IdeaOne(Idea):
    """Idea number 1: train a regressor on single transitions, to predict deltas at successive timestamps.

    This is a far easier task than predicting entire trajectories, and maximises data usage (because each transition becomes a data point, instead of a whole trajectory).
    """

    def __init__(self, config: DictConfig):
        super().__init__(config)
        self.p_net = ParallelNet(config["nets"])

        self.n_epochs = self.conf["n_epochs"]
        self.optimizer = OPTIM_MAP[self.conf["optimiser"]](
            self.p_net.parameters(), lr=self.conf["optim_lr"]
        )
        self.criterion = LOSS_MAP[self.conf["loss"]]()

    def preprocess_data(self) -> None:
        preprocess_data_streaming(self.train_data_path)
        preprocess_data_streaming(self.test_data_path)

    def train_model(self) -> None:
        # val_dataset = EventsTrajDataset(
        #     self.train_data_path, split="val", shuffle=True
        # )
        train_dataset = EventsTrajDataset(
            self.train_data_path, split="train", val_ratio=0.95, shuffle=True
        )
        # set batch_size to 1 because event stacks have variable lengths
        train_loader = DataLoader(train_dataset, batch_size=1)

        writer = SummaryWriter()

        epoch_loss, res_epoch_loss = 0.0, 0.0
        for epoch in range(self.n_epochs):
            with tqdm(
                train_loader,
                desc=f"\t[+] Epoch {epoch + 1}/{self.n_epochs}",
                unit="batch",
            ) as tqdm_ctxt:
                acc_loss, res_acc_loss, total_samples = self._one_epoch_train_(
                    tqdm_ctxt
                )
                epoch_loss += acc_loss
                res_epoch_loss += res_acc_loss if res_acc_loss else 0

            avg_loss = epoch_loss / total_samples
            writer.add_scalar("Loss/train", avg_loss, epoch)
            print(f"\t[+] Epoch {epoch + 1} - Average Loss: {avg_loss:.5f}")

    @torch.no_grad
    def run_model(self) -> Dict[int, Dict[str, List[float]]]:
        # do not shuffle the test set!
        test_dataset = EventsTrajDataset(self.test_data_path, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=1)

        # defaultdict would be nice, but it's annoying to JSON-serialise
        out_dict = {}
        # ignore the labels (which would be N/A anyhow)
        i = 0
        for feats, _, fnum in test_loader:
            # write text to start of line, overwriting the previous one
            sys.stdout.write("\r" + f"\t[+] File {fnum.item()}, datapoint {i}")
            sys.stdout.flush()
            self._fill_out_dict(out_dict, feats, fnum.item())
            i += 1
        sys.stdout.write("\n")

        return out_dict

    def _fill_out_dict(
        self,
        data_dict: Dict[int, Dict[str, List[float]]],
        feats: Dict[str, torch.Tensor],
        fnum: int,
    ):
        """Fills the output dictionary incrementally, making sure it satisfies the output format specified on Kelvins (https://kelvins.esa.int/elope/submission-rules/)."""
        preds = self.p_net(feats)  # [1, 6]
        vs = preds[0][3:].tolist()  # [vx, vy, vz]
        # xs = preds[0][:3].tolist()  # [x, y, z]

        # ensure the dict structure exists
        data = data_dict.setdefault(fnum, {k: [] for k in ("vx", "vy", "vz")})
        for k, v in zip(("vx", "vy", "vz"), vs):
            data[k].append(v)

    def _one_epoch_train_(self, tqdm_ctxt: tqdm) -> Tuple[float, float]:
        """Trains the network for a single epoch (i.e., a single pass over all the trianing data).

        The final underscore indicates that this function modifies its inputs as a side-effect. In this case, this is done to update the `tqdm_ctxt`.
        """
        total_samples = 0
        acc_samples, acc_labels = [], []
        # ignore the file number during training
        for X_batch, y_batch, _ in tqdm_ctxt:
            acc_samples.append(X_batch)
            acc_labels.append(y_batch)
            if len(acc_samples) == self.conf["acc_steps"]:
                # TODO this takes a long time => lambda and/or simplify architecture (bottle neck is the conv part)
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
        """Processes a single accumulated batch of data."""
        total_loss = 0

        self.optimizer.zero_grad()
        for sample, label in zip(samples, labels):
            # handles single samples to preserve temporal and spatial semantics
            # (inefficient, but the alternative approaches are not convincing)
            pred = self.p_net(sample)
            loss = self.criterion(pred, label)
            loss.backward()
            total_loss += loss.item()
        self.optimizer.step()

        # mean-reduce the loss
        return total_loss / len(samples)
