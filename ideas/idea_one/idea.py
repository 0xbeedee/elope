import sys
from typing import Dict, List, Tuple

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from ideas.base import Idea
from ideas.idea_one.nets.parallel_net import ParallelNet
from ideas.common import (
    EventsTrajDataset,
    collate_fn,
    preprocess_data_streaming,
)

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

    def __init__(self, config: DictConfig) -> None:
        super().__init__(config)
        self.net = ParallelNet(config["nets"])

        self.n_epochs = self.conf["n_epochs"]
        self.optimizer = OPTIM_MAP[self.conf["optimiser"]](
            self.net.parameters(), lr=self.conf["optim_lr"]
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
        batch_size = self.conf.get("batch_size", 1)
        num_workers = self.conf.get("num_workers", 0)
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

        writer = SummaryWriter()

        for epoch in range(self.n_epochs):
            with tqdm(
                train_loader,
                desc=f"\t[+] Epoch {epoch + 1}/{self.n_epochs}",
                unit="batch",
            ) as tqdm_ctxt:
                epoch_loss, total_samples = self._one_epoch_train_(tqdm_ctxt)

            if total_samples > 0:
                avg_loss = epoch_loss / total_samples
                writer.add_scalar("Loss/train", avg_loss, epoch)
                print(f"\t[+] Epoch {epoch + 1} - Average Loss: {avg_loss:.5f}")

    @torch.no_grad
    def run_model(self) -> Dict[int, Dict[str, List[float]]]:
        # do not shuffle the test set!
        test_dataset = EventsTrajDataset(self.test_data_path, shuffle=False)
        batch_size = self.conf.get("batch_size", 1)
        num_workers = self.conf.get("num_workers", 0)
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            num_workers=num_workers,
        )

        # defaultdict would be nice, but it's annoying to JSON-serialise
        out_dict = {}
        # ignore the labels (which would be N/A anyhow)
        i = 0
        for feats, _, fnums in test_loader:
            # write text to start of line, overwriting the previous one
            sys.stdout.write("\r" + f"\t[+] Batch {i}, samples {len(fnums)}")
            sys.stdout.flush()
            self._fill_out_dict_batch(out_dict, feats, fnums)
            i += 1
        sys.stdout.write("\n")

        return out_dict

    def _fill_out_dict_batch(
        self,
        data_dict: Dict[int, Dict[str, List[float]]],
        feats: Dict[str, torch.Tensor],
        fnums: torch.Tensor,
    ):
        """Fills the output dictionary for a batch of predictions."""
        preds = self.net(feats)  # (B, 6)

        for j in range(preds.shape[0]):
            fnum = fnums[j].item()
            vs = preds[j, 3:].tolist()  # [vx, vy, vz]
            # ensure the dict structure exists
            data = data_dict.setdefault(fnum, {k: [] for k in ("vx", "vy", "vz")})
            for k, v in zip(("vx", "vy", "vz"), vs):
                data[k].append(v)

    def _one_epoch_train_(self, tqdm_ctxt: tqdm) -> Tuple[float, int]:
        """Trains the network for a single epoch (i.e., a single pass over all the training data).

        The final underscore indicates that this function modifies its inputs as a side-effect. In this case, this is done to update the `tqdm_ctxt`.
        """
        total_samples = 0
        total_loss = 0.0
        acc_batches, acc_labels = [], []
        acc_batch_samples = 0
        last_batch_loss = 0.0
        # ignore the file number during training
        for X_batch, y_batch, _ in tqdm_ctxt:
            batch_size = y_batch.shape[0]
            acc_batches.append(X_batch)
            acc_labels.append(y_batch)
            acc_batch_samples += batch_size

            if len(acc_batches) == self.conf["acc_steps"]:
                last_batch_loss = self._acc_batch_train(acc_batches, acc_labels)
                total_loss += last_batch_loss * acc_batch_samples
                total_samples += acc_batch_samples
                acc_batches.clear()
                acc_labels.clear()
                acc_batch_samples = 0
                tqdm_ctxt.set_postfix({"loss": last_batch_loss})

        if acc_batches:
            # handle the remaining batches
            last_batch_loss = self._acc_batch_train(acc_batches, acc_labels)
            total_loss += last_batch_loss * acc_batch_samples
            total_samples += acc_batch_samples
            tqdm_ctxt.set_postfix({"loss": last_batch_loss})

        return total_loss, total_samples

    def _acc_batch_train(self, batches: list, labels: list) -> float:
        """Processes accumulated batches of data."""
        total_loss = 0.0
        total_samples = 0

        self.optimizer.zero_grad()
        for batch, label in zip(batches, labels):
            batch_size = label.shape[0]
            pred = self.net(batch)
            loss = self.criterion(pred, label)
            loss.backward()
            total_loss += loss.item() * batch_size
            total_samples += batch_size
        self.optimizer.step()

        # return mean loss per sample
        return total_loss / total_samples if total_samples > 0 else 0.0
