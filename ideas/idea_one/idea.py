from omegaconf import DictConfig
from tqdm import tqdm

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

    def train_net(self) -> None:
        train_dataset = EventsTrajDataset(self.train_data_path, shuffle=True)
        # set batch_size to 1 because event stacks have variable lengths
        train_loader = DataLoader(train_dataset, batch_size=1)

        writer = SummaryWriter()
        for epoch in range(self.n_epochs):
            epoch_loss, res_epoch_loss = 0, 0
            with tqdm(
                train_loader,
                desc=f"\t[+] Epoch {epoch + 1}/{self.n_epochs}",
                unit="batch",
            ) as tqdm_ctxt:
                acc_loss, res_acc_loss = self._one_epoch_train_(tqdm_ctxt)
                epoch_loss += acc_loss
                res_epoch_loss += res_acc_loss if res_acc_loss else 0

            avg_loss = epoch_loss / len(train_loader)
            writer.add_scalar("Loss/train", avg_loss, epoch)
            print(f"\t[+] Epoch {epoch + 1} - Average Loss: {avg_loss:.5f}")

    def run(self) -> None:
        test_dataset = EventsTrajDataset(self.test_data_path)
        # TODO save output in desired JSON format
        pass

    def _one_epoch_train_(self, tqdm_ctxt: tqdm) -> None:
        """Trains the network for a single epoch (i.e., a single pass over all the trianing data).

        The final underscore indicates that this function modifies its inputs as a side-effect. In this case, this is done to update the `tqdm_ctxt`.
        """
        acc_samples, acc_labels = [], []
        for X_batch, y_batch in tqdm_ctxt:
            acc_samples.append(X_batch)
            acc_labels.append(y_batch)
            if len(acc_samples) == self.conf["acc_steps"]:
                acc_loss = self._acc_batch_train(acc_samples, acc_labels)
                acc_samples.clear()
                acc_labels.clear()

        if acc_samples:
            # handle the remaining samples
            res_acc_loss = self._acc_batch_train(acc_samples, acc_labels)
        tqdm_ctxt.set_postfix({"loss": acc_loss, "res_los": res_acc_loss})

        return acc_loss, res_acc_loss

    def _acc_batch_train(self, samples: list, labels: list) -> float:
        """Processes a single accumulated batch of data."""
        total_loss = 0

        self.optimizer.zero_grad()
        for sample, label in zip(samples, labels):
            # have the network handle single sample to both preserve temporal and spatial semantics
            pred = self.p_net(sample)
            loss = self.criterion(pred, label)
            loss.backward()
            total_loss += loss.item()
        self.optimizer.step()

        # mean-reduce the loss
        return total_loss / len(samples)
