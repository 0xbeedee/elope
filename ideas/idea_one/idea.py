from ideas.base import Idea
from ideas.idea_one.nets.parallel_net import ParallelNet
from ideas.idea_one.utils import EventsTrajDataset, preprocess_data_streaming

import torch
from torch.utils.data import DataLoader


class IdeaOne(Idea):
    """Idea number 1: train a regressor on single transitions, to predict deltas at successive timestamps.

    This is a far easier task than predicting entire trajectories, and maximises data usage (because each transition becomes a data point, instead of a whole trajectory).
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        # TODO actually build the net...
        # self.p_net = ParallelNet()

        self.optimizer = None  # TODO: optimizer
        self.criterion = None  # TODO: loss criterion

    def preprocess_data(self) -> None:
        preprocess_data_streaming(self.train_data_path)
        preprocess_data_streaming(self.test_data_path)

    # TODO put accumulation_steps into the YAML
    def train_net(self, accumulation_steps: int = 8) -> None:
        train_dataset = EventsTrajDataset(self.train_data_path)
        # set batch_size to 1 because event stacks have variable lengths
        train_loader = DataLoader(train_dataset, batch_size=1)

        acc_samples, acc_labels = [], []
        for X_batch, y_batch in train_loader:
            acc_samples.append(X_batch)
            acc_labels.append(y_batch)
            if len(acc_samples) == accumulation_steps:
                self._acc_batch_train(acc_samples, acc_labels)
                acc_samples.clear()
                acc_labels.clear()

        if acc_samples:
            # handle the remaining samples
            self._acc_batch_train(acc_samples, acc_labels)

    def run(self) -> None:
        test_dataset = EventsTrajDataset(self.test_data_path)
        # TODO save output in desired JSON format
        pass

    # TODO type annotations
    def _acc_batch_train(self, samples, labels):
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
