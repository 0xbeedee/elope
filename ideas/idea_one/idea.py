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
        self.p_net = ParallelNet()

    def preprocess_data(self) -> None:
        print(f"[+] {self.__name__}: Preprocessing the training data...")
        preprocess_data_streaming(self.train_data_path)

        print(f"[+] {self.__name__}: Preprocessing the test data...")
        preprocess_data_streaming(self.test_data_path)

    def train_net(self) -> None:
        print(f"[+] {self.__name__}: Training the neural net...")
        train_dataset = EventsTrajDataset(self.train_data_path)
        train_loader = DataLoader(train_dataset, batch_size=8)

        for X_batch, y_batch in train_loader:
            print(X_batch, y_batch)
        #     optimizer.zero_grad()
        #     preds_batch = self.p_net(X_batch)
        #     loss = criterion(preds_batch, y_batch)
        #     loss.backward()
        #     optimizer.step()

    def run(self) -> None:
        test_dataset = EventsTrajDataset(self.test_data_path)
        # TODO save output in desired JSON format
        pass
