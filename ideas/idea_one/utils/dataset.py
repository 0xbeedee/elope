import os
import random
from typing import Dict, Iterator, Tuple

import h5py
import torch
from torch.utils.data import IterableDataset


class EventsTrajDataset(IterableDataset):
    """An iterable dataset containing events and trajectory data to be wrapped in a torch DataLoader."""

    def __init__(
        self,
        data_path: str,
        *,
        split="train",
        val_ratio=0.2,
        shuffle=False,
        seed=None,
        transform=None
    ) -> None:
        super().__init__()

        self.preprocessed_path = os.path.join(data_path, "preprocessed")
        # the directory must exist and it must contain data
        assert os.path.exists(self.preprocessed_path) and os.listdir(
            self.preprocessed_path
        ), "[!] Preprocessed data does not exist"

        file_list = os.listdir(self.preprocessed_path)
        if "test" in self.preprocessed_path:
            # do not split the test dataset
            self.file_list = file_list
        else:
            split_idx = int(len(file_list) * (1 - val_ratio))
            # split the dataset by file to maintain internal consistency
            if split == "train":
                self.file_list = file_list[:split_idx]
            else:
                self.file_list = file_list[split_idx:]

        if seed:
            random.seed(seed)

        # because we only implement __iter__, we need to shuffle manually (instead of relying on DataLoader's shuffle)
        if shuffle:
            # (do not shuffle the single datapoints to maintain the temporal sequence intact)
            random.shuffle(self.file_list)

        self.transform = transform

    def __iter__(self) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor, int]]:
        for fname in self.file_list:
            if not fname.endswith(".h5"):
                # ignore all non-h5py files
                continue

            fpath = os.path.join(self.preprocessed_path, fname)
            with h5py.File(fpath, "r") as f:
                X_group = f["X"]  # data
                y_group = f["y"]  # labels

                indices = sorted(list(X_group.keys()), key=int)
                for i in indices:
                    X_i = X_group[i]
                    event_stack = X_i["event_stack"][:]
                    trajectory = X_i["trajectory"][:]
                    rangemeter = X_i["rangemeter"][:]
                    labels = y_group[i][:]

                    # convert to tensors
                    features = {
                        "event_stack": torch.from_numpy(event_stack).float(),
                        "trajectory": torch.from_numpy(trajectory).float(),
                        "rangemeter": torch.from_numpy(rangemeter).float(),
                    }
                    labels = torch.from_numpy(labels).float()
                    # apply the transform, if one is specified
                    if self.transform:
                        features = self.transform(features)

                    yield features, labels, int(fname[:-3])
