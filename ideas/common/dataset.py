import os
import random
from typing import Dict, Iterator, List, Tuple

import h5py
import torch
from torch.utils.data import IterableDataset, get_worker_info


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

    def _get_worker_file_list(self) -> List[str]:
        """Split file list across workers for multi-worker loading."""
        worker_info = get_worker_info()
        if worker_info is None:
            # single-process loading
            return self.file_list

        # multi-process loading, split files across workers
        per_worker = len(self.file_list) // worker_info.num_workers
        remainder = len(self.file_list) % worker_info.num_workers
        start = worker_info.id * per_worker + min(worker_info.id, remainder)
        end = start + per_worker + (1 if worker_info.id < remainder else 0)
        return self.file_list[start:end]

    def __iter__(self) -> Iterator[Tuple[Dict[str, torch.Tensor], torch.Tensor, int]]:
        file_list = self._get_worker_file_list()
        for fname in file_list:
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


def collate_fn(
    batch: List[Tuple[Dict[str, torch.Tensor], torch.Tensor, int]],
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
    """Collate function that pads variable-length tensors for batching.

    Returns:
        features: Dict with padded tensors and lengths for variable-size inputs
        labels: Stacked labels tensor
        fnums: Tensor of file numbers
    """
    features_list, labels_list, fnums_list = zip(*batch)

    # stack fixed-size tensors
    trajectory = torch.stack([f["trajectory"] for f in features_list])  # (B, 6)
    labels = torch.stack(labels_list)  # (B, 6)
    fnums = torch.tensor(fnums_list)  # (B,)

    # event_stack (variable T dimension): each is (T_i, H, W)
    event_stacks = [f["event_stack"] for f in features_list]
    event_lengths = torch.tensor([e.shape[0] for e in event_stacks])
    max_T = event_lengths.max().item()
    H, W = event_stacks[0].shape[1], event_stacks[0].shape[2]
    # pad to (max_T, H, W) then stack to (B, max_T, H, W)
    padded_events = torch.zeros(len(batch), max_T, H, W)
    for i, (events, length) in enumerate(zip(event_stacks, event_lengths)):
        padded_events[i, :length] = events

    # rangemeter (variable sequence length): each is (S_i,)
    rangemeters = [f["rangemeter"] for f in features_list]
    range_lengths = torch.tensor([r.shape[0] for r in rangemeters])
    max_S = range_lengths.max().item()
    # pad to (max_S,) then stack to (B, max_S)
    padded_range = torch.zeros(len(batch), max_S)
    for i, (rm, length) in enumerate(zip(rangemeters, range_lengths)):
        padded_range[i, :length] = rm

    features = {
        "event_stack": padded_events,
        "event_lengths": event_lengths,
        "trajectory": trajectory,
        "rangemeter": padded_range,
        "range_lengths": range_lengths,
    }

    return features, labels, fnums
