from abc import ABC, abstractmethod
from omegaconf import DictConfig
import os


class Idea(ABC):
    """Base class for all the ideas."""

    def __init__(self, config: DictConfig):
        # these are useful to all ideas (and remain constant)
        self.train_data_path = os.path.join(os.getcwd(), "data/train")
        self.test_data_path = os.path.join(os.getcwd(), "data/test")

        # all the ideas must have a "global" entry (and possibly other entries)
        self.conf = config["global"]

    @abstractmethod
    def preprocess_data(self) -> None:
        pass

    @abstractmethod
    def train_net(self) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass
