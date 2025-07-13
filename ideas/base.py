from abc import ABC, abstractmethod
import os


class Idea(ABC):
    """Base class for all the ideas."""

    def __init__(self):
        # these are useful to all ideas (and remain constant)
        self.train_data_path = os.path.join(os.getcwd(), "data/train")
        self.test_data_path = os.path.join(os.getcwd(), "data/test")

    @abstractmethod
    def preprocess_data(self) -> None:
        pass

    @abstractmethod
    def train_net(self) -> None:
        pass

    @abstractmethod
    def run(self) -> None:
        pass
