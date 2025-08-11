from typing import Dict, List
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
        """Preprocesses the training and test data."""

    @abstractmethod
    def train_model(self) -> None:
        """Trains the model."""

    @abstractmethod
    def run_model(self) -> Dict[int, Dict[str, List[float]]]:
        """Runs the model on the test data."""
