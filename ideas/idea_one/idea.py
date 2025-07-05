from ideas.base import Idea
import torch

# TODO perhaps i can clean up these imports?
from ideas.idea_one.nets.parallel_net import ParallelNet
from ideas.idea_one.utils.preprocessing import *


# TODO more descriptive names? or docstrings already suffice?
class IdeaOne(Idea):
    """Idea number 1: train a regressor on single transitions, to predict deltas at successive timestamps.

    This is a far easier task than predicting entire trajectories, and maximises data usage (because each transition becomes a data point, instead of a whole trajectory).
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.p_net = ParallelNet()

    def run(self):
        # TODO does the whole processing/training/whatever, and save the JSON file somewhere
        preprocess_data(self.train_data_path)
        pass
