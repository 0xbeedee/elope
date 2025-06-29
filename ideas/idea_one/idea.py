from ideas.base import Idea
import torch


# TODO more descriptive names? or docstrings already suffice?
class IdeaOne(Idea):
    """Idea number 1: train a regressor on single transitions, to predict deltas at successive timestamps.

    This is a far easier task than predicting entire trajectories, and maximises data usage (because each transition becomes a data point, instead of a whole trajectory).
    """

    def __init__(self):
        super().__init__()

    def run():
        # TODO does the whole processing/training/whatever, and save the JSON file somewhere
        pass
