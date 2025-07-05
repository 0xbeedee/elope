from typing import Tuple

import os
from collections import namedtuple

import numpy as np
import pandas as pd


# TODO this approach takes way too much RAM (>100GB), perhaps stream the data with generators? => or use generators to write data to files, and then stream from these files to the neural net?!
def preprocess_data(data_path: str):
    """Preprocesses the train and test data."""
    np_data = []
    for f in os.listdir(data_path):
        if not f.endswith(".npz"):
            # ignore all non-npz fles
            continue
        np_data.append(np.load(f))

    events_data = []  # events from the DVS camera
    trajectory_data = []  # trajectory + rangemeter data
    # TODO possibly include in the loop above?
    for data in np_data:
        time_s = data["timestamps"]  # in seconds
        events_data.append(extract_2D_event_stacks(data["events"], time_s))
        # TODO ignore the rangemeter for now (it operates with different timestamps, which creates quite the headache)

    clean_data = namedtuple("Data", "events trajectory")
    return clean_data(events_data, trajectory_data)


def extract_2D_event_stacks(events: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Extracts stacks of events, i.e., 3-tensors, where each slice in the stack is a 2D array indicating the polarities at each timestep (in microseconds).
    Each stack is made up of these 2D slices, up to a time specified in seconds (by `time_s`).

    The function returns a list of these stacks, as with as many 3-tensors as there are timesteps in `time_s`.
    """
    t = 1
    last_ts = events[0][3]
    events_stack_2D, events_stack_list = [], []
    event_canvas = np.zeros((200, 200), dtype=np.int8)
    positions = []
    polarities = []

    for event in events:
        x, y, polarity, ts = event
        if ts / 1e6 < time_s[t]:
            if last_ts != ts:
                xs, ys = zip(*positions)
                # 1 for True, -1 for False
                event_canvas[xs, ys] = np.where(polarities, 1, -1)
                events_stack_2D.append(event_canvas.copy())

                event_canvas.fill(0)
                positions.clear()
                polarities.clear()
                last_ts = ts
            positions.append((x, y))
            polarities.append(polarity)
        else:
            t += 1
            print("==> ", t)
            events_stack_list.append(np.array(events_stack_2D))
        if t >= len(time_s):
            break

    return events_stack_list
