from typing import Generator

import os
from math import ceil

import numpy as np
import h5py


def preprocess_data_streaming(data_path: str) -> None:
    """Preprocesses data in a streaming fashion, saving it incrementally to a file."""
    # base path to save to
    out_path = os.path.join(data_path, "preprocessed")
    os.makedirs(out_path, exist_ok=True)

    for fname in os.listdir(data_path):
        if not fname.endswith(".npz"):
            # ignore all non-npz fles
            continue

        fpath = os.path.join(data_path, fname)
        out_file = os.path.join(out_path, f"{fname.replace('.npz', '.h5')}")
        if os.path.exists(out_file):
            # to enable checkpointing, we skip already-present files
            # (assumes that, if it exists, the file has the correct data)
            continue
        _process_single_file_streaming(fpath, out_file)


def _process_single_file_streaming(fpath: str, out_file: str) -> None:
    data = np.load(fpath)
    time_s = data["timestamps"]  # in seconds

    events = data["events"]  # in microseconds
    traj_data = data["traj"]  # in seconds
    rangemeter_data = data["range_meter"]  # in seconds (sampled at 10Hz)

    ev_gen = _event_stacks_generator(events, time_s)
    with h5py.File(out_file, "w") as f:
        X_group = f.create_group("X")  # data
        y_group = f.create_group("y")  # labels

        start_rm_idx = 0
        for i, event_stack in enumerate(ev_gen):
            # so that we can index into X, i.e., do X[i]
            X_i = X_group.create_group(f"{i}")
            # assign event data
            X_i.create_dataset(f"event_stack", data=event_stack, compression="gzip")
            # assign trajectory data
            # (phi, theta, psi and p, q, r are for training)
            X_i.create_dataset(f"trajectory", data=traj_data[i, 6:], compression="gzip")
            # assign rangemeter data
            slice = _get_rangemeter_slice_(
                i, start_rm_idx, len(traj_data), len(rangemeter_data)
            )
            X_i.create_dataset(
                f"rangemeter",
                data=rangemeter_data[slice],
                compression="gzip",
            )

            # so that we can index into y, i.e., do y[i]
            # (x, y, z and vx, vy, vz are for supervision)
            y_group.create_dataset(f"{i}", data=traj_data[i, :6], compression="gzip")


def _event_stacks_generator(
    events: np.ndarray, time_s: np.ndarray
) -> Generator[np.ndarray]:
    """Extracts stacks of events, i.e., 3-tensors, where each slice in the stack is a 2D array indicating the polarities at each timestep (in microseconds).
    Each stack is made up of these 2D slices, up to a time specified in seconds (by `time_s`).

    The generator yields a list of these stacks, as with as many 3-tensors as there are timesteps in `time_s`.
    """
    t = 1
    last_ts = events[0][3]
    events_stack_2D = []
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
            if events_stack_2D:
                yield np.array(events_stack_2D)
                events_stack_2D.clear()  # free memory

            t += 1
        if t >= len(time_s):
            break

    if events_stack_2D:
        # yield any remaining stack
        yield np.array(events_stack_2D)
        events_stack_2D.clear()


def _get_rangemeter_slice_(
    cur_traj_idx: int,
    start_rm_idx: int,
    traj_len: int,
    rangemeter_len: int,
) -> int:
    """Calculates the rangemeter slice necessary to match the timestamps of the trajectory.

    This is necessary because the rangemeter samples data at 10Hz, while the trajectory is sampled at 120 points in time, which do not map one-to-one to rangemeter data.

    The final underscore indicates that this function modifies `start_rm_idx`.
    """
    # rangemeter data must be funer than trajectory data (need >=12s of data)
    assert rangemeter_len > traj_len

    # max number of rangemeter datapoints for one traj datapoint
    max_n_rm = ceil(rangemeter_len / traj_len)
    # the index at which to assign n_rm datapoints to each rangemeter slice
    switching_idx = max_n_rm * traj_len - rangemeter_len

    start_idx = start_rm_idx
    if cur_traj_idx < switching_idx:
        end_idx = start_idx + (max_n_rm - 1)
    else:
        end_idx = start_idx + max_n_rm
    # update the starting rangemeter index
    start_rm_idx = end_idx

    return slice(start_idx, end_idx)
