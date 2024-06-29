"""Utility functions."""

import csv
import json
import os
import pickle
import random
from typing import Union

import numpy as np
import torch
import yaml


def seed_everything(seed: int) -> None:
    """Seed every randomness to seed"""
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def read_json(fname: str) -> dict:
    """Read json"""
    with open(fname, "r") as stream:
        return json.load(stream)


def write_json(content: dict, fname: str) -> None:
    """Write json"""
    with open(fname, "w") as stream:
        json.dump(content, stream, indent=4, sort_keys=False)


def read_yaml(fname: str) -> dict:
    """Read yaml."""
    with open(fname, "r") as stream:
        return yaml.safe_load(stream)


def write_yaml(content: dict, fname: str) -> None:
    """write yaml."""
    with open(fname, "w") as stream:
        yaml.dump(content, stream, indent=2, sort_keys=False)


def write_pickle(to_pickle: object, fname: str):
    """Read pickle"""
    with open(fname, "wb") as stream:
        foo = pickle.dump(to_pickle, stream)
    return foo


def read_pickle(fname: str):
    """Read pickle"""
    with open(fname, "rb") as stream:
        foo = pickle.load(stream)
    return foo


def write_csv(content: list, fname: str) -> None:
    with open(fname, "w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerows(content)


def read_data(data_path: str) -> dict:
    """Read train, val, test spilts.

    Args:
        data_path: path to data.

    Returns:
        data: {'train': list of training obs,
            'val': list of val obs,
            'test': list of test obs}

    """
    data = read_json(data_path)

    return data


def load_questions(path: str) -> dict:
    """Load premade questions.

    Args:
        path: path to the question json file.

    """
    questions = read_json(path)

    return questions


def argmax(iterable):
    """argmax"""
    return max(enumerate(iterable), key=lambda x: x[1])[0]


def get_duplicate_dicts(search: dict, target: list) -> list:
    """Find if there are duplicate dicts.

    Args:
        search: dict
        target: target list to look up.

    Returns:
        duplicates: a list of dicts or None

    """
    assert isinstance(search, dict)
    duplicates = []

    for candidate in target:
        assert isinstance(candidate, dict)
        if set(search).issubset(set(candidate)):
            if all([val == candidate[key] for key, val in search.items()]):
                duplicates.append(candidate)

    return duplicates


def list_duplicates_of(seq, item) -> list:
    # https://stackoverflow.com/questions/5419204/index-of-duplicates-items-in-a-python-list
    start_at = -1
    locs = []
    while True:
        try:
            loc = seq.index(item, start_at + 1)
        except ValueError:
            break
        else:
            locs.append(loc)
            start_at = loc
    return locs


def is_running_notebook() -> bool:
    """See if the code is running in a notebook or not."""
    try:
        shell = get_ipython().__class__.__name__
        if shell == "ZMQInteractiveShell":
            return True  # Jupyter notebook or qtconsole
        elif shell == "TerminalInteractiveShell":
            return False  # Terminal running IPython
        else:
            return False  # Other type (?)
    except NameError:
        return False  # Probably standard Python interpreter


def positional_encoding(
    positions: int,
    dimensions: int,
    scaling_factor: float = 10000.0,
    return_tensor: bool = False,
) -> Union[np.ndarray, torch.Tensor]:
    """
    Generate sinusoidal positional encoding.

    Parameters:
    positions (int): The number of positions in the sequence.
    dimensions (int): The dimension of the embedding vectors.
    scaling_factor (float): The scaling factor used in the sinusoidal functions.
    return_tensor (bool): If True, return a PyTorch tensor; otherwise, return a NumPy array.

    Returns:
    Union[np.ndarray, torch.Tensor]: A positional encoding in the form of either a NumPy array or a PyTorch tensor.
    """
    # Ensure the number of dimensions is even
    assert dimensions % 2 == 0, "The dimension must be even."

    # Initialize a matrix of position encodings with zeros
    pos_enc = np.zeros((positions, dimensions))

    # Compute the positional encodings
    for pos in range(positions):
        for i in range(0, dimensions, 2):
            pos_enc[pos, i] = np.sin(pos / (scaling_factor ** ((2 * i) / dimensions)))
            pos_enc[pos, i + 1] = np.cos(
                pos / (scaling_factor ** ((2 * (i + 1)) / dimensions))
            )

    # Return as PyTorch tensor if requested
    if return_tensor:
        return torch.from_numpy(pos_enc)

    return pos_enc
