import csv
from collections import Counter, OrderedDict, defaultdict
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


def safe_copy(tensor_dict):
    if not isinstance(tensor_dict, dict):
        msg = "Input should be a dictionary"
        raise ValueError(msg)

    new_dict = {}
    for key, tensor in tensor_dict.items():
        if not isinstance(tensor, torch.Tensor):
            msg = f"Value for key '{key}' is not a PyTorch tensor"
            raise ValueError(msg)
        new_dict[key] = tensor.detach().clone().cpu()
    return new_dict


def extract_model_state_dict_from_ckpt(ckpt: dict[str, Any]):
    """Extract model state dict from ckpt.
    Note: the key of model_state_dict is the name of your
    model variable in the task.
    """
    if "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        # assume ckpt is state_dict
        state_dict = ckpt
    model_state_dict = {}
    for key, state in state_dict.items():
        key = key.split(".")
        model = key[0]
        layer = ".".join(key[1:])
        if model not in model_state_dict:
            model_state_dict[model] = OrderedDict()
        model_state_dict[model][layer] = state.clone()
    return model_state_dict


def dict_list_to_csv(path: str, dict_list: list[dict[str, Any]]) -> None:
    """Write a dictionary to CSV file.

    Keyword arguments:
        path: path to CSV file to be written to
        dict: dictionary whose values are to be stored
    """
    with open(path, "w") as f:
        dict_writer = csv.DictWriter(f, dict_list[0].keys())
        dict_writer.writeheader()
        dict_writer.writerows(dict_list)


def csv_to_dict_list(path: str) -> None:
    """Read a dictionary from a CSV file.

    Keyword arguments:
        path: path to CSV file to be read from
    """
    with open(path) as f:
        dict_reader = csv.DictReader(f)
        return list(dict_reader)


def compute_dataset_stats_simple(dataset: Dataset, num_channels: int = 3) -> None:
    """Compute the min, max, mean, and std of a given PyTorch compatible
    datset. Assumes that the data tensor is set as (C, H, W).
    """
    num_pixels = 0
    channels_sum = torch.zeros(num_channels)
    channels_squared_sum = torch.zeros(num_channels)

    dl = DataLoader(dataset, batch_size=256, num_workers=64)
    for _, (data, _) in enumerate(tqdm(dl)):
        channels_sum += torch.sum(data, (0, 2, 3))
        channels_squared_sum += torch.sum(data**2, (0, 2, 3))
        num_pixels += data.size(0) * data.size(2) * data.size(3)

    mean = channels_sum / num_pixels
    std = torch.sqrt((channels_squared_sum / num_pixels) - (mean**2))
    print(f"Dataset len: {len(dataset)}")
    print(f"Mean: {mean}")
    print(f"Std: {std}")


def compute_dataset_stats(dataset: Dataset, num_workers) -> None:
    """Compute the min, max, mean, and std of a given PyTorch compatible
    datset. Assumes that the dataset returns a dictionary with bands as keys and values as tensors (B, 1, H, W).
    """
    num_pixels = Counter()
    channels_sum = defaultdict(lambda: torch.tensor([0.0]))
    channels_squared_sum = defaultdict(lambda: torch.tensor([0.0]))

    dl = DataLoader(dataset, batch_size=None, collate_fn=dataset.collate_fn, num_workers=num_workers)
    for i, data in enumerate(tqdm(dl)):
        for key, value in data.items():
            channels_sum[key] += torch.sum(value, (0, 2, 3))
            channels_squared_sum[key] += torch.sum(value**2, (0, 2, 3))
            num_pixels[key] += value.size(0) * value.size(2) * value.size(3)

        if i % 1000 == 0:
            for key in channels_sum.keys():
                mean = channels_sum[key] / num_pixels[key]
                std = torch.sqrt((channels_squared_sum[key] / num_pixels[key]) - (mean**2))
                print(f"Band: {key} \n Mean: {mean.item():.4f} \n Std: {std.item():.4f}")

    print("Finished")
    for key in channels_sum.keys():
        mean = channels_sum[key] / num_pixels[key]
        std = torch.sqrt((channels_squared_sum[key] / num_pixels[key]) - (mean**2))
        print(f"Band: {key} \n Mean: {mean.item():.4f} \n Std: {std.item():.4f}")
