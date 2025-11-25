import json
from pathlib import Path

import fire
import torch
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from tqdm import tqdm

from thor.core.serialization import read_yaml
from thor.data.thor_footprint_builder import FootprintBuilder


def parse_products(products: list[str] | dict[str, str | None]) -> dict[str, str]:
    if isinstance(products, dict):
        return products
    p = {}
    for prod in products:
        if ":" in prod:
            product, resample = prod.split(":")
            p[product] = resample
        else:
            p[prod] = None

    return p


def build_index(dataset_cfg: dict | str | Path, num_workers=7):
    if isinstance(dataset_cfg, str | Path):
        cfg = read_yaml(dataset_cfg)
        dataset_cfg = cfg["dataset"]["train_kwargs"]
    dataset_cfg["products"] = parse_products(dataset_cfg["products"])
    overwrite = dataset_cfg.get("overwrite", False)
    data_dir = dataset_cfg["paths"]
    dataset = FootprintBuilder(**dataset_cfg)
    print("number of unique samples", len(dataset))

    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        num_workers=num_workers,
        drop_last=False,
    )
    out_geometa_file = f"{data_dir}_geometa.json"
    out_geometa_file = Path(out_geometa_file)

    if out_geometa_file.exists() and not overwrite:
        with open(out_geometa_file) as f:
            dataset_metadata = json.load(f)
    else:
        dataset_metadata = {}
    misses = 0
    misses_files = []
    found_files = []
    torch.manual_seed(402)  # 22)
    for sample_b in tqdm(dataloader, total=len(dataloader), desc="Building dataset"):
        sample = sample_b[0]
        file_path, gdf, total_area, success = sample
        if not success:
            misses += 1
            misses_files.append(str(file_path))
        else:
            found_files.append(str(file_path))
            if total_area is not None:
                dataset_metadata[str(file_path)] = total_area

    print(f"Misses: {misses}")
    print(f"Missing percentage: {misses / len(dataloader) * 100:.2f}%")
    print(f"Misses files: {misses_files}")
    print("===============================")
    print(f"Found {len(found_files)} files")

    with open(out_geometa_file, "w") as f:
        json.dump(dataset_metadata, f)


if __name__ == "__main__":
    mp.set_start_method("spawn")
    fire.Fire(build_index)
