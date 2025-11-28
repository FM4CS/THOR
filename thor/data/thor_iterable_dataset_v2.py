import json
import logging
import math
import os
import random
import signal
import traceback
from collections import Counter
from collections.abc import Generator
from datetime import datetime
from functools import lru_cache, wraps
from pathlib import Path
from typing import Literal

import geopandas as gpd
import numpy as np
import torch
import torchvision.transforms.functional as F
import xarray as xr
from geopandas.tools._random import uniform
from pyproj import Proj, Transformer
from torch.utils.data import IterableDataset, get_worker_info
from torchvision.transforms import InterpolationMode

from thor.core.dataset_registry import DATASETS
from thor.data.multi_look import MulitLook
from thor.data.thor_dataset_base import MetaData, ProductDataBatch, THORDatasetBase, _to_tensor

logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def timeout(seconds=300):
    """Timeout decorator that raises TimeoutError if the function takes longer than specified seconds.

    Args:
        seconds (int): Timeout in seconds (default: 300 = 5 minutes)
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            def handler(signum, frame):
                error_msg = f"Function {func.__name__} timed out after {seconds} seconds"
                raise TimeoutError(error_msg)

            # Set the timeout handler
            old_handler = signal.signal(signal.SIGALRM, handler)
            # Set alarm
            signal.alarm(seconds)

            try:
                result = func(*args, **kwargs)
            finally:
                # Restore previous signal handler and cancel alarm
                signal.signal(signal.SIGALRM, old_handler)
                signal.alarm(0)
            return result

        return wrapper

    return decorator


@DATASETS.register(override=True)
class THORIterableDatasetV2(IterableDataset, THORDatasetBase):
    def __init__(
        self,
        paths: str,
        ground_covers: int | list[int],
        ground_cover_weights: list[float] | None = None,
        split: str = "train",
        products: list | dict = [  # noqa: B006
            "S2-10m",
            "S2-20m",
            "S2-60m",
            "S1-10m",
        ],
        discard_bands: list = [],
        standardize: bool = True,
        full_return: bool = False,
        data_percent: float = 1.0,
        max_nan_ratio: float = 0.05,
        metadata_dir=None,
        strict_data_mode=False,
        include_filter: str | None = None,
        exclude_filter: str | None = None,
        legacy_nan_handling=False,
        sample_ratio: float = 1.0,
        batch_size=1,
        global_rank=0,
        world_size=1,
        random_seed=42,
        current_epoch=0,
        weighted_sampling=False,
        land_cover_products: list[str] | None = None,
        land_cover_dir: str | None = None,
        multilook_S1=False,
        S1_multilook_factors=[1, 2, 3, 4],
        dem_products: list[str] | None = None,
        dem_dir: str | None = None,
        dem_normalize: dict[
            str, Literal["local_min", "local_min_max", "local_mean", "local_median", "local_mean_std"]
        ] = {  # noqa: B006
            "elevation": "local_min",
            "slope": "none",
        },
        dem_type: Literal["elevation", "slope", "both"] = "both",
        use_incidence_angle=False,
        aggregate_incidence_angle=False,
        **kwargs,
    ):
        if isinstance(ground_covers, int):
            ground_covers = [ground_covers]
        ground_covers = sorted(ground_covers)

        if ground_cover_weights is None:
            ground_cover_weights = [1.0 / len(ground_covers)] * len(ground_covers)
        else:
            assert len(ground_cover_weights) == len(ground_covers), (
                f"Ground cover weights must be the same length as ground covers. "
                f"Got {len(ground_cover_weights)} weights for {len(ground_covers)} ground covers."
            )
            ground_cover_weights = np.array(ground_cover_weights) / np.sum(ground_cover_weights)
        self.ground_cover_weights = ground_cover_weights

        if land_cover_dir is None:
            land_cover_dir = Path(paths).parent / "world_cover"
        self.land_cover_dir = Path(land_cover_dir)

        if land_cover_products is None:
            land_cover_products = []

        if isinstance(land_cover_products, dict):
            land_cover_products = {
                f"{lc_product.split('-')[0]}-{self.LAND_COVER_GSD_MAP[lc_product.split('-')[0]]}m": v
                for lc_product, v in land_cover_products.items()
            }
            land_cover_product_changes = land_cover_products
            land_cover_products = list(land_cover_product_changes.keys())
            for lc_product in land_cover_product_changes:
                if land_cover_product_changes[lc_product] is None:
                    land_cover_product_changes[lc_product] = lc_product
            self.land_cover_product_changes = land_cover_product_changes
        else:
            land_cover_products = [
                f"{lc_product.split('-')[0]}-{self.LAND_COVER_GSD_MAP[lc_product]}m"
                for lc_product in land_cover_products
            ]
            self.land_cover_product_changes = {lc_product: lc_product for lc_product in land_cover_products}

        if len(land_cover_products) > 0:
            assert all(land_cover_product in self.LAND_COVER_PRODUCTS for land_cover_product in land_cover_products), (
                f"Invalid land cover products: {land_cover_products}. Must be one of {self.LAND_COVER_PRODUCTS}"
            )
        self.land_cover_products = land_cover_products

        if "SCL-20m" in self.land_cover_products:
            assert "S2-20m" in products, "SCL mask requires the S2-20m product"

        if dem_products is None:
            dem_products = []

        self.dem_products = [dem_product.lower() for dem_product in dem_products]
        assert all(dem_product in self.DEM_PRODUCTS for dem_product in self.dem_products), (
            f"Invalid DEM products: {self.dem_products}. Must be one of {self.DEM_PRODUCTS}"
        )
        self.dem_product_changes = {dem_product: dem_product for dem_product in self.dem_products}
        if dem_dir is None:
            dem_dir = Path(paths).parent / "dem"
        self.dem_dir = Path(dem_dir)
        self.dem_normalize = dem_normalize
        self.dem_type = dem_type
        if use_incidence_angle:
            self.incidence_angle_products = ["INC-10m", "INC-60m"]
        else:
            self.incidence_angle_products = []
        self.incidence_angle_product_changes = {
            inc_product: inc_product for inc_product in self.incidence_angle_products
        }
        self.aggregate_incidence_angle = aggregate_incidence_angle

        super().__init__(
            paths=paths,
            split=split,
            products=products,
            discard_bands=discard_bands,
            ground_covers=ground_covers,
            standardize=standardize,
            full_return=full_return,
            data_percent=data_percent,
            max_nan_ratio=max_nan_ratio,
            include_filter=include_filter,
            exclude_filter=exclude_filter,
            legacy_nan_handling=legacy_nan_handling,
            random_seed=random_seed,
            **kwargs,
        )

        self.strict_data_mode = strict_data_mode
        self.sample_ratio = sample_ratio

        self.batch_size = batch_size
        self.global_rank = global_rank
        self.world_size = world_size
        self.epoch = current_epoch
        self.weighted_sampling = weighted_sampling

        all_products = self.products + self.land_cover_products + self.dem_products + self.incidence_angle_products
        all_product_changes = {
            **self.product_changes,
            **self.land_cover_product_changes,
            **self.dem_product_changes,
            **self.incidence_angle_product_changes,
        }

        self.custom_transform = self._build_transforms(
            split == "train",
            all_products,
            all_product_changes,
        )

        if metadata_dir is None:
            metadata_dir = self.get_metadata_dir()
        self.metadata_dir = Path(metadata_dir)

        self.multilook_S1 = multilook_S1
        self.s1_multilook_factors = S1_multilook_factors

        if self.multilook_S1:
            self.multilook = MulitLook(gsd=10, multilook_factors=self.s1_multilook_factors)

        logger.info(f"Using metadata dir: {self.metadata_dir}")

        self.metadata_cache_name = self.get_metadata_cache_name()

        self.file_paths, _ = self.load_files(split, strict=False)
        logger.info(f"Loaded {len(self.file_paths)} files for split '{split}'")
        self.file_paths = np.array([str(fp) for fp in self.file_paths])

        if self.weighted_sampling:
            weights_file = f"{paths}_geometa.json"
            with open(weights_file) as f:
                weights = json.load(f)
                median_weight = np.median(np.array(list(weights.values())))
            self.weights = np.array([weights[fp] if fp in weights else median_weight for fp in self.file_paths])

        # DEBUG
        self._hits = 0
        self._bad_hits = Counter()
        self._per_file_bad_hits = Counter()
        self._per_product_bad_hits = Counter()
        self._per_full_path_bad_hits = Counter()
        self.debug_datetime = datetime.now().strftime("%Y%m%d-%H%M%S")

    def set_epoch(self, i):
        self.epoch = int(i)

    def _print_bad_hits(self):
        total_hits = self._bad_hits.total()
        hits = [
            f"{key}: {value / total_hits * 100:.2f}% [{value} / {total_hits}]" for key, value in self._bad_hits.items()
        ]
        per_file_hits = self._per_file_bad_hits.most_common(20)
        per_product_hits = self._per_product_bad_hits.most_common(20)
        per_full_path_hits = self._per_full_path_bad_hits.most_common(20)
        nl = "\n"
        print_str = (
            f"=================Bad hits for epoch {self.epoch}================="
            f"\n{nl.join(hits)}\n"
            f"======================================================\n"
            f"Per file hits: \n{nl.join([f'{file}: {hits}' for (file, hits) in per_file_hits])}\n"
            f"Per product hits: \n{nl.join([f'{product}: {hits}' for (product, hits) in per_product_hits])}\n"
            f"Per full path hits: \n{nl.join([f'{full_path}: {hits}' for (full_path, hits) in per_full_path_hits])}\n"
            f"======================================================\n"
        )
        logger.info(print_str)

    def _log_bad_hit(
        self,
        file_path: str | Path,
        hit_type: str,
        product_relative_path: str | Path | None = None,
        product: str | None = None,
        log: bool = False,
    ):
        if log:
            log_path = Path(file_path)
            if product_relative_path is not None:
                log_path = log_path / product_relative_path
            if product is not None:
                log_str = f" {hit_type} for product {product} at path: {log_path!s}"
            else:
                log_str = f" {hit_type} at path: {log_path!s}"
            logger.debug(log_str)

        try:
            file_path = Path(file_path).relative_to(self.paths.parent)
        except Exception:
            file_path = Path(file_path)

        self._bad_hits[hit_type] += 1
        self._per_file_bad_hits[str(file_path)] += 1
        if product_relative_path is not None:
            self._per_full_path_bad_hits[str(Path(file_path) / product_relative_path)] += 1
        if product is not None:
            self._per_product_bad_hits[product] += 1

    def __len__(self):
        # Using length of perfectly divisible square tiling of a tile
        num_tiles = (100_000 / min(self.ground_covers)) ** 2  # 100km x 100km / (gc x gc)
        return int(len(self.file_paths) * num_tiles * self.sample_ratio / (self.batch_size * self.world_size))

    def _load_img(self, path, group=None):
        return xr.open_dataset(
            path,
            group=group,
            cache=False,
            lock=False,
            engine="h5netcdf",
        )

    @lru_cache(maxsize=1024)
    def _load_metadata(self, metadata_path):
        return gpd.read_file(metadata_path)

    def load_sample_areas(self, file_path, ground_cover: int | None = None):
        """
        Load all possible sample areas from the metadata file
        """

        fp = Path(file_path)
        rel_path = fp.relative_to(fp.parents[3])
        metadata_path = Path(self.metadata_dir) / rel_path / self.metadata_cache_name

        if metadata_path.exists():
            footprint_df = self._load_metadata(metadata_path)
            footprint_df = footprint_df.sample(
                frac=1, weights="area_prod", ignore_index=True, random_state=np.random.get_state()[1]
            )

            # select a random ground cover size
            if ground_cover is None:
                if "max_ground_cover" in footprint_df:
                    max_ground_cover = footprint_df["max_ground_cover"].max()
                    ground_cover_idx = [i for i, gc in enumerate(self.ground_covers) if gc <= max_ground_cover]
                    ground_cover_selection = [self.ground_covers[i] for i in ground_cover_idx]
                    ground_cover_selection_weights = [self.ground_cover_weights[i] for i in ground_cover_idx]
                else:
                    ground_cover_selection = self.ground_covers
                    ground_cover_selection_weights = self.ground_cover_weights

                if len(ground_cover_selection) == 0:
                    return None, None

                ground_cover_selection_weights = np.array(ground_cover_selection_weights) / np.sum(
                    ground_cover_selection_weights
                )

                ground_cover = ground_cover_selection[
                    np.random.choice(len(ground_cover_selection), p=ground_cover_selection_weights)
                ]

            crop_crs_wkt = footprint_df.crs.to_wkt()

            # create a buffer around the footprint
            footprint_df["geometry"] = footprint_df.geometry.buffer(
                -ground_cover / 2**0.5,
                cap_style="square",
                join_style="bevel",
                resolution=0,
            ).set_crs(crop_crs_wkt)
            # drop rows where geometry is empty
            footprint_df = footprint_df[~footprint_df.geometry.is_empty]
            if footprint_df.empty:
                return None, None
            return footprint_df, ground_cover
        else:
            return None, None

    def __iter__(self) -> Generator[ProductDataBatch]:
        device_rank = self.global_rank
        worker_info = get_worker_info()
        worker_rank = worker_info.id if worker_info else 0

        if worker_rank == 0 and device_rank == 0:
            logger.info(f"Shuffling file paths for epoch {self.epoch} with random seed {self.random_seed + self.epoch}")

        np.random.default_rng(self.random_seed + self.epoch).shuffle(self.file_paths)

        if self.weighted_sampling:
            if worker_rank == 0 and device_rank == 0:
                logger.info(
                    f"Using weighted sampling for epoch {self.epoch} with random seed {self.random_seed + self.epoch}"
                )
            np.random.default_rng(self.random_seed + self.epoch).shuffle(self.weights)
            return self.weighted_iter()
        else:
            if worker_rank == 0 and device_rank == 0:
                logger.info(
                    f"Using simple sampling for epoch {self.epoch} with random seed {self.random_seed + self.epoch}"
                )
            return self.simple_iter()

    def weighted_iter(self) -> Generator[ProductDataBatch]:
        # devices split
        device_rank, num_devices = self.global_rank, self.world_size

        # workers split
        worker_info = get_worker_info()
        worker_rank, num_workers = (worker_info.id, worker_info.num_workers) if worker_info else (0, 1)

        # total (devices + workers) split by device, then by worker
        num_replicas = num_workers * num_devices
        replica_rank = worker_rank * num_devices + device_rank
        # by worker, then device would be:
        # rank = device_rank * num_workers + worker_rank

        num_tiles = (100_000 / min(self.ground_covers)) ** 2  # 100km x 100km / (gc x gc)
        num_samples = int(len(self.file_paths) * num_tiles * self.sample_ratio / (self.batch_size * num_replicas))
        total_size = num_samples * num_replicas

        replacement = True

        indices = np.arange(total_size)
        indices = indices % len(self.file_paths)

        # subsample
        indices = indices[replica_rank:total_size:num_replicas]
        assert len(indices) == num_samples

        # Weights for this rank
        weights = torch.tensor(self.weights[indices])

        # Subsample indices
        subsample_weighted_samples = torch.multinomial(weights, num_samples, replacement)
        dataset_indices = iter(torch.tensor(indices)[subsample_weighted_samples].tolist())

        for sample in dataset_indices:
            yield self.__getitem__(sample)

    def simple_iter(self) -> Generator[ProductDataBatch]:
        # devices split
        device_rank, num_devices = self.global_rank, self.world_size

        # workers split
        worker_info = get_worker_info()
        worker_rank, num_workers = (worker_info.id, worker_info.num_workers) if worker_info else (0, 1)

        # total (devices + workers) split by device, then by worker
        num_replicas = num_workers * num_devices
        replica_rank = worker_rank * num_devices + device_rank
        # by worker, then device would be:
        # rank = device_rank * num_workers + worker_rank

        i = 0
        num_samples = 0
        while True:
            if i % num_replicas == replica_rank:
                try:
                    s = self.__getitem__(i)
                    num_samples += 1
                    yield s
                except TimeoutError:
                    fp = self.file_paths[i % len(self.file_paths)]
                    logger.error(
                        f"TimeoutError at index {i}, skipping sample for device {device_rank}, worker {worker_rank}, file path: {self.file_paths[i % len(self.file_paths)]}"
                    )
                    self._log_bad_hit(fp, "timeout_error", log=True)
                except Exception as e:
                    logger.error(
                        f"ERROR: at index {i} for device {device_rank}, worker {worker_rank}, file path: {self.file_paths[i % len(self.file_paths)]}"
                    )
                    exc_type, exc_value, exc_traceback = e.__class__, e, e.__traceback__
                    logger.error(f"Exception type: {exc_type.__name__}, value: {exc_value}")
                    logger.error(f"Traceback: {traceback.format_exception(exc_type, exc_value, exc_traceback)}")
                    fp = self.file_paths[i % len(self.file_paths)]
                    self._log_bad_hit(fp, "load_error", log=True)
            if num_samples >= (len(self) * num_devices) // num_replicas:
                logger.info(f"Device: {device_rank}, worker {worker_rank} finished after {num_samples} samples")
                break
            i += 1

    @timeout(seconds=30)
    def __getitem__(self, idx, get_ground_cover=None, crop_center_points_override=None) -> ProductDataBatch:
        worker_info = get_worker_info()
        num_workers = worker_info.num_workers if worker_info is not None else 1
        worker_id = worker_info.id if worker_info is not None else 0

        # Fetches one batch of data, from one single crop
        batch_size = self.batch_size

        MAX_TRIES = 5
        for t in range(MAX_TRIES):
            file_path = self.file_paths[idx % len(self.file_paths)]

            footprints_df, ground_cover = self.load_sample_areas(file_path, get_ground_cover)

            if footprints_df is None:
                self._log_bad_hit(file_path, "no_footprint")
                logger.info(
                    f"no footprint for file {file_path}, with ground cover {ground_cover}, trying again with get ground cover {get_ground_cover}"
                )

                idx += self.world_size * num_workers + 1
                if t == MAX_TRIES - 1:
                    logger.info(
                        f"Failed to find a footprint for file {file_path} after {MAX_TRIES} tries, doing a hail mary pass"
                    )
                    return self.__getitem__(random.randint(0, len(self.file_paths) - 1), get_ground_cover)
            else:
                if t > 0:
                    logger.info(f"Used {t} tries to find a footprint for file {file_path}, ground cover {ground_cover}")
                break

        found_all_products = False
        found_important_products = False
        for sample_try in range(len(footprints_df)):
            product_imgs = {}
            crop_bounds = None
            crop_center_points = None
            crop_crs_wkt = None
            single_bands_s1 = []
            skip_products = []
            cloud_masks_scda2 = []
            cloud_masks_scl = []
            scl_crops = []
            # layover_shadow_masks = []
            incidence_angles = []
            aggregated_incidence_angles = None
            slstr_snow_masks = []
            s1_orbit_direction = None

            prod_imgs = footprints_df.iloc[sample_try].to_dict()

            # Get products for this ground cover
            products = self.ground_cover_products[ground_cover]

            # Filter out products that are not in the metadata
            filtered_products = [p for p in products if p in prod_imgs and prod_imgs[p] is not None]

            # Remove S1 60m if S1 10m is present
            if any("S1" in p and "60m" in p for p in filtered_products) and any(
                "S1" in p and "10m" in p for p in filtered_products
            ):
                filtered_products = [p for p in filtered_products if not ("S1" in p and "60m" in p)]

            s1_product = next((p for p in filtered_products if "S1" in p), None)

            if len(filtered_products) == 0:
                self._log_bad_hit(file_path, "no_filtered_products")
                continue

            if self.strict_data_mode and not all(any(s in p for s in self.sensors) for p in filtered_products):
                self._log_bad_hit(file_path, "missing_products")
                continue

            found_important_products = False
            found_all_products = True
            # NOTE: the order that we iterate through products needs to match
            #       the order that we initialized ConsistentRandomTransform so
            #       the correct scale is applied to each image
            for product in filtered_products:
                product_crops = []
                if any(skip_product in product for skip_product in skip_products):
                    continue
                with self._load_img(
                    os.path.join(file_path, prod_imgs[product]),
                    product.split("-")[-1] if product not in self.SEPARATE_FOLDER_PRODUCTS else None,
                ) as prod_img_dataset:
                    if crop_bounds is None:
                        largest_prod = prod_img_dataset
                        largest_GSD = self.GSD(filtered_products[0])

                        # Offset the geometry by the ground cover size and sample a random crops
                        if crop_center_points_override is not None:
                            crop_center_points = crop_center_points_override
                        else:
                            crop_center_points = uniform(
                                footprints_df.iloc[sample_try]["geometry"], batch_size, rng=np.random.get_state()[1]
                            )
                            if self.batch_size == 1:
                                crop_center_points = np.array([crop_center_points.xy])
                            else:
                                crop_center_points = np.array([p.xy for p in crop_center_points.geoms])  # [B, 2, 1]

                        # Make sure that N and S UTM zones are handled correctly
                        if Proj(footprints_df.crs) != Proj(largest_prod.crs.crs_wkt):
                            transformer = Transformer.from_crs(
                                footprints_df.crs, largest_prod.crs.crs_wkt, always_xy=True
                            )

                            crop_center_points = transformer.transform(
                                crop_center_points[:, 0, 0], crop_center_points[:, 1, 0]
                            )
                            crop_center_points = np.stack(crop_center_points, axis=1).reshape(batch_size, 2, 1)

                        crop_corner_points = crop_center_points - ground_cover / 2

                        # Pixel align
                        crop_aligned_min_x = (
                            largest_prod.x.sel(x=crop_corner_points[:, 0, 0] + largest_GSD / 2, method="nearest").values
                            - largest_GSD / 2
                        )

                        crop_aligned_min_y = (
                            largest_prod.y.sel(y=crop_corner_points[:, 1, 0] + largest_GSD / 2, method="nearest").values
                            - largest_GSD / 2
                        )

                        crop_aligned_min_xy = np.stack([crop_aligned_min_x, crop_aligned_min_y], axis=1).astype(int)

                        crop_bounds = np.stack(
                            [
                                crop_aligned_min_xy[:, 0],
                                crop_aligned_min_xy[:, 1],
                                crop_aligned_min_xy[:, 0] + ground_cover,
                                crop_aligned_min_xy[:, 1] + ground_cover,
                            ],
                            axis=1,
                        )

                        crop_crs_wkt = prod_img_dataset.crs.crs_wkt

                    product_gsd = self.GSD(product)
                    img_size = math.ceil(ground_cover / product_gsd)

                    X, Y = self.get_crop(
                        product_gsd=product_gsd,
                        product_crs_wkt=prod_img_dataset.crs.crs_wkt,
                        crop_bounds=crop_bounds,
                        crop_crs_wkt=crop_crs_wkt,
                        batch_size=batch_size,
                    )

                    if "S1" in product:
                        if "1SSH" in prod_imgs[product]:
                            single_bands_s1.append((f"{product}-1SSH", product))
                        elif "1SSV" in prod_imgs[product]:
                            single_bands_s1.append((f"{product}-1SSV", product))

                        s1_orbit_dir = prod_img_dataset.source_meta.attrs["orbitdirection"]
                        s1_orbit_direction = 0 if s1_orbit_dir == "ascending" else 1

                    if abs(Y).max() > abs(prod_img_dataset.y).max() or abs(X).max() > abs(prod_img_dataset.x).max():
                        self._log_bad_hit(file_path, f"out_of_bounds_{product}", prod_imgs[product], product)
                        found_all_products = False
                        if "S2" in product:
                            skip_products.append(product)
                        break

                    for b in range(batch_size):
                        x = X[b]
                        y = Y[b]

                        # Extract the indices of the image
                        prod_img = prod_img_dataset.sel(x=x, y=y, method="nearest")

                        if prod_img.sizes["x"] != img_size or prod_img.sizes["y"] != img_size:
                            self._log_bad_hit(file_path, f"size_mismatch_{product}", prod_imgs[product], product)
                            # breakpoint()
                            found_all_products = False
                            if "S2" in product:
                                skip_products.append(product)
                            break

                        if product == "S2-20m" and (
                            "SCL-20m" in self.land_cover_products
                            or "WC-10m" in self.land_cover_products
                            or len(self.dem_products) > 0
                        ):
                            try:
                                scl_data = prod_img.SCL.values
                            except Exception as e:
                                logger.error(
                                    f"corrupt file error for SCL file path: {os.path.join(file_path, prod_imgs[product])} error: {e}",
                                )
                                self._log_bad_hit(file_path, "corrupt_file_SCL", prod_imgs[product], product)
                                found_all_products = False
                                if "S2" in product:
                                    skip_products.append(product)
                                if "WC-10m" in self.land_cover_products or len(self.dem_products) > 0:
                                    cloud_masks_scl.append(np.zeros((img_size, img_size), dtype=np.bool_))
                                break
                            if "SCL-20m" in self.land_cover_products:
                                scl_crops.append(scl_data)

                            if "WC-10m" in self.land_cover_products or len(self.dem_products) > 0:
                                cloud_mask = np.zeros((img_size, img_size), dtype=np.bool_)
                                # Ignore no data, saturated, dark area, cloud high prob and snow
                                cloud_mask[scl_data <= 1] = 1
                                # cloud_mask[scl_data == 3] = 1
                                cloud_mask[scl_data == 9] = 1
                                cloud_mask[scl_data == 11] = 1

                                cloud_masks_scl.append(cloud_mask)

                        elif product == "S3-500m" and (
                            "MCD-500m" in self.land_cover_products or "GC-250m" in self.land_cover_products
                        ):
                            # get scda2 cloud mask
                            try:
                                cloud_mask = prod_img.scda2.values
                                cloud_masks_scda2.append(cloud_mask)
                            except Exception as e:
                                logger.error(
                                    f"corrupt file error for cloud mask file path: {os.path.join(file_path, prod_imgs[product])} error: {e}",
                                )
                                self._log_bad_hit(file_path, "corrupt_file_cloud_mask", prod_imgs[product], product)
                                cloud_masks_scda2.append(np.zeros((img_size, img_size), dtype=np.bool_))

                        elif (
                            "S1" in product and self.incidence_angle_products and "incidenceangle" in prod_img.data_vars
                        ):
                            # Try get s1 incidence angle
                            try:
                                incidence_angles.append(prod_img.incidenceangle.values)
                            except Exception:
                                logger.error(
                                    f"corrupt file error for incidence angle file path: {os.path.join(file_path, prod_imgs[product])}"
                                )
                                self._log_bad_hit(
                                    file_path, "corrupt_file_incidence_angle", prod_imgs[product], product
                                )

                                incidence_angles.append(np.zeros((img_size, img_size), dtype=np.bool_))

                        # elif "S1" in product and (
                        #     "WC-10m" in self.land_cover_products
                        #     or "SCL-20m" in self.land_cover_products
                        #     or "GC-250m" in self.land_cover_products
                        # ):
                        #     # try get s1 layover shadow mask
                        #     try:
                        #         layover_shadow_mask = prod_img.layover_shadow.values
                        #         layover_shadow_masks.append(layover_shadow_mask)
                        #     except Exception as e:
                        #         logger.error(
                        #             f"corrupt file error for layover shadow mask file path: {os.path.join(file_path, prod_imgs[product])}"
                        #         )
                        #         self._log_bad_hit(
                        #             file_path,
                        #             f"corrupt_file_layover_shadow_mask"
                        #         layover_shadow_masks.append(np.zeros((img_size, img_size), dtype=np.bool_))

                        # Convert to numpy array
                        try:
                            prod_img = prod_img.bands.values
                        except Exception as e:
                            logger.error(
                                f"corrupt file error for prod file path: {os.path.join(file_path, prod_imgs[product])} , error: {e}",
                            )
                            self._log_bad_hit(file_path, f"corrupt_file_{product}", prod_imgs[product], product)
                            found_all_products = False
                            if "S2" in product:
                                skip_products.append(product)
                            break

                        nan_ratio = np.isnan(prod_img).sum(axis=(1, 2)) / (prod_img.shape[1] * prod_img.shape[2])
                        if (nan_ratio > self.max_nan_ratio).any():
                            self._log_bad_hit(file_path, f"nan_ratio_{product}", prod_imgs[product], product)
                            found_all_products = False
                            if "S2" in product:
                                skip_products.append(product)
                            break

                        if product == "S3-500m" and (
                            "MCD-500m" in self.land_cover_products or "GC-250m" in self.land_cover_products
                        ):
                            # NDSI snow  mask
                            try:
                                snow_threshold = 0.5
                                snow_mask = (prod_img[0] - prod_img[4]) / (prod_img[0] + prod_img[4]) > snow_threshold
                                slstr_snow_masks.append(snow_mask)
                            except Exception as e:
                                logger.error(
                                    f"corrupt file error for snow mask file path: {os.path.join(file_path, prod_imgs[product])} error: {e}",
                                )
                                self._log_bad_hit(file_path, "corrupt_file_snow_mask", prod_imgs[product], product)
                                slstr_snow_masks.append(np.zeros((img_size, img_size), dtype=np.bool_))

                        product_crops.append(prod_img)

                if len(product_crops) == batch_size:
                    product_imgs[product] = product_crops
                if len(scl_crops) == batch_size:
                    product_imgs["SCL-20m"] = scl_crops
                if len(incidence_angles) == batch_size and s1_product is not None:
                    if not self.aggregate_incidence_angle:
                        if "10m" in s1_product:
                            product_imgs["INC-10m"] = incidence_angles
                        elif "60m" in s1_product:
                            product_imgs["INC-60m"] = incidence_angles
                    else:
                        aggregated_incidence_angles = np.nanmean(np.stack(incidence_angles, axis=0), axis=(1, 2))
                        if np.isnan(aggregated_incidence_angles).any():
                            self._log_bad_hit(
                                file_path, f"aggregated_incidence_angle_nan_{s1_product}", prod_imgs[s1_product]
                            )
                            aggregated_incidence_angles = None

            if found_all_products:  # Found all products for a given sample tile, finish the loop
                # the intersection of the products found (i.e, if we don't find S1 thats ok...)
                break

            # NOTE: When we are not strict, we only look for 'important products' and finish the loop
            if not self.strict_data_mode:
                for product in product_imgs.keys():
                    if any(important_product in product for important_product in self.important_products):
                        found_important_products = True
                        break

                if found_important_products:
                    break

        if len(cloud_masks_scda2) < batch_size and (
            "GC-250m" in self.land_cover_products or "MCD-500m" in self.land_cover_products
        ):
            img_size = math.ceil(ground_cover / 500)
            cloud_masks_scda2 = np.zeros((batch_size, img_size, img_size), dtype=np.bool_)
        if len(slstr_snow_masks) < batch_size and (
            "GC-250m" in self.land_cover_products or "MCD-500m" in self.land_cover_products
        ):
            img_size = math.ceil(ground_cover / 500)
            slstr_snow_masks = np.zeros((batch_size, img_size, img_size), dtype=np.bool_)
        if len(cloud_masks_scl) < batch_size and "WC-10m" in self.land_cover_products:
            img_size = math.ceil(ground_cover / 20)
            cloud_masks_scl = np.zeros((batch_size, img_size, img_size), dtype=np.bool_)
        # if len(layover_shadow_masks) < batch_size and (
        #     "WC-10m" in self.land_cover_products or "GC-250m" in self.land_cover_products or "SCL-20m" in self.land_cover_products
        # ):
        #
        #     img_size = math.ceil(ground_cover / 20)
        #     layover_shadow_masks = np.zeros((batch_size, img_size, img_size), dtype=np.bool_)

        if (not found_all_products and self.strict_data_mode) or len(product_imgs) == 0:
            self._log_bad_hit(file_path, "no_products")
            return self.__getitem__(idx + self.world_size * num_workers + 1, get_ground_cover)
        elif not (found_important_products or found_all_products) and not self.strict_data_mode:
            self._log_bad_hit(file_path, "no_important_products")
            return self.__getitem__(idx + self.world_size * num_workers + 1, get_ground_cover)

        tile = Path(file_path).parents[2].name
        land_cover_path = self.land_cover_dir / f"{tile}_maps.nc"

        if crop_crs_wkt is not None:
            land_cover_products = [
                lcp
                for lcp in self.land_cover_products
                if lcp in self.ground_cover_products[ground_cover] and "SCL" not in lcp
            ]
        else:
            land_cover_products = []

        for land_cover_product_gsd in land_cover_products:
            land_cover_product_crops = []
            land_cover_product, land_cover_gsd_name = land_cover_product_gsd.split("-")
            if not land_cover_path.exists():
                continue
            try:
                with self._load_img(land_cover_path, group=land_cover_gsd_name) as land_cover_dataset:
                    land_cover_gsd = self.GSD(land_cover_product_gsd)
                    img_size = math.ceil(ground_cover / land_cover_gsd)

                    X, Y = self.get_crop(
                        product_gsd=land_cover_gsd,
                        product_crs_wkt=land_cover_dataset.crs.crs_wkt,
                        crop_bounds=crop_bounds,
                        crop_crs_wkt=crop_crs_wkt,
                        batch_size=batch_size,
                    )

                    if abs(Y).max() > abs(land_cover_dataset.y).max() or abs(X).max() > abs(land_cover_dataset.x).max():
                        self._log_bad_hit(file_path, f"out_of_bounds_{land_cover_product}", log=True)
                        continue

                    # Map values to range 0 to max classes
                    map_values = {
                        class_value: i for i, class_value in enumerate(getattr(self, f"{land_cover_product}_CLASSES"))
                    }
                    if hasattr(self, f"{land_cover_product}_IGNORE_CLASSES"):
                        for class_value in getattr(self, f"{land_cover_product}_IGNORE_CLASSES"):
                            map_values[class_value] = 0

                    for b in range(batch_size):
                        x = X[b]
                        y = Y[b]

                        lc_prod_img = land_cover_dataset.sel(x=x, y=y, method="nearest")

                        if lc_prod_img.sizes["x"] != img_size or lc_prod_img.sizes["y"] != img_size:
                            self._log_bad_hit(file_path, f"size_mismatch_{land_cover_product}")
                            continue

                        # Convert to numpy array
                        lc_prod_img = lc_prod_img[land_cover_product].values

                        if land_cover_product == "MCD":
                            lc_prod_img[cloud_masks_scda2[b] == 1] = 0
                            lc_prod_img[slstr_snow_masks[b] == 1] = 0
                        elif land_cover_product == "GC":
                            # resample cloud mask from 500m to 250m
                            c_mask = cloud_masks_scda2[b]
                            c_mask = (
                                F.resize(
                                    torch.tensor(c_mask).long().unsqueeze(0),
                                    size=(img_size, img_size),
                                    interpolation=InterpolationMode.NEAREST_EXACT,
                                )
                                .squeeze(0)
                                .bool()
                                .numpy()
                            )

                            lc_prod_img[c_mask == 1] = 0
                            s_mask = slstr_snow_masks[b]
                            s_mask = (
                                F.resize(
                                    torch.tensor(s_mask).long().unsqueeze(0),
                                    size=(img_size, img_size),
                                    interpolation=InterpolationMode.NEAREST_EXACT,
                                )
                                .squeeze(0)
                                .bool()
                                .numpy()
                            )
                            lc_prod_img[s_mask == 1] = 0
                        elif land_cover_product == "WC":
                            # resample from 20m to 10m
                            c_mask = cloud_masks_scl[b]
                            c_mask = np.repeat(c_mask, 2, axis=0)
                            c_mask = np.repeat(c_mask, 2, axis=1)
                            lc_prod_img[c_mask == 1] = 0

                        lc_prod_img = np.vectorize(map_values.get)(lc_prod_img)
                        land_cover_product_crops.append(lc_prod_img)

            except Exception as e:
                logger.error(
                    f"load file error for land cover product file path: {file_path}, land cover product: {land_cover_product_gsd} , error: {e}"
                )
                self._log_bad_hit(file_path, f"load_error_{land_cover_product}")

            if len(land_cover_product_crops) == batch_size:
                product_imgs[land_cover_product_gsd] = land_cover_product_crops

        if self.dem_products and self.dem_dir is not None and crop_bounds is not None:
            try:
                dem_product, dem_data = self.get_dem_data(
                    file_path=file_path,
                    ground_cover=ground_cover,
                    crop_bounds=crop_bounds,
                    crop_crs_wkt=crop_crs_wkt,
                    batch_size=batch_size,
                    cloud_masks_scl=cloud_masks_scl,
                )
                if dem_data is not None:
                    product_imgs[dem_product] = dem_data

            except Exception as e:
                logger.error(f"DEM ERROR, {e}")
                self._log_bad_hit(file_path, "load_error_dem", log=True)

        # Load ERA5 Land data
        era5_land_data = None
        if self.era5_land_products is not None and crop_center_points is not None:
            fp = Path(file_path)
            try:
                file_date = footprints_df.iloc[sample_try]["timestamp"]
            except Exception as e:
                logger.error(f"timestamp error for era5 file path: {fp}, error: {e}")
                file_date = datetime.strptime(str(fp.relative_to(fp.parents[2])), "%Y/%m/%d")
            try:
                era5_land_data = self.get_era5_product(
                    file_date, ground_cover, crop_center_points, crop_crs_wkt, era5_land=True
                )
            except Exception as e:
                logger.error(f"load file error for era5 file date: {file_date}, path {fp}, error: {e}")
                era5_land_data = None

        center_coords = Transformer.from_crs(crop_crs_wkt, "epsg:4326", always_xy=True).transform(
            crop_center_points[:, 0, 0], crop_center_points[:, 1, 0]
        )
        center_coords = np.stack(center_coords, axis=1).reshape(batch_size, 2)

        month = float(Path(file_path).parts[-2])

        product_metadata = {}

        # NOTE: the order that we iterate through products needs to match
        #       the order that we initialized ConsistentRandomTransform so
        #       the correct scale is applied to each image
        # consistent cropping in custom transforms has an internal counter,
        # only trigger counting once we are sure we have all the product images
        # NOTE: from here the product will be potentially resized
        # Generating random crop params consistent across all products
        self.cont_cons_rand_crops[ground_cover].generate_crop_params()

        for product, prod_img in product_imgs.items():
            if isinstance(prod_img, list):
                prod_img = np.stack(prod_img, axis=0)
            for single_band_s1 in single_bands_s1:
                if single_band_s1[1] == product:
                    prod_img = np.concatenate([prod_img, prod_img.copy()], axis=1)

            prod_img = _to_tensor(prod_img)

            ###############################
            # NOTE: ideally we should multilook before applying the custom transform
            #       it is ok as along as we are only cropping, flipping etc.
            #       but if we are doing something like resizing or blurring, we should multilook first
            prod_img = self.custom_transform[ground_cover][product](prod_img)
            ###############################

            # Potentially resampled
            GSD = self.GSD(product, resampled=True)

            if self.multilook_S1 and "S1" in product:
                possible_factors = [
                    factor
                    for factor in self.s1_multilook_factors
                    if prod_img.shape[-2] % factor == 0 and prod_img.shape[-2] // factor >= self.IMAGE_MIN_LIMIT
                ]
                multilook_factor = random.choice(possible_factors)
                if multilook_factor > 1:
                    prod_img = self.multilook(prod_img, mulitlook_factor=multilook_factor, pad=False)
                GSD *= multilook_factor

            product_imgs[product] = prod_img

            product_metadata[product] = {
                "img_size": prod_img.shape[-2:],
                "GSD": GSD,
            }

        self.cons_horiz_flips[ground_cover].count = 0  # Reset the counter to ensure consistent flip
        self.cons_vertical_flips[ground_cover].count = 0  # Reset the counter to ensure consistent flip

        # NOTE: transform / normalize era5 data is done automatically in get_era5_product

        # Hack to handle single band S1 products
        if single_bands_s1:
            for single_band_s1 in single_bands_s1:
                if single_band_s1[1] in product_imgs:
                    product_imgs[single_band_s1[0]] = product_imgs[single_band_s1[1]]
                    del product_imgs[single_band_s1[1]]

        self._hits += 1
        if self._hits % 25 == 0:
            worker_info = get_worker_info()
            worker_id = 0 if worker_info is None else worker_info.id

            if worker_id == 0:
                logger.info(f"Worker {worker_id}, idx: {idx}, hits: {self._hits}")
                logger.info(
                    f"crop hit rate: {self._hits / (self._hits + self._per_full_path_bad_hits.total()) * 100:.2f}%"
                )
                self._print_bad_hits()

        data = ProductDataBatch(
            product_imgs=product_imgs,
            metadata=MetaData(
                channel_params=product_metadata,
                ground_cover=ground_cover,
                center_coords=torch.tensor(center_coords),
                month=torch.tensor(month).unsqueeze(0).expand(batch_size, -1),
                s1_orbit_direction=torch.tensor(s1_orbit_direction).unsqueeze(0).expand(batch_size, -1)
                if s1_orbit_direction is not None
                else None,
                s1_incidence_angles=torch.tensor(aggregated_incidence_angles).unsqueeze(1)
                if aggregated_incidence_angles is not None
                else None,
                file_path=str(file_path) if self.full_return else None,
            ),
            era5_land_data=era5_land_data,
        )

        return data

    def get_dem_data(
        self, file_path, ground_cover, crop_bounds, crop_crs_wkt, batch_size, cloud_masks_scl=None
    ) -> dict[str, torch.Tensor]:
        if "dem-60m" in self.dem_products and "dem-60m" in self.ground_cover_products[ground_cover]:
            dem_product = "dem-60m"
            dem_gsd = 60
        elif "dem-10m" in self.dem_products and "dem-10m" in self.ground_cover_products[ground_cover]:
            dem_product = "dem-10m"
            dem_gsd = 10
        else:
            # logger.debug(f"No DEM product found for ground cover {ground_cover} in file {file_path}")
            return "dem", None

        tile = Path(file_path).parents[2].name[1:]
        dem_path = self.dem_dir / f"{dem_gsd}" / f"{tile}_dem.nc"
        if not dem_path.exists():
            return dem_product, None

        dem_product_key = dem_product.split("-")[0]
        dem_crops = []
        dem_type = self.dem_type  # slope or elevation or both

        with self._load_img(dem_path, group=None) as dem_dataset:
            img_size = math.ceil(ground_cover / dem_gsd)
            X, Y = self.get_crop(
                product_gsd=dem_gsd,
                product_crs_wkt=dem_dataset.crs.crs_wkt,
                crop_bounds=crop_bounds,
                crop_crs_wkt=crop_crs_wkt,
                batch_size=batch_size,
            )

            if abs(Y).max() > abs(dem_dataset.y).max() or abs(X).max() > abs(dem_dataset.x).max():
                self._log_bad_hit(dem_path, f"out_of_bounds_{dem_product}", product=dem_product, log=True)
                return dem_product, None

            dem_slope = None
            for b in range(batch_size):
                x = X[b]
                y = Y[b]

                dem_prod_img = dem_dataset.sel(x=x, y=y, method="nearest")

                if dem_prod_img.sizes["x"] != img_size or dem_prod_img.sizes["y"] != img_size:
                    self._log_bad_hit(dem_path, f"size_mismatch_{dem_product}", product=dem_product, log=True)
                    return dem_product, None

                # Convert to numpy array
                dem_prod_img = dem_prod_img[dem_product_key].values

                if dem_type in ["slope", "both"]:
                    # Calculate gradients in x and y directions
                    dy, dx = np.gradient(dem_prod_img, axis=(1, 2))
                    # Compute slope magnitude (squared gradients)
                    dem_slope = dx**2 + dy**2
                    dem_slope = np.sqrt(dem_slope)

                    if dem_type == "both":
                        # Stack elevation and slope as separate channels
                        dem_prod_img = np.concatenate([dem_prod_img, dem_slope], axis=0)
                    elif dem_type == "slope":
                        dem_prod_img = dem_slope

                def normalize(arr, normalize, index=0):
                    if normalize == "local_min":
                        arr[index] = arr[index] - arr[index].min()
                    elif normalize == "local_median":
                        arr[index] = arr[index] - arr[index].median()
                    elif normalize == "local_mean":
                        arr[index] = arr[index] - arr[index].mean()
                    elif normalize == "local_min_max":
                        arr[index] = (arr[index] - arr[index].min()) / (arr[index].max() - arr[index].min())
                    elif normalize == "local_mean_std":
                        arr[index] = (arr[index] - arr[index].mean()) / arr[index].std()
                    elif normalize == "none":
                        pass
                    else:
                        raise ValueError(f"Unknown normalization type: {normalize}")

                    return arr

                if dem_type in ["elevation", "both"]:
                    dem_prod_img = normalize(dem_prod_img, self.dem_normalize["elevation"], index=0)

                if dem_type in ["slope", "both"]:
                    slope_index = 1 if dem_type == "both" else 0
                    dem_prod_img = normalize(dem_prod_img, self.dem_normalize["slope"], index=slope_index)

                if dem_gsd == 10 and len(cloud_masks_scl) == batch_size:
                    # resample from 20m to 10m
                    c_mask = cloud_masks_scl[b]
                    c_mask = np.repeat(c_mask, 2, axis=0)
                    c_mask = np.repeat(c_mask, 2, axis=1)
                    c_mask = c_mask[None, :, :]
                    if c_mask.shape[0] != dem_prod_img.shape[0]:
                        c_mask = np.repeat(c_mask, dem_prod_img.shape[0], axis=0)
                    dem_prod_img[c_mask == 1] = float("nan")  # Will be masked out in the loss
                if dem_gsd == 60 and len(cloud_masks_scl) == batch_size:
                    # resample from 20m to 60m
                    c_mask = cloud_masks_scl[b]
                    c_mask = (
                        F.resize(
                            torch.tensor(c_mask).long().unsqueeze(0),
                            size=(img_size, img_size),
                            interpolation=InterpolationMode.NEAREST_EXACT,
                        )
                        .bool()
                        .numpy()
                    )
                    if c_mask.shape[0] != dem_prod_img.shape[0]:
                        c_mask = np.repeat(c_mask, dem_prod_img.shape[0], axis=0)
                    dem_prod_img[c_mask == 1] = float("nan")  # Will be masked out in the loss

                nan_ratio = np.isnan(dem_prod_img).sum(axis=(1, 2)) / (dem_prod_img.shape[1] * dem_prod_img.shape[2])
                if (nan_ratio > 0.5).any():
                    # NOTE: it does not really matter if there is a lot of Nans, as they will be masked out
                    self._log_bad_hit(dem_path, f"nan_ratio_{dem_product}", product=dem_product)
                    return dem_product, None

                dem_crops.append(dem_prod_img)

        if len(dem_crops) != batch_size:
            return dem_product, None

        return dem_product, dem_crops

    def get_crop(
        self,
        product_gsd: float,
        product_crs_wkt: str,
        crop_bounds: np.ndarray,
        crop_crs_wkt: str,
        batch_size: int,
    ) -> tuple[np.ndarray, np.ndarray, int]:
        """
        Generate crop indices for a product.

        Args:
            product_gsd: Ground Sample Distance of the product
            product_crs_wkt: Coordinate Reference System of the product in WKT format
            crop_bounds: Bounds of the crop as [x_min, y_min, x_max, y_max] for each sample in batch
            crop_crs_wkt: CRS of the crop bounds in WKT format
            batch_size: Number of samples in batch

        Returns:
            X: Array of x-coordinates for each sample
            Y: Array of y-coordinates for each sample
        """
        # Handle N and S UTMs difference
        if product_crs_wkt != crop_crs_wkt:
            transformer = Transformer.from_crs(crop_crs_wkt, product_crs_wkt, always_xy=True)

            # Transform the coordinates of the bounding box
            x_min = crop_bounds[:, 0]
            y_min = crop_bounds[:, 1]
            x_max = crop_bounds[:, 2]
            y_max = crop_bounds[:, 3]
            x_min_transformed, y_min_transformed = transformer.transform(x_min, y_min)
            x_max_transformed, y_max_transformed = transformer.transform(x_max, y_max)

            product_crop_bounds = np.round(
                np.stack([x_min_transformed, y_min_transformed, x_max_transformed, y_max_transformed], axis=1)
            ).astype(np.int32)
        else:
            product_crop_bounds = crop_bounds

        X = np.stack(
            [np.arange(product_crop_bounds[i, 0], product_crop_bounds[i, 2], product_gsd) for i in range(batch_size)]
        )

        Y = np.stack(
            [np.arange(product_crop_bounds[i, 1], product_crop_bounds[i, 3], product_gsd) for i in range(batch_size)]
        )

        # Add half the GSD to get pixel centers
        X += int(product_gsd / 2)
        Y += int(product_gsd / 2)

        return X, Y
