import logging
import os
import re
import warnings
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Union

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shapely
import xarray as xr
from shapely import MultiPoint
from shapely.geometry import Polygon, box
from torch.utils.data import Dataset, get_worker_info

from thor.data.thor_dataset_base import THORDatasetBase

logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def print_node(node, indent: int = 0, node_str=""):
    match node:
        case RootNode(children=children):
            node_str += "RootNode:"
            for child in children:
                node_str += "\n"
                node_str += print_node(child, indent + 1)
        case Node(file=file, children=children, footprint=footprint):
            node_str += f"{'  ' * indent}Node({Path(file).name}, {footprint.area.item()}):"
            for child in children:
                node_str += "\n"
                node_str += print_node(child, indent + 1)
        case Leaf(file=file, footprint=footprint):
            return f"{'  ' * indent}Leaf({Path(file).name}, {footprint.area.item()})"
    return node_str


@dataclass
class RootNode:
    children: list[Union["Node", "Leaf"]]
    level: int = 0
    footprint: Polygon | None = None
    crs: str | None = None
    timestamp: str | None = None

    def __repr__(self):
        return print_node(self)


@dataclass
class Node:
    file: str
    products: list[str]
    children: list[Union["Node", "Leaf"]]
    footprint: Polygon | None = None
    crs: str | None = None
    timestamp: str | None = None
    level: int | None = None
    parent: Union["Node", "RootNode", None] = None

    def __repr__(self):
        return print_node(self)


@dataclass
class Leaf:
    file: str
    products: list[str]
    footprint: Polygon | None = None
    crs: str | None = None
    timestamp: str | None = None
    level: int | None = None
    parent: Union["Node", "RootNode"] | None = None

    def __repr__(self):
        return print_node(self)


SampleNode = RootNode | Node | Leaf


class FootprintBuilder(Dataset, THORDatasetBase):
    def __init__(
        self,
        paths: str | Iterable[str],
        ground_covers: int | list[int],
        kml_path: str | None = None,
        out_dir: str | None = None,
        split: str = "train",
        products: list | dict = [  # noqa: B006
            "S2-10m",
            "S2-20m",
            "S2-60m",
            "S1-10m",
            "S1-60m",
        ],
        data_percent: float = 1.0,
        include_filter: str | None = None,
        exclude_filter: str | None = None,
        important_products: list[str] | None = None,
        save_fig=False,
        overwrite=False,
        verbose=True,
        use_concave_hull=True,
        hull_sample_step=100,
        min_area_ratio: float | None = 0.10,
        min_length: int | None = None,  # 5000
        only_check_for_valid_footprints: bool = False,
        **kwargs,
    ):
        super().__init__(
            paths=paths,
            split=split,
            products=products,
            ground_covers=ground_covers,
            data_percent=data_percent,
            include_filter=include_filter,
            exclude_filter=exclude_filter,
            important_products=important_products,
            **kwargs,
        )

        self.save_fig = save_fig
        self.verbose = verbose
        self.overwrite = overwrite
        self.use_concave_hull = use_concave_hull
        self.hull_sample_step = hull_sample_step
        self.min_area_ratio = min_area_ratio  # Minimum area ratio for footprint to be considered valid
        self.min_length = min_length  # Minimum length from centroid to boundary for footprint to be considered valid
        self.only_check_for_valid_footprints = (
            only_check_for_valid_footprints  # If True, only check for valid footprints and do not build sample graph
        )

        self.file_paths, self.file_path_map = self.load_files(split=split, strict=False)
        logger.info(f"Loaded {len(self.file_paths)} files")

        self._possible_too_small_files_logger = None
        self._possible_nan_files_logger = None

        if out_dir is None:
            out_dir = self.get_metadata_dir()
        self.out_dir = Path(out_dir)

        self._tile_bounds = None
        self.kml_path = kml_path

    @property
    def possible_nan_files_logger(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        if self._possible_nan_files_logger is None:
            current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
            # Set up file loggers
            possible_nan_files_logger = logging.getLogger("possible_nan_files")
            possible_nan_files_logger.setLevel(logging.INFO)
            possible_nan_files_handler = logging.FileHandler(f"possible_nan_files_{current_time}_{worker_id}.log")
            possible_nan_files_handler.setLevel(logging.INFO)
            possible_nan_files_formatter = logging.Formatter("%(message)s")
            possible_nan_files_handler.setFormatter(possible_nan_files_formatter)
            possible_nan_files_logger.addHandler(possible_nan_files_handler)
            self._possible_nan_files_logger = possible_nan_files_logger
        return self._possible_nan_files_logger

    @property
    def possible_too_small_files_logger(self):
        worker_info = get_worker_info()
        worker_id = worker_info.id if worker_info is not None else 0
        if self._possible_too_small_files_logger is None:
            current_time = datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")
            # Set up file loggers
            possible_too_small_files_logger = logging.getLogger("possible_too_small_files")
            possible_too_small_files_logger.setLevel(logging.INFO)
            possible_too_small_files_handler = logging.FileHandler(
                f"possible_too_small_files_{current_time}_{worker_id}.log"
            )
            possible_too_small_files_handler.setLevel(logging.INFO)
            possible_too_small_files_formatter = logging.Formatter("%(message)s")
            possible_too_small_files_handler.setFormatter(possible_too_small_files_formatter)
            possible_too_small_files_logger.addHandler(possible_too_small_files_handler)
            self._possible_too_small_files_logger = possible_too_small_files_logger
        return self._possible_too_small_files_logger

    @property
    def tile_bounds(self):
        if self._tile_bounds is None:
            if self.kml_path is not None:
                import fiona

                fiona.drvsupport.supported_drivers["kml"] = "rw"
                fiona.drvsupport.supported_drivers["KML"] = "rw"
                kml_gdf = gpd.read_file(self.kml_path, driver="LIBKML")
                self._tile_bounds = kml_gdf
                return self._tile_bounds
            else:
                return None
        else:
            return self._tile_bounds

    def get_tile_footprint(self, tile):
        def extract_epsg(html_description):
            # Define the regex pattern to extract the EPSG value
            pattern = re.compile(r"<b>EPSG</b></font></td><td.*?> <font.*?>(\d+)</font></td>")

            # Search for the pattern in the HTML description
            match = pattern.search(html_description)

            # If a match is found, return the EPSG value, else return None
            if match:
                return match.group(1)
            else:
                return None

        tile_data = self.tile_bounds[self.tile_bounds["Name"] == tile]
        if tile_data.empty:
            logger.warning(f"Could not find tile {tile}")
            return None
        desc = tile_data["Description"].item()
        geom = tile_data["geometry"].item()

        dst_epsg = extract_epsg(desc)
        footprint = next(iter(geom.geoms))
        footprint = Polygon([(x, y) for x, y, z in footprint.exterior.coords])  # Drop z

        src_epsg = "4326"  # WGS84

        footprint = gpd.GeoSeries([footprint], crs=f"EPSG:{src_epsg}")
        footprint = footprint.to_crs(f"EPSG:{dst_epsg}")

        return footprint

    def __len__(self):
        return int(len(self.file_paths) * self.data_percent)

    def load_img_paths(self, file_path, product):
        """
        Finds all images for a given product and product gsd at a given file path
        """
        product_gsd = product.split("-")[-1]

        # Build list of image paths for the product
        if product in self.file_path_map[file_path]:
            img_paths = [
                os.path.join(file_path, rel_img_path) for rel_img_path in self.file_path_map[file_path][product]
            ]
        else:
            img_paths = []

        # filter files on our product gsd
        netcdf_product_group = product_gsd
        if product in self.SEPARATE_FOLDER_PRODUCTS:
            # These products are stored in separate folders
            netcdf_product_group = None

        return img_paths, netcdf_product_group

    def get_footprint(self, prod_img_meta, crop_crs_wkt):
        footprint_meta = prod_img_meta.source_meta.footprint
        if not isinstance(footprint_meta, list):
            footprint_meta = [footprint_meta]

        new_footprint = gpd.GeoSeries.from_wkt(footprint_meta, crs=prod_img_meta.source_meta.footprint_srs)

        new_footprint = new_footprint.to_crs(crop_crs_wkt)

        if len(new_footprint) > 1:
            new_footprint = new_footprint.buffer(10, cap_style="square", join_style="bevel", resolution=0)
            new_footprint = gpd.GeoSeries(new_footprint.union_all()).set_crs(crop_crs_wkt)
            new_footprint = new_footprint.buffer(-10, cap_style="square", join_style="bevel", resolution=0).set_crs(
                crop_crs_wkt
            )
        return new_footprint

    def get_data_concave_hull(self, img_path: str, netcdf_group: str | None = None, ratio=0.1) -> Polygon | None:
        """
        Load data from image file and compute its concave hull.

        Args:
            img_path: Path to the netCDF image file
            netcdf_group: NetCDF group to read from

        Returns:
            concave hull polygon in the image's CRS, or None if failed
        """
        try:
            with xr.open_dataset(img_path, group=netcdf_group, cache=False, engine="h5netcdf") as ds:
                data_array = ds.bands

                # Ensure we have spatial reference
                if hasattr(ds, "crs") and hasattr(data_array, "rio"):
                    data_array.rio.write_crs(ds.crs.crs_wkt, inplace=True)

                sample_step = self.hull_sample_step

                gsd = data_array.coords["x"].values[1] - data_array.coords["x"].values[0]

                sample_step = max(int(sample_step * 10 / gsd), 10)

                x_samples = data_array.coords["x"].values[::sample_step]
                y_samples = data_array.coords["y"].values[::sample_step]

                points = data_array.sel(y=y_samples, x=x_samples).values.transpose(0, 2, 1)

                # Find valid (non-NaN) points
                valid_mask = np.all(~np.isnan(points), axis=0)
                valid_indices = np.where(valid_mask)

                points = np.column_stack((x_samples[valid_indices[0]], y_samples[valid_indices[1]]))

                if len(points) < 3:
                    if self.verbose:
                        logger.warning(f"Insufficient valid points for concave hull: {len(valid_indices[0])}")
                    return None

                x = points[:, 0]
                y = points[:, 1]

                points = gpd.GeoSeries(MultiPoint(gpd.points_from_xy(x, y, crs=ds.crs.crs_wkt)), crs=ds.crs.crs_wkt)

                # hull = points.convex_hull
                hull = points.concave_hull(
                    ratio=ratio,  # Lower is more aggressive, higher is more conservative
                    allow_holes=False,  # Don't Allow holes in the hull
                )

                if hull is not None and hasattr(ds, "crs"):
                    hull = hull.set_crs(ds.crs.crs_wkt)

                return hull

        except Exception as e:
            if self.verbose:
                logger.warning(f"Error computing data concave hull for {img_path}: {e}")
            return None

    def get_enhanced_footprint(self, prod_img_meta, crop_crs_wkt, img_path: str, netcdf_group: str | None = None):
        """
        Enhanced footprint generation combining metadata footprint with data concave hull.

        Args:
            prod_img_meta: Product image metadata
            crop_crs_wkt: Target CRS
            img_path: Path to image file
            netcdf_group: NetCDF group

        Returns:
            Enhanced footprint as GeoSeries
        """
        original_footprint = self.get_footprint(prod_img_meta, crop_crs_wkt)

        data_hull = self.get_data_concave_hull(img_path, netcdf_group)

        if data_hull is None:
            if self.verbose:
                logger.warning(f"Could not compute data concave hull for {img_path}, skipping")
            return None

        # Convert hull to same CRS as footprint
        data_hull = data_hull.to_crs(crop_crs_wkt)

        # Intersect metadata footprint with data concave hull
        try:
            intersected = self.intersect_footprints(original_footprint, data_hull, crop_crs_wkt, limit_area=False)

            if intersected is not None and not intersected.is_empty.any():
                return intersected
            else:
                if self.verbose:
                    logger.warning(f"No intersection between metadata footprint and data hull for {img_path}, skipping")
                return None

        except Exception as e:
            if self.verbose:
                logger.warning(f"Error intersecting footprints for {img_path}: {e}\n Skipping")
            return None

    def build_sample_graph(self, products: list[str], file_path: str, node: SampleNode, level=0, crop_crs_wkt=None):
        if level >= len(products):  # Ugly
            return

        product = products[level]

        try:
            # Find all images for the product
            img_paths, netcdf_group = self.load_img_paths(file_path, product)

            if len(img_paths) == 0:
                if self.verbose:
                    logger.warning(f"Could not find any images for product {product} at file_path: {file_path}")
                return self.build_sample_graph(products, file_path, node, level + 1, crop_crs_wkt)

            if crop_crs_wkt is None:
                with xr.open_dataset(
                    img_paths[0],
                    group=netcdf_group,
                    # decode_coords="all",
                    engine="h5netcdf",
                    cache=False,
                ) as prod_img_dataset:
                    crop_crs_wkt = prod_img_dataset.crs.crs_wkt

        except Exception as e:
            logger.warning(f"Error loading images for product {product} file_path {file_path}: {e}")

            return self.build_sample_graph(products, file_path, node, level + 1, crop_crs_wkt)

        found_at_least_one = False
        for img_path in img_paths:
            try:
                # Get source metadata footprint, NB! the netcdf might contain cropped images, but the footprints are not updated!
                group = netcdf_group if product in self.SEPARATE_FOLDER_PRODUCTS else None
                with xr.open_dataset(img_path, group=group, cache=False, engine="h5netcdf") as prod_img_meta:
                    if ("S1" in product or "S2" in product) and self.use_concave_hull:
                        # Use enhanced footprint that combines metadata with data concave hull
                        footprint = self.get_enhanced_footprint(prod_img_meta, crop_crs_wkt, img_path, netcdf_group)
                        if footprint is None:
                            logger.warning(f"Could not compute enhanced footprint for {img_path}, skipping")
                            self.possible_nan_files_logger.info(img_path)
                            continue
                    else:
                        footprint = self.get_footprint(prod_img_meta, crop_crs_wkt)

                    timestamp = None
                    try:
                        timestamp_strs = prod_img_meta.source_meta.endposition
                        if not isinstance(timestamp_strs, list):
                            timestamp_strs = [timestamp_strs]
                        for timestamp_str in timestamp_strs:
                            if timestamp_str.endswith("Z"):
                                timestamp_str = timestamp_str[:-1]
                            ts = datetime.strptime(timestamp_str, "%Y-%m-%dT%H:%M:%S.%f")
                            if timestamp is None or ts > timestamp:
                                timestamp = ts
                    except Exception as e:
                        logger.warning(f"Error parsing timestamp for {img_path}: {e}")

                # Get bounding box of the image
                with xr.open_dataset(img_path, group=netcdf_group, cache=False, engine="h5netcdf") as prod_img:
                    prod_img.rio.write_crs(prod_img.crs.crs_wkt, inplace=True)
                    bounds = prod_img.rio.bounds()
                    bounds = gpd.GeoSeries(box(*bounds), crs=prod_img.crs.crs_wkt)

                # Intersect the footprint with the bounding box
                footprint = self.intersect_footprints(footprint, bounds, crop_crs_wkt, limit_area=False)

            except Exception as e:
                logger.warning(f"Error loading image {img_path}: {e}")
                continue

            if footprint is None:
                logger.warning(f"Footprint is None for image {img_path}")
                continue

            sanity_footprint = footprint.copy()
            try:
                sanity_footprint = sanity_footprint.to_crs(sanity_footprint.estimate_utm_crs())
            except Exception as e:
                logger.info(f"Error converting footprint to UTM CRS for {img_path}: {e}")
            sanity_footprint_area = sanity_footprint.area.item() / (1000**2)  # Convert to km^2
            sanity_footprint_shortest_length = (
                sanity_footprint.boundary.shortest_line(sanity_footprint.centroid).length.item() / 1000
            )
            area = 100**2 if "S3" not in product else 300**2
            if self.min_area_ratio is not None and sanity_footprint_area < area * self.min_area_ratio:
                wrn_msg = f"Footprint area is less than {self.min_area_ratio * 100:.2f}% of 100km^2 for image {img_path}, area: {sanity_footprint_area:.2f}km^2"
                self.possible_too_small_files_logger.info(img_path)
                logger.warning(wrn_msg)
                continue

            if self.min_length is not None and sanity_footprint_shortest_length < self.min_length / 1000:
                wrn_msg = f"Footprint is too small, boundary length is less than {self.min_length / 1000:.2f}km for image {img_path}, length: {sanity_footprint_shortest_length:.2f}km"
                logger.warning(wrn_msg)
                self.possible_too_small_files_logger.info(img_path)
                continue

            if self.only_check_for_valid_footprints:
                # If we are only checking for valid footprints, we can skip the rest of the processing
                if self.verbose:
                    logger.info(
                        f"Skipping further processing for {img_path} as we are only checking for valid footprints"
                    )
                continue

            found_at_least_one = True
            node_products = [prod for prod in self.products if self.SUBDIR_MAP[prod] == self.SUBDIR_MAP[product]]

            if "S1" in product:
                if "1SDH" in img_path or "1SSH" in img_path:
                    node_products = [prod for prod in node_products if "HH" in prod]
                elif "1SDV" in img_path or "1SSV" in img_path:
                    node_products = [prod for prod in node_products if "VV" in prod]

            if level == len(products) - 1:
                leaf = Leaf(
                    file=str(Path(img_path).relative_to(file_path)),
                    products=node_products,
                    footprint=footprint,
                    crs=crop_crs_wkt,
                    timestamp=timestamp,
                    level=level,
                    parent=node,
                )
                node.children.append(leaf)
            else:
                new_node = Node(
                    file=str(Path(img_path).relative_to(file_path)),
                    products=node_products,
                    children=[],
                    footprint=footprint,
                    crs=crop_crs_wkt,
                    timestamp=timestamp,
                    level=level,
                    parent=node,
                )
                node.children.append(new_node)
                self.build_sample_graph(products, file_path, new_node, level + 1, crop_crs_wkt)

        # Skip product and continue
        if not found_at_least_one:
            return self.build_sample_graph(products, file_path, node, level + 1, crop_crs_wkt)

    def intersect_footprints(self, f1, f2, crs, limit_area=True):
        f1 = f1.to_crs(crs)
        f2 = f2.to_crs(crs)
        if not f1.intersects(f2).values[0]:
            return None

        new_footprint = shapely.intersection(f1, f2)
        new_footprint = new_footprint.set_crs(crs)

        if limit_area and (
            (new_footprint.area < 10_000**2).item()
            or (new_footprint.boundary.shortest_line(new_footprint.centroid).length < 5_000).item()
        ):
            return None

        return new_footprint

    def find_all_sample_areas(
        self,
        node: SampleNode,
        sample_areas: list[Leaf],
    ) -> bool:
        match node:
            case RootNode(children=children, footprint=footprint):
                # Iterate over all children
                for child in children:
                    child.timestamp = (
                        max(child.timestamp, node.timestamp) if child.timestamp is not None else node.timestamp
                    )
                    self.find_all_sample_areas(child, sample_areas)

            case Node(children=children, footprint=footprint, parent=parent):
                # If we have a parent node, try to intersect footprint
                if isinstance(parent, Node):
                    if "S1" in node.products[0] and "S1" in parent.products[0]:
                        # Check if we have the same path, if not, we should skip
                        if not Path(node.file).name == Path(parent.file).name:
                            return False

                    new_footprint = self.intersect_footprints(footprint, parent.footprint, parent.crs)
                    # Continue with the new footprint
                    if new_footprint is not None:
                        node.footprint = new_footprint

                    # If we cannot intersect, try other permutations, either (node.parent -> node.children) or (node -> node.children, exluding parent)
                    else:
                        # Create a new node with our parent as node, and our children as children, skipping us, and then continue
                        new_node = Node(
                            file=parent.file,
                            products=parent.products,
                            children=deepcopy(children),
                            footprint=parent.footprint,
                            crs=parent.crs,
                            parent=parent.parent,
                            timestamp=parent.timestamp,
                        )
                        for child in new_node.children:
                            child.parent = new_node

                        success_new_node = self.find_all_sample_areas(new_node, sample_areas)

                        # Create a new node with our children as children, and our parent's parent as parent, skipping our parent and then continue
                        other_new_node = deepcopy(node)
                        other_new_node.parent = parent.parent
                        success_other_new_node = False
                        # If we have a parent, we should intersect with the parent's footprint,
                        if other_new_node.parent.footprint is not None:
                            other_new_footprint = self.intersect_footprints(
                                other_new_node.footprint, other_new_node.parent.footprint, other_new_node.crs
                            )
                            # Only continue if we have a valid intersection
                            if other_new_footprint is not None:
                                other_new_node.footprint = other_new_footprint

                                for child in other_new_node.children:
                                    child.parent = other_new_node

                                success_other_new_node = self.find_all_sample_areas(other_new_node, sample_areas)

                        # If we have at least one success, we are not a leaf node
                        if not (success_new_node or success_other_new_node):
                            # We are now a leaf node, and we should add our footprint to the sample areas
                            new_leaf = Leaf(
                                file=node.file,
                                products=node.products,
                                footprint=node.footprint,
                                crs=node.crs,
                                timestamp=node.timestamp,
                                parent=node.parent,
                            )
                            sample_areas.append(new_leaf)

                        return True

                elif isinstance(parent, RootNode) and parent.footprint is not None:
                    new_footprint = self.intersect_footprints(footprint, parent.footprint, node.crs)
                    if new_footprint is None:
                        msg = (
                            "Something is wrong!, our parent (a rootnode) does not have an intersecting footprint with us!"
                            f"RootNode footprint, crs: {parent.footprint, parent.crs} Node file, products, footprint, crs: {node.file, node.products, node.footprint, node.crs}"
                        )
                        raise ValueError(msg)
                    node.footprint = new_footprint

                node.timestamp = (
                    max(node.timestamp, parent.timestamp) if node.timestamp is not None else parent.timestamp
                )

                # Parent is root node or we have successfully intersected footprints
                successes = []
                for child in children:
                    success = self.find_all_sample_areas(child, sample_areas)
                    successes.append(success)

                # We are now a leaf node, and we should add our footprint to the sample areas
                if not any(successes):
                    new_leaf = Leaf(
                        file=node.file,
                        products=node.products,
                        footprint=node.footprint,
                        crs=node.crs,
                        timestamp=node.timestamp,
                        parent=node.parent,
                    )
                    sample_areas.append(new_leaf)

                return True

            case Leaf(footprint=footprint, parent=parent):
                if isinstance(parent, Node) and "S1" in node.products[0] and "S1" in parent.products[0]:
                    # Check if we have the same path, if not, we should skip
                    if not Path(node.file).name == Path(parent.file).name:
                        return False

                if isinstance(parent, Node) or (isinstance(parent, RootNode) and parent.footprint is not None):
                    new_footprint = self.intersect_footprints(footprint, parent.footprint, parent.crs)
                    if new_footprint is None:
                        return False

                    node.footprint = new_footprint

                node.timestamp = (
                    max(node.timestamp, parent.timestamp) if node.timestamp is not None else parent.timestamp
                )

                sample_areas.append(node)
                return True

    @staticmethod
    def _get_samples(node):
        # Sanity checks
        if not isinstance(node, Leaf):
            warnings.warn(f"Node is not a leaf node, but a {type(node)}", stacklevel=2)

        _iter_node = node
        _iter_node_parent = _iter_node.parent
        while _iter_node_parent.footprint is not None:
            if (_iter_node.footprint.area.item() - _iter_node_parent.footprint.area.item()) > 100:
                warnings.warn(
                    f"Child footprint is larger than parent footprint, node parent file: {_iter_node_parent.file} \n {_iter_node_parent}",
                    stacklevel=2,
                )
                break
            # _iter_node = _iter_node_parent
            if isinstance(_iter_node_parent, RootNode) or _iter_node_parent.footprint is None:
                break
            _iter_node_parent = _iter_node_parent.parent

        sample_traj = {
            **dict.fromkeys(node.products, node.file),
            "footprint": node.footprint[0],
            "area": node.footprint.area.item(),
            "timestamp": node.timestamp.strftime("%Y-%m-%dT%H:%M:%S.%f") if node.timestamp is not None else None,
            "crs": node.crs,
        }

        traverse_node = node
        while not isinstance(traverse_node, RootNode):
            for prod in traverse_node.products:
                sample_traj[prod] = traverse_node.file
            traverse_node = traverse_node.parent

        return sample_traj

    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        tile = Path(file_path).parents[2].name[1:]

        out_dir = self.out_dir
        fp = Path(file_path)

        file_date = datetime.strptime(str(fp.relative_to(fp.parents[2])), "%Y/%m/%d")

        relative_path = fp.relative_to(fp.parents[3])
        out_dir = out_dir / relative_path

        out_filename = self.get_metadata_cache_name()

        out_file = out_dir / out_filename

        if out_file.exists() and not self.overwrite:
            return (file_path, None, None, True)

        out_dir.mkdir(parents=True, exist_ok=True)

        # Sorting products after SAMPLE_ORDER
        sample_products = sorted(self.products, key=lambda x: self.SAMPLE_ORDER.index(x.split("-")[0]))

        filtered_sample_products = []
        for sample_product in sample_products:
            product_subdir = self.SUBDIR_MAP[sample_product]
            if not any(product_subdir == self.SUBDIR_MAP[prod] for prod in filtered_sample_products):
                filtered_sample_products.append(sample_product)

        sample_products = filtered_sample_products

        # Build the sample graph
        sample_graph = RootNode(children=[], level=0, timestamp=file_date)

        if self.tile_bounds is not None:
            sample_graph.footprint = self.get_tile_footprint(tile)
            if sample_graph.footprint is not None:
                sample_graph.crs = sample_graph.footprint.crs

        self.build_sample_graph(sample_products, file_path, sample_graph)

        # Finding a suitable bounding box to sample from, from the products, possibly multiple files for each sensor
        sample_graph_traversed = []
        self.find_all_sample_areas(sample_graph, sample_graph_traversed)

        prod_imgs = []
        for sample_traversed in sample_graph_traversed:
            sample = self._get_samples(sample_traversed)
            if sample not in prod_imgs:
                prod_imgs.append(sample)

        # Convert the dictionary to a pandas DataFrame
        df = pd.DataFrame(prod_imgs)

        if "footprint" not in df.columns:
            # logger.info(f"file path: {file_path}")
            return (file_path, df, 0, False)

        # Convert bounding box to shapely geometry
        if isinstance(df["footprint"][0], (list, tuple)):
            df["geometry"] = df["footprint"].apply(lambda bbox: box(*bbox))
        else:
            df["geometry"] = df["footprint"]

        df = df.drop(columns=["footprint"])

        # ensure all products are in the dataframe
        for product in self.products:
            if product not in df.columns:
                df[product] = None

        # Area times number of products for sampling weighting
        # df["not_null"] = len(self.products) - df[self.products].apply(lambda row: sum(row.isnull()), axis=1)

        # Existing sensors, (not products)
        existing_sensors = df[self.products].T.groupby(df[self.products].columns.str.split("-").str[0]).any().T

        area_prod_score = existing_sensors.sum(axis=1)
        if "S1" in existing_sensors.columns:
            area_prod_score += 1 * existing_sensors["S1"]  # Make area prod which contains S1 twice as as large

        # df["area_prod"] = (df["area"] / 100_000**2) * existing_sensors.sum(axis=1)
        # df["area_prod"] *= 1 * existing_sensors["S1"] + 1  # Make area prod which contains S1 twice as as large
        df["area_prod"] = (df["area"] / 100_000**2) * area_prod_score
        # df = df.drop(columns=["not_null"])

        # # Convert the DataFrame to a GeoDataFrame
        gdf = gpd.GeoDataFrame(df, geometry="geometry", crs=df["crs"][0])

        gdf = gdf.drop(gdf[gdf["area_prod"] == 0].index)

        # take union of all geometries and get the total area
        union_footprint = gdf["geometry"].union_all()
        total_area = union_footprint.area / 100_000**2

        # multiply the area with the number of sensors
        total_existing_sensors = existing_sensors.any(axis=0)
        score = total_existing_sensors.sum()
        # score *= 1 * total_existing_sensors["S1"] + 1  # Make area prod which contains S1 twice as as large
        if "S1" in total_existing_sensors:
            score += 1 * total_existing_sensors["S1"]  # Make area prod which contains S1 twice as as large

        total_area *= score

        # total_area = gdf["area_prod"].sum()

        if total_area == 0:
            return (file_path, df, 0, False)

        # Create a heurisitc indicator variable for largest reasonable footprint we can sample from
        gdf["max_ground_cover"] = 0
        for gc in self.ground_covers:
            gc_area_limit = (2 * gc) ** 2
            gc_centroid_limit = gc
            gc_mask = (gdf.area >= gc_area_limit) & (
                gdf.geometry.boundary.shortest_line(gdf.geometry.centroid).length >= gc_centroid_limit
            )
            gdf.loc[gc_mask, "max_ground_cover"] = gc

        # Save the GeoDataFrame to the specified file path
        gdf.to_file(out_file, driver="GeoJSON")

        if self.save_fig:
            gdf["id"] = gdf.index
            # plot the bounding boxes
            fig, ax = plt.subplots()
            gdf.plot(
                ax=ax,
                column="id",
                categorical=True,
                alpha=0.15,
                # cmap=plt.cm.tab20,
                edgecolor="black",
                legend=True,
            )

            if sample_graph.footprint is not None:
                gpd.GeoSeries(sample_graph.footprint).plot(
                    ax=ax,
                    alpha=0.1,
                    edgecolor="black",
                    color="red",
                    label="Tile",
                )
            plt.savefig(out_file.with_suffix(".png"))
            plt.close(fig)

        return (file_path, gdf, total_area, True)

    def collate_fn(self, batch):
        return batch
