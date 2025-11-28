import glob
import logging
import math
import multiprocessing as mp
import os
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import ClassVar

import matplotlib.pyplot as plt
import numpy as np
import torch
import xarray as xr
from pyproj import Transformer
from torchvision import transforms
from tqdm import tqdm

from thor.utils.sentinel import (
    CallWrapper,
    ConsistentRadomHorizontalFlip,
    ConsistentRadomVerticalFlip,
    ControlledConsistentRandomCrop,
)

logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class MetaData:
    channel_params: dict[str, dict[str, int]]
    ground_cover: int
    month: torch.Tensor
    center_coords: torch.Tensor
    file_path: str | None = None
    s1_orbit_direction: int | None = None
    s1_incidence_angles: torch.Tensor | None = None


@dataclass(slots=True)
class ProductDataBatch:
    product_imgs: dict[str, torch.Tensor]
    metadata: MetaData
    era5_land_data: dict[str, torch.Tensor] | None = None


@dataclass(slots=True)
class BandDataBatch:
    band_imgs: dict[str, torch.Tensor]
    metadata: MetaData
    era5_land_data: dict[str, torch.Tensor] | None = None


def _to_tensor(x):
    return torch.from_numpy(x.astype(np.float32))


def clamp_5_sigma(tensor, sigma=1.0):
    return torch.clamp(tensor, -5.0 * sigma, 5.0 * sigma)


class THORDatasetBase:
    FILENAME_GLOB = "T[0-9]*/[0-9]*/[0-9]*/[0-9]*"

    # Names used within netCDF files
    NETCDF_PRODUCT_BAND_MAP = {  # noqa: RUF012
        # NOTE: VH+VV and HV+HH are never available at the same time
        "S1-IW-VV-10m": {
            0: "sigma0_vh",
            1: "sigma0_vv",
        },
        "S1-IW-HH-10m": {
            0: "sigma0_hh",
            1: "sigma0_hv",
        },
        "S1-IW-VV-60m": {
            0: "sigma0_vh",
            1: "sigma0_vv",
        },
        "S1-IW-HH-60m": {
            0: "sigma0_hh",
            1: "sigma0_hv",
        },
        "S1-EW-VV-10m": {
            0: "sigma0_vh",
            1: "sigma0_vv",
        },
        "S1-EW-HH-10m": {
            0: "sigma0_hh",
            1: "sigma0_hv",
        },
        "S1-EW-VV-60m": {
            0: "sigma0_vh",
            1: "sigma0_vv",
        },
        "S1-EW-HH-60m": {
            0: "sigma0_hh",
            1: "sigma0_hv",
        },
        "S2-10m": {0: "B02", 1: "B03", 2: "B04", 3: "B08"},
        "S2-20m": {
            0: "B05",
            1: "B06",
            2: "B07",
            3: "B8A",
            4: "B11",
            5: "B12",
        },
        "S2-60m": {
            0: "B01",
            1: "B09",
        },
        "S3-250m": {  # OLCI resampled to 250m
            0: "Oa01_reflectance",
            1: "Oa02_reflectance",
            2: "Oa03_reflectance",
            3: "Oa04_reflectance",
            4: "Oa05_reflectance",
            5: "Oa06_reflectance",
            6: "Oa07_reflectance",
            7: "Oa08_reflectance",
            8: "Oa09_reflectance",
            9: "Oa10_reflectance",
            10: "Oa11_reflectance",
            11: "Oa12_reflectance",
            12: "Oa13_reflectance",
            13: "Oa14_reflectance",
            14: "Oa15_reflectance",
            15: "Oa16_reflectance",
            16: "Oa17_reflectance",
            17: "Oa18_reflectance",
            18: "Oa19_reflectance",
            19: "Oa20_reflectance",
            20: "Oa21_reflectance",
        },
        "S3-500m": {
            0: "S1_reflectance_an",
            1: "S2_reflectance_an",
            2: "S3_reflectance_an",
            3: "S4_reflectance_an",
            4: "S5_reflectance_an",
            5: "S6_reflectance_an",
        },
        "S3-1000m": {
            0: "S7_BT_in",
            1: "S8_BT_in",
            2: "S9_BT_in",
        },
    }

    # Names used within the dataset
    PRODUCT_BAND_NAME_MAP = {  # noqa: RUF012
        # NOTE: VH+VV and HV+HH are never available at the same time
        "S1-IW-VV-10m": {0: "S1:IW-VH", 1: "S1:IW-VV"},
        "S1-IW-HH-10m": {0: "S1:IW-HH", 1: "S1:IW-HV"},
        "S1-EW-VV-10m": {0: "S1:EW-VH", 1: "S1:EW-VV"},
        "S1-EW-HH-10m": {0: "S1:EW-HH", 1: "S1:EW-HV"},
        "S1-IW-VV-60m": {0: "S1:IW-VH", 1: "S1:IW-VV"},
        "S1-IW-HH-60m": {0: "S1:IW-HH", 1: "S1:IW-HV"},
        "S1-EW-VV-60m": {0: "S1:EW-VH", 1: "S1:EW-VV"},
        "S1-EW-HH-60m": {0: "S1:EW-HH", 1: "S1:EW-HV"},
        "S2-10m": {0: "S2:Blue", 1: "S2:Green", 2: "S2:Red", 3: "S2:NIR"},
        "S2-20m": {
            0: "S2:RE1",
            1: "S2:RE2",
            2: "S2:RE3",
            3: "S2:RE4",
            4: "S2:SWIR1",
            5: "S2:SWIR2",
        },
        "S2-60m": {
            0: "S2:CoastAerosal",
            1: "S2:WaterVapor",
        },
        "S3-250m": {  # OLCI resampled to 250m
            0: "S3:Oa01_reflectance",
            1: "S3:Oa02_reflectance",
            2: "S3:Oa03_reflectance",
            3: "S3:Oa04_reflectance",
            4: "S3:Oa05_reflectance",
            5: "S3:Oa06_reflectance",
            6: "S3:Oa07_reflectance",
            7: "S3:Oa08_reflectance",
            8: "S3:Oa09_reflectance",
            9: "S3:Oa10_reflectance",
            10: "S3:Oa11_reflectance",
            11: "S3:Oa12_reflectance",
            12: "S3:Oa13_reflectance",
            13: "S3:Oa14_reflectance",
            14: "S3:Oa15_reflectance",
            15: "S3:Oa16_reflectance",
            16: "S3:Oa17_reflectance",
            17: "S3:Oa18_reflectance",
            18: "S3:Oa19_reflectance",
            19: "S3:Oa20_reflectance",
            20: "S3:Oa21_reflectance",
        },
        "S3-500m": {
            0: "S3:S1_reflectance_an",
            1: "S3:S2_reflectance_an",
            2: "S3:S3_reflectance_an",
            3: "S3:S4_reflectance_an",
            4: "S3:S5_reflectance_an",
            5: "S3:S6_reflectance_an",
        },
        "S3-1000m": {
            0: "S3:S7_BT_in",
            1: "S3:S8_BT_in",
            2: "S3:S9_BT_in",
        },
        "dem-10m": {
            "0": "elevation",
            "1": "slope",
        },
        "dem-60m": {
            "0": "elevation",
            "1": "slope",
        },
        "INC-10m": {
            "0": "incidence_angle",
        },
        "INC-60m": {
            "0": "incidence_angle",
        },
    }

    # NOTE: using dataset band names to be able to differentiate between S1 10m and 60m bands
    ALL_BAND_NAMES: ClassVar = [name for _, bands in PRODUCT_BAND_NAME_MAP.items() for _, name in bands.items()]

    RGB_BANDS = ("S2:Red", "S2:Green", "S2:Blue")

    # Subdirectories within each tile + date directory
    SUBDIR_MAP = {  # noqa: RUF012
        "S1-IW-VV-10m": "S1/IW_GRDH_1S/10m",
        "S1-IW-HH-10m": "S1/IW_GRDH_1S/10m",
        "S1-IW-VV-60m": "S1/IW_GRDH_1S/60m",
        "S1-IW-HH-60m": "S1/IW_GRDH_1S/60m",
        "S1-EW-VV-10m": "S1/EW_GRDM_1S/10m",
        "S1-EW-HH-10m": "S1/EW_GRDM_1S/10m",
        "S1-EW-VV-60m": "S1/EW_GRDM_1S/60m",
        "S1-EW-HH-60m": "S1/EW_GRDM_1S/60m",
        "S2-10m": "S2/S2MSI2A",
        "S2-20m": "S2/S2MSI2A",
        "S2-60m": "S2/S2MSI2A",
        "S3-250m": "S3/OL_1_EFR",
        "S3-500m": "S3/SL_1_RBT",
        "S3-1000m": "S3/SL_1_RBT",
    }

    ERA5_LAND_GSD = 11132  # m

    ERA5_LAND_VARIABLES = [  # noqa: RUF012
        "volumetric_soil_water_layer_1",
        "volumetric_soil_water_layer_4",
        "skin_temperature",
        "dewpoint_temperature_2m",
        "temperature_2m",
        "soil_temperature_level_1",
        "soil_temperature_level_4",
        "snow_cover",
        "snow_depth_water_equivalent",
        "snowfall_sum",
        "snow_depth",
        "leaf_area_index_high_vegetation",
        "leaf_area_index_low_vegetation",
        "surface_pressure",
        "total_precipitation_sum",
        "surface_runoff_sum",
        "total_evaporation_sum",
    ]

    ERA5_LAND_PRODUCT_MAP = {  # noqa: RUF012
        k: k for k in ERA5_LAND_VARIABLES
    }

    NORMALIZATION = {  # noqa: RUF012
        "S2-10m": {
            "S2:Blue": {"mean": 0.176620, "std": 0.264520},
            "S2:Green": {"mean": 0.195923, "std": 0.252949},
            "S2:Red": {"mean": 0.213948, "std": 0.259180},
            "S2:NIR": {"mean": 0.308133, "std": 0.226434},
        },
        "S2-20m": {
            "S2:RE1": {"mean": 0.263378, "std": 0.272771},
            "S2:RE2": {"mean": 0.300818, "std": 0.248175},
            "S2:RE3": {"mean": 0.313144, "std": 0.235432},
            "S2:RE4": {"mean": 0.320993, "std": 0.223274},
            "S2:SWIR1": {"mean": 0.221550, "std": 0.171606},
            "S2:SWIR2": {"mean": 0.175772, "std": 0.156223},
        },
        "S2-60m": {
            "S2:CoastAerosal": {"mean": 0.182569, "std": 0.282463},
            "S2:WaterVapor": {"mean": 0.322589, "std": 0.235857},
        },
        "S1-IW-VV-60m": {
            "S1:IW-VH": {"mean": -20.6672, "std": 5.9634},
            "S1:IW-VV": {"mean": -13.0095, "std": 5.2439},
        },
        "S1-IW-VV-10m": {
            "S1:IW-VH": {"mean": -20.6958, "std": 5.8688},
            "S1:IW-VV": {"mean": -12.9850, "std": 5.0062},
        },
        "S1-IW-HH-60m": {
            "S1:IW-HH": {"mean": -13.0126, "std": 7.1580},
            "S1:IW-HV": {"mean": -21.2863, "std": 7.6221},
        },
        "S1-IW-HH-10m": {
            "S1:IW-HH": {"mean": -13.9485, "std": 6.8015},
            "S1:IW-HV": {"mean": -22.4575, "std": 6.9106},
        },
        "S1-EW-VV-60m": {
            "S1:EW-VH": {"mean": -23.2024, "std": 7.0584},
            "S1:EW-VV": {"mean": -13.1240, "std": 5.5013},
        },
        "S1-EW-VV-10m": {
            "S1:EW-VH": {"mean": -23.5719, "std": 6.8895},
            "S1:EW-VV": {"mean": -13.9046, "std": 6.3085},
        },
        "S1-EW-HH-60m": {
            "S1:EW-HH": {"mean": -12.1138, "std": 6.5830},
            "S1:EW-HV": {"mean": -21.7450, "std": 7.4658},
        },
        "S1-EW-HH-10m": {
            "S1:EW-HH": {"mean": -12.7691, "std": 6.6416},
            "S1:EW-HV": {"mean": -22.6922, "std": 7.2472},
        },
        ###############
        "S3-250m": {
            "S3:Oa01_reflectance": {"mean": 0.418360, "std": 0.271155},
            "S3:Oa02_reflectance": {"mean": 0.407687, "std": 0.276213},
            "S3:Oa03_reflectance": {"mean": 0.384837, "std": 0.286278},
            "S3:Oa04_reflectance": {"mean": 0.356827, "std": 0.291605},
            "S3:Oa05_reflectance": {"mean": 0.342093, "std": 0.284960},
            "S3:Oa06_reflectance": {"mean": 0.313561, "std": 0.259517},
            "S3:Oa07_reflectance": {"mean": 0.305947, "std": 0.260534},
            "S3:Oa08_reflectance": {"mean": 0.326632, "std": 0.286289},
            "S3:Oa09_reflectance": {"mean": 0.329984, "std": 0.290304},
            "S3:Oa10_reflectance": {"mean": 0.331824, "std": 0.291953},
            "S3:Oa11_reflectance": {"mean": 0.348113, "std": 0.281539},
            "S3:Oa12_reflectance": {"mean": 0.394400, "std": 0.265238},
            "S3:Oa13_reflectance": {"mean": 0.101062, "std": 0.066239},
            "S3:Oa14_reflectance": {"mean": 0.188930, "std": 0.122599},
            "S3:Oa15_reflectance": {"mean": 0.352127, "std": 0.233035},
            "S3:Oa16_reflectance": {"mean": 0.394914, "std": 0.258835},
            "S3:Oa17_reflectance": {"mean": 0.402817, "std": 0.251323},
            "S3:Oa18_reflectance": {"mean": 0.398475, "std": 0.244922},
            "S3:Oa19_reflectance": {"mean": 0.312429, "std": 0.213306},
            "S3:Oa20_reflectance": {"mean": 0.166702, "std": 0.156878},
            "S3:Oa21_reflectance": {"mean": 0.380989, "std": 0.209298},
        },
        ###########
        "S3-500m": {
            "S3:S1_reflectance_an": {"mean": 0.338953, "std": 0.280222},
            "S3:S2_reflectance_an": {"mean": 0.339034, "std": 0.294075},
            "S3:S3_reflectance_an": {"mean": 0.405393, "std": 0.262460},
            "S3:S4_reflectance_an": {"mean": 0.010942, "std": 0.030842},
            "S3:S5_reflectance_an": {"mean": 0.184803, "std": 0.140885},
            "S3:S6_reflectance_an": {"mean": 0.137453, "std": 0.109526},
        },
        ##########
        "S3-1000m": {
            "S3:S7_BT_in": {"mean": 286.190167, "std": 21.070093},
            "S3:S8_BT_in": {"mean": 278.423815, "std": 22.109991},
            "S3:S9_BT_in": {"mean": 277.117165, "std": 21.526558},
        },
        # Stats from ERA5-Land 2017-2023 10% random files daily
        "ERA5-Land": {
            "volumetric_soil_water_layer_1": {"mean": 0.260158410201127, "std": 0.1339651949378201},
            "volumetric_soil_water_layer_4": {"mean": 0.24631001867522678, "std": 0.13773668029335784},
            "skin_temperature": {"mean": 268.41312361456494, "std": 28.297447290102216},
            "dewpoint_temperature_2m": {"mean": 262.1750659089831, "std": 24.41657962234843},
            "temperature_2m": {"mean": 268.9595305911954, "std": 26.624741966852582},
            "soil_temperature_level_1": {"mean": 270.2222789617739, "std": 27.193818752358226},
            "soil_temperature_level_4": {"mean": 269.965937748763, "std": 26.576650049964233},
            "snow_cover": {"mean": 48.510896397836355, "std": 48.81570793830908},
            "snow_depth_water_equivalent": {"mean": 3.3941514765115945, "std": 4.706830804123465},
            "snowfall_sum": {"mean": 0.0003619569084225213, "std": 0.001528487125944941},
            "snow_depth": {"mean": 11.32951351334466, "std": 15.672580045541395},
            "leaf_area_index_high_vegetation": {"mean": 1.158268796192595, "std": 1.6527017289361292},
            "leaf_area_index_low_vegetation": {"mean": 0.7789587799935094, "std": 0.9551067955624861},
            "surface_pressure": {"mean": 88360.89609089476, "std": 12771.240215809312},
            "total_precipitation_sum": {"mean": 0.001637288803682897, "std": 0.004863235423371868},
            "surface_runoff_sum": {"mean": 0.0001738812386621543, "std": 0.0016216913481759804},
            "total_evaporation_sum": {"mean": -0.0009354519852274739, "std": 0.0013856140564146471},
        },
        "dem-10m": {
            "elevation": {"mean": 79.2351, "std": 166.2252},
            "slope": {"mean": 2.1897, "std": 5.7798},
        },
        "dem-60m": {
            "elevation": {"mean": 79.2351, "std": 166.2252},
            "slope": {"mean": 2.1897, "std": 5.7798},
        },
        "INC-10m": {"incidence_angle": {"mean": 33.887590498890795, "std": 8.378258197166582}},
        "INC-60m": {"incidence_angle": {"mean": 33.59603978264744, "std": 7.601051398951261}},
    }

    LAND_COVER_PRODUCTS: ClassVar = ["SCL-20m", "WC-10m", "GC-250m", "MCD-500m"]

    DEM_PRODUCTS: ClassVar = ["dem-10m", "dem-60m"]

    INCIDENCE_ANGLE_PRODUCTS: ClassVar = ["INC-10m", "INC-60m"]

    LAND_COVER_GSD_MAP = {  # noqa: RUF012
        "SCL": 20,
        "WC": 10,
        "GC": 250,
        "MCD": 500,
    }

    # Sentinel-2 Scene Classification Layer (SCL) classes
    SCL_CLASSES = {  # noqa: RUF012
        0: "No data",
        1: "Saturated or defective",
        2: "Dark area pixels",
        3: "Cloud shadows",
        4: "Vegetation",
        5: "Not vegetated",
        6: "Water",
        7: "Unclassified",
        8: "Cloud medium probability",
        9: "Cloud high probability",
        10: "Thin cirrus",
        11: "Snow or ice",
    }

    SCL_COLORS = [  # noqa: RUF012
        (0, 0, 0),
        (255, 0, 0),
        (47, 47, 47),
        (100, 50, 0),
        (0, 160, 0),
        (255, 230, 90),
        (0, 0, 255),
        (128, 128, 128),
        (192, 192, 192),
        (255, 255, 255),
        (100, 200, 255),
        (255, 150, 255),
    ]

    WC_CLASSES = {  # noqa: RUF012
        0: "No data",
        10: "Tree cover",
        20: "Shrubland",
        30: "Grassland",
        40: "Cropland",
        50: "Built-up",
        60: "Bare / sparse vegetation",
        70: "Snow and ice",
        80: "Permanent water bodies",
        90: "Herbaceous wetland",
        95: "Mangroves",
        100: "Moss and lichen",
    }

    WC_IGNORE_CLASSES = [70]  # noqa: RUF012

    WC_COLORS = [  # noqa: RUF012
        (0, 0, 0),  # No data
        (0, 100, 0),  # #006400 Tree cover
        (255, 187, 34),  # #ffbb22 Shrubland
        (255, 255, 76),  # #ffff4c Grassland
        (240, 150, 255),  # #f096ff Cropland
        (250, 0, 0),  # #fa0000 Built-up
        (180, 180, 180),  # #b4b4b4 Bare / sparse vegetation
        (240, 240, 240),  # #f0f0f0 Snow and ice
        (0, 100, 200),  # #0064c8 Permanent water bodies
        (0, 150, 160),  # #0096a0 Herbaceous wetland
        (0, 207, 117),  # #00cf75 Mangroves
        (250, 230, 160),  # #fae6a0 Moss and lichen
    ]

    # Glob cover
    GC_CLASSES = {  # noqa: RUF012
        0: "No data",
        11: "Irrigated croplands",
        14: "Rainfed croplands",
        20: "Mosaic: cropland(50-70%)/vegetation(20-50%)",
        30: "Mosaic: vegetation(50-70%)/cropland(20-50%)",
        40: "Open broadleaved evergreen forest",
        50: "Closed broadleaved deciduous forest",
        60: "Open broadleaved deciduous forest",
        70: "Closed needleleaved evergreen forest",
        90: "Open needleleaved forest",
        100: "Mixed forest",
        110: "Mosaic: forest(50-70%)/grassland(20-50%)",
        120: "Mosaic: grassland(50-70%)/forest(20-50%)",
        130: "Shrubland",
        140: "Herbaceous vegetation",
        150: "Sparse vegetation",
        160: "Flooded broadleaved forest - Fresh water",
        170: "Flooded forest/shrubland - Brackish water",
        180: "Flooded grassland/vegetation",
        190: "Urban areas",
        200: "Bare areas",
        210: "Water bodies",
        220: "Snow and ice",
        # 230: "No data",
    }

    GC_IGNORE_CLASSES = [230]  # noqa: RUF012

    GC_COLORS = [  # noqa: RUF012
        (116, 52, 17),  # #743411 No data
        (170, 239, 239),  # #aaefef Post-flooding or irrigated croplands
        (255, 255, 99),  # #ffff63 Rainfed croplands
        (220, 239, 99),  # #dcef63 Mosaic cropland (50-70%) / vegetation (20-50%)
        (205, 205, 100),  # #cdcd64 Mosaic vegetation (50-70%) / cropland (20-50%)
        (0, 99, 0),  # #006300 Closed to open broadleaved evergreen forest
        (0, 159, 0),  # #009f00 Closed broadleaved deciduous forest
        (170, 199, 0),  # #aac700 Open broadleaved deciduous forest
        (0, 59, 0),  # #003b00 Closed needleleaved evergreen forest
        (40, 99, 0),  # #286300 Open needleleaved forest
        (120, 131, 0),  # #788300 Mixed broadleaved and needleleaved forest
        (141, 159, 0),  # #8d9f00 Mosaic forest-shrubland / grassland
        (189, 149, 0),  # #bd9500 Mosaic grassland / forest-shrubland
        (149, 99, 0),  # #956300 Shrubland
        (255, 180, 49),  # #ffb431 Grassland
        (255, 235, 174),  # #ffebae Sparse vegetation
        (0, 120, 90),  # #00785a Flooded broadleaved forest - Fresh water
        (0, 149, 120),  # #009578 Flooded forest - saline water
        (0, 220, 131),  # #00dc83 Flooded vegetation
        (195, 19, 0),  # #c31300 Urban areas
        (255, 245, 214),  # #fff5d6 Bare areas
        (0, 70, 199),  # #0046c7 Water bodies
        (255, 255, 255),  # #ffffff Permanent snow and ice
        # (116, 52, 17),  # #743411 Unclassified
    ]

    MCD_CLASSES = {  # noqa: RUF012
        0: "No data",
        1: "Evergreen Needleleaf Forest",
        2: "Evergreen Broadleaf Forest",
        3: "Deciduous Needleleaf Forest",
        4: "Deciduous Broadleaf Forest",
        5: "Mixed Forest",
        6: "Closed Shrubland",
        7: "Open Shrubland",
        8: "Woody Savanna",
        9: "Savanna",
        10: "Grassland",
        11: "Permanent Wetland",
        12: "Cropland",
        13: "Urban",
        14: "Cropland/Natural Mosaic",
        15: "Snow and Ice",
        16: "Barren",
        17: "Water Body",
    }

    MCD_COLORS = [  # noqa: RUF012
        (0, 0, 0),  # #000000 No data
        (5, 69, 10),  # #05450a Evergreen Needleleaf Forest
        (8, 106, 16),  # #086a10 Evergreen Broadleaf Forest
        (84, 167, 8),  # #54a708 Deciduous Needleleaf Forest
        (120, 210, 3),  # #78d203 Deciduous Broadleaf Forest
        (0, 153, 0),  # #009900 Mixed Forest
        (198, 176, 68),  # #c6b044 Closed Shrubland
        (220, 209, 89),  # #dcd159 Open Shrubland
        (218, 222, 72),  # #dade48 Woody Savanna
        (251, 255, 19),  # #fbff13 Savanna
        (182, 255, 5),  # #b6ff05 Grassland
        (39, 255, 135),  # #27ff87 Permanent Wetland
        (194, 79, 68),  # #c24f44 Cropland
        (165, 165, 165),  # #a5a5a5 Urban
        (255, 109, 76),  # #ff6d4c Cropland/Natural Mosaic
        (105, 255, 248),  # #69fff8 Snow and Ice
        (249, 255, 164),  # #f9ffa4 Barren
        (28, 13, 255),  # #1c0dff Water Body
    ]

    S1_MODES = ("IW", "EW")
    S1_POLARIZATIONS = ("VV", "HH")

    SAMPLE_ORDER = ("S1", "S2", "S3")

    SEPARATE_FOLDER_PRODUCTS = (
        "S3-250m",
        "S1-IW-VV-10m",
        "S1-IW-HH-10m",
        "S1-EW-VV-10m",
        "S1-EW-HH-10m",
        "S1-IW-VV-60m",
        "S1-IW-HH-60m",
        "S1-EW-VV-60m",
        "S1-EW-HH-60m",
    )

    # Image pixel limits for all products
    IMAGE_MAX_LIMIT = 288
    IMAGE_MIN_LIMIT = 24

    def __init__(
        self,
        paths: str,
        ground_covers: int | list[int],
        split: str = "train",
        products: list | dict = [  # noqa: B006
            "S2-10m",
            "S2-20m",
            "S2-60m",
            "S1-10m",
        ],
        discard_bands: list = [],
        standardize: bool = False,
        full_return: bool = False,
        data_percent: float = 1.0,
        max_nan_ratio: float = 0.05,
        include_filter: str | None = None,
        exclude_filter: str | None = None,
        era5_data_dir: str | None = None,
        era5_land_products: list[str] | None = None,
        important_products: list[str] | None = None,
        legacy_nan_handling: bool = False,
        random_seed: int = 42,
        **kwargs,
    ) -> None:
        if isinstance(ground_covers, int):
            ground_covers = [ground_covers]
        ground_covers = sorted(ground_covers)

        if isinstance(products, dict):
            product_changes = products
            products = list(product_changes.keys())
            for product in products:
                if product_changes[product] is None:
                    product_changes[product] = product
            self.product_changes = product_changes
        else:
            self.product_changes = {product: product for product in products}

        for product in products:
            if "S1" in product and (
                not any(polarization in product for polarization in self.S1_POLARIZATIONS)
                or not any(mode in product for mode in self.S1_MODES)
            ):
                msg = f"S1 product {product} will be expanded to include all polarizations"
                warnings.warn(msg, UserWarning)

        self.product_changes = dict(
            zip(
                self.expand_products(self.product_changes.keys()),
                self.expand_products(self.product_changes.values()),
                strict=False,
            )
        )

        products = self.expand_products(products)

        all_products = products.copy()
        all_product_changes = self.product_changes.copy()

        if hasattr(self, "land_cover_products"):
            all_products.extend(self.land_cover_products)
            all_product_changes.update(self.land_cover_product_changes)
        if hasattr(self, "dem_products"):
            all_products.extend(self.dem_products)
            all_product_changes.update(self.dem_product_changes)
        if hasattr(self, "incidence_angle_products"):
            all_products.extend(self.incidence_angle_products)
            all_product_changes.update(self.incidence_angle_product_changes)

        # Map of products for each ground cover
        ground_cover_products = defaultdict(list)
        # Map of ground covers  for each product
        product_ground_covers = defaultdict(list)

        # Iterate through ground covers and check against min and max image size
        for ground_cover in ground_covers:
            for product in all_products:
                gsd = self.GSD(all_product_changes[product])

                max_limit = self.IMAGE_MAX_LIMIT
                if "S1" in product and "60m" in product:
                    # Lets allow 2x the image size for S1 60m
                    max_limit *= 2
                if "S2" in product and "10m" in product:
                    max_limit *= 2
                if "dem" in product:
                    max_limit *= 2
                if "INC" in product:
                    max_limit *= 2
                if ground_cover // gsd < self.IMAGE_MIN_LIMIT or ground_cover // gsd > max_limit:
                    msg = f"Ground cover {ground_cover} is not valid for product {product} with GSD {gsd}"
                    f" due to image size being {ground_cover // gsd} and the product will not be used for this ground cover"
                    logger.debug(msg)
                    continue
                else:
                    ground_cover_products[ground_cover].append(product)
                    product_ground_covers[product].append(ground_cover)

        # Sort the ground cover products
        for ground_cover in ground_cover_products:
            ground_cover_products[ground_cover] = sorted(
                ground_cover_products[ground_cover], key=lambda x: self.GSD(x), reverse=True
            )

        # Sort the product ground covers
        for product in product_ground_covers:
            product_ground_covers[product] = sorted(product_ground_covers[product], reverse=True)

        # Check if all products have at least one ground cover
        if len(product_ground_covers) != len(all_products):
            msg = f"Not all products are valid for the given ground covers: {sorted(product_ground_covers)} vs {sorted(all_products)}"
            raise ValueError(msg)
        # Check if all ground covers have at least one product
        if len(ground_cover_products) != len(ground_covers):
            msg = f"Not all ground covers are valid for the given products: {sorted(ground_cover_products)} vs {sorted(ground_covers)}"
            raise ValueError(msg)

        # Print the ground cover products
        logger.info("#" * 80)
        logger.info("Ground cover products:")
        for ground_cover, gc_products in ground_cover_products.items():
            logger.info(f"  {ground_cover:<6}m: {sorted(gc_products, key=lambda x: self.GSD(all_product_changes[x]))}")
        logger.info("#" * 80)

        self.ground_cover_products = ground_cover_products
        self.product_ground_covers = product_ground_covers

        if important_products is not None:
            self.important_products = important_products
        elif not any("S2" in product for product in products):
            # Typical low res mode
            self.important_products = ("S1", "S3")
        elif not any("S1" in product for product in products):
            # Assuming high res mode
            self.important_products = ("S2",)
        elif not any("S2" in product for product in products) and not any("S1" in product for product in products):
            # Only one left :)
            self.important_products = ("S3",)
        else:
            # Using all, assuming high res mode
            self.important_products = ("S1", "S2")

        logger.info(f"products: {products}")
        logger.info(f"product changes: {self.product_changes}")
        logger.info(f"important products: {self.important_products}")

        assert all(product in self.NETCDF_PRODUCT_BAND_MAP for product in products), (
            f"Invalid product in {products}, available products: {self.NETCDF_PRODUCT_BAND_MAP.keys()}"
        )
        assert all(band in self.ALL_BAND_NAMES for band in discard_bands), (
            f"Invalid band in {discard_bands}, available bands: {self.ALL_BAND_NAMES}"
        )

        assert not (era5_land_products and era5_data_dir is None), (
            "era5_data_dir must be specified if era5_land_products is used"
        )

        largest_to_smallest_gsd_products = sorted(products, key=lambda x: self.GSD(x), reverse=True)
        logger.info(f"largest to smallest gsd products: {largest_to_smallest_gsd_products}")
        self.paths = paths
        self.products = largest_to_smallest_gsd_products
        self.split = split
        self.ground_covers = ground_covers
        self.standardize = standardize
        self.full_return = full_return
        self.data_percent = data_percent
        self.discard_bands = discard_bands
        self.max_nan_ratio = max_nan_ratio
        self.legacy_nan_handling = legacy_nan_handling
        self.era5_data_dir = era5_data_dir
        self.era5_land_products = era5_land_products

        self.random_seed = random_seed

        self.sensors = {p.split("-")[0] for p in self.products}

        logger.info(f"Split: {self.split}")

        self.include_tiles = None
        if include_filter is not None:
            with open(include_filter) as f:
                self.include_tiles = f.readlines()
            self.include_tiles = [
                f.strip().removesuffix("/**").removesuffix("/*").removesuffix("/")
                for f in self.include_tiles
                if "#" not in f
            ]
            logger.info(f"include_tiles: {self.include_tiles}")

        self.exclude_tiles = None
        if exclude_filter is not None:
            with open(exclude_filter) as f:
                self.exclude_tiles = f.readlines()
            self.exclude_tiles = [
                f.strip().removesuffix("/**").removesuffix("/*").removesuffix("/")
                for f in self.exclude_tiles
                if "#" not in f
            ]
            logger.info(f"exclude_tiles: {self.exclude_tiles}")

        self.largest_gsd_product_shape = None

        logger.info(f"Ground covers: {ground_covers}")

    def get_era5_land_patch_size(self, ground_cover: int) -> int:
        """
        Get the patch size for ERA5-Land data.
        """
        return math.ceil(ground_cover / self.ERA5_LAND_GSD)

    def get_metadata_dir(self) -> Path:
        out_products = sorted({self.SUBDIR_MAP[prod] for prod in self.products})
        out_footprint_key = f"{Path(self.paths).name}_geometa_{'_'.join(out_products)}"
        out_footprint_key = out_footprint_key.replace("/", "_")
        out_dir = Path(self.paths).parent / out_footprint_key

        return out_dir

    def get_metadata_cache_name(self) -> str:
        out_products = sorted({self.SUBDIR_MAP[prod] for prod in self.products})
        out_filename = f"{'_'.join(out_products)}.geojson"
        out_filename = out_filename.replace("/", "_")
        return out_filename

    def expand_products(self, products):
        """
        Expand products to include all modes and polarizations if not already included.
        """

        expanded_products = []
        for product in products:
            if "S1" in product and (
                not any(polarization in product for polarization in self.S1_POLARIZATIONS)
                or not any(mode in product for mode in self.S1_MODES)
            ):
                mode, polarization = None, None
                p = product.split("-")
                if len(p) == 2:
                    product_name, product_gsd = (*p,)
                elif len(p) == 3:
                    product_name, unknown, product_gsd = (*p,)
                    if unknown in self.S1_MODES:
                        mode = unknown
                    elif unknown in self.S1_POLARIZATIONS:
                        polarization = unknown
                    else:
                        msg = f"Unknown part in product name: {unknown}"
                        raise ValueError(msg)
                else:
                    product_name, mode, polarization, product_gsd = (*p,)
                    logger.info(
                        f"product_name: {product_name}, mode: {mode}, polarization: {polarization}, product_gsd: {product_gsd}"
                    )

                if mode is not None:
                    modes = [mode]
                else:
                    modes = self.S1_MODES

                if polarization is not None:
                    polarizations = [polarization]
                else:
                    polarizations = self.S1_POLARIZATIONS

                for mode in modes:
                    for polarization in polarizations:
                        if len(p) == 2:
                            # Legacy
                            if mode == self.S1_MODES[-1] and product_gsd in ["10m", "60m"]:
                                logger.info(
                                    f"skipping product: [{product_name}-{mode}-{polarization}-{product_gsd}] during expansion"
                                )
                                continue

                        product_key = f"{product_name}-{mode}-{polarization}-{product_gsd}"

                        expanded_products.append(product_key)
            else:
                expanded_products.append(product)
        return expanded_products

    def _get_product_bands(self, product, netcdf=False):
        if netcdf:
            return list(self.NETCDF_PRODUCT_BAND_MAP[product].values())
        return list(self.PRODUCT_BAND_NAME_MAP[product].values())

    def GSD(self, product, resampled=False):
        if resampled:
            if product in self.product_changes:
                return int(self.product_changes[product].split("-")[-1].replace("m", ""))
            elif hasattr(self, "land_cover_product_changes") and product in self.land_cover_product_changes:
                return int(self.land_cover_product_changes[product].split("-")[-1].replace("m", ""))
        return int(product.split("-")[-1].replace("m", ""))

    @staticmethod
    def process_tile_dir(
        path_dir: Path,
        dataset_path: Path,
        subdir_to_products_map: dict,
        important_products: tuple,
        strict: bool,
        include_tiles: list | None,
        exclude_tiles: list | None,
        split_tiles: list | None,
        metadata_dir: Path | None,
        metadata_cache_name: str | None,
    ):
        path_dir = Path(path_dir)
        missing_products = Counter()

        # Filter on split
        if split_tiles is not None:
            if path_dir.parents[2].name not in split_tiles:
                # logger.info(f"skipping path: {path_dir} because it is not in split_tiles")
                return

        # Filter on all products existing
        if strict:
            if not all((path_dir / product_subdir).exists() for product_subdir in subdir_to_products_map.keys()):
                # logger.info(f"skipping path: {path_dir} because not all products exist")
                return

        # Filter on metadata
        if metadata_dir is not None:
            if not (metadata_dir / path_dir.relative_to(dataset_path) / metadata_cache_name).exists():
                # logger.info(f"skipping path: {path_dir} because metadata does not exist")
                return

        if include_tiles is not None:
            if not ("/".join(path_dir.parts[-4:]) in include_tiles or path_dir.parents[2].name in include_tiles):
                return
        if exclude_tiles is not None:
            if "/".join(path_dir.parts[-4:]) in exclude_tiles or path_dir.parents[2].name in exclude_tiles:
                return

        product_files_map = {}

        found_all_products = True
        found_S1_IW = True
        found_S1_EW = True
        found_important_products = False
        for product_subdir, products in subdir_to_products_map.items():
            product_files = [str(p.relative_to(path_dir)) for p in (path_dir / product_subdir).glob("*.nc")]
            for product in products:
                if len(product_files) > 0:
                    product_files_map[product] = product_files
                    if any(important_product in product for important_product in important_products) and not strict:
                        found_important_products = True
                else:
                    missing_products[product] += 1

                    if "S1" in product:
                        if "IW" in product:
                            found_S1_IW = False
                        elif "EW" in product:
                            found_S1_EW = False
                    else:
                        found_all_products = False
                        if strict:
                            logger.info(f"skipping path: {path_dir} because product {product} does not exist")

                            break

        if not found_S1_IW and not found_S1_EW:
            found_all_products = False

        if found_all_products or found_important_products:
            return path_dir, product_files_map, missing_products

    def load_files(self, split=None, strict=True) -> tuple[list[str], dict[str, dict[str, list[str]]]]:
        """A list of all files in the dataset.

        Returns:
            All files in the dataset.

        """

        subdir_to_products_map = {}
        for product, subdir in self.SUBDIR_MAP.items():
            if product not in self.products:
                continue
            if subdir not in subdir_to_products_map:
                subdir_to_products_map[subdir] = []
            subdir_to_products_map[subdir].append(product)

        file_dirs = []
        file_path_map = {}
        missing_products = Counter()
        path = Path(self.paths)

        if not path.exists() or not path.is_dir():
            warnings.warn(
                f"Could not find any relevant files for provided path '{path}'. Path was ignored.",
                UserWarning,
            )

        split_tiles = None
        if split is not None:
            split_file = f"{path}_{split}_tiles.txt"
            if os.path.exists(split_file):
                with open(split_file) as f:
                    split_tiles = f.read().splitlines()
            else:
                warnings.warn(
                    f"Could not find split file '{split_file}'. Ignoring split.",
                    UserWarning,
                )

        process_func = partial(
            self.process_tile_dir,
            dataset_path=path,
            subdir_to_products_map=subdir_to_products_map,
            important_products=self.important_products,
            strict=strict,
            include_tiles=self.include_tiles,
            exclude_tiles=self.exclude_tiles,
            split_tiles=split_tiles,
            metadata_dir=None if not hasattr(self, "metadata_dir") else self.metadata_dir,
            metadata_cache_name=None if not hasattr(self, "metadata_cache_name") else self.metadata_cache_name,
        )

        pathname = os.path.join(str(path), self.FILENAME_GLOB)
        logger.info(f"Searching for files in {path} with glob {self.FILENAME_GLOB}")
        # for r in tqdm(map(process_func, glob.iglob(pathname, recursive=False))):
        with mp.Pool(8) as pool:
            for r in tqdm(
                pool.imap_unordered(process_func, glob.iglob(pathname, recursive=False), chunksize=2),
                desc="Loading files",
                disable=(hasattr(self, "global_rank") and self.global_rank != 0),
            ):
                if r is not None:
                    file_dirs.append(r[0])
                    file_path_map[r[0]] = r[1]
                    missing_products += r[2]

        logger.info(f"missing product stats: {missing_products}")

        file_dirs.sort()
        np.random.default_rng(self.random_seed).shuffle(file_dirs)
        file_dirs = file_dirs[: int(self.data_percent * len(file_dirs))]
        file_dirs.sort()
        file_path_map = {p: file_path_map[p] for p in file_dirs}

        return file_dirs, file_path_map

    def _build_transforms(self, is_train, products, product_changes):
        custom_transforms = {}

        self.cons_vertical_flips = {}
        self.cons_horiz_flips = {}
        self.cont_cons_rand_crops = {}

        for ground_cover in self.ground_covers:
            logger.debug(f"building transforms for ground cover: {ground_cover}")
            custom_transforms[ground_cover] = {}

            # Assume the order is sorted from smallest to largest img_size b/c
            # we sorted in class init
            img_sizes = [
                math.ceil(ground_cover / self.GSD(product_changes[product]))
                for product in products
                if product in self.ground_cover_products[ground_cover]
            ]
            logger.debug(f"ground cover: {ground_cover}, img_sizes: {img_sizes}")

            cont_cons_rand_crop = ControlledConsistentRandomCrop(
                img_sizes, pad_if_needed=True, padding_mode="constant", fill=0
            )

            self.cont_cons_rand_crops[ground_cover] = cont_cons_rand_crop
            cons_horiz_flip = ConsistentRadomHorizontalFlip(len(img_sizes))
            self.cons_horiz_flips[ground_cover] = cons_horiz_flip
            cons_vertical_flip = ConsistentRadomVerticalFlip(len(img_sizes))
            self.cons_vertical_flips[ground_cover] = cons_vertical_flip
            i = 0
            for product in products:
                if product not in self.ground_cover_products[ground_cover]:
                    logger.debug(f"skipping product: {product} for ground cover {ground_cover}")
                    continue

                t = []

                if self.standardize and (product not in self.land_cover_products):
                    if product in self.NORMALIZATION:
                        mean = [
                            self.NORMALIZATION[product][band]["mean"]
                            for band in self.PRODUCT_BAND_NAME_MAP[product].values()
                        ]
                        std = [
                            self.NORMALIZATION[product][band]["std"]
                            for band in self.PRODUCT_BAND_NAME_MAP[product].values()
                        ]
                    else:
                        mean = [0.0 for _ in self.PRODUCT_BAND_NAME_MAP[product]]
                        std = [1.0 for _ in self.PRODUCT_BAND_NAME_MAP[product]]
                        warnings.warn(
                            f"No normalization values found for product {product}, using mean=0.0, std=1.0",
                            UserWarning,
                        )
                    t.append(transforms.Normalize(mean, std))
                    t.append(clamp_5_sigma)

                if product_changes[product] != product:
                    # calculate resampled img size
                    (product_change_name, *_) = product_changes[product].split("-")
                    product_change_gsd = self.GSD(product_changes[product])
                    img_size = math.ceil(ground_cover / product_change_gsd)
                    logger.info(f"resampling product: {product}, to: {product_changes[product]}, img_size: {img_size}")
                    t.append(
                        transforms.Resize(
                            (img_size, img_size),
                            interpolation=transforms.InterpolationMode.BILINEAR
                            if product not in self.land_cover_products
                            else transforms.InterpolationMode.NEAREST,
                        )
                    )

                if is_train:
                    call_wrapper = CallWrapper(cont_cons_rand_crop.forward, i)

                    t.append(call_wrapper)
                    t.append(cons_horiz_flip)
                    t.append(cons_vertical_flip)
                else:
                    # For each grouped product, we will need a different input size
                    t.append(transforms.CenterCrop(img_sizes[i]))

                custom_transforms[ground_cover][product] = transforms.Compose(t)
                logger.debug(f"transforms for product {product}: {custom_transforms[ground_cover][product]}")
                i += 1

        return custom_transforms

    # stacked products
    def get_era5_product(
        self,
        date: datetime,
        ground_cover: int,
        crop_center_points: np.ndarray,
        crop_crs_wkt: str,
        products: list[str] | tuple[str] | None = None,
        interpolate: bool = False,
        era5_land: bool = False,
    ) -> dict[str, np.ndarray] | None:
        batch_size = len(crop_center_points)

        if era5_land:
            era5_data_path = (
                Path(self.era5_data_dir) / f"{date.year}" / f"ERA5_Land_{date.year}_{date.month:02d}_{date.day:02d}.nc"
            )
            era5_patch_size = self.get_era5_land_patch_size(ground_cover)
            era5_normalization = self.NORMALIZATION["ERA5-Land"]
            era5_products = self.era5_land_products
            era5_gsd = self.ERA5_LAND_GSD

        if not era5_data_path.exists():
            return None

        if era5_patch_size == 1:
            era5_points_lonlat = Transformer.from_crs(crop_crs_wkt, "epsg:4326", always_xy=True).transform(
                crop_center_points[:, 0, 0], crop_center_points[:, 1, 0]
            )

        else:
            i = np.arange(-math.floor(era5_patch_size / 2), math.ceil(era5_patch_size / 2), dtype=np.float32)
            j = np.arange(-math.floor(era5_patch_size / 2), math.ceil(era5_patch_size / 2), dtype=np.float32)
            if era5_patch_size % 2 == 0:
                i += 0.5
                j += 0.5
            i *= era5_gsd
            j *= era5_gsd

            crop_x_points = crop_center_points[:, 0, 0, None] + i[None]
            crop_y_points = crop_center_points[:, 1, 0, None] + j[None]

            crop_x_points = crop_x_points.flatten()
            crop_y_points = crop_y_points.flatten()

            era5_points_lonlat = Transformer.from_crs(crop_crs_wkt, "epsg:4326", always_xy=True).transform(
                crop_x_points, crop_y_points
            )

            era5_points_lonlat = (
                era5_points_lonlat[0].reshape(batch_size, era5_patch_size),
                era5_points_lonlat[1].reshape(batch_size, era5_patch_size),
            )

        lon, lat = era5_points_lonlat

        with xr.open_dataset(
            era5_data_path, engine="h5netcdf", lock=False, cache=False, decode_cf=False, group="era5_land"
        ) as era5_data:
            if era5_patch_size == 1:
                if interpolate:
                    era5_vars = era5_data.interp(y=lat, x=lon)
                else:
                    era5_vars = era5_data.sel(
                        y=xr.DataArray(lat, dims="points"),
                        x=xr.DataArray(lon, dims="points"),
                        method="nearest",
                    )

                if products is None:
                    products = list(era5_data.data_vars)
                    # remove spatial_ref
                    products.remove("spatial_ref")

                if self.standardize:
                    era5_vars = {
                        product: (
                            _to_tensor(
                                np.expand_dims(
                                    era5_vars[product].values,
                                    (
                                        1,
                                        2,
                                    ),
                                )
                            )
                            - era5_normalization[product]["mean"]
                        )
                        / era5_normalization[product]["std"]
                        for product in products
                    }
                else:
                    era5_vars = {
                        product: _to_tensor(np.expand_dims(era5_vars[product].values, (1, 2))) for product in products
                    }

            else:
                era5_vars_total = [
                    era5_data.sel(y=lat[b], x=lon[b], method="nearest").drop_vars(
                        [
                            "y",
                            "x",
                            "spatial_ref",
                        ]
                    )
                    for b in range(lat.shape[0])
                ]
                era5_vars_total = xr.concat(era5_vars_total, dim="points", coords="minimal", compat="override")

                if products is None:
                    products = list(era5_data.data_vars)
                    # remove spatial_ref
                    products.remove("spatial_ref")

                if self.standardize:
                    era5_vars = {
                        product: (_to_tensor(era5_vars_total[product].values) - era5_normalization[product]["mean"])
                        / era5_normalization[product]["std"]
                        for product in products
                    }
                else:
                    era5_vars = {product: _to_tensor(era5_vars_total[product].values) for product in products}

            era5_vars = {
                product_name: era5_vars[product_name]
                if product_name in era5_vars
                else torch.zeros((lat.shape[0], era5_patch_size, era5_patch_size))
                for product_name in era5_products
            }

            return era5_vars

    def collate_fn(self, batch: ProductDataBatch) -> BandDataBatch:
        all_product_imgs = batch.product_imgs

        collated_imgs = {}
        band_metadata = {}
        for product in all_product_imgs:
            product_imgs = all_product_imgs[product]
            single_band_S1 = None
            if "1SSH" in product:
                product = product.removesuffix("-1SSH")
                single_band_S1 = "sigma0_hh"
            elif "1SSV" in product:
                product = product.removesuffix("-1SSV")
                single_band_S1 = "sigma0_vv"

            if product in self.land_cover_products:
                collated_imgs[product.split("-")[0]] = product_imgs[:, None, ...]
                continue

            if product in self.dem_products:
                collated_imgs[product.split("-")[0]] = product_imgs[:]
                continue

            if product in self.incidence_angle_products:
                collated_imgs[product.split("-")[0]] = product_imgs[:, None, ...]
                continue

            # Need product name to differentiate between S1 multilooked to different resolutions
            for band_name, prod_band_name in zip(
                self.NETCDF_PRODUCT_BAND_MAP[product].values(),
                self.PRODUCT_BAND_NAME_MAP[product].values(),
                strict=False,
            ):
                prod_bands = self._get_product_bands(product, netcdf=True)

                if single_band_S1:
                    prod_bands = [single_band_S1]
                    if single_band_S1 != band_name:
                        continue

                band_index = prod_bands.index(band_name)
                band_imgs = product_imgs[:, band_index]

                if band_name not in self.discard_bands:
                    if (nan_index := torch.isnan(band_imgs)).any():
                        if self.legacy_nan_handling:
                            band_imgs[nan_index] = -0.1
                        else:
                            band_means = torch.nanmean(band_imgs, dim=(1, 2))[:, None, None].repeat(
                                1, band_imgs.shape[1], band_imgs.shape[2]
                            )
                            band_imgs[nan_index] = band_means[nan_index]

                    collated_imgs[prod_band_name] = band_imgs[:, None, ...]
                    band_metadata[prod_band_name] = batch.metadata.channel_params[product].copy()

        thor_data = BandDataBatch(
            band_imgs=collated_imgs,
            metadata=replace(batch.metadata, channel_params=band_metadata),
            era5_land_data=batch.era5_land_data,
        )

        return thor_data

    def plot(self, sample, metadata: MetaData | None = None, num_samples=None):
        B = sample[next(iter(sample.keys()))].shape[0] if num_samples is None else num_samples

        filtered_products = (
            self.products
            + self.land_cover_products
            + (["dem"] if len(self.dem_products) > 0 else [])
            + (["INC"] if len(self.incidence_angle_products) else [])
        )
        if any("S1" in p and "60m" in p for p in filtered_products) and any(
            "S1" in p and "10m" in p for p in filtered_products
        ):
            filtered_products = [p for p in filtered_products if not ("S1" in p and "60m" in p)]

        P = len(filtered_products)
        fig, ax = plt.subplots(B, P, figsize=(P * 4, B * 4), squeeze=False)

        if metadata is not None:
            ground_cover = metadata.ground_cover
            center_coords = metadata.center_coords
            month = metadata.month
            s1_orbit_direction = (
                metadata.s1_orbit_direction[0, 0].item() if metadata.s1_orbit_direction is not None else None
            )
            s1_incidence_angles = (
                metadata.s1_incidence_angles[0, 0].item() if metadata.s1_incidence_angles is not None else None
            )
            logger.info(
                f"ground cover: {ground_cover}, center coords: {center_coords[:num_samples].round(decimals=2)}, month: {month[0, 0].item()}, s1 orbit direction: {s1_orbit_direction}, s1 incidence angle: {s1_incidence_angles}"
            )

            fig.suptitle(
                f"Ground cover: {ground_cover}m, Center coords: {center_coords[:num_samples].round(decimals=2).tolist()}, Month: {month[0, 0].item()}, S1 orbit direction: {s1_orbit_direction}, S1 incidence angle: {s1_incidence_angles}",
                fontsize=16,
            )

        min_values = [1] * len(filtered_products)
        max_values = [0] * len(filtered_products)

        for b in range(B):
            for i, product in enumerate(filtered_products):
                if product == "S2-10m":
                    bands = self.RGB_BANDS

                elif product in self.land_cover_products:
                    bands = [product]
                elif product == "dem":
                    bands = [product]
                elif product == "INC":
                    bands = [product]

                else:
                    bands = []
                    prod_bands = [band for band in self._get_product_bands(product) if band in sample]
                    if len(prod_bands) == 0:
                        continue

                    if len(prod_bands) == 2:
                        prod_bands = [prod_bands[0], prod_bands[1], prod_bands[1]]
                    elif len(prod_bands) > 3:
                        prod_bands = prod_bands[:3]

                    for prod_band_name in prod_bands:
                        bands.append(prod_band_name)

                if bands[0] not in sample:
                    continue

                image = torch.stack([sample[band][b] for band in bands])

                if product not in self.land_cover_products:
                    image_min = torch.min(image).item()
                    image_max = torch.max(image).item()
                    min_values[i] = min(image_min, min_values[i])
                    max_values[i] = max(image_max, max_values[i])

                image = torch.squeeze(image, dim=(1)).cpu().numpy()
                image = image.transpose(1, 2, 0)

                image = np.ma.masked_invalid(image)

                if product not in self.land_cover_products:
                    # calculate 99th percentile
                    image_min_p = np.percentile(image, 1)
                    image_max_p = np.percentile(image, 99)

                    image = (image - image_min_p) / (image_max_p - image_min_p)

                # rescale to 0-1
                image = (image - image.min()) / (image.max() - image.min())

                product_name = product
                if "S1" in product:
                    product_name = "-".join(product.split("-")[:-1])

                ax[b, i].imshow(image)
                if product not in self.land_cover_products:
                    ax[b, i].set_title(
                        f"{product_name} shape {image.shape} \n min: {min_values[i]:.2f}, max: {max_values[i]:.2f} \n ({', '.join(b.split(':')[-1] for b in bands)})"
                    )
                else:
                    ax[b, i].set_title(f"{product}")

        fig.tight_layout()
        plt.show()

        return fig
