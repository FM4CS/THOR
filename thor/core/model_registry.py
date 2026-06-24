import logging
import re
import warnings

import torch
from torch import nn

from thor.utils.helper import extract_model_state_dict_from_ckpt
from thor.utils.patch_embed import pi_resize_patch_embed
from thor.utils.pos_embed import interpolate_pos_embed_thor

logging.basicConfig(
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class ModelRegistry:
    def __init__(self):
        self.models = {}

    def _register(self, model_name: str, model: nn.Module):
        if model_name is None:
            model_name = model.__name__

        if model_name in self.models:
            msg = f"Model {model_name} already registered"
            raise ValueError(msg)

        self.models[model_name] = model

    def register(self, model_name: str | None = None, model: nn.Module = None):
        def _register_wrapper(model):
            self._register(model_name, model)
            return model

        return _register_wrapper

    def get_model(self, model_name: str) -> nn.Module:
        return self.models[model_name]

    def _apply_ckpt_init_from(self, model, model_state_dict, ckpt_init_from):
        """Initialize new modality weights from existing checkpoint weights.

        Config format:
            ckpt_init_from:
                channels:
                    "NewSensor:BandA": "S2:Red"     # Clone Conv2d patch embed weights
                    "NewSensor:BandB": "S2:Green"
                groups:
                    "group10": "group0"             # Clone band_embed from source group
                clone_prod_embed: false             # If true, clone prod_embed from source product;
                                                    # default false uses model's sincos init
        """
        channel_map = ckpt_init_from.get("channels", {})
        group_map = ckpt_init_from.get("groups", {})
        clone_prod_embed = ckpt_init_from.get("clone_prod_embed", False)

        # Resolve channel_rename_map if available
        rename_map = {}
        if hasattr(model, "ind_patch_embed") and hasattr(model.ind_patch_embed, "channel_rename_map"):
            rename_map = model.ind_patch_embed.channel_rename_map
        elif hasattr(model, "channel_rename_map"):
            rename_map = model.channel_rename_map

        # Clone patch embed Conv2d weights for new channels
        for dst_channel, src_channel in channel_map.items():
            dst_name = rename_map.get(dst_channel, dst_channel)
            src_name = rename_map.get(src_channel, src_channel)

            for suffix in ("weight", "bias"):
                src_key = f"ind_patch_embed.patch_embed.{src_name}.{suffix}"
                dst_key = f"ind_patch_embed.patch_embed.{dst_name}.{suffix}"

                if src_key not in model_state_dict:
                    logger.warning(
                        f"ckpt_init_from: source key '{src_key}' not found in checkpoint, "
                        f"skipping clone for '{dst_channel}' -> '{src_channel}'"
                    )
                    continue

                # Only clone if the destination key is expected by the model
                if dst_key in model.state_dict():
                    cloned = model_state_dict[src_key].clone()
                    if suffix == "weight":
                        dst_shape = model.state_dict()[dst_key].shape
                        if cloned.shape[2:] != dst_shape[2:]:
                            cloned = pi_resize_patch_embed(cloned, tuple(dst_shape[2:]))
                    model_state_dict[dst_key] = cloned
                    logger.info(f"ckpt_init_from: cloned {src_key} -> {dst_key}")

        # Clone band_embed for new groups
        for dst_group, src_group in group_map.items():
            src_key = f"band_embed.{src_group}"
            dst_key = f"band_embed.{dst_group}"

            if src_key not in model_state_dict:
                logger.warning(
                    f"ckpt_init_from: source key '{src_key}' not found in checkpoint, "
                    f"skipping clone for '{dst_group}' -> '{src_group}'"
                )
                continue

            if dst_key in model.state_dict():
                model_state_dict[dst_key] = model_state_dict[src_key].clone()
                logger.info(f"ckpt_init_from: cloned {src_key} -> {dst_key}")

        if channel_map:
            model_sd = model.state_dict()

            # Build channel -> group lookup from the model
            channel_to_group: dict[str, str] = {}
            groups_attr = None
            if hasattr(model, "ind_patch_embed") and hasattr(model.ind_patch_embed, "groups"):
                groups_attr = model.ind_patch_embed.groups
            elif hasattr(model, "groups"):
                groups_attr = model.groups
            if groups_attr is not None:
                for group_name, members in groups_attr.items():
                    for member in members:
                        channel_to_group[member] = group_name

            # Auto-fill band_embed for new groups introduced by channel_map.
            # Explicit group_map entries already handle the clone case; here we
            # use the model's default sincos init for groups not otherwise covered.
            for dst_channel in channel_map:
                group_name = channel_to_group.get(dst_channel)
                if group_name is None:
                    continue
                band_embed_key = f"band_embed.{group_name}"
                if band_embed_key in model_sd and band_embed_key not in model_state_dict:
                    model_state_dict[band_embed_key] = model_sd[band_embed_key].clone()
                    logger.info(
                        f"ckpt_init_from: initialized band_embed for new group '{group_name}' from model default"
                    )

            # Fill prod_embed for new products.
            # clone_prod_embed=True  → clone from source product in model_state_dict
            # clone_prod_embed=False → use model's default sincos init (default)
            for dst_channel, src_channel in channel_map.items():
                dst_product = dst_channel.split(":")[0]
                src_product = src_channel.split(":")[0]
                prod_embed_key = f"prod_embed.{dst_product}"
                if prod_embed_key not in model_sd or prod_embed_key in model_state_dict:
                    continue
                if clone_prod_embed:
                    src_prod_key = f"prod_embed.{src_product}"
                    if src_prod_key in model_state_dict:
                        model_state_dict[prod_embed_key] = model_state_dict[src_prod_key].clone()
                        logger.info(f"ckpt_init_from: cloned prod_embed {src_prod_key} -> {prod_embed_key}")
                    else:
                        logger.warning(
                            f"ckpt_init_from: clone_prod_embed=True but '{src_prod_key}' not in checkpoint, "
                            f"falling back to model default init for '{prod_embed_key}'"
                        )
                        model_state_dict[prod_embed_key] = model_sd[prod_embed_key].clone()
                else:
                    model_state_dict[prod_embed_key] = model_sd[prod_embed_key].clone()
                    logger.info(
                        f"ckpt_init_from: initialized prod_embed for new product '{dst_product}' from model default"
                    )

    def build(self, model_cfgs) -> nn.Module:
        if model_cfgs.get("name", None) is not None:  # single model config
            model_cfgs = {model_cfgs["name"]: model_cfgs}

        models = {}

        for model_name, model_cfg in model_cfgs.items():
            model_type = model_cfg.get("type")

            if model_type not in self.models:
                msg = f"Model {model_name} not found in registry, available models: {self.models}"
                raise ValueError(msg)

            input_params = model_cfg.get("input_params", {})
            model_kwargs = model_cfg.get("kwargs", {})

            logger.debug(f"Building model {model_name} with input params: {input_params}, kwargs {model_kwargs}")

            model = self.get_model(model_type)(input_params, **model_kwargs)

            ckpt = model_cfg.get("ckpt", None)
            ckpt_ignore = model_cfg.get("ckpt_ignore", [])
            ckpt_copy = model_cfg.get("ckpt_copy", [])
            ckpt_remap = model_cfg.get("ckpt_remap", {})
            ckpt_init_from = model_cfg.get("ckpt_init_from", {})
            strict = model_cfg.get("strict", True)
            resize_patch_embed = model_cfg.get("resize_patch_embed", False)
            target_model = model_cfg.get("target_model", model_name)

            if ckpt is not None:
                logger.info(f"Loading custom weight for {model_name} from {ckpt}")
                ckpt = torch.load(ckpt, map_location="cpu")
                model_state_dict = extract_model_state_dict_from_ckpt(ckpt)[target_model]

                # Pop all of the keys that can be ignored
                new_keys = list(model_state_dict.keys())
                for rgx_item in ckpt_ignore:
                    re_expr = re.compile(rgx_item)
                    new_keys = [key for key in new_keys if not re_expr.match(key)]
                model_state_dict = {k: model_state_dict[k] for k in new_keys}

                # Add in all keys you want to copy the default param for
                for copy_key in ckpt_copy:
                    logger.info(f"Skipping model load for: {copy_key}")
                    model_state_dict[copy_key] = model.state_dict()[copy_key]

                # Remap certain keys for multi sensor pretrain to single sensor finetune
                for key, cfg_map in ckpt_remap.items():
                    logger.info(f"Remapping key for custom load: {key}")
                    old_val = model_state_dict.pop(key)
                    new_val = old_val
                    # Apply modifications
                    new_name = cfg_map.get("name", key)
                    params = cfg_map.get("params", {})
                    func = cfg_map.get("func", None)
                    if func == "index_select":
                        if isinstance(params["indices"], str):
                            start, stop = params["indices"].split(":")
                            params["indices"] = torch.arange(int(start), int(stop))
                        new_val = torch.index_select(old_val, params["dim"], torch.tensor(params["indices"]))
                    elif func == "concat_passthrough":
                        new_val = torch.cat(
                            (
                                new_val,
                                torch.index_select(
                                    model.state_dict()[new_name], params["dim"], torch.tensor(params["index"])
                                ),
                            ),
                            dim=params["dim"],
                        )
                    # elif func == "replace":
                    #     new_val = model.state_dict()[new_name]
                    # elif func == "resize":
                    #     new_val = pi_resize_patch_embed(old_val, params["new_patch_size"])

                    model_state_dict[new_name] = new_val

                if resize_patch_embed:
                    channels = input_params.get("channels", None)
                    ground_cover = input_params.get("ground_cover", None)
                    if ground_cover is None:
                        ground_cover = input_params.get("ground_covers", None)
                        if isinstance(ground_cover, list):
                            ground_cover = max(ground_cover)

                    patch_embed_keys = [k for k in model_state_dict.keys() if "patch_embed" in k and "weight" in k]

                    def _get_patch_size(ground_cover: int, GSD: int, num_patch: int) -> tuple[int, int]:
                        patch_size = int(ground_cover / (GSD * num_patch))
                        assert patch_size * GSD * num_patch == ground_cover, (
                            f"Patch size {patch_size} does not divide ground cover {ground_cover} evenly"
                        )
                        return patch_size, patch_size

                    for patch_embed_key in patch_embed_keys:
                        channel_key = patch_embed_key.split(".")[-2]
                        if channel_key not in channels:
                            warnings.warn(
                                f"Channel {channel_key} not found in input params, skipping resizing of patch embed",
                                stacklevel=2,
                            )
                            continue
                        new_patch_size = _get_patch_size(ground_cover, **channels[channel_key])

                        if model_state_dict[patch_embed_key].shape[2:] != new_patch_size:
                            logger.info(f"Resizing patch embed for {patch_embed_key}")
                            model_state_dict[patch_embed_key] = pi_resize_patch_embed(
                                model_state_dict[patch_embed_key], new_patch_size
                            )

                # Initialize new modalities from existing checkpoint weights
                if ckpt_init_from:
                    self._apply_ckpt_init_from(model, model_state_dict, ckpt_init_from)

                # Interpolate pos embedding if necessary
                pos_embed_keys = [f"pos_embed.{k}" for k in model.pos_embed.keys()]
                pos_embeds_needs_reinit = False
                if len(pos_embed_keys) > 0:
                    ref_pos_embed_key = sorted(pos_embed_keys, key=lambda x: int(x.split(".")[-1]))[0]
                    if (
                        ref_pos_embed_key in model_state_dict
                        and model.state_dict()[ref_pos_embed_key].shape != model_state_dict[ref_pos_embed_key].shape
                    ):
                        logger.info("interpolating pos_embed")

                        # interpolating the ref pos embed (lowest GSD)
                        interpolate_pos_embed_thor(model, model_state_dict, ref_pos_embed_key)

                        # remove the other pos embeds, as they will be reinitialized afterwards
                        for k in pos_embed_keys:
                            if k in model_state_dict:
                                del model_state_dict[k]

                        pos_embeds_needs_reinit = True

                # Check for missing and unexpected keys
                model_keys = set(model.state_dict().keys())
                ckpt_keys = set(model_state_dict.keys())

                missing_keys = model_keys - ckpt_keys
                unexpected_keys = ckpt_keys - model_keys

                logger.info(f"Key comparison for {model_name}:")
                logger.info(f"  Model has {len(model_keys)} parameters")
                logger.info(f"  Checkpoint has {len(ckpt_keys)} parameters")

                if missing_keys:
                    logger.info(f"  Missing from checkpoint ({len(missing_keys)} keys):")
                    for key in sorted(missing_keys):
                        shape = list(model.state_dict()[key].shape)
                        logger.info(f"    - {key} {shape}")

                if unexpected_keys:
                    logger.info(f"  Unexpected in checkpoint ({len(unexpected_keys)} keys):")
                    for key in sorted(unexpected_keys):
                        shape = list(model_state_dict[key].shape)
                        logger.info(f"    + {key} {shape}")

                model.load_state_dict(model_state_dict, strict=strict)
                logger.info(f"Custom weight loaded for {model_name}")

                if pos_embeds_needs_reinit:
                    model.init_embeds(pos_only=True)

            models[model_name] = model

        return models


# global model registry
MODELS = ModelRegistry()
