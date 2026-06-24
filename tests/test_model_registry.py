from __future__ import annotations

import torch
from torch import nn

from thor.core.model_registry import ModelRegistry
from thor.utils.patch_embed import pi_resize_patch_embed


class DummyPatchEmbedModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.ind_patch_embed = nn.Module()
        self.ind_patch_embed.patch_embed = nn.ModuleDict(
            {
                "src": nn.Conv2d(1, 4, kernel_size=8, stride=8, bias=True),
                "dst": nn.Conv2d(1, 4, kernel_size=4, stride=4, bias=True),
            }
        )
        self.ind_patch_embed.channel_rename_map = {
            "Sensor:Source": "src",
            "Sensor:Destination": "dst",
        }


class DummyFullModel(nn.Module):
    """Model with patch_embed, band_embed, and prod_embed — mirrors the real ThorViT structure."""

    EMBED_DIM = 16

    def __init__(self) -> None:
        super().__init__()
        self.ind_patch_embed = nn.Module()
        self.ind_patch_embed.patch_embed = nn.ModuleDict(
            {
                "src": nn.Conv2d(1, self.EMBED_DIM, kernel_size=8, stride=8, bias=True),
                "dst": nn.Conv2d(1, self.EMBED_DIM, kernel_size=4, stride=4, bias=True),
            }
        )
        self.ind_patch_embed.channel_rename_map = {
            "Sensor:Source": "src",
            "NewSensor:Destination": "dst",
        }
        self.ind_patch_embed.groups = {
            "group0": ["Sensor:Source"],
            "group1": ["NewSensor:Destination"],
        }
        self.band_embed = nn.ParameterDict(
            {
                "group0": nn.Parameter(torch.randn(self.EMBED_DIM), requires_grad=False),
                "group1": nn.Parameter(torch.randn(self.EMBED_DIM), requires_grad=False),
            }
        )
        self.prod_embed = nn.ParameterDict(
            {
                "Sensor": nn.Parameter(torch.randn(self.EMBED_DIM), requires_grad=False),
                "NewSensor": nn.Parameter(torch.randn(self.EMBED_DIM), requires_grad=False),
            }
        )


def test_apply_ckpt_init_from_resizes_cloned_patch_embed_weight() -> None:
    model = DummyPatchEmbedModel()
    registry = ModelRegistry()
    src_weight_key = "ind_patch_embed.patch_embed.src.weight"
    src_bias_key = "ind_patch_embed.patch_embed.src.bias"
    dst_weight_key = "ind_patch_embed.patch_embed.dst.weight"
    dst_bias_key = "ind_patch_embed.patch_embed.dst.bias"

    src_weight = torch.randn_like(model.state_dict()[src_weight_key])
    src_bias = torch.randn_like(model.state_dict()[src_bias_key])
    model_state_dict = {
        src_weight_key: src_weight,
        src_bias_key: src_bias,
    }

    registry._apply_ckpt_init_from(
        model,
        model_state_dict,
        {"channels": {"Sensor:Destination": "Sensor:Source"}},
    )

    expected_weight = pi_resize_patch_embed(src_weight, tuple(model.state_dict()[dst_weight_key].shape[2:]))

    assert model_state_dict[dst_weight_key].shape == model.state_dict()[dst_weight_key].shape
    assert torch.allclose(model_state_dict[dst_weight_key], expected_weight)
    assert torch.equal(model_state_dict[dst_bias_key], src_bias)
    assert model_state_dict[dst_bias_key] is not src_bias


def test_apply_ckpt_init_from_fills_band_embed_for_new_group() -> None:
    model = DummyFullModel()
    registry = ModelRegistry()

    src_weight = torch.randn_like(model.state_dict()["ind_patch_embed.patch_embed.src.weight"])
    src_bias = torch.randn_like(model.state_dict()["ind_patch_embed.patch_embed.src.bias"])
    # Checkpoint only has group0's band_embed, not group1's
    src_band_embed = torch.randn_like(model.state_dict()["band_embed.group0"])
    model_state_dict = {
        "ind_patch_embed.patch_embed.src.weight": src_weight,
        "ind_patch_embed.patch_embed.src.bias": src_bias,
        "band_embed.group0": src_band_embed,
    }

    registry._apply_ckpt_init_from(
        model,
        model_state_dict,
        {"channels": {"NewSensor:Destination": "Sensor:Source"}},
    )

    # band_embed for group1 should be auto-filled from model's default init
    assert "band_embed.group1" in model_state_dict
    assert model_state_dict["band_embed.group1"].shape == model.state_dict()["band_embed.group1"].shape
    assert torch.equal(model_state_dict["band_embed.group1"], model.state_dict()["band_embed.group1"])
    # group0's band_embed must be untouched
    assert torch.equal(model_state_dict["band_embed.group0"], src_band_embed)


def test_apply_ckpt_init_from_fills_prod_embed_default_init() -> None:
    model = DummyFullModel()
    registry = ModelRegistry()

    src_weight = torch.randn_like(model.state_dict()["ind_patch_embed.patch_embed.src.weight"])
    # Checkpoint has Sensor's prod_embed but not NewSensor's
    src_prod_embed = torch.randn_like(model.state_dict()["prod_embed.Sensor"])
    model_state_dict = {
        "ind_patch_embed.patch_embed.src.weight": src_weight,
        "ind_patch_embed.patch_embed.src.bias": torch.randn_like(
            model.state_dict()["ind_patch_embed.patch_embed.src.bias"]
        ),
        "prod_embed.Sensor": src_prod_embed,
    }

    registry._apply_ckpt_init_from(
        model,
        model_state_dict,
        {"channels": {"NewSensor:Destination": "Sensor:Source"}},
    )

    assert "prod_embed.NewSensor" in model_state_dict
    assert torch.equal(model_state_dict["prod_embed.NewSensor"], model.state_dict()["prod_embed.NewSensor"])
    # Source product must be untouched
    assert torch.equal(model_state_dict["prod_embed.Sensor"], src_prod_embed)


def test_apply_ckpt_init_from_fills_prod_embed_clone() -> None:
    model = DummyFullModel()
    registry = ModelRegistry()

    src_weight = torch.randn_like(model.state_dict()["ind_patch_embed.patch_embed.src.weight"])
    src_prod_embed = torch.randn_like(model.state_dict()["prod_embed.Sensor"])
    model_state_dict = {
        "ind_patch_embed.patch_embed.src.weight": src_weight,
        "ind_patch_embed.patch_embed.src.bias": torch.randn_like(
            model.state_dict()["ind_patch_embed.patch_embed.src.bias"]
        ),
        "prod_embed.Sensor": src_prod_embed,
    }

    registry._apply_ckpt_init_from(
        model,
        model_state_dict,
        {"channels": {"NewSensor:Destination": "Sensor:Source"}, "clone_prod_embed": True},
    )

    assert "prod_embed.NewSensor" in model_state_dict
    assert torch.equal(model_state_dict["prod_embed.NewSensor"], src_prod_embed)
    assert model_state_dict["prod_embed.NewSensor"] is not src_prod_embed  # must be a clone
