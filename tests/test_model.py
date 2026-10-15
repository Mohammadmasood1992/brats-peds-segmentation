"""Forward/backward shape checks of the U-Net across its main options (tiny models, CPU-friendly)."""

import pytest
import torch

from brats_peds_seg.config import N_CLASSES, load_config, resolve_config
from brats_peds_seg.losses import SegmentationLoss
from brats_peds_seg.model import UNet

VARIANTS = [
    {},
    {"bottleneck": "transformer", "transformer_heads": 4, "skip_attention": True, "block_attention": "cbam"},
    {"bottleneck": "aspp", "block_attention": "se", "downsample": "maxpool", "upsample": "interpolate", "norm": "group",
     "residual": False, "dropout": 0.1},
    {"deep_supervision": False, "norm": "batch", "activation": "gelu"},
]


def tiny_config(dim, **overrides):
    return resolve_config(load_config(overrides={"dim": dim, "base_channels": 4, "max_channels": 32, "depth": 4,
                                                 "channels_last": False, **overrides}))


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("dim", ["2d", "3d"])
def test_unet_shapes_and_loss(dim, variant):
    cfg = tiny_config(dim, **variant)
    model = UNet(cfg).train()
    size = (32,) * cfg["spatial_dims"]
    x = torch.randn(2, cfg["in_channels"], *size)
    outputs = model(x)
    outputs = outputs if isinstance(outputs, list) else [outputs]
    assert outputs[0].shape == (2, N_CLASSES, *size)
    expected_levels = min(cfg["deep_supervision_levels"], cfg["depth"] - 1) if cfg["deep_supervision"] else 1
    assert len(outputs) == expected_levels
    target = torch.randint(0, N_CLASSES, (2, 1, *size)).float()
    loss = SegmentationLoss(cfg)(outputs, target)
    loss.backward()
    assert torch.isfinite(loss)
    model.eval()
    with torch.no_grad():
        assert model(x).shape == (2, N_CLASSES, *size)


def test_context_slices_set_input_channels():
    cfg = tiny_config("2d", context_slices=2, modalities=["t1c", "t2f"])
    assert cfg["in_channels"] == 2 * 5
    assert cfg["channel_indices"] == [1, 3]
