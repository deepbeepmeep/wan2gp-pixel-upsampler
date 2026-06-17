from __future__ import annotations

from typing import Any

import torch

from postprocessing.spatial_upsamplers import SimpleScaleSuffixMixin, UPSAMPLER_PROFILE_VIDEO, UPSAMPLER_TYPE_POSTPROCESSING


def duplicate_pixels(sample: torch.Tensor, scale: int) -> torch.Tensor:
    """Nearest-neighbor integer upsampling for WanGP media tensors.

    WanGP passes decoded post-processing media as a tensor shaped [C, T, H, W]:
    channels, frames, height, width. Image post-processing uses the same shape
    with a single frame. Repeating only the two spatial dimensions preserves
    channels, frame count, dtype, and device.
    """
    if scale == 1:
        return sample
    return sample.repeat_interleave(scale, dim=-2).repeat_interleave(scale, dim=-1).contiguous()


class PixelDuplicateUpsampler(SimpleScaleSuffixMixin):
    """Minimal spatial upsampler plugin handler.

    This class is intentionally small so plugin authors can copy it as a
    starting point. The API is the same one used by built-in upsamplers:
    declare capabilities with query_upsampler_def(), validate a selected value,
    and return (output_tensor, continue_cache) from upscale().
    """

    METHOD = "pixel"
    MULTIPLIERS = (1.0, 2.0, 3.0, 4.0)
    batch_image_inputs = True

    def __init__(self, server_config: dict[str, Any] | None = None, files_locator=None):
        self.server_config = server_config
        self.files_locator = files_locator

    @classmethod
    def query_upsampler_def(cls) -> dict[str, Any]:
        return {
            "name": "Pixel Upsampler Template",
            "upsampler_types": (UPSAMPLER_TYPE_POSTPROCESSING,),
            "media": ("video", "image"),
            "profile": UPSAMPLER_PROFILE_VIDEO,
            "config_key": "pixel",
            "pos": 900,
            "method_pos": {cls.METHOD: 900},
            "methods": [("Pixel Duplicate", cls.METHOD)],
            "vae_methods": [],
            "multipliers": {cls.METHOD: cls.MULTIPLIERS},
            "default_spatial_upsampling": "pixel2",
        }

    def validate_upsampling(self, spatial_upsampling, image_mode: int) -> str:
        split = self.split_value(spatial_upsampling)
        if split is None:
            return ""
        _, scale = split
        if scale not in self.MULTIPLIERS:
            supported = ", ".join(f"x{int(value)}" for value in self.MULTIPLIERS)
            return f"Pixel Duplicate only supports {supported}"
        return ""

    def download(self, process_files, send_cmd=None, status_text: str | None = None, spatial_upsampling=None) -> bool:
        """No-op download hook.

        Keep this method when your plugin has optional assets: return True only
        when you actually invoked process_files(...). This template has no model
        files, so returning False tells WanGP that nothing was downloaded.
        """
        return False

    def release_vram(self) -> None:
        """No-op release hook.

        If an upsampler creates an MMGP offload object, register it with
        shared.utils.offload_registry and release it here. This template only
        uses tensor operations, so there is nothing persistent to release.
        """
        return None

    def upscale(self, sample: torch.Tensor, spatial_upsampling, *, abort_callback=None, progress_callback=None, **kwargs):
        split = self.split_value(spatial_upsampling)
        if split is None:
            raise ValueError(f"Unknown Pixel Duplicate upsampling value: {spatial_upsampling}")
        _, scale = split
        if scale not in self.MULTIPLIERS:
            raise ValueError(self.validate_upsampling(spatial_upsampling, image_mode=0))
        if callable(abort_callback) and abort_callback():
            return None, None
        if callable(progress_callback):
            progress_callback("Pixel Duplicate Upsampling", 0, 1)
        output = duplicate_pixels(sample, int(scale))
        if callable(progress_callback):
            progress_callback("Pixel Duplicate Upsampling", 1, 1)
        return output, None

