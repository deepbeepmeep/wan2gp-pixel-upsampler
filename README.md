# Pixel Upsampler Template

This plugin is a reference implementation for WanGP spatial upsampler plugins. It
adds a simple post-processing upsampler named **Pixel Duplicate**. The upsampler
does nearest-neighbor integer scaling by duplicating pixels, so it is useful for
testing plugin registration and API behavior rather than for quality.

The plugin is metadata-only: there is no `plugin.py` and no UI tab. WanGP discovers
the upsampler from `plugin_info.json`:

```json
{
  "type": "spatial_upsampler",
  "spatial_upsampler_handlers": [".upsampler.PixelDuplicateUpsampler"]
}
```

Relative handler paths are resolved from the plugin package root. The entry above
loads `plugins/wan2gp-pixel-upsampler/upsampler.py` and instantiates
`PixelDuplicateUpsampler(server_config, files_locator)`.

## Files

- `plugin_info.json`: plugin metadata and the `spatial_upsampler_handlers` list.
- `upsampler.py`: the actual handler class and pixel duplication function.
- `README.md`: notes for plugin authors.

## Handler Checklist

`PixelDuplicateUpsampler` demonstrates the core post-processing upsampler surface:

- Inherit `SimpleScaleSuffixMixin` when your serialized values follow the common
  `<method><scale>` convention, for example `pixel2` or `pixel4`.
- Implement `query_upsampler_def()` to declare display labels, method ids,
  supported media, supported multipliers, dropdown ordering, and the config key.
- Implement `validate_upsampling(...)` to return an empty string when the selection
  is valid, or a user-facing error message when it is not.
- Implement `upscale(...)` and return `(output_tensor, continue_cache)`. Most
  upsamplers return `None` for `continue_cache`; FlashVSR-style streaming
  upsamplers can return state there.
- Implement `download(...)` if the plugin needs checkpoints or other assets.
  Return `False` when nothing was downloaded.
- Implement `release_vram()` if the plugin owns models, CUDA memory, or an MMGP
  offload object.

## Tensor Contract

Post-processing upsamplers receive decoded media tensors shaped:

```text
[channels, frames, height, width]
```

Image mode uses the same shape with one frame. This template repeats only the
height and width dimensions:

```python
sample.repeat_interleave(scale, dim=-2).repeat_interleave(scale, dim=-1)
```

That preserves channel count, frame count, dtype, and device. Real plugins should
avoid unnecessary CPU/GPU transfers and should keep the output layout identical to
the input layout except for the spatial size.

## Values and Multipliers

The method id is `pixel`, so the UI stores values such as:

- `pixel1`
- `pixel2`
- `pixel3`
- `pixel4`

The method id must be multiplier-free in `methods`, `method_pos`, and
`multipliers`. Only serialized selections contain the multiplier suffix.

## Config Guidance

This template intentionally has no configuration section. Only add config when
there is real shared extension behavior to store.

If your upsampler needs shared config, put it under:

```json
{
  "spatial_upsamplers": {
    "your_config_key": {
      "your_option": "value"
    }
  }
}
```

Do not add spatial upsampler options to `models/_settings.json`; settings are
per-model generation defaults, while spatial upsampler config is shared extension
configuration.

## Progress and Abort

`upscale(...)` receives optional callbacks:

- `abort_callback()`: return early when it becomes true.
- `progress_callback(status, current_step, total_steps)`: report integer progress.

This template reports a one-step operation. Longer upsamplers should report
meaningful phases such as tile encoding, model inference, or frame stitching.

## Downloads and Offload Objects

If a plugin needs files, declare them in `download(...)` and use the provided
`process_files(...)` function. If a plugin creates an MMGP offload object, register
it with `shared.utils.offload_registry` so WanGP's unload tools can release it.

This template has no downloads and no persistent VRAM state.

## Smoke Test

From the WanGP repo root:

```powershell
C:\Users\Marc\anaconda3\envs\py311\python.exe -m py_compile plugins\wan2gp-pixel-upsampler\upsampler.py
```

You can also instantiate the handler directly and verify a tiny tensor:

```python
import importlib
import sys
import torch

sys.path.insert(0, "plugins")
module = importlib.import_module("wan2gp-pixel-upsampler.upsampler")
handler = module.PixelDuplicateUpsampler({}, None)
sample = torch.arange(1 * 1 * 2 * 3).reshape(1, 1, 2, 3)
output, cache = handler.upscale(sample, "pixel2")
assert output.shape == (1, 1, 4, 6)
assert cache is None
```

After enabling the plugin in WanGP and restarting, **Pixel Duplicate** appears as a
post-processing spatial upsampler for both images and videos.
