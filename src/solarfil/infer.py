"""Dense inference: full-frame probability maps with flip test-time augmentation, and their PNG storage."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch

from .frames import Frame, input_planes

FLIPS = ((), (3,), (2,), (2, 3))  # identity, horizontal, vertical, both (dims of an NCHW tensor)


def _view(x: torch.Tensor, dims: tuple, transpose: bool) -> torch.Tensor:
    x = torch.flip(x, dims) if dims else x
    return x.transpose(2, 3) if transpose else x


def _unview(y: torch.Tensor, dims: tuple, transpose: bool) -> torch.Tensor:
    y = y.transpose(2, 3) if transpose else y
    return torch.flip(y, dims) if dims else y


def _rows_up(y: torch.Tensor, d: float) -> torch.Tensor:
    """Map content moved up by `d` pixels (0 <= d < 1): out(r) = (1 - d) y(r) + d y(r + 1), as `frames.shift_rows`."""
    assert 0 <= d < 1, d
    return (1 - d) * y + d * torch.cat([y[..., 1:, :], torch.zeros_like(y[..., :1, :])], dim=-2)


def _mps_pool_supported(source: tuple, target: tuple) -> bool:
    """MPS adaptive average pooling supports integer ratios in one direction."""
    return (all(a >= b and a % b == 0 for a, b in zip(source, target))
            or all(a <= b and b % a == 0 for a, b in zip(source, target)))


def _pool_area(y: torch.Tensor, size: tuple) -> torch.Tensor:
    """The same adaptive-average operator on CPU for unsupported MPS ratios.

    Only the completed probability map travels; the network still runs on
    the GPU. This preserves area averaging instead of substituting another
    interpolation rule. Native-scale inference never calls this function.
    """
    if y.device.type == "mps" and not _mps_pool_supported(y.shape[-2:], size):
        return torch.nn.functional.adaptive_avg_pool2d(y.cpu(), size).to(y.device)
    return torch.nn.functional.adaptive_avg_pool2d(y, size)


@torch.no_grad()
def tiled_sigmoid(m: torch.nn.Module, x: torch.Tensor, tile: int) -> torch.Tensor:
    """The network's sigmoid map of `x`, whole when `tile` is 0 or covers it, else from `tile` px windows at half a
    tile's stride, each weighted by a ramp that falls linearly to its border and the sum renormalised. A network
    whose attention uses a fixed number of grid cells (MaxViT) then sees the spacing it was trained at."""
    h, w = x.shape[-2:]
    if tile <= 0 or (h <= tile and w <= tile):
        return m(x).float().sigmoid()

    def origins(n: int) -> list[int]:
        """Window origins along a side of `n` px: half-tile steps, the last window flush with the end."""
        o = list(range(0, n - tile + 1, tile // 2))
        return o if o[-1] == n - tile else o + [n - tile]

    r = torch.minimum(torch.arange(1, tile + 1), torch.arange(tile, 0, -1)).float().to(x.device)
    win = (r[:, None] * r[None, :]) / float(r.max() ** 2)
    acc, wsum = None, torch.zeros((1, 1, h, w), device=x.device)
    for y in origins(h):
        for z in origins(w):
            out = m(x[..., y:y + tile, z:z + tile]).float().sigmoid()
            if acc is None:
                acc = torch.zeros((x.shape[0], out.shape[1], h, w), device=x.device)
            acc[..., y:y + tile, z:z + tile] += out * win
            wsum[..., y:y + tile, z:z + tile] += win
    return acc / wsum


def predict_frame(models: list[torch.nn.Module], fr: Frame, dev: torch.device, n_flips: int = 4,
                  amp_dtype: torch.dtype | None = None, label_dys: list[float] | None = None,
                  zoom: float = 1.0, tiles: list[int] | None = None,
                  weights: list[float] | None = None) -> np.ndarray:
    """(C, S, S) float32 sigmoid maps (filament, and spine if the models have it) at the model scale, averaged over
    models and views.

    `n_flips` = 1, 2 or 4 flips, or 8 for the full dihedral group (the four flips with and without a transpose).
    The coordinate planes are transformed with the pixels, as in training. `label_dys`: each network's recipe
    `label_dy` at the model scale; a network trained on targets moved down onto the image has its map moved back up
    by that much, into the annotators' frame, before the average. `zoom` > 1 enlarges the input planes
    (bilinear) before the networks see them and brings each network's map back to the model scale by area
    averaging, so a filament covers more of the networks' pixels (an inference-time test of resolution). `tiles`:
    each network's recipe `infer_tile` (`tiled_sigmoid`; 0 for the whole frame). `weights`: each network's share of
    the average (`blend_maps.py --weights`; equal shares without it)."""
    x = torch.from_numpy(input_planes(fr))[None].to(dev)
    size = x.shape[-2:]
    if zoom != 1.0:
        x = torch.nn.functional.interpolate(x, scale_factor=zoom, mode="bilinear", align_corners=False)
    acc = 0.0
    views = [(d, False) for d in FLIPS[:min(n_flips, 4)]] + ([(d, True) for d in FLIPS] if n_flips == 8 else [])
    for i, m in enumerate(models):
        acc_m = 0.0
        for dims, tr in views:
            with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                out = tiled_sigmoid(m, _view(x, dims, tr), tiles[i] if tiles else 0)
            acc_m += _unview(out, dims, tr)
        if zoom != 1.0:
            acc_m = _pool_area(acc_m, size)
        d = label_dys[i] if label_dys else 0.0
        w = weights[i] if weights else 1.0
        acc += w * (_rows_up(acc_m, d) if d else acc_m)
    return (acc / ((sum(weights) if weights else len(models)) * len(views)))[0].cpu().numpy()


def to_native(p: np.ndarray, scale: int) -> np.ndarray:
    """Bilinear upsampling of a model-scale map to 2048 px (pixel-centre aligned, as INTER_AREA downsampling)."""
    if scale == 1:
        return p
    s = p.shape[-1] * scale
    return cv2.resize(p, (s, s), interpolation=cv2.INTER_LINEAR)


def to_u8(p: np.ndarray) -> np.ndarray:
    """A [0, 1] map as uint8 (x 255, rounded and clipped)."""
    return np.clip(np.rint(p * 255), 0, 255).astype(np.uint8)


def write_map_png(path: Path | str, pm: np.ndarray) -> None:
    """Store (C, S, S) maps as an RGB PNG: R = filament, G = spine (x 255; 0 without a spine map), B = 0."""
    zero = np.zeros(pm.shape[1:], np.uint8)
    rgb = np.stack([to_u8(pm[0]), to_u8(pm[1]) if len(pm) > 1 else zero, zero], axis=-1)
    cv2.imwrite(str(path), rgb[:, :, ::-1])  # cv2 writes BGR


def read_map_png(path: Path | str, native: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Filament and spine probabilities from a map PNG, upsampled to 2048 px unless `native` is False."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(path)
    p = bgr[:, :, 2].astype(np.float32) / 255.0
    s = bgr[:, :, 1].astype(np.float32) / 255.0
    if native:
        scale = 2048 // bgr.shape[0]
        p, s = to_native(p, scale), to_native(s, scale)
    return p, s
