"""An independent instance model for the scorer: Mask R-CNN detects filaments on the flat field, and every candidate
of the component tree gets descriptors of the detections that overlap it.

Version 4 of the public notebook tushhaaarrrr/solar-filament-segmentation-2026-0-40 measured +0.0034 PQ, positive in
5 of 5 partitions, from an instance model's evidence in the same kind of decision layer. The model here is
torchvision's Mask R-CNN v2 (BSD-3 code, COCO weights) at 1024 px, with 56 x 56 masks (the default 28 x 28 cannot
hold a filament 5 px wide across a diagonal box). Each training sample shows the instances of one annotator record
drawn at random, so a detection's score estimates how often an annotator draws that filament, as the scorer's target
does.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torchvision.models.detection import MaskRCNN_ResNet50_FPN_V2_Weights, maskrcnn_resnet50_fpn_v2
from torchvision.models.detection.anchor_utils import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.ops import MultiScaleRoIAlign

from .tree import Tree

SCALE = 2  # the model sees the 2048 px flat field at 1024 px
SIZE = 2048 // SCALE
COLUMNS = ["i_iou", "i_score", "i_wscore", "i_cover", "i_n", "i_mask"]
FLIPS = ((), (-1,), (-2,), (-2, -1))  # the dense model's four views (infer.FLIPS): identity, horizontal, vertical, both


def build(coco: bool = True, anchor: int = 32) -> torch.nn.Module:
    """Mask R-CNN v2 for one class at 1024 px: COCO weights for the backbone, the region proposals and the box
    and mask heads (their last layers new), and 56 x 56 masks. `anchor`: the side of the smallest anchor box,
    doubling over the five pyramid levels; torchvision's 32 leaves 64% of the smallest third of filaments without
    an anchor at IoU 0.5 (their boxes' short side is 12-26 px at 1024 px), and 16 leaves 4.5%. Three aspect ratios at
    every size either way, so the COCO proposal head keeps its shape."""
    m = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if coco else None,
                                 weights_backbone=None, min_size=SIZE, max_size=SIZE, box_nms_thresh=0.7,
                                 box_score_thresh=0.05, box_detections_per_img=100)
    if anchor != 32:
        m.rpn.anchor_generator = AnchorGenerator(tuple((anchor * 2 ** i,) for i in range(5)), ((0.5, 1.0, 2.0),) * 5)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    m.roi_heads.mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=28,
                                                   sampling_ratio=2)
    m.roi_heads.mask_predictor = MaskRCNNPredictor(256, 256, 2)
    return m


def half(a: np.ndarray) -> np.ndarray:
    """A 2048 px image (uint8) or mask (bool) at 1024 px: the mean of each 2 x 2 block (a mask keeps blocks at least
    half covered)."""
    m = a.reshape(SIZE, SCALE, SIZE, SCALE).mean((1, 3), dtype=np.float32)
    return m >= 0.5 if a.dtype == bool else m.astype(np.float32) / 255.0


def to_input(flat_1024: np.ndarray) -> torch.Tensor:
    """(3, 1024, 1024) float input from the flat field in [0, 1] (the backbone's normalisation is the model's)."""
    return torch.from_numpy(np.ascontiguousarray(flat_1024))[None].expand(3, -1, -1).contiguous()


def targets(masks: list[np.ndarray]) -> dict:
    """Mask R-CNN targets of a record's filaments (bool masks at 1024 px); empty records are allowed."""
    keep = [m for m in masks if m.any()]
    if not keep:
        return dict(boxes=torch.zeros((0, 4)), labels=torch.zeros(0, dtype=torch.int64),
                    masks=torch.zeros((0, SIZE, SIZE), dtype=torch.uint8))
    boxes = []
    for m in keep:
        ys, xs = np.nonzero(m)
        boxes.append([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1])
    return dict(boxes=torch.tensor(boxes, dtype=torch.float32), labels=torch.ones(len(keep), dtype=torch.int64),
                masks=torch.from_numpy(np.stack(keep).astype(np.uint8)))


def pack(masks: np.ndarray) -> np.ndarray:
    """(N, 1024, 1024) bool masks as (N, 131072) bytes, for storage and between processes (N may be 0)."""
    if len(masks) == 0:
        return np.zeros((0, SIZE * SIZE // 8), np.uint8)
    return np.packbits(masks.reshape(len(masks), -1), axis=1)


def unpack(packed: np.ndarray) -> np.ndarray:
    """The inverse of `pack`: (N, 1024, 1024) bool masks."""
    return np.unpackbits(packed, axis=1, count=SIZE * SIZE).astype(bool).reshape(len(packed), SIZE, SIZE)


def flip(a: np.ndarray, axes: tuple) -> np.ndarray:
    """`a` flipped along `axes` (one of FLIPS); every flip is its own inverse."""
    return np.ascontiguousarray(np.flip(a, axes)) if axes else a


@torch.no_grad()
def detect(model: torch.nn.Module, flat_u8: np.ndarray, dev: torch.device,
           amp_dtype: torch.dtype | None = None, views: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores, bit-packed masks (probability > 0.5) and maskness of a frame's detections, from its 8-bit flat field at
    2048 px, as the out-of-fold detections were made. With `views` > 1 the model also sees the frame under the next
    flips of FLIPS (it trained under all eight flips and transposes), and the detections of every view, each flipped
    back onto the frame, are pooled: a filament one view misses can be found by another, and the count of
    detections over a candidate becomes a vote across views."""
    img = half(flat_u8)
    scores, packed, maskness = [], [], []
    for axes in FLIPS[:views]:
        x = to_input(flip(img, axes)).to(dev)
        with torch.autocast(dev.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            det = model([x])[0]
        prob = det["masks"][:, 0].float()
        on = prob > 0.5
        maskness.append(((prob * on).sum((1, 2)) / on.sum((1, 2)).clamp(min=1)).cpu().numpy())
        scores.append(det["scores"].float().cpu().numpy())
        packed.append(pack(flip(on.cpu().numpy(), axes)))
    return np.concatenate(scores), np.concatenate(packed), np.concatenate(maskness)


def save_instances(path: Path | str, scores: np.ndarray, masks: np.ndarray, maskness: np.ndarray | None = None) -> None:
    """Detections of one frame: scores (N,), bool masks (N, 1024, 1024) or the same bit-packed (`pack`), and each
    mask's maskness, the mean mask probability inside the binarised mask (Sartorius 1st, 298869: box score x maskness
    rescoring)."""
    maskness = np.ones(len(scores), np.float32) if maskness is None else maskness
    np.savez_compressed(path, scores=scores.astype(np.float32), maskness=maskness.astype(np.float32),
                        masks=masks if masks.dtype == np.uint8 else pack(masks))


def load_instances(path: Path | str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Scores, masks and maskness of a frame's detections (maskness 1 for files written without it)."""
    with np.load(path) as z:
        scores, packed = z["scores"], z["masks"]
        maskness = z["maskness"] if "maskness" in z.files else np.ones(len(scores), np.float32)
    return scores, unpack(packed), maskness


def describe(tree: Tree, scores: np.ndarray, masks: np.ndarray, maskness: np.ndarray | None = None) -> np.ndarray:
    """(n, 6) instance columns of every node of `tree` (2048 px) from detections at 1024 px: the best IoU of the node
    with a detection, that detection's score, the best score x IoU, the best score x share of the node a detection
    covers, the number of detections with IoU above 0.3, and the best score x maskness x IoU."""
    maskness = np.ones(len(scores), np.float32) if maskness is None else maskness
    n = tree.n
    X = np.zeros((n, len(COLUMNS)), np.float32)
    if n == 0 or len(scores) == 0:
        return X
    y0, _, x0, _ = tree.box
    ww = tree.crop_shape[1]
    sy, sx = np.divmod(tree.support, ww)
    hy, hx = (sy + y0) // SCALE, (sx + x0) // SCALE  # each support pixel's cell at 1024 px
    born = [(tree.node_of[k] >= 0) & (tree.level[np.maximum(tree.node_of[k], 0)] == k) for k in range(len(tree.levels))]
    a_node = tree.area.astype(np.float64)
    best_iou, best_score, wscore, cover, count, mscore = (np.zeros(n) for _ in range(6))
    for s, m, q in zip(scores, masks, maskness):
        idx = np.flatnonzero(m[hy, hx])  # support pixels inside the detection
        if len(idx) == 0:
            continue
        inter = np.zeros(n)
        for k, b in enumerate(born):
            bk = b[idx]
            if bk.any():
                inter += np.bincount(tree.node_of[k][idx[bk]], minlength=n)
        a_det = float(m.sum()) * SCALE * SCALE
        iou = inter / (a_node + a_det - inter)
        better = iou > best_iou
        best_iou[better], best_score[better] = iou[better], s
        wscore = np.maximum(wscore, s * iou)
        cover = np.maximum(cover, s * inter / a_node)
        count += iou > 0.3
        mscore = np.maximum(mscore, s * q * iou)
    X[:, 0], X[:, 1], X[:, 2], X[:, 3], X[:, 4], X[:, 5] = best_iou, best_score, wscore, cover, count, mscore
    return X
