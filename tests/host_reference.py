"""The host's scoring code, copied from the Kaggle notebook `azimahmadzadeh/self-evaluation-notebook`
(Kaggle notebooks are Apache-2.0). Used only as the reference in the parity tests."""
import numpy as np
import pandas as pd
import torch
from pycocotools import mask as mask_util


def get_overlap_matrices(gt_layers, pred_layers):
    n_gt, height, width = gt_layers.shape
    n_pred = pred_layers.shape[0]
    gt_flat = gt_layers.reshape(n_gt, height * width)
    pred_flat = pred_layers.reshape(n_pred, height * width)
    intersection = torch.matmul(gt_flat, pred_flat.t())
    gt_areas = gt_flat.sum(dim=1).view(-1, 1)
    pred_areas = pred_flat.sum(dim=1).view(1, -1)
    union = gt_areas + pred_areas - intersection
    iou_matrix = torch.where(union == 0, torch.tensor(0.0), intersection / union)
    dice_matrix = torch.where(union == 0, torch.tensor(0.0), 2 * intersection / (gt_areas + pred_areas))
    return iou_matrix, dice_matrix


def rles_to_layers(rles, height=2048, width=2048):
    if not rles:
        return np.zeros((0, height, width), dtype=np.float32)
    rle_dicts = [{"size": [height, width], "counts": rle} for rle in rles]
    masks = mask_util.decode(rle_dicts)
    return masks.transpose(2, 0, 1).astype(np.float32)


def process_entry(annotator_image, gt_annotator_image_ids, pred_image_ids, gt_df, pred_df):
    image_id = annotator_image.split("-", maxsplit=1)[1]
    gt_rles = gt_df.loc[gt_annotator_image_ids == annotator_image, "segmentation_rle"].tolist()
    gt_layers = torch.from_numpy(rles_to_layers(gt_rles))
    pred_rles = pred_df.loc[pred_image_ids == image_id, "segmentation_rle"].tolist()
    pred_layers = torch.from_numpy(rles_to_layers(pred_rles))
    iou_matrix, dice_matrix = get_overlap_matrices(gt_layers, pred_layers)
    return iou_matrix, dice_matrix, len(gt_rles), len(pred_rles)


def get_overlap_df(gt_df, pred_df):
    gt_annotator_image_ids = gt_df["filament_id"].str.split("_", n=1).str[0]
    pred_image_ids = pred_df["filament_id"].str.split("_", n=1).str[0]
    overlap_df = pd.DataFrame({"annotator_image": gt_annotator_image_ids.unique()})
    results = overlap_df["annotator_image"].apply(
        process_entry, gt_annotator_image_ids=gt_annotator_image_ids, pred_image_ids=pred_image_ids,
        gt_df=gt_df, pred_df=pred_df)
    overlap_df["iou_matrix"], overlap_df["dice_matrix"], overlap_df["n_gt"], overlap_df["n_pred"] = zip(*results)
    return overlap_df


def get_pq_score(overlap_df):
    tp_iou_scores, fp_count, fn_count = [], 0, 0
    for row in overlap_df.itertuples(index=False):
        iou_matrix, n_gt, n_pred = row.iou_matrix, row.n_gt, row.n_pred
        if n_gt == 0:
            fp_count += n_pred
            continue
        if n_pred == 0:
            fn_count += n_gt
            continue
        hit_matrix = iou_matrix > 0.5
        tp_iou_scores.extend(iou_matrix[hit_matrix].tolist())
        fp_count += (hit_matrix.sum(dim=0) == 0).sum().item()
        fn_count += (hit_matrix.sum(dim=1) == 0).sum().item()
    denominator = len(tp_iou_scores) + 0.5 * fp_count + 0.5 * fn_count
    return sum(tp_iou_scores) / denominator if denominator > 0 else 0.0
