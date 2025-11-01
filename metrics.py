# metrics_safe.py
# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Iterable, List, Optional, Tuple

import numpy as np
import cv2
import pandas as pd
from sklearn import metrics as sk_metrics
from skimage import measure


__all__ = [
    "compute_best_pr_re",
    "compute_imagewise_retrieval_metrics",
    "compute_pixelwise_retrieval_metrics",
    "compute_pro",
]


# ---------------------------
# Helpers
# ---------------------------

def _binarize_mask(m: np.ndarray) -> np.ndarray:
    """
    Any positive -> 1.0, else 0.0 (float32)
    Accepts {0,255} or {0,1} or grayscale.
    """
    m = np.asarray(m)
    if m.ndim == 3:
        # if BGR/GRY with 3 channels, pick first channel
        m = m[..., 0]
    if m.dtype != np.float32:
        m = m.astype(np.float32)
    # treat any positive value as 1.0
    if m.max() > 1.0:
        m = (m > 0).astype(np.float32)
    else:
        m = (m >= 0.5).astype(np.float32)
    return m


def _resize_to(pred_hw: Tuple[int, int], arr: np.ndarray, interp: int) -> np.ndarray:
    """Resize arr to (H, W) if needed."""
    H, W = pred_hw
    if arr.shape[:2] == (H, W):
        return arr
    return cv2.resize(arr, (W, H), interpolation=interp)


def _safe_auc_ap(y_true: np.ndarray, y_score: np.ndarray) -> Tuple[float, float]:
    """
    AUROC/AP 안전 계산:
      - 클래스가 한쪽뿐이면 (양/음성 모두 없어지면) AUROC=nan, AP=0.0 반환
      - y_true in {0,1}, y_score in [any real], 자동 float32/ints로 캐스팅
    """
    y_true = np.asarray(y_true).astype(np.int32).ravel()
    y_score = np.asarray(y_score).astype(np.float32).ravel()

    n_pos = int((y_true == 1).sum())
    n_neg = int((y_true == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan"), 0.0

    # sklearn이 내부적으로 안전 처리
    try:
        auroc = float(sk_metrics.roc_auc_score(y_true, y_score))
    except Exception:
        auroc = float("nan")
    try:
        ap = float(sk_metrics.average_precision_score(y_true, y_score))
    except Exception:
        ap = 0.0
    return auroc, ap


def _normalize_per_image(a: np.ndarray) -> np.ndarray:
    """
    Per-image min-max normalization for anomaly maps/scores.
    a: (H, W) or (N, H, W)
    """
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if a.ndim == 2:
        amin, amax = float(a.min()), float(a.max())
        if amax - amin < 1e-12:
            return np.zeros_like(a, dtype=np.float32)
        return (a - amin) / (amax - amin + 1e-12)
    elif a.ndim == 3:
        N = a.shape[0]
        a_ = a.reshape(N, -1)
        mins = a_.min(axis=1, keepdims=True)
        maxs = a_.max(axis=1, keepdims=True)
        denom = np.maximum(maxs - mins, 1e-12)
        return ((a_ - mins) / denom).reshape(a.shape).astype(np.float32)
    else:
        raise AssertionError("normalize expects (H,W) or (N,H,W)")


# ---------------------------
# Public metrics
# ---------------------------

def compute_best_pr_re(
    anomaly_ground_truth_labels: Iterable[int],
    anomaly_prediction_weights: Iterable[float],
) -> Tuple[float, float, float]:
    """
    Best F1 operating point from PR curve.
    Returns: (best_threshold, precision_at_best, recall_at_best)

    NOTE:
      - If only one class present, returns (0.5, 0.0, 0.0).
      - Threshold is approximated from sklearn's precision_recall_curve thresholds.
    """
    y = np.asarray(anomaly_ground_truth_labels).astype(int).ravel()
    s = np.asarray(anomaly_prediction_weights).astype(np.float32).ravel()

    if len(np.unique(y)) < 2:
        return 0.5, 0.0, 0.0

    precision, recall, thresholds = sk_metrics.precision_recall_curve(y, s)
    # precision/recall length = thresholds length + 1
    # compute F1 on valid pairs
    denom = precision + recall
    denom[denom == 0] = 1e-12
    f1 = 2.0 * (precision * recall) / denom

    # For threshold indexing, align to thresholds length
    # We'll use f1[:-1] to match thresholds
    if len(thresholds) > 0:
        idx = int(np.argmax(f1[:-1]))
        best_th = float(thresholds[idx])
        return best_th, float(precision[idx]), float(recall[idx])
    else:
        # Degenerate case (constant scores)
        return 0.5, float(precision[np.argmax(f1)]), float(recall[np.argmax(f1)])


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights: Iterable[float],
    anomaly_ground_truth_labels: Iterable[int],
    path: str = "training",
) -> dict:
    """
    AUROC/AP for image-level scores.

    If `path == "training"`, AP는 0으로 간주(일부 벤치마크 관례).
    """
    y = np.asarray(anomaly_ground_truth_labels).astype(int).ravel()
    s = np.asarray(anomaly_prediction_weights).astype(np.float32).ravel()

    auroc, ap = _safe_auc_ap(y, s)
    if path.lower().startswith("train"):
        ap = 0.0
    return {"auroc": auroc, "ap": ap}


def compute_pixelwise_retrieval_metrics(
    anomaly_segmentations: np.ndarray | List[np.ndarray],
    ground_truth_masks: np.ndarray | List[np.ndarray],
    path: str = "train",
) -> dict:
    """
    Pixel-wise AUROC/AP.

    - anomaly_segmentations: (N,H,W) float or list of (H,W)
    - ground_truth_masks    : (N,H,W) or list of (H,W) binary-like; 크기 달라도 자동 리사이즈
    - 내부적으로 (N,H,W)로 통일 후 평탄화하여 계산
    """
    # preds: unify to (N,H,W)
    if isinstance(anomaly_segmentations, list):
        preds = np.stack(anomaly_segmentations, axis=0)
    else:
        preds = np.asarray(anomaly_segmentations)
    assert preds.ndim == 3, "anomaly_segmentations must be (N,H,W) or list of (H,W)"
    N, H, W = preds.shape
    preds = preds.astype(np.float32)

    # gts: unify to (N,H,W) and binarize
    if isinstance(ground_truth_masks, list):
        gt_stack = []
        for g in ground_truth_masks:
            g = np.asarray(g)
            if g.ndim == 3:
                g = g[..., 0]
            g = _resize_to((H, W), g, cv2.INTER_NEAREST)
            gt_stack.append(_binarize_mask(g))
        gts = np.stack(gt_stack, axis=0)
    else:
        gtmp = np.asarray(ground_truth_masks)
        if gtmp.ndim == 3:
            gts_list = []
            for i in range(gtmp.shape[0]):
                g = gtmp[i]
                if g.ndim == 3:
                    g = g[..., 0]
                g = _resize_to((H, W), g, cv2.INTER_NEAREST)
                gts_list.append(_binarize_mask(g))
            gts = np.stack(gts_list, axis=0)
        elif gtmp.ndim == 2:
            g = _resize_to((H, W), gtmp, cv2.INTER_NEAREST)
            gts = _binarize_mask(g)[None, ...]
        else:
            raise AssertionError("ground_truth_masks must be (N,H,W) or list of (H,W)")

    # flatten
    flat_pred = preds.reshape(-1).astype(np.float32)
    flat_gt = gts.reshape(-1).astype(np.int32)

    auroc, ap = _safe_auc_ap(flat_gt, flat_pred)
    if path.lower().startswith("train"):
        ap = 0.0
    return {"auroc": auroc, "ap": ap}


def compute_pro(
    masks: np.ndarray,
    amaps: np.ndarray,
    num_th: int = 200,
    dilate_ksize: int = 5,
    fpr_max: float = 0.3,
    per_image_minmax: bool = True,
) -> float:
    """
    Per-Region Overlap (PRO) AUC 근사 (PaDiM 등 정의와 호환되도록 설계).

    Args
    ----
    masks: (N,H,W) binary-like (0/1 or 0/255). 자동 이진화.
    amaps: (N,H,W) float anomaly map (logit/score 무관).
    num_th: 임계값 샘플 개수
    dilate_ksize: dilation 커널 크기 (검출 경계 완화)
    fpr_max: FPR < fpr_max 영역에서만 AUC 계산
    per_image_minmax: 각 이미지별 [min,max] 정규화 후 임계 스윕

    Returns
    -------
    pro_auc: float
    """
    masks = np.asarray(masks)
    amaps = np.asarray(amaps)
    assert masks.shape == amaps.shape, "masks and amaps must have same shape (N,H,W)"
    assert masks.ndim == 3, "expected (N,H,W)"

    N, H, W = masks.shape
    masks = _binarize_mask(masks).astype(np.uint8)

    if per_image_minmax:
        amaps = _normalize_per_image(amaps)
    else:
        amaps = np.nan_to_num(amaps, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    min_th = float(amaps.min())
    max_th = float(amaps.max())
    if num_th <= 1 or not np.isfinite(min_th) or not np.isfinite(max_th) or max_th <= min_th + 1e-12:
        return 0.0
    delta = (max_th - min_th) / num_th

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_ksize, dilate_ksize))

    rows: List[Tuple[float, float, float]] = []
    binary = np.zeros_like(amaps, dtype=np.uint8)

    # 임계값 sweep
    th = min_th
    while th < max_th:
        # threshold: amaps > th
        np.greater(amaps, th, out=binary)

        pros: List[float] = []
        fp = 0
        tn_plus_fp = 0

        for i in range(N):
            bd = cv2.dilate(binary[i], k)  # predicted positive map (dilated)
            m = masks[i]

            # PRO: 각 GT region 별 overlap 비율 평균
            lbl = measure.label(m, connectivity=1)
            if lbl.max() > 0:
                for region in measure.regionprops(lbl):
                    coords = region.coords  # (K, 2) -> rows, cols
                    rr, cc = coords[:, 0], coords[:, 1]
                    area = int(region.area)
                    if area <= 0:
                        continue
                    tp_pixels = int(bd[rr, cc].sum())
                    pros.append(tp_pixels / max(area, 1))

            # FPR 계산
            inv = (1 - m).astype(bool)
            fp += int(np.logical_and(inv, bd.astype(bool)).sum())
            tn_plus_fp += int(inv.sum())

        pro = float(np.mean(pros)) if len(pros) > 0 else 0.0
        fpr = (fp / max(tn_plus_fp, 1)) if tn_plus_fp > 0 else 0.0
        rows.append((pro, fpr, float(th)))

        th += delta

    if len(rows) == 0:
        return 0.0

    df = pd.DataFrame(rows, columns=["pro", "fpr", "threshold"])
    df = df[df["fpr"] < fpr_max]
    if len(df) == 0:
        return 0.0

    # fpr 정규화 후 AUC
    fmin, fmax = float(df["fpr"].min()), float(df["fpr"].max())
    if not np.isfinite(fmin) or not np.isfinite(fmax) or fmax <= fmin + 1e-12:
        return 0.0
    fpr_norm = (df["fpr"] - fmin) / (fmax - fmin + 1e-12)
    pro_auc = sk_metrics.auc(fpr_norm, df["pro"].to_numpy(np.float32))
    return float(pro_auc)