from sklearn import metrics as sk_metrics
from skimage import measure
import pandas as pd
import numpy as np
import cv2

def compute_best_pr_re(anomaly_ground_truth_labels, anomaly_prediction_weights):
    """
    Computes the best precision, recall and threshold for a given set of
    anomaly ground truth labels and anomaly prediction weights.
    """
    precision, recall, thresholds = sk_metrics.precision_recall_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    # 방어: division by zero 회피
    denom = (precision + recall)
    denom[denom == 0] = 1e-12
    f1_scores = 2 * (precision * recall) / denom

    idx = int(np.argmax(f1_scores))
    # sklearn은 thresholds 길이가 (len(precision)-1) 이라 주의
    best_threshold = thresholds[min(idx, len(thresholds) - 1)] if len(thresholds) > 0 else 0.5
    best_precision = float(precision[idx])
    best_recall = float(recall[idx])
    print(best_threshold, best_precision, best_recall)

    return best_threshold, best_precision, best_recall


def compute_imagewise_retrieval_metrics(anomaly_prediction_weights, anomaly_ground_truth_labels, path='training'):
    """
    Computes retrieval statistics (AUROC, AP).
    """
    auroc = sk_metrics.roc_auc_score(anomaly_ground_truth_labels, anomaly_prediction_weights)
    ap = 0.0 if path == 'training' else sk_metrics.average_precision_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    return {"auroc": auroc, "ap": ap}


def compute_pixelwise_retrieval_metrics(anomaly_segmentations, ground_truth_masks, path='train'):
    """
    Computes pixel-wise statistics (AUROC, AP) for anomaly segmentations
    and ground truth segmentation masks.
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    flat_anomaly_segmentations = anomaly_segmentations.ravel()
    flat_ground_truth_masks = ground_truth_masks.ravel().astype(int)

    auroc = sk_metrics.roc_auc_score(flat_ground_truth_masks, flat_anomaly_segmentations)
    ap = 0.0 if path == 'training' else sk_metrics.average_precision_score(
        flat_ground_truth_masks, flat_anomaly_segmentations
    )
    return {"auroc": auroc, "ap": ap}


def compute_pro(masks, amaps, num_th=200):
    """
    Per-Region Overlap AUC
    masks: (N,H,W) binary
    amaps: (N,H,W) continuous
    """
    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = float(amaps.min())
    max_th = float(amaps.max())
    if num_th <= 1 or max_th <= min_th + 1e-12:
        return 0.0
    delta = (max_th - min_th) / num_th

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    for th in np.arange(min_th, max_th, delta):
        binary_amaps[...] = amaps > th

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            binary_amap_d = cv2.dilate(binary_amap.astype(np.uint8), k)
            lbl = measure.label(mask.astype(np.uint8), connectivity=1)
            for region in measure.regionprops(lbl):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap_d[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / max(region.area, 1))

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks.astype(bool), binary_amaps).sum()
        fpr = fp_pixels / max(inverse_masks.sum(), 1)

        df = pd.concat([df, pd.DataFrame({"pro": np.mean(pros) if pros else 0.0,
                                          "fpr": fpr,
                                          "threshold": th}, index=[0])],
                       ignore_index=True)

    # 안전 필터링 & 정규화
    df = df[df["fpr"] < 0.3]
    if len(df) == 0:
        return 0.0
    denom = (df["fpr"].max() - df["fpr"].min())
    if denom <= 0:
        return 0.0
    df["fpr"] = (df["fpr"] - df["fpr"].min()) / denom

    pro_auc = sk_metrics.auc(df["fpr"], df["pro"])
    return float(pro_auc)