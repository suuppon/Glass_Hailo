#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eval MVTec metal_nut (TEST) with Hailo backbone (HEF) + ONNX head.
- No PyTorch dependency in data/ops path (uses NumPy, PIL/OpenCV, SciPy, onnxruntime, HailoRT)
- Metrics/CSV/vis compatible with your utils.py / metrics.py style.

CLI example:
python eval_hailo_onnx.py \
  --data_path /path/to/MVTec_AD \
  --hef ckpt/glass_backbone.hef \
  --head ckpt/glass_head.onnx \
  --results_path results_eval_hailo \
  --save_vis

Requirements:
  pip install onnxruntime scipy opencv-python pillow scikit-image scikit-learn hailo_platform
"""

import os
import sys
import time
import glob
import csv
import math
import json
import argparse
import logging
from typing import List, Tuple, Optional

import numpy as np
import onnxruntime as ort
import cv2
from PIL import Image
from scipy import ndimage

# local modules you already have
import metrics  # from your snippet (NumPy/Sklearn-based)
import utils    # from your snippet (CSV, image utils, etc.)

# HailoRT
from hailo_platform import VDevice, HailoSchedulingAlgorithm

LOGGER = logging.getLogger("eval_hailo_onnx")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ----------- Preprocess constants (ImageNet) -----------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


# =========================
# Data utilities (MVTec)
# =========================
def list_metal_nut_test(data_root: str) -> List[Tuple[str, int, Optional[str]]]:
    """
    Return list of (image_path, is_anomaly(0/1), mask_path or None) for metal_nut TEST.
    MVTec structure:
      data_root/metal_nut/test/<type>/*.png
      data_root/metal_nut/ground_truth/<type>/*_mask.png  (only for anomaly types)
    """
    cls = "metal_nut"
    test_dir = os.path.join(data_root, cls, "test")
    gt_dir   = os.path.join(data_root, cls, "ground_truth")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Not found: {test_dir}")

    results = []
    for anomaly_type in sorted(os.listdir(test_dir)):
        img_dir = os.path.join(test_dir, anomaly_type)
        if not os.path.isdir(img_dir):
            continue
        img_files = sorted([
            os.path.join(img_dir, f) for f in os.listdir(img_dir)
            if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))
        ])
        if anomaly_type == "good":
            for p in img_files:
                results.append((p, 0, None))
        else:
            m_dir = os.path.join(gt_dir, anomaly_type)
            m_files = sorted([
                os.path.join(m_dir, f) for f in os.listdir(m_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))
            ])
            if len(m_files) != len(img_files):
                LOGGER.warning(f"[WARN] #masks != #images for type={anomaly_type}: {len(m_files)} vs {len(img_files)}")
            # pair by index ordering
            for i, p in enumerate(img_files):
                mask_p = m_files[i] if i < len(m_files) else None
                results.append((p, 1, mask_p))
    return results


def resize_then_center_crop(img: np.ndarray, target: int = 288) -> np.ndarray:
    """
    img: HxWxC (RGB float32 [0..1] or uint8 [0..255])
    Mimic torchvision: Resize(resize) then CenterCrop(imagesize).
    Here resize == imagesize == target (default 288). If you later want different values,
    change 'resize' first, then center-crop to 'target'.
    """
    # 1) Resize shortest side to 'target', keeping aspect
    h, w, _ = img.shape
    if h == target and w == target:
        resized = img
    else:
        # torchvision.transforms.Resize(int) -> shorter side == target
        if h < w:
            new_h = target
            new_w = int(round(w * (target / h)))
        else:
            new_w = target
            new_h = int(round(h * (target / w)))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 2) CenterCrop to (target, target)
    h2, w2, _ = resized.shape
    top = max((h2 - target) // 2, 0)
    left = max((w2 - target) // 2, 0)
    crop = resized[top: top + target, left: left + target]
    if crop.shape[0] != target or crop.shape[1] != target:
        # pad if edge case
        pad_img = np.zeros((target, target, 3), dtype=resized.dtype)
        pad_img[:crop.shape[0], :crop.shape[1]] = crop
        crop = pad_img
    return crop


def preprocess_image_to_bchw(path: str, size: int = 288) -> np.ndarray:
    """
    Load RGB, Resize→CenterCrop(size), to [0..1] float -> normalize -> NCHW float32 (1,3,H,W)
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        # PIL fallback
        pil = Image.open(path).convert("RGB")
        img = np.array(pil)[:, :, ::-1]  # RGB->BGR then back to align below? We'll use cv2 pipeline, so convert to RGB now:
    # BGR -> RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    img = resize_then_center_crop(img, target=size)
    x = np.transpose(img, (2, 0, 1))[None, ...]  # (1,3,H,W)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.astype(np.float32)


def load_mask_to_hw(path: Optional[str], size: int = 288) -> np.ndarray:
    """
    Load GT mask as float in [0..1], after Resize→CenterCrop(size). If path is None -> zeros.
    """
    if path is None or (not os.path.isfile(path)):
        return np.zeros((size, size), dtype=np.float32)

    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        m = np.array(Image.open(path).convert("L"))
    # resize shortest side then center crop (like images)
    h, w = m.shape
    if h == size and w == size:
        resized = m
    else:
        if h < w:
            new_h = size
            new_w = int(round(w * (size / h)))
        else:
            new_w = size
            new_h = int(round(h * (size / w)))
        resized = cv2.resize(m, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    h2, w2 = resized.shape
    top = max((h2 - size) // 2, 0)
    left = max((w2 - size) // 2, 0)
    crop = resized[top: top + size, left: left + size]
    if crop.shape[0] != size or crop.shape[1] != size:
        pad = np.zeros((size, size), dtype=resized.dtype)
        pad[:crop.shape[0], :crop.shape[1]] = crop
        crop = pad

    # To float 0..1
    if crop.max() > 1:
        crop = (crop > 0).astype(np.float32)  # MVTec masks are 0/255 → binarize
    else:
        crop = crop.astype(np.float32)
    return crop


# =========================
# Hailo backbone runner
# =========================
def get_shape_from_meta(meta):
    for attr in ("shape", "dims", "dimensions"):
        v = getattr(meta, attr, None)
        if v is not None:
            v = v() if callable(v) else v
            try:
                return tuple(v)
            except Exception:
                pass
    b = getattr(meta, "batch", 1)
    h = getattr(meta, "height", None)
    w = getattr(meta, "width", None)
    c = getattr(meta, "channels", None)
    if all(v is not None for v in (b, h, w, c)):
        return (int(b), int(h), int(w), int(c))
    raise RuntimeError(f"Cannot infer output shape from meta: {meta}")


def get_scale_zp(qinfo):
    if qinfo is None:
        return 1.0, 0.0
    for s_name, z_name in [
        ("scale", "zero_point"),
        ("scale", "zp"),
        ("qp_scale", "qp_zp"),
        ("quant_scale", "quant_zp"),
    ]:
        s = getattr(qinfo, s_name, None)
        z = getattr(qinfo, z_name, None)
        if s is not None and z is not None:
            try:
                return float(s), float(z)
            except Exception:
                pass
    try:
        if isinstance(qinfo, dict):
            s = qinfo.get("scale", qinfo.get("qp_scale", 1.0))
            z = qinfo.get("zero_point", qinfo.get("zp", qinfo.get("qp_zp", 0.0)))
            return float(s), float(z)
    except Exception:
        pass
    return 1.0, 0.0


def dequantize(arr_uint8, scale, zp):
    return scale * (arr_uint8.astype(np.float32) - zp)


def run_backbone_hailo_async(hef_path: str, x_bchw_f32: np.ndarray):
    """
    Return layer2, layer3 as float32 NCHW for a single image (batch=1).
    """
    if not os.path.isfile(hef_path):
        raise FileNotFoundError(hef_path)

    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    with VDevice(params) as vdev:
        infer_model = vdev.create_infer_model(hef_path)
        with infer_model.configure() as cmodel:
            bindings = cmodel.create_bindings()

            # Input quant
            in_meta = infer_model.input()
            in_q = getattr(in_meta, "quant_info", None)
            in_scale, in_zp = get_scale_zp(in_q)

            # quantize to feed
            x_q = np.clip(np.round(x_bchw_f32 / max(in_scale, 1e-12) + in_zp), 0, 255).astype(np.uint8)
            in_shape = tuple(in_meta.shape)
            if len(in_shape) == 4 and in_shape[1] in (288, 224) and in_shape[-1] == 3:
                # NHWC stream → transpose
                x_q_feed = np.transpose(x_q, (0, 2, 3, 1)).copy()
            else:
                x_q_feed = x_q
            bindings.input().set_buffer(np.ascontiguousarray(x_q_feed, dtype=np.uint8))

            # outputs meta & buffers
            outputs_attr = getattr(infer_model, 'outputs', None)
            if callable(outputs_attr):
                outs_meta = list(outputs_attr())
            elif isinstance(outputs_attr, (list, tuple)):
                outs_meta = list(outputs_attr)
            else:
                get_single = getattr(infer_model, 'output', None)
                if callable(get_single):
                    outs_meta = [get_single()]
                else:
                    raise RuntimeError("Cannot discover model outputs from infer_model.")

            get_outputs = getattr(bindings, 'outputs', None)
            if callable(get_outputs):
                out_bindings = list(get_outputs())
            else:
                out_bindings = []
                get_output = getattr(bindings, 'output', None)
                if callable(get_output) and len(outs_meta) > 0:
                    for meta in outs_meta:
                        name = getattr(meta, "name", None)
                        if name is None:
                            name = str(meta)
                        out_bindings.append(get_output(name))
                else:
                    raise RuntimeError("Bindings exposes neither outputs() nor output(name).")

            for i, ob in enumerate(out_bindings):
                shp = get_shape_from_meta(outs_meta[i])
                ob.set_buffer(np.empty(shp, dtype=np.uint8, order='C'))

            job = cmodel.run_async([bindings])
            job.wait(10_000)

            outputs_np = []
            for i, ob in enumerate(out_bindings):
                raw = ob.get_buffer()  # uint8
                ometa = outs_meta[i]
                s, zp = get_scale_zp(getattr(ometa, "quant_info", None))
                fp = dequantize(raw, s, zp)

                shp = tuple(fp.shape)
                if len(shp) == 4 and shp[-1] in (512, 1024):
                    fp = np.transpose(fp, (0, 3, 1, 2)).copy()  # NHWC → NCHW
                elif len(shp) == 3 and shp[-1] in (512, 1024):
                    fp = np.transpose(fp, (2, 0, 1))[None, ...].copy()
                elif len(shp) == 4 and shp[1] in (512, 1024):
                    fp = fp.copy()  # already NCHW
                else:
                    raise RuntimeError(f"Unexpected output shape {shp}; cannot normalize to NCHW.")
                outputs_np.append(fp.astype(np.float32))

            l2, l3 = None, None
            for arr in outputs_np:
                _, C, H, W = arr.shape
                if (C, H, W) == (512, 36, 36):
                    l2 = arr
                elif (C, H, W) == (1024, 18, 18):
                    l3 = arr
            if l2 is None or l3 is None:
                raise RuntimeError(f"Failed to identify L2/L3 from shapes {[a.shape for a in outputs_np]}")
            return l2, l3


# =========================
# ONNX Head
# =========================
def run_head_onnx(head_onnx_path: str, layer2: np.ndarray, layer3: np.ndarray) -> np.ndarray:
    """
    Returns patch_scores (B,H',W'), float32
    """
    sess = ort.InferenceSession(head_onnx_path, providers=["CPUExecutionProvider"])
    inputs = {sess.get_inputs()[0].name: layer2.astype(np.float32),
              sess.get_inputs()[1].name: layer3.astype(np.float32)}
    (out,) = sess.run([sess.get_outputs()[0].name], inputs)
    out = np.asarray(out)
    if out.ndim == 2:
        out = out[None, ...]
    return out.astype(np.float32, copy=False)


# =========================
# Postprocess
# =========================
def upsample_and_smooth(patch_scores_bhw: np.ndarray, size: int = 288, sigma: float = 4.0) -> List[np.ndarray]:
    """
    patch_scores_bhw: (B,H',W') float32 in [0..1] (assumed)
    Returns list of (H,W) float32 (smoothed)
    """
    B, h, w = patch_scores_bhw.shape
    resized = [cv2.resize(patch_scores_bhw[i], (size, size), interpolation=cv2.INTER_LINEAR) for i in range(B)]
    smoothed = [ndimage.gaussian_filter(m.astype(np.float32, copy=False), sigma=sigma) for m in resized]
    return smoothed


# =========================
# Eval loop
# =========================
def evaluate_dataset(
    data_root: str,
    hef_path: str,
    head_onnx: str,
    results_path: str,
    save_vis: bool = False,
    run_name: str = "eval_hailo_onnx",
    imagesize: int = 288
):
    os.makedirs(results_path, exist_ok=True)
    vis_dir = os.path.join(results_path, "vis", "mvtec_metal_nut")
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    # build test list
    items = list_metal_nut_test(data_root)  # (img_path, is_anomaly, mask_path)
    LOGGER.info(f"[DATA] metal_nut TEST: {len(items)} images")

    labels_gt: List[int] = []
    masks_gt: List[np.ndarray] = []
    all_image_scores: List[float] = []
    all_masks: List[np.ndarray] = []

    # measure time (end-to-end per image, incl. Hailo + head + postproc)
    t0 = time.perf_counter()

    idx_vis = 0
    for idx, (img_p, y, m_p) in enumerate(items):
        x = preprocess_image_to_bchw(img_p, size=imagesize)           # (1,3,288,288)
        l2, l3 = run_backbone_hailo_async(hef_path, x)                # (1,512,36,36), (1,1024,18,18)
        patch_scores = run_head_onnx(head_onnx, l2, l3)               # (1,36,36)
        masks = upsample_and_smooth(patch_scores, size=imagesize, sigma=4.0)  # list of (288,288)
        m = masks[0]

        img_score = float(patch_scores.max())  # image-level score = max over patch scores

        labels_gt.append(int(y))
        if m_p is not None:
            masks_gt.append(load_mask_to_hw(m_p, size=imagesize))
        else:
            masks_gt.append(np.zeros((imagesize, imagesize), dtype=np.float32))
        all_image_scores.append(img_score)
        all_masks.append(m)

        if save_vis:
            heat = (np.clip(m, 0, 1) * 255.0).astype(np.uint8)
            heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(vis_dir, f"{idx_vis:04d}.png"), heat)
            idx_vis += 1

    t1 = time.perf_counter()
    elapsed = t1 - t0
    num_images = len(items)
    latency = elapsed / max(num_images, 1)
    fps = num_images / max(elapsed, 1e-9)

    # metrics
    scores_arr = np.asarray(all_image_scores, dtype=np.float32)
    img_metrics = metrics.compute_imagewise_retrieval_metrics(scores_arr, labels_gt, path='eval')
    i_auroc = float(img_metrics["auroc"])
    i_ap    = float(img_metrics["ap"])

    if len(masks_gt) > 0:
        seg_arr = np.stack(all_masks, axis=0).astype(np.float32)
        pix = metrics.compute_pixelwise_retrieval_metrics(seg_arr, masks_gt, path='eval')
        p_auroc = float(pix["auroc"])
        p_ap    = float(pix["ap"])
        try:
            p_pro = float(metrics.compute_pro(np.squeeze(np.array(masks_gt)), seg_arr))
        except Exception:
            p_pro = 0.0
    else:
        p_auroc = p_ap = p_pro = -1.0

    LOGGER.info(f"[METRICS] image_auroc={i_auroc:.4f} image_ap={i_ap:.4f}  "
                f"pixel_auroc={p_auroc:.4f} pixel_ap={p_ap:.4f} pixel_pro={p_pro:.4f}")
    LOGGER.info(f"[SPEED] Elapsed={elapsed:.3f}s  Images={num_images}  "
                f"Latency={latency*1000:.3f} ms/img  FPS={fps:.2f}")

    # CSV (same columns/order as your eval.py)
    utils.compute_and_store_final_results(
        results_path=results_path,
        results=[[i_auroc, i_ap, p_auroc, p_ap, p_pro, latency, fps]],
        column_names=["image_auroc", "image_ap", "pixel_auroc", "pixel_ap", "pixel_pro", "latency_sec", "fps"],
        row_names=["mvtec_metal_nut"]
    )

    LOGGER.info(f"[DONE] Results saved under: {results_path}")


def parse_args():
    ap = argparse.ArgumentParser("Eval MVTec metal_nut with Hailo backbone + ONNX head (no PyTorch)")
    ap.add_argument("--data_path", required=True, help="MVTec root (contains metal_nut/...)")
    ap.add_argument("--hef", required=True, help="Path to glass_backbone.hef")
    ap.add_argument("--head", required=True, help="Path to glass_head.onnx")
    ap.add_argument("--results_path", default="results_eval_hailo", help="Where to save CSV/vis")
    ap.add_argument("--imagesize", type=int, default=288, help="Resize+CenterCrop size (image & mask)")
    ap.add_argument("--save_vis", action="store_true", help="Save heatmaps (JET)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_dataset(
        data_root=args.data_path,
        hef_path=args.hef,
        head_onnx=args.head,
        results_path=args.results_path,
        save_vis=args.save_vis,
        imagesize=args.imagesize
    )