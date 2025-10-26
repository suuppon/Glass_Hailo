#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Eval MVTec metal_nut (TEST) with Hailo backbone (HEF) + ONNX head.

- Hailo VDevice/configure를 1회만 열고 전체 테스트셋을 순회 (속도↑)
- ONNXRuntime 세션 재사용
- ONNX head가 1입력/2입력 모두 자동 대응
- PyTorch 미의존 (NumPy/OpenCV/PIL/SciPy/onnxruntime/HailoRT만 사용)
- metrics.py / utils.py (네가 준 버전)과 결과 형식 호환

예:
python eval_hailo_onnx.py \
  --data_path /path/to/MVTec_AD \
  --hef ckpt/glass_backbone.hef \
  --head ckpt/glass_head.onnx \
  --results_path results_eval_hailo \
  --save_vis
"""

import os
import time
import argparse
import logging
from typing import List, Tuple, Optional

import numpy as np
import cv2
from PIL import Image
from scipy import ndimage
import onnxruntime as ort

# local
import metrics   # 너의 파일
import utils     # 너의 파일

# HailoRT
from hailo_platform import VDevice, HailoSchedulingAlgorithm

LOGGER = logging.getLogger("eval_hailo_onnx")
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ----------- ImageNet norm -----------
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


# =========================
# MVTec loader (metal_nut)
# =========================
def list_metal_nut_test(data_root: str) -> List[Tuple[str, int, Optional[str]]]:
    cls = "metal_nut"
    test_dir = os.path.join(data_root, cls, "test")
    gt_dir   = os.path.join(data_root, cls, "ground_truth")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Not found: {test_dir}")

    items: List[Tuple[str, int, Optional[str]]] = []
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
                items.append((p, 0, None))
        else:
            m_dir = os.path.join(gt_dir, anomaly_type)
            m_files = sorted([
                os.path.join(m_dir, f) for f in os.listdir(m_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif"))
            ])
            if len(m_files) != len(img_files):
                LOGGER.warning(f"[WARN] #masks != #images for type={anomaly_type}: {len(m_files)} vs {len(img_files)}")
            for i, p in enumerate(img_files):
                items.append((p, 1, m_files[i] if i < len(m_files) else None))
    return items


# =========================
# Pre/Post
# =========================
def _resize_then_center_crop(img: np.ndarray, target: int) -> np.ndarray:
    h, w = img.shape[:2]
    if h == target and w == target:
        resized = img
    else:
        if h < w:
            new_h = target
            new_w = int(round(w * (target / h)))
        else:
            new_w = target
            new_h = int(round(h * (target / w)))
        resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    h2, w2 = resized.shape[:2]
    top = max((h2 - target) // 2, 0)
    left = max((w2 - target) // 2, 0)
    crop = resized[top: top + target, left: left + target]
    if crop.shape[0] != target or crop.shape[1] != target:
        pad = np.zeros((target, target, resized.shape[2] if resized.ndim == 3 else 1), dtype=resized.dtype)
        pad[:crop.shape[0], :crop.shape[1]] = crop
        crop = pad
    return crop


def preprocess_image_to_bchw(path: str, size: int = 288) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        img = np.array(Image.open(path).convert("RGB"))[:, :, ::-1]  # RGB->BGR
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    img = _resize_then_center_crop(img, size)
    x = np.transpose(img, (2, 0, 1))[None, ...]
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.astype(np.float32, copy=False)


def load_mask_to_hw(path: Optional[str], size: int = 288) -> np.ndarray:
    if path is None or (not os.path.isfile(path)):
        return np.zeros((size, size), dtype=np.float32)
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        m = np.array(Image.open(path).convert("L"))
    m = _resize_then_center_crop(m[..., None], size)[..., 0]  # keep 2D
    m = (m > 0).astype(np.float32) if m.max() > 1 else m.astype(np.float32)
    return m


def upsample_and_smooth(patch_scores_bhw: np.ndarray, size: int = 288, sigma: float = 4.0) -> np.ndarray:
    """(B,H',W') -> (B,size,size) + Gaussian blur"""
    B, h, w = patch_scores_bhw.shape
    out = np.empty((B, size, size), dtype=np.float32)
    for i in range(B):
        up = cv2.resize(patch_scores_bhw[i], (size, size), interpolation=cv2.INTER_LINEAR)
        out[i] = ndimage.gaussian_filter(up.astype(np.float32, copy=False), sigma=sigma)
    return out


# =========================
# Hailo backbone runner
# =========================
def _get_scale_zp(qinfo):
    if qinfo is None:
        return 1.0, 0.0
    for s_name, z_name in [("scale", "zero_point"), ("scale", "zp"),
                           ("qp_scale", "qp_zp"), ("quant_scale", "quant_zp")]:
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


class HailoBackbone:
    """VDevice + configured model 재사용"""
    def __init__(self, hef_path: str):
        if not os.path.isfile(hef_path):
            raise FileNotFoundError(hef_path)
        self.hef_path = hef_path
        self.vdev = None
        self.infer_model = None
        self.cmodel = None
        self.in_meta = None
        self.outs_meta = None

    def __enter__(self):
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.vdev = VDevice(params)
        self.vdev.__enter__()
        self.infer_model = self.vdev.create_infer_model(self.hef_path)
        self.cmodel_ctx = self.infer_model.configure()
        self.cmodel = self.cmodel_ctx.__enter__()
        # input meta
        self.in_meta = self.infer_model.input() if callable(getattr(self.infer_model, "input", None)) \
            else getattr(self.infer_model, "input")
        # outputs meta
        outs_attr = getattr(self.infer_model, "outputs", None)
        if callable(outs_attr):
            self.outs_meta = list(outs_attr())
        elif isinstance(outs_attr, (list, tuple)):
            self.outs_meta = list(outs_attr)
        else:
            get_single = getattr(self.infer_model, "output", None)
            if callable(get_single):
                self.outs_meta = [get_single()]
            else:
                raise RuntimeError("Cannot discover model outputs from infer_model.")
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            if self.cmodel is not None:
                self.cmodel_ctx.__exit__(exc_type, exc, tb)
        finally:
            if self.vdev is not None:
                self.vdev.__exit__(exc_type, exc, tb)

    @staticmethod
    def _dequantize(u8: np.ndarray, scale: float, zp: float) -> np.ndarray:
        return scale * (u8.astype(np.float32) - zp)

    def infer_one(self, x_bchw_f32: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """x: (1,3,288,288) float32 -> returns (l2:1x512x36x36, l3:1x1024x18x18) float32"""
        bindings = self.cmodel.create_bindings()

        # input quant & layout
        in_scale, in_zp = _get_scale_zp(getattr(self.in_meta, "quant_info", None))
        x_q = np.clip(np.round(x_bchw_f32 / max(in_scale, 1e-12) + in_zp), 0, 255).astype(np.uint8)

        in_shape = tuple(self.in_meta.shape) if hasattr(self.in_meta, "shape") else None
        if in_shape and len(in_shape) == 4 and in_shape[1] in (288, 224) and in_shape[-1] == 3:
            x_q_feed = np.transpose(x_q, (0, 2, 3, 1))
        else:
            x_q_feed = x_q
        x_q_feed = np.ascontiguousarray(x_q_feed, dtype=np.uint8)
        bindings.input().set_buffer(x_q_feed)

        # outputs buffer
        out_bindings = []
        get_outputs = getattr(bindings, "outputs", None)
        if callable(get_outputs):
            out_bindings = list(get_outputs())
        else:
            get_output = getattr(bindings, "output", None)
            if callable(get_output):
                for meta in self.outs_meta:
                    name = getattr(meta, "name", None) or str(meta)
                    out_bindings.append(get_output(name))
            else:
                raise RuntimeError("Bindings exposes neither outputs() nor output(name).")

        for i, ob in enumerate(out_bindings):
            shp = tuple(getattr(self.outs_meta[i], "shape")()) if callable(getattr(self.outs_meta[i], "shape", None)) \
                else tuple(getattr(self.outs_meta[i], "shape"))
            ob.set_buffer(np.empty(shp, dtype=np.uint8, order="C"))

        # run
        job = self.cmodel.run_async([bindings])
        job.wait(10_000)

        # fetch & normalize
        outs = []
        for i, ob in enumerate(out_bindings):
            raw = ob.get_buffer()
            s, zp = _get_scale_zp(getattr(self.outs_meta[i], "quant_info", None))
            fp = self._dequantize(raw, s, zp)
            shp = fp.shape
            if len(shp) == 4 and shp[-1] in (512, 1024):
                fp = np.transpose(fp, (0, 3, 1, 2))  # NHWC->NCHW
            elif len(shp) == 3 and shp[-1] in (512, 1024):
                fp = np.transpose(fp, (2, 0, 1))[None, ...]
            outs.append(fp.astype(np.float32, copy=False))

        l2, l3 = None, None
        for arr in outs:
            _, C, H, W = arr.shape if arr.ndim == 4 else (1,) + arr.shape
            if (C, H, W) == (512, 36, 36):
                l2 = arr
            elif (C, H, W) == (1024, 18, 18):
                l3 = arr
        if l2 is None or l3 is None:
            raise RuntimeError(f"Failed to identify L2/L3: {[a.shape for a in outs]}")
        return l2, l3


# =========================
# ONNX Head (reusable)
# =========================
class ONNXHead:
    def __init__(self, onnx_path: str, provider: str = "CPUExecutionProvider"):
        if not os.path.isfile(onnx_path):
            raise FileNotFoundError(onnx_path)
        self.sess = ort.InferenceSession(onnx_path, providers=[provider])
        self.inputs = self.sess.get_inputs()
        self.outputs = self.sess.get_outputs()
        self.in_names = [i.name for i in self.inputs]
        self.out_name = self.outputs[0].name

    @staticmethod
    def _to_bchw(x: np.ndarray) -> np.ndarray:
        return np.ascontiguousarray(x.astype(np.float32, copy=False))

    def __call__(self, l2: np.ndarray, l3: np.ndarray) -> np.ndarray:
        """
        Returns patch_scores (1,H',W') float32.
        - Two-input model: feed {in0:l2, in1:l3}
        - One-input model:
            * expect C == 512 -> feed l2
            * expect C == 1024 -> feed l3
            * expect C == 1536 -> upsample l3 to 36x36 and concat with l2
        """
        if len(self.inputs) == 2:
            in0, in1 = self.in_names
            (out,) = self.sess.run([self.out_name], {in0: self._to_bchw(l2), in1: self._to_bchw(l3)})
        else:
            # single input heuristic
            in0 = self.in_names[0]
            ishape = self.inputs[0].shape  # [1, C, H, W] or dynamic
            # infer expected C/H/W if possible
            exp_C = None
            exp_H = None
            exp_W = None
            if len(ishape) == 4:
                exp_C = ishape[1] if isinstance(ishape[1], int) else None
                exp_H = ishape[2] if isinstance(ishape[2], int) else None
                exp_W = ishape[3] if isinstance(ishape[3], int) else None

            # Decide what to feed
            if exp_C in (512, None):
                feed = l2
            elif exp_C == 1024:
                feed = l3
            elif exp_C == 1536:
                # upsample l3 to 36x36 then concat on C
                B, C3, H3, W3 = l3.shape
                up_l3 = np.empty((B, C3, 36, 36), dtype=np.float32)
                for i in range(C3):
                    up_l3[:, i] = cv2.resize(l3[0, i], (36, 36), interpolation=cv2.INTER_LINEAR)[None, ...]
                feed = np.concatenate([l2, up_l3], axis=1)
            else:
                # fallback to try concat as common case
                B, C3, H3, W3 = l3.shape
                up_l3 = np.empty((B, C3, 36, 36), dtype=np.float32)
                for i in range(C3):
                    up_l3[:, i] = cv2.resize(l3[0, i], (36, 36), interpolation=cv2.INTER_LINEAR)[None, ...]
                feed = np.concatenate([l2, up_l3], axis=1)

            (out,) = self.sess.run([self.out_name], {in0: self._to_bchw(feed)})

        out = np.asarray(out)
        if out.ndim == 2:  # (H',W') -> (1,H',W')
            out = out[None, ...]
        return out.astype(np.float32, copy=False)


# =========================
# Eval
# =========================
def evaluate_dataset(
    data_root: str,
    hef_path: str,
    head_onnx: str,
    results_path: str,
    imagesize: int = 288,
    save_vis: bool = False,
    overlay: bool = False,
):
    os.makedirs(results_path, exist_ok=True)
    vis_dir = os.path.join(results_path, "vis", "mvtec_metal_nut")
    if save_vis:
        os.makedirs(vis_dir, exist_ok=True)

    items = list_metal_nut_test(data_root)
    LOGGER.info(f"[DATA] metal_nut TEST: {len(items)} images")

    labels_gt: List[int] = []
    masks_gt: List[np.ndarray] = []
    image_scores: List[float] = []
    masks_pred: List[np.ndarray] = []

    head = ONNXHead(head_onnx, provider="CPUExecutionProvider")

    t0 = time.perf_counter()
    with HailoBackbone(hef_path) as backbone:
        for idx, (img_p, y, m_p) in enumerate(items):
            x = preprocess_image_to_bchw(img_p, size=imagesize)          # (1,3,288,288)
            l2, l3 = backbone.infer_one(x)                                # (1,512,36,36), (1,1024,18,18)
            patch_scores = head(l2, l3)                                   # (1,H',W')
            mask_ups = upsample_and_smooth(patch_scores, size=imagesize)  # (1,288,288)
            m = mask_ups[0]

            image_scores.append(float(patch_scores.max()))
            labels_gt.append(int(y))
            masks_gt.append(load_mask_to_hw(m_p, size=imagesize))
            masks_pred.append(m)

            if save_vis:
                heat = (np.clip(m, 0, 1) * 255.0).astype(np.uint8)
                heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                if overlay:
                    # 원본 불러 overlay
                    bgr = cv2.imread(img_p, cv2.IMREAD_COLOR)
                    if bgr is None:
                        bgr = np.zeros((imagesize, imagesize, 3), np.uint8)
                    bgr = cv2.cvtColor(cv2.resize(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), (imagesize, imagesize)),
                                       cv2.COLOR_RGB2BGR)
                    vis = cv2.addWeighted(bgr, 0.5, heat, 0.5, 0)
                else:
                    vis = heat
                cv2.imwrite(os.path.join(vis_dir, f"{idx:04d}.png"), vis)

    elapsed = time.perf_counter() - t0
    n = max(len(items), 1)
    latency = elapsed / n
    fps = len(items) / max(elapsed, 1e-9)

    # --- metrics ---
    scores_arr = np.asarray(image_scores, dtype=np.float32)
    i_res = metrics.compute_imagewise_retrieval_metrics(scores_arr, labels_gt, path="eval")
    i_auc = float(i_res["auroc"])
    i_ap  = float(i_res["ap"])

    seg_arr = np.stack(masks_pred, axis=0).astype(np.float32)
    p_res = metrics.compute_pixelwise_retrieval_metrics(seg_arr, masks_gt, path="eval")
    p_auc = float(p_res["auroc"])
    p_ap  = float(p_res["ap"])
    try:
        p_pro = float(metrics.compute_pro(np.squeeze(np.array(masks_gt)), seg_arr))
    except Exception:
        p_pro = 0.0

    LOGGER.info(f"[METRICS] image_auroc={i_auc:.4f} image_ap={i_ap:.4f}  "
                f"pixel_auroc={p_auc:.4f} pixel_ap={p_ap:.4f} pixel_pro={p_pro:.4f}")
    LOGGER.info(f"[SPEED] Elapsed={elapsed:.3f}s  N={len(items)}  Latency={latency*1000:.2f} ms/img  FPS={fps:.2f}")

    # --- CSV ---
    utils.compute_and_store_final_results(
        results_path=results_path,
        results=[[i_auc, i_ap, p_auc, p_ap, p_pro, latency, fps]],
        column_names=["image_auroc", "image_ap", "pixel_auroc", "pixel_ap", "pixel_pro", "latency_sec", "fps"],
        row_names=["mvtec_metal_nut"],
    )

    LOGGER.info(f"[DONE] Results saved at: {results_path}")


def parse_args():
    ap = argparse.ArgumentParser("Eval MVTec metal_nut with Hailo backbone + ONNX head")
    ap.add_argument("--data_path", required=True, help="MVTec root (contains metal_nut/...)")
    ap.add_argument("--hef", required=True, help="Path to glass_backbone.hef")
    ap.add_argument("--head", required=True, help="Path to glass_head.onnx")
    ap.add_argument("--results_path", default="results_eval_hailo", help="Where to save CSV/vis")
    ap.add_argument("--imagesize", type=int, default=288, help="Resize+CenterCrop size")
    ap.add_argument("--save_vis", action="store_true", help="Save heatmaps")
    ap.add_argument("--overlay", action="store_true", help="Save overlay (orig + heatmap)")
    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate_dataset(
        data_root=args.data_path,
        hef_path=args.hef,
        head_onnx=args.head,
        results_path=args.results_path,
        imagesize=args.imagesize,
        save_vis=args.save_vis,
        overlay=args.overlay,
    )