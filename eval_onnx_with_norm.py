# =============================
# eval_onnx_with_norm.py
# =============================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX-only evaluation for GLASS-style anomaly detection (no PyTorch dependency),
assuming the backbone ONNX already includes ImageNet normalization internally.

Two execution modes:
  1) Split: --backbone_onnx_with_norm + --head_onnx
     - Backbone ONNX input: RGB, NCHW, values in [0..255], dtype {uint8|float32}
     - Head ONNX input: layer2/layer3 (float32)
  2) Full : --full_onnx (end-to-end; input same as above)

Datasets (no torch):
  - MVTec AD:
      <data_path>/<classname>/test/<good|defect>/*.png|jpg
      masks: <data_path>/<classname>/ground_truth/<defect>/*_mask.png or *.png
  - VisA (basic):
      <data_path>/<classname>/test/<good|defect>/*.jpg|png
      (masks optional; if absent, pixel metrics are skipped)

If your layout differs, provide a CSV index via --index_csv:
  columns: image_path,label(0|1),mask_path(optional)

Example:
  python eval_onnx_with_norm.py \
    --data_path /datasets/MVTec_AD \
    --dataset mvtec -d bottle -d cable \
    --backbone_onnx_with_norm ckpt_onnx/glass_backbone_with_norm.onnx \
    --head_onnx ckpt_onnx/glass_head.onnx \
    --results_path results_eval_onnx --save_vis
"""
import os
import sys
import time
import csv
import glob
import logging
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple
from datetime import datetime

import click
import numpy as np
import onnxruntime as ort
from scipy import ndimage
import cv2

import utils
import metrics  # <- 안전 버전(metrics_safe.py 내용)으로 대체되어 있다고 가정

LOGGER = logging.getLogger("eval_onnx_with_norm")


# ----------------------------
# NumPy-only RescaleSegmentor
# ----------------------------
class RescaleSegmentor:
    """NumPy-only GLASS-style patch map upscaling & smoothing."""
    def __init__(self, target_size=288, smoothing=4):
        if isinstance(target_size, int):
            self.target_size = (target_size, target_size)
        else:
            self.target_size = tuple(target_size)
        self.smoothing = smoothing

    def convert_to_segmentation(self, patch_scores):
        """
        patch_scores: np.ndarray (B,H',W') or (H',W') or list of (H',W')
        returns: list of np.ndarray (H,W), float32
        """
        if isinstance(patch_scores, (list, tuple)):
            patch_scores = np.stack(patch_scores, axis=0)
        patch_scores = np.asarray(patch_scores, dtype=np.float32)

        if patch_scores.ndim == 2:
            patch_scores = patch_scores[None, ...]

        outs = []
        H, W = self.target_size
        for ps in patch_scores:
            # bilinear resize
            mask = cv2.resize(ps, (W, H), interpolation=cv2.INTER_LINEAR)
            # gaussian smoothing (sigma = 4 is GLASS default)
            if self.smoothing and self.smoothing > 0:
                mask = ndimage.gaussian_filter(mask, sigma=self.smoothing)
            outs.append(mask.astype(np.float32, copy=False))
        return outs


# ----------------------------
# ORT Helpers
# ----------------------------
def _best_ort_providers():
    prov = []
    try:
        avail = set(ort.get_available_providers())
        if "CUDAExecutionProvider" in avail:
            prov.append("CUDAExecutionProvider")
        if "OpenVINOExecutionProvider" in avail:
            prov.append("OpenVINOExecutionProvider")
    except Exception:
        pass
    prov.append("CPUExecutionProvider")
    return prov


def _ort_session(path: str, providers: list):
    # 성능 옵션 세팅
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        return ort.InferenceSession(path, sess_options=so, providers=providers)
    except Exception as e:
        LOGGER.warning(f"[ORT] Failed to init session with {providers}: {e}\n -> Fallback to CPUExecutionProvider")
        return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


def _onnx_first_io_dtypes(sess):
    """Return (input_dtype, output_dtype) as numpy dtype objects for the first input/output."""
    def _to_np_dtype(type_str: str):
        s = type_str.lower()
        if "float16" in s: return np.float16
        if "float"   in s: return np.float32
        if "double"  in s: return np.float64
        if "uint8"   in s: return np.uint8
        if "int64"   in s: return np.int64
        if "int32"   in s: return np.int32
        raise ValueError(f"Unsupported dtype string: {type_str}")
    in_dtype  = _to_np_dtype(sess.get_inputs()[0].type)
    out_dtype = _to_np_dtype(sess.get_outputs()[0].type)
    return in_dtype, out_dtype


def _run_head_safe(hsess, h_inputs, h_outputs, layer2, layer3):
    """Run ONNX Head robustly, falling back to per-sample if needed."""
    def _run(l2, l3):
        feed = {h_inputs[0]: l2.astype(np.float32, copy=False),
                h_inputs[1]: l3.astype(np.float32, copy=False)}
        (ps,) = hsess.run(h_outputs, feed)
        ps = np.asarray(ps)
        if ps.ndim == 2:
            ps = ps[None, ...]
        return ps

    B = layer2.shape[0]
    ps = _run(layer2, layer3)
    if ps.shape[0] == B:
        return ps.astype(np.float32, copy=False)

    out = []
    for i in range(B):
        out.append(_run(layer2[i:i+1], layer3[i:i+1])[0])
    return np.stack(out, axis=0).astype(np.float32, copy=False)


def _run_full_safe(fsess, f_inputs, f_outputs, images):
    """Run Full ONNX robustly; returns (B,H',W')."""
    def _run(img):
        feed = {f_inputs[0]: img}
        (ps,) = fsess.run(f_outputs, feed)
        ps = np.asarray(ps)
        if ps.ndim == 2:
            ps = ps[None, ...]
        return ps

    B = images.shape[0]
    ps = _run(images)
    if ps.shape[0] == B:
        return ps.astype(np.float32, copy=False)

    out = []
    for i in range(B):
        out.append(_run(images[i:i+1])[0])
    return np.stack(out, axis=0).astype(np.float32, copy=False)


# ----------------------------
# Data loading (no torch)
# ----------------------------
@dataclass
class Sample:
    image_path: str
    label: int           # 0 = good, 1 = anomaly
    mask_path: Optional[str]  # None if not available


def _discover_mvtec(data_root: str, classname: str) -> List[Sample]:
    root = os.path.join(data_root, classname)
    test_dir = os.path.join(root, "test")
    gt_dir   = os.path.join(root, "ground_truth")

    samples: List[Sample] = []
    for defect_dir in sorted(glob.glob(os.path.join(test_dir, "*"))):
        defect = os.path.basename(defect_dir)
        img_paths = sorted(glob.glob(os.path.join(defect_dir, "*.png")) +
                           glob.glob(os.path.join(defect_dir, "*.jpg")))
        is_good = (defect == "good")
        for ip in img_paths:
            mp: Optional[str] = None
            if not is_good:
                base = os.path.splitext(os.path.basename(ip))[0]
                cand1 = os.path.join(gt_dir, defect, base + "_mask.png")
                cand2 = os.path.join(gt_dir, defect, base + ".png")
                mp = cand1 if os.path.exists(cand1) else (cand2 if os.path.exists(cand2) else None)
            samples.append(Sample(image_path=ip, label=0 if is_good else 1, mask_path=mp))
    return samples


def _discover_visa(data_root: str, classname: str) -> List[Sample]:
    root = os.path.join(data_root, classname)
    test_dir = os.path.join(root, "test")
    samples: List[Sample] = []
    for defect_dir in sorted(glob.glob(os.path.join(test_dir, "*"))):
        defect = os.path.basename(defect_dir)
        img_paths = sorted(glob.glob(os.path.join(defect_dir, "*.png")) +
                           glob.glob(os.path.join(defect_dir, "*.jpg")))
        is_good = (defect.lower() == "good")
        for ip in img_paths:
            samples.append(Sample(image_path=ip, label=0 if is_good else 1, mask_path=None))
    return samples


def _read_index_csv(csv_path: str) -> List[Sample]:
    rows: List[Sample] = []
    with open(csv_path, "r") as f:
        r = csv.DictReader(f)
        for row in r:
            ip = row["image_path"].strip()
            lab = int(row.get("label", 0))
            mp = row.get("mask_path")
            mp = mp.strip() if (mp and len(mp.strip()) > 0) else None
            rows.append(Sample(ip, lab, mp))
    return rows


def _preprocess_raw_nchw(img_bgr: np.ndarray, imagesize: int, out_dtype: np.dtype) -> np.ndarray:
    """
    Resize -> BGR2RGB -> NCHW.
    IMPORTANT:
      - No normalization here (backbone ONNX already normalizes internally).
      - Values expected in [0..255].
      - Dtype matches the ONNX session input (uint8 or float32).
    """
    img = cv2.resize(img_bgr, (imagesize, imagesize), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # RGB
    if out_dtype == np.uint8:
        chw = np.transpose(img, (2, 0, 1)).astype(np.uint8, copy=False)
    else:
        chw = np.transpose(img, (2, 0, 1)).astype(np.float32, copy=False)
    return chw


def _load_mask(mask_path: Optional[str], imagesize: int) -> Optional[np.ndarray]:
    if mask_path is None or (not os.path.exists(mask_path)):
        return None
    m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        return None
    m = cv2.resize(m, (imagesize, imagesize), interpolation=cv2.INTER_NEAREST)
    m = (m > 0).astype(np.uint8)
    return m


def _yield_minibatches(samples: List[Sample], batch_size: int, imagesize: int, x_dtype: np.dtype):
    for i in range(0, len(samples), batch_size):
        chunk = samples[i:i+batch_size]
        imgs = []
        labels = []
        masks = []
        paths = []
        for s in chunk:
            bgr = cv2.imread(s.image_path, cv2.IMREAD_COLOR)
            if bgr is None:
                continue
            chw = _preprocess_raw_nchw(bgr, imagesize, x_dtype)
            imgs.append(chw)
            labels.append(s.label)
            m = _load_mask(s.mask_path, imagesize)
            masks.append(m)
            paths.append(s.image_path)
        if not imgs:
            continue
        batch = {
            "image": np.stack(imgs, axis=0),        # (B,3,H,W) uint8|float32 in [0..255]
            "is_anomaly": np.array(labels, dtype=np.int32),
            "mask_gt": masks,                       # list[Optional[np.ndarray]]
            "paths": paths,
        }
        yield batch


# ----------------------------
# Main
# ----------------------------
@click.command(context_settings=dict(ignore_unknown_options=True, allow_extra_args=True))
# 데이터
@click.option("--data_path", type=click.Path(exists=True, file_okay=False), required=True, help="데이터 루트")
@click.option("--dataset", "dataset_name_key", type=click.Choice(["mvtec", "visa", "mpdd", "wfdd"]), required=True)
@click.option("--subdatasets", "-d", multiple=True, required=True, help="클래스/서브데이터셋들 (예: -d bottle -d cable)")
@click.option("--index_csv", type=click.Path(exists=True, dir_okay=False), required=False, help="(옵션) image_path,label,mask_path CSV")
# ONNX 경로
@click.option("--backbone_onnx_with_norm", type=click.Path(exists=True, dir_okay=False), required=False,
              help="Backbone ONNX with internal normalization (input: RGB [0..255], dtype {uint8|float32})")
@click.option("--head_onnx", type=click.Path(exists=True, dir_okay=False), required=False,
              help="HeadOnly ONNX (입력: layer2, layer3 → 출력: patch_scores(B,H',W'))")
@click.option("--full_onnx", type=click.Path(exists=True, dir_okay=False), required=False,
              help="FullGLASS ONNX (입력: image(B,3,H,W) raw RGB [0..255], 출력: patch_scores(B,H',W'))")
# 러닝/환경
@click.option("--results_path", default="results_eval_onnx", show_default=True, help="결과 저장 루트")
@click.option("--run_name", default="eval_onnx_with_norm", show_default=True, help="런 이름 (폴더명 일부)")
@click.option("--group", "log_group", default="group", show_default=True)
@click.option("--project", "log_project", default="project", show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
# 하이퍼(입력/로더)
@click.option("--imagesize", type=int, default=288, show_default=True)
@click.option("--batch_size", type=int, default=8, show_default=True)
# 시각화 여부
@click.option("--save_vis", is_flag=True, help="결과 이미지 저장 (heatmap 오버레이)")
def main(
    data_path, dataset_name_key, subdatasets, index_csv,
    backbone_onnx_with_norm, head_onnx, full_onnx,
    results_path, run_name, log_group, log_project, seed,
    imagesize, batch_size,
    save_vis
):
    """Torch-free ONNX evaluation (backbone ONNX contains normalization)."""
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LOGGER.info("Args: %s", " ".join(sys.argv))

    # 0) 결과/시드/프로바이더
    run_save_path = utils.create_storage_folder(results_path, log_project, log_group, run_name, mode="overwrite")
    os.makedirs(run_save_path, exist_ok=True)

    utils.fix_seeds(seed, with_torch=False, with_cuda=False)

    providers = _best_ort_providers()
    LOGGER.info(f"[ORT] providers: {providers}")

    # 1) 데이터 수집 (TEST 전용)
    def collect_for_class(sub: str) -> List[Sample]:
        if index_csv:
            rows = [r for r in _read_index_csv(index_csv)
                    if os.path.basename(os.path.dirname(r.image_path)) == sub or sub in r.image_path]
            return rows
        if dataset_name_key in ("mvtec", "mpdd", "wfdd"):
            return _discover_mvtec(data_path, sub)
        elif dataset_name_key == "visa":
            return _discover_visa(data_path, sub)
        else:
            raise SystemExit(f"Unsupported dataset: {dataset_name_key}")

    datasets = {}
    for sub in subdatasets:
        samples = collect_for_class(sub)
        if not samples:
            LOGGER.warning(f"[DATA] No samples found for class '{sub}' under {data_path}")
        datasets[sub] = samples
        LOGGER.info(f"[DATA] {sub:>12s}: test={len(samples)}")

    # 2) ONNX 세션 및 입력 dtype 결정
    use_full = full_onnx is not None
    x_dtype_for_loader = np.uint8  # 기본값

    if use_full:
        fsess = _ort_session(full_onnx, providers)
        f_inputs  = [i.name for i in fsess.get_inputs()]
        f_outputs = [o.name for o in fsess.get_outputs()]
        f_in_dtype, f_out_dtype = _onnx_first_io_dtypes(fsess)
        x_dtype_for_loader = f_in_dtype  # uint8 또는 float32
        LOGGER.info(f"[MODEL] Using FULL ONNX: {full_onnx}  inputs={f_inputs} outputs={f_outputs}  in_dtype={f_in_dtype} out_dtype={f_out_dtype}")
    else:
        if not (backbone_onnx_with_norm and head_onnx):
            raise SystemExit("Provide --full_onnx OR both --backbone_onnx_with_norm and --head_onnx.")
        bsess = _ort_session(backbone_onnx_with_norm, providers)
        hsess = _ort_session(head_onnx, providers)
        b_inputs  = [i.name for i in bsess.get_inputs()]
        b_outputs = [o.name for o in bsess.get_outputs()]
        h_inputs  = [i.name for i in hsess.get_inputs()]
        h_outputs = [o.name for o in hsess.get_outputs()]
        b_in_dtype, _ = _onnx_first_io_dtypes(bsess)
        x_dtype_for_loader = b_in_dtype  # uint8 또는 float32
        LOGGER.info(f"[MODEL] Using SPLIT ONNX: backbone(norm)={backbone_onnx_with_norm}, head={head_onnx}  backbone_in_dtype={b_in_dtype}")

    # NumPy-only segmentor
    segmentor = RescaleSegmentor(target_size=imagesize, smoothing=4)

    # 3) 평가 per class
    for idx, (classname, samples) in enumerate(datasets.items(), start=1):
        LOGGER.info(f"[EVAL-ONNX] {dataset_name_key}_{classname} ({idx}/{len(datasets)}) {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # 이미지 레벨 메트릭용
        all_image_scores: List[float] = []
        labels_gt: List[int] = []

        # 픽셀 메트릭/PRO용 (마스크 있는 샘플만 수집)
        seg_for_metric: List[np.ndarray] = []
        gt_for_metric:  List[np.ndarray] = []

        # 시각화 저장용 (전체 이미지)
        all_masks_vis: List[np.ndarray] = []

        vis_dir = os.path.join(run_save_path, "vis", f"{dataset_name_key}_{classname}")
        if save_vis:
            os.makedirs(vis_dir, exist_ok=True)

        t0 = time.perf_counter()
        num_images = 0

        for batch in _yield_minibatches(samples, batch_size, imagesize, x_dtype_for_loader):
            inp    = batch["image"]            # (B,3,H,W) uint8|float32 in [0..255]
            labels = batch["is_anomaly"].tolist()
            paths  = batch["paths"]
            gts    = batch["mask_gt"]          # list[Optional[np.ndarray]]

            labels_gt.extend(labels)

            if use_full:
                # 세션 입력 dtype으로 캐스팅만 보장
                if inp.dtype != x_dtype_for_loader:
                    inp_cast = inp.astype(x_dtype_for_loader, copy=False)
                else:
                    inp_cast = inp
                patch_scores = _run_full_safe(fsess, f_inputs, f_outputs, inp_cast)  # (B,h',w')
            else:
                bfeed = {b_inputs[0]: inp.astype(x_dtype_for_loader, copy=False)}
                layer2, layer3 = bsess.run(b_outputs, bfeed)
                patch_scores = _run_head_safe(hsess, h_inputs, h_outputs, layer2, layer3)  # (B,h',w')

            masks_rescaled = segmentor.convert_to_segmentation(patch_scores)  # list of (H,W)

            B = inp.shape[0]
            for bi in range(B):
                ps = patch_scores[bi]
                img_score = float(ps.max())   # image-level score: max anomaly
                all_image_scores.append(img_score)

                seg = masks_rescaled[bi]
                all_masks_vis.append(seg)

                gt = gts[bi]
                if isinstance(gt, np.ndarray):
                    # 픽셀 메트릭/PRO는 마스크 존재하는 샘플만 정렬 수집
                    seg_for_metric.append(seg)
                    gt_for_metric.append(gt)

            num_images += B

        t1 = time.perf_counter()
        elapsed = t1 - t0
        latency = elapsed / max(num_images, 1)
        fps = num_images / max(elapsed, 1e-9)

        # 4) 메트릭 계산
        scores_arr = np.asarray(all_image_scores, dtype=np.float32)
        image_scores = metrics.compute_imagewise_retrieval_metrics(scores_arr, labels_gt, path='eval_onnx')
        i_auroc = float(image_scores["auroc"])
        i_ap    = float(image_scores["ap"])

        if len(gt_for_metric) > 0:
            seg_arr = np.stack(seg_for_metric, axis=0).astype(np.float32)  # (N_pos,H,W)
            # 픽셀 AUROC/AP
            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(seg_arr, gt_for_metric, path='eval_onnx')
            p_auroc = float(pixel_scores["auroc"])
            p_ap    = float(pixel_scores["ap"])
            # PRO
            try:
                masks_np = np.stack(gt_for_metric, axis=0).astype(np.uint8)
                p_pro = float(metrics.compute_pro(masks_np, seg_arr, fpr_max=0.3, per_image_minmax=True))
            except Exception as e:
                LOGGER.warning(f"[METRIC] PRO computation failed: {e}")
                p_pro = 0.0
        else:
            p_auroc = p_ap = p_pro = -1.0  # 마스크가 없으면 스킵

        LOGGER.info(
            f"  image_auroc: {i_auroc:.4f}  image_ap: {i_ap:.4f}  "
            f"pixel_auroc: {p_auroc:.4f}  pixel_ap: {p_ap:.4f}  pixel_pro: {p_pro:.4f}"
        )
        LOGGER.info(f"  [SPEED] Elapsed={elapsed:.3f}s  Images={num_images}  "
                    f"Latency={latency*1000:.3f} ms/img  FPS={fps:.2f}")

        # 5) 시각화 저장 (옵션)
        if save_vis:
            for i, m in enumerate(all_masks_vis):
                # m은 0~1 스케일이 아닐 수 있으니 안전 정규화 후 표시
                mm = m.astype(np.float32)
                mmin, mmax = float(mm.min()), float(mm.max())
                if mmax - mmin > 1e-12:
                    mm = (mm - mmin) / (mmax - mmin)
                else:
                    mm = np.zeros_like(mm, dtype=np.float32)
                heat = (mm * 255.0).astype(np.uint8)
                heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(vis_dir, f"{i:04d}.png"), heat)

        # 6) CSV 저장
        utils.compute_and_store_final_results(
            results_path=run_save_path,
            results=[[i_auroc, i_ap, p_auroc, p_ap, p_pro, latency, fps]],
            column_names=["image_auroc", "image_ap", "pixel_auroc", "pixel_ap", "pixel_pro", "latency_sec", "fps"],
            row_names=[f"{dataset_name_key}_{classname}"],
        )

    LOGGER.info(f"[DONE] Results saved under: {run_save_path}")


if __name__ == "__main__":
    main()