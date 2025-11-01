# =============================
# eval_onnx_with_norm.py  (HEF backbone supported)
# =============================
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ONNX-only evaluation for GLASS-style anomaly detection (no PyTorch dependency),
assuming the backbone (ONNX or HEF) already includes ImageNet normalization internally.

Execution modes (choose one):
  1) Split-ONNX : --backbone_onnx_with_norm + --head_onnx
       - Backbone ONNX input: RGB, NCHW, [0..255], dtype {uint8|float32}
       - Head ONNX input: layer2/layer3 (float32)
  2) Full-ONNX  : --full_onnx (end-to-end; input same as above)
  3) Split-HEF  : --backbone_hef + --head_onnx
       - Backbone HEF input: RGB, NCHW, [0..255] (float32) → internally quantized by HEF
       - Head ONNX input: layer2/layer3 (float32)
  4) Split-HAR  : --backbone_har + --head_onnx
       - Backbone HAR input: RGB, NCHW, [0..255] (float32) → uses InferVStreams API
       - Head ONNX input: layer2/layer3 (float32)

Datasets (torch-free):
  - MVTec AD:
      <data_path>/<classname>/test/<good|defect>/*.png|jpg
      masks: <data_path>/<classname>/ground_truth/<defect>/*_mask.png (or .png)
  - VisA (basic):
      <data_path>/<classname>/test/<good|defect>/*.jpg|png
      (masks optional)

If layout differs, provide --index_csv with columns:
  image_path,label(0|1),mask_path(optional)
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
import metrics  # safe metrics (your metrics_safe.py)

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
        returns: list[np.ndarray(H,W) float32]
        """
        if isinstance(patch_scores, (list, tuple)):
            patch_scores = np.stack(patch_scores, axis=0)
        patch_scores = np.asarray(patch_scores, dtype=np.float32)
        if patch_scores.ndim == 2:
            patch_scores = patch_scores[0][None, ...]
        outs = []
        H, W = self.target_size
        for ps in patch_scores:
            m = cv2.resize(ps, (W, H), interpolation=cv2.INTER_LINEAR)
            if self.smoothing and self.smoothing > 0:
                m = ndimage.gaussian_filter(m, sigma=self.smoothing)
            outs.append(m.astype(np.float32, copy=False))
        return outs


# ----------------------------
# ONNX Runtime helpers
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
    so = ort.SessionOptions()
    so.intra_op_num_threads = max(1, (os.cpu_count() or 4) // 2)
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    try:
        return ort.InferenceSession(path, sess_options=so, providers=providers)
    except Exception as e:
        LOGGER.warning(f"[ORT] init failed for providers={providers}: {e}\n -> fallback CPU")
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
    """Run ONNX Head robustly; returns (B,H',W')."""
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
# Hailo HEF helpers (lazy import)
# ----------------------------
def _get_shape_from_meta(meta):
    shp = getattr(meta, "shape", None)
    if callable(shp):
        shp = tuple(shp())
    elif shp is not None:
        shp = tuple(shp)
    return shp

def _get_scale_zp(qinfo):
    # qinfo may expose .qp_scale / .qp_zp or nested attrs; be defensive
    if qinfo is None:
        return 1.0, 0.0
    scale = getattr(qinfo, "qp_scale", None)
    zp    = getattr(qinfo, "qp_zp", None)
    if scale is None:
        scale = getattr(qinfo, "scale", 1.0)
    if zp is None:
        zp = getattr(qinfo, "zero_point", 0.0)
    try:
        scale = float(scale)
    except Exception:
        scale = 1.0
    try:
        zp = float(zp)
    except Exception:
        zp = 0.0
    return max(scale, 1e-12), zp

def _dequantize_u8(arr_u8: np.ndarray, scale: float, zp: float) -> np.ndarray:
    return (arr_u8.astype(np.float32) - zp) * scale

def _hailo_run_backbone_har(har_path: str, x_bchw_f32: np.ndarray):
    """
    Run HAR backbone using InferVStreams API.
    Input : float32 (1,3,288,288) NCHW, values [0..255] (raw RGB)
    Output: layer2 (1,512,36,36) float32 NCHW, layer3 (1,1024,18,18) float32 NCHW
    """
    if not os.path.isfile(har_path):
        raise FileNotFoundError(har_path)
    
    import hailo_platform as hpf
    
    # Load HAR
    har = hpf.HAR(har_path)
    
    with hpf.VDevice() as target:
        # Configure network
        configure_params = hpf.ConfigureParams.create_from_har(har)
        network_group = target.configure(har, configure_params)[0]
        network_group_params = network_group.create_params()
        
        # Get input/output vstream info
        input_vstream_infos = har.get_input_vstream_infos()
        output_vstream_infos = har.get_output_vstream_infos()
        
        # Setup input/output params
        input_vstreams_params = hpf.InputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
        )
        output_vstreams_params = hpf.OutputVStreamParams.make_from_network_group(
            network_group, quantized=False, format_type=hpf.FormatType.FLOAT32
        )
        
        with network_group.activate(network_group_params):
            with hpf.InferVStreams(network_group, input_vstreams_params, output_vstreams_params) as infer_pipeline:
                # Prepare input: NCHW [0..255] → NCHW float32 [0..1] range
                # (Since HAR expects normalized float32)
                x_normalized = x_bchw_f32 / 255.0  # [0..255] → [0..1]
                
                # Create input dict
                input_data = {}
                for in_info in input_vstream_infos:
                    input_data[in_info.name] = x_normalized
                
                # Run inference
                results = infer_pipeline.infer(input_data)
                
                # Extract outputs
                l2, l3 = None, None
                for out_info in output_vstream_infos:
                    output = results[out_info.name]
                    _, C, H, W = output.shape
                    if (C, H, W) == (512, 36, 36):
                        l2 = output
                    elif (C, H, W) == (1024, 18, 18):
                        l3 = output
                
                if l2 is None or l3 is None:
                    shapes = {out_info.name: results[out_info.name].shape for out_info in output_vstream_infos}
                    raise RuntimeError(f"Failed to identify L2/L3. Output shapes: {shapes}")
                
                # Scale back to [0..255] range for compatibility
                l2 = (l2 * 255.0).astype(np.float32, copy=False)
                l3 = (l3 * 255.0).astype(np.float32, copy=False)
                
                return l2, l3

def _hailo_run_backbone_hef(hef_path: str, x_bchw_f32: np.ndarray):
    # x_bchw_f32: (1,3,288,288) float32, [0..255], RGB, 정규화/표준화 없음
    import hailo_platform as hpf
    if not os.path.isfile(hef_path):
        raise FileNotFoundError(hef_path)

    hef = hpf.HEF(hef_path)
    with hpf.VDevice() as target:
        cfg = hpf.ConfigureParams.create_from_hef(hef=hef, interface=hpf.HailoStreamInterface.PCIe)
        ng = target.configure(hef, cfg)[0]
        ng_params = ng.create_params()

        # 입력/출력 스트림 메타
        in_infos  = hef.get_input_vstream_infos()
        out_infos = hef.get_output_vstream_infos()

        # 입력/출력 스트림 파라미터: quantized=False, FLOAT32  -> SDK가 내부에서 올바른 지점에 양자화
        in_params  = hpf.InputVStreamParams.make(ng, format_type=hpf.FormatType.FLOAT32)
        out_params = hpf.OutputVStreamParams.make(ng, format_type=hpf.FormatType.FLOAT32)

        # 레이아웃 맞추기 (보통 NHWC). in_info.shape 를 신뢰.
        in_info = in_infos[0]
        shp = tuple(in_info.shape)  # e.g. (1, 288, 288, 3)  or (1, 3, 288, 288)
        if len(shp) == 4 and shp[-1] == 3:
            x_feed = np.transpose(x_bchw_f32, (0, 2, 3, 1))  # NCHW -> NHWC
        else:
            x_feed = x_bchw_f32  # 이미 NCHW

        with hpf.InferVStreams(ng, in_params, out_params) as pipe:
            with ng.activate(ng_params):
                results = pipe.infer({in_info.name: x_feed.astype(np.float32, copy=False)})

        # 출력들에서 (512,36,36), (1024,18,18) 채널/공간 찾고, NHWC면 NCHW로 전치
        l2 = l3 = None
        for oi in out_infos:
            y = results[oi.name]
            if y.ndim == 4 and y.shape[-1] in (512, 1024):   # NHWC -> NCHW
                y = np.transpose(y, (0, 3, 1, 2))
            c, h, w = y.shape[1], y.shape[2], y.shape[3]
            if (c, h, w) == (512, 36, 36): l2 = y.astype(np.float32, copy=False)
            if (c, h, w) == (1024, 18, 18): l3 = y.astype(np.float32, copy=False)

        if l2 is None or l3 is None:
            all_shapes = {oi.name: tuple(results[oi.name].shape) for oi in out_infos}
            raise RuntimeError(f"L2/L3 식별 실패. 출력 shape들: {all_shapes}")
        return l2, l3


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
    Resize -> BGR2RGB -> NCHW. No normalization. Values in [0..255].
    """
    img = cv2.resize(img_bgr, (imagesize, imagesize), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
# 백본 / 헤드
@click.option("--backbone_onnx_with_norm", type=click.Path(exists=True, dir_okay=False), required=False,
              help="Backbone ONNX with internal normalization (input: RGB [0..255], dtype {uint8|float32})")
@click.option("--backbone_hef", type=click.Path(exists=True, dir_okay=False), required=False,
              help="Backbone HEF (quantized), expects RGB NCHW float32 [0..255]; internal normalization present.")
@click.option("--backbone_har", type=click.Path(exists=True, dir_okay=False), required=False,
              help="Backbone HAR (Hailo Archive), expects RGB NCHW float32 [0..255]; uses InferVStreams API.")
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
    backbone_onnx_with_norm, backbone_hef, backbone_har, head_onnx, full_onnx,
    results_path, run_name, log_group, log_project, seed,
    imagesize, batch_size,
    save_vis
):
    """Torch-free evaluation (ONNX OR HEF OR HAR backbone with internal normalization)."""
    warnings.filterwarnings("ignore")
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    LOGGER.info("Args: %s", " ".join(sys.argv))

    # 0) 결과/시드/프로바이더
    run_save_path = utils.create_storage_folder(results_path, log_project, log_group, run_name, mode="overwrite")
    os.makedirs(run_save_path, exist_ok=True)
    utils.fix_seeds(seed, with_torch=False, with_cuda=False)

    providers = _best_ort_providers()
    LOGGER.info(f"[ORT] providers: {providers}")

    # 1) 데이터 수집
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

    # 2) 모드 선택 & 세션 구성
    use_full = full_onnx is not None
    use_hef  = (backbone_hef is not None)
    use_har  = (backbone_har is not None)
    x_dtype_for_loader = np.uint8  # default

    if use_full:
        fsess = _ort_session(full_onnx, providers)
        f_inputs  = [i.name for i in fsess.get_inputs()]
        f_outputs = [o.name for o in fsess.get_outputs()]
        f_in_dtype, f_out_dtype = _onnx_first_io_dtypes(fsess)
        x_dtype_for_loader = f_in_dtype  # uint8 or float32
        LOGGER.info(f"[MODEL] Using FULL ONNX: {full_onnx}  in_dtype={f_in_dtype} out_dtype={f_out_dtype}")

    elif use_hef:
        if not head_onnx:
            raise SystemExit("HEF backbone requires --head_onnx.")
        # Head ONNX session (CPU EP ok)
        hsess = _ort_session(head_onnx, providers=["CPUExecutionProvider"])
        h_inputs  = [i.name for i in hsess.get_inputs()]
        h_outputs = [o.name for o in hsess.get_outputs()]
        # Loader must deliver float32 raw [0..255] for HEF helper
        x_dtype_for_loader = np.float32
        LOGGER.info(f"[MODEL] Using SPLIT HEF backbone={backbone_hef}, head={head_onnx}")

    elif use_har:
        if not head_onnx:
            raise SystemExit("HAR backbone requires --head_onnx.")
        # Head ONNX session (CPU EP ok)
        hsess = _ort_session(head_onnx, providers=["CPUExecutionProvider"])
        h_inputs  = [i.name for i in hsess.get_inputs()]
        h_outputs = [o.name for o in hsess.get_outputs()]
        # Loader must deliver float32 raw [0..255] for HAR helper
        x_dtype_for_loader = np.float32
        LOGGER.info(f"[MODEL] Using SPLIT HAR backbone={backbone_har}, head={head_onnx}")

    else:
        # Split ONNX
        if not (backbone_onnx_with_norm and head_onnx):
            raise SystemExit("Provide --full_onnx OR (--backbone_onnx_with_norm & --head_onnx) OR (--backbone_hef & --head_onnx) OR (--backbone_har & --head_onnx).")
        bsess = _ort_session(backbone_onnx_with_norm, providers)
        hsess = _ort_session(head_onnx, providers)
        b_inputs  = [i.name for i in bsess.get_inputs()]
        b_outputs = [o.name for o in bsess.get_outputs()]
        h_inputs  = [i.name for i in hsess.get_inputs()]
        h_outputs = [o.name for o in hsess.get_outputs()]
        b_in_dtype, _ = _onnx_first_io_dtypes(bsess)
        x_dtype_for_loader = b_in_dtype
        LOGGER.info(f"[MODEL] Using SPLIT ONNX backbone(norm)={backbone_onnx_with_norm}, head={head_onnx}  backbone_in_dtype={b_in_dtype}")

    # NumPy-only segmentor
    segmentor = RescaleSegmentor(target_size=imagesize, smoothing=4)

    # 3) 평가 per class
    for idx, (classname, samples) in enumerate(datasets.items(), start=1):
        LOGGER.info(f"[EVAL] {dataset_name_key}_{classname} ({idx}/{len(datasets)}) {datetime.now():%Y-%m-%d %H:%M:%S}")

        all_image_scores: List[float] = []
        labels_gt: List[int] = []

        seg_for_metric: List[np.ndarray] = []
        gt_for_metric:  List[np.ndarray] = []
        all_masks_vis:  List[np.ndarray] = []

        vis_dir = os.path.join(run_save_path, "vis", f"{dataset_name_key}_{classname}")
        if save_vis:
            os.makedirs(vis_dir, exist_ok=True)

        t0 = time.perf_counter()
        num_images = 0

        for batch in _yield_minibatches(samples, batch_size, imagesize, x_dtype_for_loader):
            inp    = batch["image"]            # (B,3,H,W) uint8|float32 in [0..255]
            labels = batch["is_anomaly"].tolist()
            gts    = batch["mask_gt"]
            labels_gt.extend(labels)

            if use_full:
                inp_cast = inp.astype(x_dtype_for_loader, copy=False)
                patch_scores = _run_full_safe(fsess, f_inputs, f_outputs, inp_cast)

            elif use_hef:
                # HEF backbone: run per-sample (batch=1)
                outs = []
                for bi in range(inp.shape[0]):
                    x1 = inp[bi:bi+1].astype(np.float32, copy=False)  # (1,3,H,W) float32 [0..255]
                    l2, l3 = _hailo_run_backbone_hef(backbone_hef, x1)
                    ps = _run_head_safe(hsess, h_inputs, h_outputs, l2, l3)[0]
                    outs.append(ps)
                patch_scores = np.stack(outs, axis=0).astype(np.float32, copy=False)

            elif use_har:
                # HAR backbone: run per-sample (batch=1)
                outs = []
                for bi in range(inp.shape[0]):
                    x1 = inp[bi:bi+1].astype(np.float32, copy=False)  # (1,3,H,W) float32 [0..255]
                    l2, l3 = _hailo_run_backbone_har(backbone_har, x1)
                    ps = _run_head_safe(hsess, h_inputs, h_outputs, l2, l3)[0]
                    outs.append(ps)
                patch_scores = np.stack(outs, axis=0).astype(np.float32, copy=False)

            else:
                # Split ONNX
                bfeed = {b_inputs[0]: inp.astype(x_dtype_for_loader, copy=False)}
                layer2, layer3 = bsess.run(b_outputs, bfeed)
                patch_scores = _run_head_safe(hsess, h_inputs, h_outputs, layer2, layer3)

            masks_rescaled = segmentor.convert_to_segmentation(patch_scores)  # list of (H,W)

            B = inp.shape[0]
            for bi in range(B):
                ps  = patch_scores[bi]
                seg = masks_rescaled[bi]
                img_score = float(ps.max())
                all_image_scores.append(img_score)
                all_masks_vis.append(seg)
                gt = gts[bi]
                if isinstance(gt, np.ndarray):
                    seg_for_metric.append(seg)
                    gt_for_metric.append(gt)

            num_images += B

        t1 = time.perf_counter()
        elapsed = t1 - t0
        latency = elapsed / max(num_images, 1)
        fps = num_images / max(elapsed, 1e-9)

        # 4) metrics
        scores_arr = np.asarray(all_image_scores, dtype=np.float32)
        image_scores = metrics.compute_imagewise_retrieval_metrics(scores_arr, labels_gt, path='eval_onnx')
        i_auroc = float(image_scores["auroc"])
        i_ap    = float(image_scores["ap"])

        if len(gt_for_metric) > 0:
            seg_arr = np.stack(seg_for_metric, axis=0).astype(np.float32)
            pixel_scores = metrics.compute_pixelwise_retrieval_metrics(seg_arr, gt_for_metric, path='eval_onnx')
            p_auroc = float(pixel_scores["auroc"])
            p_ap    = float(pixel_scores["ap"])
            try:
                masks_np = np.stack(gt_for_metric, axis=0).astype(np.uint8)
                p_pro = float(metrics.compute_pro(masks_np, seg_arr, fpr_max=0.3, per_image_minmax=True))
            except Exception as e:
                LOGGER.warning(f"[METRIC] PRO failed: {e}")
                p_pro = 0.0
        else:
            p_auroc = p_ap = p_pro = -1.0

        LOGGER.info(
            f"  image_auroc: {i_auroc:.4f}  image_ap: {i_ap:.4f}  "
            f"pixel_auroc: {p_auroc:.4f}  pixel_ap: {p_ap:.4f}  pixel_pro: {p_pro:.4f}"
        )
        LOGGER.info(f"  [SPEED] Elapsed={elapsed:.3f}s  Images={num_images}  Latency={latency*1000:.3f} ms/img  FPS={fps:.2f}")

        # 5) save vis
        if save_vis:
            for i, m in enumerate(all_masks_vis):
                mm = m.astype(np.float32)
                mmin, mmax = float(mm.min()), float(mm.max())
                if mmax - mmin > 1e-12:
                    mm = (mm - mmin) / (mmax - mmin)
                else:
                    mm = np.zeros_like(mm, dtype=np.float32)
                heat = (mm * 255.0).astype(np.uint8)
                heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
                cv2.imwrite(os.path.join(vis_dir, f"{i:04d}.png"), heat)

        # 6) save CSV
        utils.compute_and_store_final_results(
            results_path=run_save_path,
            results=[[i_auroc, i_ap, p_auroc, p_ap, p_pro, latency, fps]],
            column_names=["image_auroc", "image_ap", "pixel_auroc", "pixel_ap", "pixel_pro", "latency_sec", "fps"],
            row_names=[f"{dataset_name_key}_{classname}"],
        )

    LOGGER.info(f"[DONE] Results saved under: {run_save_path}")


if __name__ == "__main__":
    main()