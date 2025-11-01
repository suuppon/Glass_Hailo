#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare backbone outputs between a Hailo HEF (quantized) backbone and an ONNX backbone
that already includes ImageNet normalization internally ("with_norm"), with
robust layout normalization (FCR/NHWC/NHCW -> NCHW) and channel-permutation alignment
(Hungarian; fallback greedy) to maximize cosine similarity.

Usage example:
  python compare.py \
    --hef ckpt/glass_backbone_with_norm.hef \
    --backbone_onnx_with_norm ckpt/glass_backbone_with_norm.onnx \
    --image test.jpg \
    --imagesize 288 \
    --out_dir cmp_out
"""
import os, json, argparse, logging
import numpy as np
import onnxruntime as ort
import cv2
from typing import Tuple, Dict, Any

LOGGER = logging.getLogger("cmp_hef_onnx")

# ----------------------------
# ORT helpers
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

def _onnx_first_io_dtypes(sess) -> Tuple[np.dtype, np.dtype]:
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

# ----------------------------
# Hailo helpers (InferVStreams, safe shape)
# ----------------------------
def _safe_vstream_shape(vinfo):
    shp_attr = getattr(vinfo, "shape", None)
    if callable(shp_attr):
        try:
            shp = shp_attr()
            if isinstance(shp, (list, tuple)): return tuple(shp)
        except Exception:
            pass
    if isinstance(shp_attr, (list, tuple)):
        return tuple(shp_attr)
    n = getattr(vinfo, "batch", None)
    h = getattr(vinfo, "height", None)
    w = getattr(vinfo, "width", None)
    c = getattr(vinfo, "channels", None)
    if all(isinstance(x, int) for x in (h, w, c)):
        if not isinstance(n, int) or n <= 0: n = 1
        return (n, h, w, c)
    raise RuntimeError(f"Cannot read vstream shape safely from: {vinfo}")

def _hailo_run_backbone_hef(hef_path: str, x_bchw_f32: np.ndarray):
    """
    Run HEF backbone with Hailo InferVStreams (FLOAT32, quantized=False).
    Input : float32 (1,3,H,W) NCHW, RGB, [0..255], no normalization.
    Output: l2 (1,512,36,36) float32 NCHW, l3 (1,1024,18,18) float32 NCHW
    """
    import hailo_platform as hpf
    if not os.path.isfile(hef_path): raise FileNotFoundError(hef_path)

    hef = hpf.HEF(hef_path)
    with hpf.VDevice() as target:
        cfg = hpf.ConfigureParams.create_from_hef(hef=hef, interface=hpf.HailoStreamInterface.PCIe)
        ng = target.configure(hef, cfg)[0]
        ng_params = ng.create_params()

        in_infos  = hef.get_input_vstream_infos()
        out_infos = hef.get_output_vstream_infos()
        if not in_infos: raise RuntimeError("No input vstream infos found in HEF.")
        in_info = in_infos[0]

        in_params  = hpf.InputVStreamParams.make(ng, format_type=hpf.FormatType.FLOAT32)
        out_params = hpf.OutputVStreamParams.make(ng, format_type=hpf.FormatType.FLOAT32)

        # Input NHWC/NCHW 맞추기
        shp = _safe_vstream_shape(in_info)  # (N, H, W, C) or (N, C, H, W) depending on format
        if len(shp) == 4 and shp[-1] == 3:      # NHWC 기대
            x_feed = np.transpose(x_bchw_f32, (0, 2, 3, 1))
        else:
            x_feed = x_bchw_f32

        with hpf.InferVStreams(ng, in_params, out_params) as pipe:
            with ng.activate(ng_params):
                results = pipe.infer({in_info.name: x_feed.astype(np.float32, copy=False)})

        # 결과를 그대로 반환 (후에 강건 정규화 유틸로 NCHW 맞춤)
        return results, out_infos

# ----------------------------
# Layout & channel alignment utils
# ----------------------------
def _to_nchw_from_unknown(y: np.ndarray, H: int, W: int, C: int) -> np.ndarray:
    """
    Try to robustly convert y to (N, C, H, W), covering NHWC, (N,H*W,C), (H,W,C), (H*W,C), NCHW.
    """
    y = np.asarray(y)
    if y.ndim == 4:
        # (N, H, W, C)
        if y.shape[1:] == (H, W, C): return np.transpose(y, (0, 3, 1, 2))
        # (N, C, H, W)
        if y.shape[1:] == (C, H, W): return y
        # (N, H*W, C)
        if y.shape[1:] == (H*W, C):
            y = y.reshape(y.shape[0], H, W, C)
            return np.transpose(y, (0, 3, 1, 2))
    elif y.ndim == 3:
        # (H, W, C)
        if y.shape == (H, W, C):
            return np.transpose(y, (2, 0, 1))[None, ...]
        # (C, H, W)
        if y.shape == (C, H, W):
            return y[None, ...]
        # (H*W, C)
        if y.shape == (H*W, C):
            y = y.reshape(1, H, W, C)
            return np.transpose(y, (0, 3, 1, 2))
    elif y.ndim == 2 and y.shape == (H*W, C):
        y = y.reshape(1, H, W, C)
        return np.transpose(y, (0, 3, 1, 2))
    raise ValueError(f"Unexpected shape for output: {y.shape}, expected some form of (N,*,*,C={C}) or (N,{C},*,*)")

def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na = np.linalg.norm(a) + 1e-12
    nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))

def _centered_cosine(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64); b = b.reshape(-1).astype(np.float64)
    a -= a.mean(); b -= b.mean()
    na = np.linalg.norm(a) + 1e-12; nb = np.linalg.norm(b) + 1e-12
    return float(np.dot(a, b) / (na * nb))

def _affine_fit_cosine(target: np.ndarray, source: np.ndarray) -> Dict[str, float]:
    t = target.reshape(-1).astype(np.float64)
    s = source.reshape(-1).astype(np.float64)
    X = np.vstack([s, np.ones_like(s)]).T
    sol, *_ = np.linalg.lstsq(X, t, rcond=None)
    a, b = float(sol[0]), float(sol[1])
    s_adj = a * s + b
    mae = float(np.mean(np.abs(t - s_adj)))
    mse = float(np.mean((t - s_adj) ** 2))
    cos = _cosine(t, s_adj)
    return {"a": a, "b": b, "mae": mae, "mse": mse, "cosine": cos}

def _align_channels_by_cosine(ref_nchw: np.ndarray, tgt_nchw: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Align channels of tgt to ref to maximize mean per-channel cosine similarity.
    Returns (tgt_aligned, info).
    """
    assert ref_nchw.ndim == 4 and tgt_nchw.ndim == 4
    ref = ref_nchw[0]; tgt = tgt_nchw[0]        # (C,H,W)
    C = ref.shape[0]
    ref2d = ref.reshape(C, -1).astype(np.float64)
    tgt2d = tgt.reshape(C, -1).astype(np.float64)
    # normalize rows
    ref2d /= (np.linalg.norm(ref2d, axis=1, keepdims=True) + 1e-12)
    tgt2d /= (np.linalg.norm(tgt2d, axis=1, keepdims=True) + 1e-12)
    sim = ref2d @ tgt2d.T   # (C, C)

    # Hungarian if available
    perm = None
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-sim)  # maximize sum -> minimize negative
        # r is 0..C-1 in order; build perm so that tgt[perm[c]] -> ref[c]
        # We want index i in ref to match index perm[i] in tgt
        perm = np.empty(C, dtype=np.int32)
        perm[r] = c
    except Exception:
        # Greedy fallback
        perm = -np.ones(C, dtype=np.int32)
        used = set()
        for i in range(C):
            j = int(np.argmax(sim[i]))
            while j in used:
                sim[i, j] = -1e9
                j = int(np.argmax(sim[i]))
            perm[i] = j
            used.add(j)

    aligned = tgt_nchw[:, perm, :, :].copy()
    mean_pc = float(sim[np.arange(C), perm].mean())
    return aligned, {"perm": perm.tolist(), "mean_per_channel_cos": mean_pc}

# ----------------------------
# Image preprocessing
# ----------------------------
def _preprocess_rgb_nchw(img_bgr: np.ndarray, size: int, dtype: np.dtype) -> np.ndarray:
    img = cv2.resize(img_bgr, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    chw = np.transpose(img, (2, 0, 1))
    return chw.astype(dtype, copy=False)[None, ...]

# ----------------------------
# Metrics helpers
# ----------------------------
def _tensor_stats(x: np.ndarray) -> Dict[str, Any]:
    x = x.astype(np.float64)
    return {"shape": list(x.shape),
            "min": float(np.min(x)),
            "max": float(np.max(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x))}

def _diff_metrics(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    a64 = a.astype(np.float64); b64 = b.astype(np.float64)
    diff = a64 - b64
    mae = float(np.mean(np.abs(diff)))
    mse = float(np.mean(diff ** 2))
    max_abs = float(np.max(np.abs(diff)))
    cos = _cosine(a64, b64)
    ccos = _centered_cosine(a64, b64)
    return {"mae": mae, "mse": mse, "max_abs": max_abs, "cosine": cos, "centered_cosine": ccos}

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hef", required=True, help="Path to glass_backbone_with_norm.hef")
    ap.add_argument("--backbone_onnx_with_norm", required=True, help="Path to glass_backbone_with_norm.onnx")
    ap.add_argument("--image", default="test.jpg", help="Path to input image (RGB content)")
    ap.add_argument("--imagesize", type=int, default=288)
    ap.add_argument("--out_dir", default="cmp_out")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    # Load image
    bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if bgr is None: raise SystemExit(f"Failed to read image: {args.image}")

    # --- ONNX backbone ---
    providers = ["CPUExecutionProvider"]  # 고정 권장 (재현성)
    bsess = _ort_session(args.backbone_onnx_with_norm, providers)
    b_inputs  = [i.name for i in bsess.get_inputs()]
    b_outputs = [o.name for o in bsess.get_outputs()]
    in_dtype, _ = _onnx_first_io_dtypes(bsess)

    x_onnx = _preprocess_rgb_nchw(bgr, args.imagesize, in_dtype)  # (1,3,H,W)
    
    # Save model inputs before inference
    np.save(os.path.join(args.out_dir, "input_onnx.npy"), x_onnx)
    LOGGER.info(f"[INPUT] Saved ONNX input: shape={x_onnx.shape}, dtype={x_onnx.dtype}")
    
    onnx_outs = bsess.run(b_outputs, {b_inputs[0]: x_onnx})

    onnx_l2, onnx_l3 = None, None
    for arr in onnx_outs:
        a = np.asarray(arr)
        if a.ndim == 3: a = a[None, ...]
        if a.ndim != 4: continue
        _, C, H, W = a.shape
        if   (C, H, W) == (512, 36, 36): onnx_l2 = a.astype(np.float32, copy=False)
        elif (C, H, W) == (1024, 18, 18): onnx_l3 = a.astype(np.float32, copy=False)
    if onnx_l2 is None or onnx_l3 is None:
        shapes = [np.asarray(t).shape for t in onnx_outs]
        raise SystemExit(f"ONNX backbone outputs do not include expected shapes. outputs={shapes}")

    # --- HEF backbone (raw results dict + out_infos) ---
    x_hef = _preprocess_rgb_nchw(bgr, args.imagesize, np.float32)
    
    # Save model inputs before inference
    np.save(os.path.join(args.out_dir, "input_hef.npy"), x_hef)
    LOGGER.info(f"[INPUT] Saved HEF input: shape={x_hef.shape}, dtype={x_hef.dtype}")
    
    hef_results, hef_out_infos = _hailo_run_backbone_hef(args.hef, x_hef)

    # Robust layout normalization to NCHW
    # conv24: (36,36,512), conv43: (18,18,1024)  (names may vary, we infer by dims)
    hef_l2 = hef_l3 = None
    for oi in hef_out_infos:
        y = hef_results[oi.name]
        # Try typical combos for both heads
        try:
            y_nchw = _to_nchw_from_unknown(y, 36, 36, 512)
            if y_nchw.shape[1:] == (512, 36, 36): hef_l2 = y_nchw.astype(np.float32, copy=False); continue
        except Exception:
            pass
        try:
            y_nchw = _to_nchw_from_unknown(y, 18, 18, 1024)
            if y_nchw.shape[1:] == (1024, 18, 18): hef_l3 = y_nchw.astype(np.float32, copy=False); continue
        except Exception:
            pass

    if hef_l2 is None or hef_l3 is None:
        detail = {oi.name: list(np.asarray(hef_results[oi.name]).shape) for oi in hef_out_infos}
        raise SystemExit(f"Failed to robustly normalize HEF outputs to NCHW. seen={detail}")

    # --- Baseline diffs (before any alignment) ---
    diff_l2 = _diff_metrics(onnx_l2, hef_l2)
    diff_l3 = _diff_metrics(onnx_l3, hef_l3)

    # --- Channel-permutation alignment (maximize cosine) ---
    hef_l2_aligned, info_l2 = _align_channels_by_cosine(onnx_l2, hef_l2)
    hef_l3_aligned, info_l3 = _align_channels_by_cosine(onnx_l3, hef_l3)

    diff_l2_aligned = _diff_metrics(onnx_l2, hef_l2_aligned)
    diff_l3_aligned = _diff_metrics(onnx_l3, hef_l3_aligned)

    # --- Affine-fit cosine (optional diagnostic) ---
    aff_l2 = _affine_fit_cosine(onnx_l2, hef_l2_aligned)
    aff_l3 = _affine_fit_cosine(onnx_l3, hef_l3_aligned)

    # --- Save artifacts ---
    np.save(os.path.join(args.out_dir, "onnx_l2.npy"), onnx_l2)
    np.save(os.path.join(args.out_dir, "onnx_l3.npy"), onnx_l3)
    np.save(os.path.join(args.out_dir, "hef_l2_raw.npy"), hef_l2)
    np.save(os.path.join(args.out_dir, "hef_l3_raw.npy"), hef_l3)
    np.save(os.path.join(args.out_dir, "hef_l2_aligned.npy"), hef_l2_aligned)
    np.save(os.path.join(args.out_dir, "hef_l3_aligned.npy"), hef_l3_aligned)
    np.save(os.path.join(args.out_dir, "perm_l2.npy"), np.array(info_l2["perm"], dtype=np.int32))
    np.save(os.path.join(args.out_dir, "perm_l3.npy"), np.array(info_l3["perm"], dtype=np.int32))

    report: Dict[str, Any] = {
        "image": os.path.abspath(args.image),
        "hef": os.path.abspath(args.hef),
        "onnx": os.path.abspath(args.backbone_onnx_with_norm),
        "imagesize": args.imagesize,
        "providers": ["CPUExecutionProvider"],
        "stats": {
            "onnx_l2": _tensor_stats(onnx_l2),
            "onnx_l3": _tensor_stats(onnx_l3),
            "hef_l2_raw": _tensor_stats(hef_l2),
            "hef_l3_raw": _tensor_stats(hef_l3),
            "hef_l2_aligned": _tensor_stats(hef_l2_aligned),
            "hef_l3_aligned": _tensor_stats(hef_l3_aligned),
        },
        "diff_raw": {"l2": diff_l2, "l3": diff_l3},
        "diff_aligned": {"l2": diff_l2_aligned, "l3": diff_l3_aligned},
        "channel_alignment": {
            "l2": {"mean_per_channel_cos": info_l2["mean_per_channel_cos"], "perm_saved": "perm_l2.npy"},
            "l3": {"mean_per_channel_cos": info_l3["mean_per_channel_cos"], "perm_saved": "perm_l3.npy"},
        },
        "affine_fit_on_aligned": {"l2": aff_l2, "l3": aff_l3},
        "files": {
            "input_onnx.npy": "ONNX model input (before inference)",
            "input_hef.npy": "HEF model input (before inference)",
            "onnx_l2.npy": "pre-aligned ONNX L2",
            "onnx_l3.npy": "pre-aligned ONNX L3",
            "hef_l2_raw.npy": "raw HEF L2 (NCHW normalized)",
            "hef_l3_raw.npy": "raw HEF L3 (NCHW normalized)",
            "hef_l2_aligned.npy": "channel-aligned HEF L2",
            "hef_l3_aligned.npy": "channel-aligned HEF L3",
            "perm_l2.npy": "argmax-cosine channel permutation for L2",
            "perm_l3.npy": "argmax-cosine channel permutation for L3",
        }
    }

    with open(os.path.join(args.out_dir, "report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    # Pretty print summary
    print("\n==== Comparison Summary ====")
    print("Inputs:")
    print(f"  image: {report['image']}")
    print(f"  HEF  : {report['hef']}")
    print(f"  ONNX : {report['onnx']}")
    print("Shapes:")
    print(f"  onnx_l2: {report['stats']['onnx_l2']['shape']}  hef_l2_raw: {report['stats']['hef_l2_raw']['shape']}")
    print(f"  onnx_l3: {report['stats']['onnx_l3']['shape']}  hef_l3_raw: {report['stats']['hef_l3_raw']['shape']}")
    print("Raw diffs:")
    print("  L2:", report['diff_raw']['l2'])
    print("  L3:", report['diff_raw']['l3'])
    print("Channel alignment:")
    print("  L2 mean per-channel cosine:", report['channel_alignment']['l2']['mean_per_channel_cos'])
    print("  L3 mean per-channel cosine:", report['channel_alignment']['l3']['mean_per_channel_cos'])
    print("Aligned diffs:")
    print("  L2:", report['diff_aligned']['l2'])
    print("  L3:", report['diff_aligned']['l3'])
    print("Affine-fit on aligned (a·x+b):")
    print("  L2:", report['affine_fit_on_aligned']['l2'])
    print("  L3:", report['affine_fit_on_aligned']['l3'])
    print(f"\nSaved: {os.path.abspath(args.out_dir)}  (npy tensors, perms, report.json)")

if __name__ == "__main__":
    main()