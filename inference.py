#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import onnxruntime as ort

# HailoRT Python Async InferModel API
from hailo_platform import VDevice, HailoSchedulingAlgorithm

# --- 이미지 전처리 (ImageNet norm, 288x288, NCHW float32) ---
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)


def load_image_bchw(path, size=(288, 288)):
    """RGB load -> resize -> to [0,1] -> normalize -> NCHW float32, shape (1,3,H,W)"""
    try:
        import cv2
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise RuntimeError(f"cv2.imread failed: {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
    except Exception:
        from PIL import Image
        img = Image.open(path).convert("RGB").resize(size, Image.BILINEAR)
        img = np.asarray(img).astype(np.float32) / 255.0

    x = np.transpose(img, (2, 0, 1))[None, ...]  # (1,3,H,W)
    x = (x - IMAGENET_MEAN) / IMAGENET_STD
    return x.astype(np.float32, copy=False)


# --- 양자화 보정 유틸 ---
def get_scale_zp(qinfo):
    """SDK 버전마다 다른 quant info 키를 방어적으로 처리."""
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


def _get_shape_from_meta(meta):
    """여러 SDK 객체 형태에서 안전하게 shape 튜플을 구한다."""
    for attr in ("shape", "dims", "dimensions"):
        v = getattr(meta, attr, None)
        if v is not None:
            v = v() if callable(v) else v
            try:
                return tuple(v)
            except Exception:
                pass
    # 개별 필드 폴백
    b = getattr(meta, "batch", None)
    h = getattr(meta, "height", None)
    w = getattr(meta, "width", None)
    c = getattr(meta, "channels", None)
    if all(v is not None for v in (b, h, w, c)):
        return (int(b), int(h), int(w), int(c))
    raise RuntimeError(f"Cannot infer output shape from meta: {meta}")


# --- Hailo Async Inference (백본) ---
def run_backbone_hailo_async(hef_path: str, x_bchw_f32: np.ndarray):
    """
    VDevice + InferModel(Async)로 HEF 실행.
    입력: float32 (1,3,288,288) NCHW
    출력: layer2 (1,512,36,36) float32 NCHW, layer3 (1,1024,18,18) float32 NCHW
    """
    if not os.path.isfile(hef_path):
        raise FileNotFoundError(hef_path)

    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    with VDevice(params) as vdev:
        infer_model = vdev.create_infer_model(hef_path)
        with infer_model.configure() as cmodel:
            bindings = cmodel.create_bindings()

            # --- 입력 세팅 ---
            in_meta = getattr(infer_model, "input", None)
            if callable(in_meta):
                in_meta = in_meta()
            if in_meta is None:
                raise RuntimeError("Cannot obtain input meta from infer_model.")

            in_scale, in_zp = get_scale_zp(getattr(in_meta, "quant_info", None))

            # float32 → uint8 양자화
            x_q = np.clip(np.round(x_bchw_f32 / max(in_scale, 1e-12) + in_zp), 0, 255).astype(np.uint8)

            # 스트림 레이아웃 추정
            in_shape = getattr(in_meta, "shape", None)
            if callable(in_shape):
                in_shape = tuple(in_shape())
            elif in_shape is not None:
                in_shape = tuple(in_shape)
            else:
                in_shape = None

            if in_shape is not None and len(in_shape) == 4 and in_shape[1] in (288, 224) and in_shape[-1] == 3:
                # NHWC 스트림 → transpose
                x_q_feed = np.transpose(x_q, (0, 2, 3, 1))
            else:
                # NCHW 스트림
                x_q_feed = x_q

            # C-Contiguous & writable 보장
            x_q_feed = np.ascontiguousarray(x_q_feed, dtype=np.uint8)
            x_q_feed.setflags(write=True)
            bindings.input().set_buffer(x_q_feed)

            # --- 출력 메타 수집 (함수/리스트 호환) ---
            outs_attr = getattr(infer_model, 'outputs', None)
            if callable(outs_attr):
                outs_meta = list(outs_attr())
            elif isinstance(outs_attr, (list, tuple)):
                outs_meta = list(outs_attr)
            else:
                # 단일 출력 폴백
                get_single = getattr(infer_model, 'output', None)
                if callable(get_single):
                    outs_meta = [get_single()]
                else:
                    raise RuntimeError("Cannot discover model outputs from infer_model (no outputs/output).")

            # --- 출력 바인딩 취득 (outputs() 또는 output(name) 호환) ---
            out_bindings = None
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
                if not out_bindings:
                    raise RuntimeError("Bindings exposes neither outputs() nor output(name).")

            # --- 출력 버퍼 shape을 메타에서 얻어 C-order로 할당 ---
            for i, ob in enumerate(out_bindings):
                shp = _get_shape_from_meta(outs_meta[i])  # e.g. (1,36,36,512) or (1,512,36,36)
                ob.set_buffer(np.empty(shp, dtype=np.uint8, order="C"))

            # --- async 실행 ---
            job = cmodel.run_async([bindings])
            job.wait(10_000)

            # --- 결과 취득 & dequant & 레이아웃 정규화 ---
            outputs_np = []
            for i, ob in enumerate(out_bindings):
                raw = ob.get_buffer()  # uint8 ndarray
                qinfo = getattr(outs_meta[i], "quant_info", None)
                s, zp = get_scale_zp(qinfo)
                fp = dequantize(raw, s, zp)  # float32

                shp = tuple(fp.shape)
                # NHWC → NCHW 정규화 (3D/4D 모두 처리)
                if len(shp) == 4 and shp[-1] in (512, 1024):         # (1,H,W,C)
                    fp = np.transpose(fp, (0, 3, 1, 2)).copy()
                elif len(shp) == 3 and shp[-1] in (512, 1024):       # (H,W,C)
                    fp = np.transpose(fp, (2, 0, 1))[None, ...].copy()
                # else: 이미 (1,C,H,W) 혹은 (C,H,W)면 그대로
                outputs_np.append(fp.astype(np.float32, copy=False))

            # --- 두 출력(layer2/layer3) 식별 ---
            l2, l3 = None, None
            for arr in outputs_np:
                # arr: (1,C,H,W)
                if arr.ndim == 4:
                    _, C, H, W = arr.shape
                elif arr.ndim == 3:
                    C, H, W = arr.shape
                    arr = arr[None, ...]
                else:
                    continue

                if (C, H, W) == (512, 36, 36):
                    l2 = arr
                elif (C, H, W) == (1024, 18, 18):
                    l3 = arr

            if l2 is None or l3 is None:
                # 디버깅용 힌트
                names = [getattr(m, "name", None) for m in outs_meta]
                shapes = [tuple(o.shape) if hasattr(o, "shape") else None for o in outputs_np]
                raise RuntimeError(f"Failed to identify L2/L3. outs={len(outputs_np)} names={names} shapes={shapes}")

            return l2, l3


# --- ONNX Head (CPU, onnxruntime) ---
def run_head_onnx(head_onnx_path, layer2, layer3):
    if not os.path.isfile(head_onnx_path):
        raise FileNotFoundError(head_onnx_path)
    sess = ort.InferenceSession(head_onnx_path, providers=["CPUExecutionProvider"])
    # 이름 자동 조회 (호환성)
    in0 = sess.get_inputs()[0].name
    in1 = sess.get_inputs()[1].name
    out0 = sess.get_outputs()[0].name

    l2 = np.ascontiguousarray(layer2.astype(np.float32, copy=False))
    l3 = np.ascontiguousarray(layer3.astype(np.float32, copy=False))
    (out,) = sess.run([out0], {in0: l2, in1: l3})
    out = np.asarray(out)
    if out.ndim == 2:  # (H',W') → (1,H',W')
        out = out[None, ...]
    return out.astype(np.float32, copy=False)  # (B,H',W')


def upsample_to_288(mask_bhw, size=(288, 288)):
    """bilinear upsample to (size,size) per-item"""
    B, H, W = mask_bhw.shape
    try:
        import cv2
        up = [cv2.resize(mask_bhw[i], size, interpolation=cv2.INTER_LINEAR) for i in range(B)]
    except Exception:
        from PIL import Image
        up = []
        for i in range(B):
            m = Image.fromarray(mask_bhw[i].astype(np.float32))
            m = m.resize(size, Image.BILINEAR)
            up.append(np.asarray(m).astype(np.float32))
    return np.stack(up, axis=0).astype(np.float32, copy=False)


def main():
    ap = argparse.ArgumentParser("Hailo backbone (Async) + ONNX head split inference")
    ap.add_argument("--hef", required=True, help="Path to glass_backbone.hef")
    ap.add_argument("--head", required=True, help="Path to glass_head.onnx")
    ap.add_argument("--image", required=True, help="Input image")
    ap.add_argument("--save-mask", default="", help="Optional .npy path to save 288x288 heatmap")
    args = ap.parse_args()

    # 입력 전처리
    x = load_image_bchw(args.image, size=(288, 288))

    # 백본(Hailo)
    l2, l3 = run_backbone_hailo_async(args.hef, x)
    print("layer2:", l2.shape, l2.dtype, "layer3:", l3.shape, l3.dtype)

    # 헤드(ONNXRuntime CPU)
    scores = run_head_onnx(args.head, l2, l3)
    print("patch_scores:", scores.shape, scores.dtype)

    # 업샘플 (선택)
    mask_288 = upsample_to_288(scores, size=(288, 288))
    print("upsampled mask:", mask_288.shape)

    if args.save_mask:
        np.save(args.save_mask, mask_288.astype(np.float32, copy=False))
        print("Saved:", args.save_mask)


if __name__ == "__main__":
    main()