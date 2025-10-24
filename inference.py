# inference_async_split.py
import argparse
import os
import numpy as np
import onnxruntime as ort

# HailoRT Python Async InferModel API (docs: hailo_platform)
from hailo_platform import VDevice, HailoSchedulingAlgorithm

# --- 이미지 전처리 (ImageNet norm, 288x288, NCHW float32) ---
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 3, 1, 1)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 3, 1, 1)

def load_image_bchw(path, size=(288, 288)):
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
    return x.astype(np.float32)

# --- 양자화 보정 ---
def get_scale_zp(qinfo):
    """여러 SDK 버전에서 scale/zp 명칭이 다를 수 있어 방어적으로 조회."""
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
    # dict-like fallback
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

# --- Hailo Async Inference (백본) ---
def run_backbone_hailo_async(hef_path: str, x_bchw_f32: np.ndarray):
    """
    VDevice + InferModel(Async)로 HEF 실행.
    입력은 float32(NCHW)이지만, 백엔드는 보통 입력 스트림이 uint8이므로 내부에서 정규화 전제가 없으면
    optimize 로그대로 CPU 쪽 양자화가 들어갈 수 있음. 여기서는 HEF 스케일링 메타데이터에 맞춰 자동 처리.
    출력: layer2, layer3를 float32(NCHW)로 반환.
    """
    if not os.path.isfile(hef_path):
        raise FileNotFoundError(hef_path)

    # vdevice 파라미터 (모델 스케줄러 ROUND_ROBIN 권장)
    params = VDevice.create_params()
    params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN

    with VDevice(params) as vdev:
        infer_model = vdev.create_infer_model(hef_path)
        with infer_model.configure() as cmodel:
            # --- 입력 바인딩 세팅 ---
            # 단일 입력 가정 (이름: parser에서 'input'으로 설정). 다중일 경우 cmodel.get_inputs() 사용해서 처리.
            in_binding = cmodel.create_bindings()

            # 입력 스트림(바이트 버퍼)은 일반적으로 uint8 형태. 스트림 shape은 보통 NHWC(or NCHW) 정수형.
            # HEF 스트림 shape을 읽고 거기에 맞춰 넣어야 하지만, cmodel가 내부 변환을 지원하는 경우가 많음.
            # 가장 안전하게는 스트림 shape를 확인 후 변환. 여기선 1x3x288x288 → uint8로 재양자화해서 공급.
            # (정확한 입력 양자화 scale/zp는 input().quant_info에서 획득)
            in_meta = infer_model.input()  # 단일 입력
            in_q = getattr(in_meta, "quant_info", None)
            in_scale, in_zp = get_scale_zp(in_q)

            # x_bchw_f32 → 양자화(uint8)
            x_q = np.clip(np.round(x_bchw_f32 / max(in_scale, 1e-12) + in_zp), 0, 255).astype(np.uint8)

            # Hailo 스트림 차원 순서가 NHWC일 수 있음 → shape 보고 맞추기
            # 메타 shape 조회:
            in_shape = tuple(in_meta.shape)  # e.g. (1, 288, 288, 3) 또는 (1, 3, 288, 288)
            if len(in_shape) == 4 and in_shape[1] in (288, 224) and in_shape[-1] == 3:
                # NHWC일 가능성 큼 → 변환
                x_q_feed = np.transpose(x_q, (0, 2, 3, 1)).copy()
            else:
                # NCHW로 가정
                x_q_feed = x_q

            # 입력 버퍼 세팅
            in_binding.input().set_buffer(x_q_feed)

            # --- 출력 바인딩 세팅 (다중 출력) ---
            # 단일 출력 API(docs 예시)만 보였지만, 실제론 출력이 2개.
            # configured_infer_model.create_bindings()가 단일 I/O만 래핑할 수 있어,
            # 최신 SDK에선 outputs() / output_tensors()가 제공됨. 방어적으로 처리:
            outs_meta = []
            try:
                outs_meta = list(infer_model.outputs())  # iterable
            except Exception:
                # 구버전: infer_model.output()만 있는 경우(단일). 하지만 여긴 2-output이므로 헬프 메시지.
                raise RuntimeError("InferModel shows single output API. "
                                   "Use a Hailo SDK version that exposes multi-output models.")

            # 바인딩 객체에서 개별 출력 버퍼에 접근:
            # 최신 SDK 기준: bindings.outputs() / bindings.output(<index>) 형태가 있음.
            # 여기선 outputs()가 있다고 가정하고, 없으면 output()만 제공 → 에러 처리.
            try:
                out_bindings = in_binding.outputs()
            except Exception:
                raise RuntimeError("Bindings object has no outputs(). Update Hailo SDK to a version "
                                   "that supports multi-output bindings.")

            # 출력 버퍼 준비: 각 output의 shape을 보고 NHWC/NCHW 추정
            # 나중에 읽은 뒤 dequantize+transpose 처리하므로 여기선 dtype만 맞추면 됨 (uint8 버퍼).
            for ob in out_bindings:
                shp = tuple(ob.shape())
                ob.set_buffer(np.empty(shp, dtype=np.uint8))

            # --- run (sync or async) ---
            # 튜토리얼에 맞춰 async → wait
            job = cmodel.run_async([in_binding])
            timeout_ms = 10_000
            job.wait(timeout_ms)

            # --- 결과 읽기(+ dequant) ---
            # 메타/바인딩 순서대로 결과 가져오기
            outputs_np = []
            for i, ob in enumerate(out_bindings):
                raw = ob.get_buffer()  # uint8 numpy
                ometa = outs_meta[i]
                oq = getattr(ometa, "quant_info", None)
                s, zp = get_scale_zp(oq)
                fp = dequantize(raw, s, zp)
                # NHWC → NCHW 정규화
                shp = tuple(raw.shape)
                if len(shp) == 4 and shp[-1] in (512, 1024):
                    # (1,H,W,C) → (1,C,H,W)
                    fp = np.transpose(fp, (0, 3, 1, 2)).copy()
                outputs_np.append(fp.astype(np.float32))

            # 두 출력(layer2, layer3) 식별 (채널 수로 분기)
            l2, l3 = None, None
            for arr in outputs_np:
                if arr.shape[1] == 512:
                    l2 = arr
                elif arr.shape[1] == 1024:
                    l3 = arr
            if l2 is None or l3 is None:
                # 스트림 이름 직접 매핑이 필요할 때는 아래처럼 이름으로 분기:
                # names = [o.name for o in infer_model.outputs()]
                # print(names) 후 layer2/layer3 위치를 고정해 매핑하세요.
                raise RuntimeError(f"Failed to identify outputs by channel size. Got {[a.shape for a in outputs_np]}")

            return l2, l3

# --- ONNX Head (CPU, onnxruntime) ---
def run_head_onnx(head_onnx_path, layer2, layer3):
    if not os.path.isfile(head_onnx_path):
        raise FileNotFoundError(head_onnx_path)
    sess = ort.InferenceSession(head_onnx_path, providers=["CPUExecutionProvider"])
    l2 = np.ascontiguousarray(layer2.astype(np.float32))
    l3 = np.ascontiguousarray(layer3.astype(np.float32))
    (out,) = sess.run(["patch_scores"], {"layer2": l2, "layer3": l3})
    return out  # (B, 36, 36)

def upsample_to_288(mask_bhw, size=(288, 288)):
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
    return np.stack(up, axis=0)

def main():
    ap = argparse.ArgumentParser("Hailo backbone (Async) + ONNX head split inference")
    ap.add_argument("--hef", required=True, help="Path to glass_backbone.hef")
    ap.add_argument("--head", required=True, help="Path to glass_head.onnx")
    ap.add_argument("--image", required=True, help="Input image")
    ap.add_argument("--save-mask", default="", help="Optional .npy path to save 288x288 heatmap")
    args = ap.parse_args()

    x = load_image_bchw(args.image, size=(288, 288))

    # 백본: Hailo Async
    l2, l3 = run_backbone_hailo_async(args.hef, x)
    print("layer2:", l2.shape, l2.dtype, "layer3:", l3.shape, l3.dtype)

    # 헤드: ONNX
    scores = run_head_onnx(args.head, l2, l3)
    print("patch_scores:", scores.shape, scores.dtype)

    # 업샘플
    mask_288 = upsample_to_288(scores, size=(288, 288))
    print("upsampled mask:", mask_288.shape)

    if args.save_mask:
        np.save(args.save_mask, mask_288.astype(np.float32))
        print("Saved:", args.save_mask)

if __name__ == "__main__":
    main()