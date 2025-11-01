#!/usr/bin/env bash
set -euo pipefail

# ============ GPU Configuration ============
# GPU 사용 설정 (RTX 5090 최적화)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"  # GPU 0과 1 사용
export TF_CPP_MIN_LOG_LEVEL="1"

# RTX 5090 compute capability 12.0 호환성을 위한 TensorFlow 설정
export TF_FORCE_GPU_ALLOW_GROWTH="true"
export TF_ENABLE_GPU_GARBAGE_COLLECTION="false"

# hailo_model_optimization이 TensorFlow를 초기화할 때 발생하는 
# CUDA handle 오류를 우회하기 위해 미리 TensorFlow GPU 환경 초기화
python3 <<'PYEOF'
import os
import sys

# CUDA_VISIBLE_DEVICES 설정
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

try:
    import tensorflow as tf
    # GPU 설정
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # GPU 메모리 증가 허용
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"TensorFlow GPU configured: {len(gpus)} GPU(s)")
    else:
        print("No GPU available for TensorFlow")
except Exception as e:
    print(f"TensorFlow initialization warning: {e}")
    sys.exit(0)  # 에러 무시하고 계속 진행
PYEOF

# ============ Config (env override OK) ============
HW_ARCH="${HW_ARCH:-hailo8}"          # hailo8 | hailo8r | hailo8l
INPUT_FORMAT="${INPUT_FORMAT:-NCHW}"
BATCH="${BATCH:-1}"
CHANNELS="${CHANNELS:-3}"
IMAGESIZE="${IMAGESIZE:-288}"

# calibration: .npy 파일 or .npy 디렉토리
CALIB_PATH="${CALIB_PATH:-/local/shared_with_docker/calib_npy_288}"

# quantization control
FULL_PRECISION_ONLY="${FULL_PRECISION_ONLY:-no}"  # yes | no

# ============ Args ============
if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <model.onnx>"
  echo "Env: HW_ARCH=${HW_ARCH} INPUT_FORMAT=${INPUT_FORMAT} BATCH=${BATCH} CHANNELS=${CHANNELS} IMAGESIZE=${IMAGESIZE}"
  echo "     CALIB_PATH=${CALIB_PATH}"
  echo "     FULL_PRECISION_ONLY=${FULL_PRECISION_ONLY} (yes|no, default: no)"
  echo ""
  echo "Example with full-precision (no quantization):"
  echo "  FULL_PRECISION_ONLY=yes $0 model.onnx"
  exit 1
fi

ONNX_PATH="$(realpath "$1")"
if [[ ! -f "${ONNX_PATH}" ]]; then
  echo "[ERR] ONNX not found: ${ONNX_PATH}" >&2
  exit 1
fi

ONNX_DIR="$(dirname "${ONNX_PATH}")"
BASE_NAME="$(basename "${ONNX_PATH}" .onnx)"

HAR_NATIVE="${ONNX_DIR}/${BASE_NAME}.har"
PARSE_JSON="${ONNX_DIR}/${BASE_NAME}_parse_report.json"
AUG_ONNX="${ONNX_DIR}/${BASE_NAME}_augmented.onnx"

HAR_OPT="${ONNX_DIR}/${BASE_NAME}_optimized.har"
HAR_COMP="${ONNX_DIR}/${BASE_NAME}_compiled.har"
HEF_DIR="${ONNX_DIR}/hef/${BASE_NAME}"

mkdir -p "${HEF_DIR}"

# ============ Pick calib .npy ============
CALIB_FILE=""
if [[ -d "${CALIB_PATH}" ]]; then
  CALIB_FILE="$(find "${CALIB_PATH}" -maxdepth 1 -type f -name '*.npy' | head -n1 || true)"
  if [[ -z "${CALIB_FILE}" ]]; then
    echo "[ERR] No .npy under ${CALIB_PATH}" >&2; exit 1
  fi
elif [[ -f "${CALIB_PATH}" ]]; then
  CALIB_FILE="${CALIB_PATH}"
else
  echo "[ERR] CALIB_PATH invalid: ${CALIB_PATH}" >&2; exit 1
fi

# ============ Inspect ONNX I/O names (fixed heredoc) ============
mapfile -t _IO <<< "$(python3 - "$ONNX_PATH" <<'PY'
import sys, onnx
p = sys.argv[1]
m = onnx.load(p)
ins  = [i.name for i in m.graph.input]
outs = [o.name for o in m.graph.output]
print(ins[0] if ins else "input")            # line 1: input name
print(" ".join(outs) if outs else "")        # line 2: outputs space-separated
PY
)"
INPUT_NAME="${_IO[0]}"
OUTPUT_NAMES="${_IO[1]}"

if [[ -z "${INPUT_NAME}" ]]; then
  echo "[ERR] Failed to read ONNX input name" >&2; exit 1
fi
if [[ -z "${OUTPUT_NAMES}" ]]; then
  echo "[ERR] Failed to read ONNX output names" >&2; exit 1
fi

echo ">> ONNX: ${ONNX_PATH}"
echo "   Input  : ${INPUT_NAME}"
echo "   Outputs: ${OUTPUT_NAMES}"

TENSOR_SPEC="${INPUT_NAME}=[${BATCH},${CHANNELS},${IMAGESIZE},${IMAGESIZE}]"
INPUT_FMT_SPEC="${INPUT_NAME}=${INPUT_FORMAT}"

# ============ 1) Parser ============
echo -e "\n=== [1/3] Parsing to HAR ===\n"
hailo parser onnx "${ONNX_PATH}" \
  --net-name "${BASE_NAME}" \
  --har-path "${HAR_NATIVE}" \
  --tensor-shapes "${TENSOR_SPEC}" \
  --input-format "${INPUT_FMT_SPEC}" \
  --start-node-names "${INPUT_NAME}" \
  --end-node-names l2_out l3_out \
  --hw-arch "${HW_ARCH}" \
  -y \
  --parsing-report-path "${PARSE_JSON}" \
  --augmented-path "${AUG_ONNX}"

echo "[OK] HAR: ${HAR_NATIVE}"
echo "[OK] Parse report: ${PARSE_JSON}"
echo "[OK] Augmented ONNX: ${AUG_ONNX}"

# ============ 2) Optimize ============
if [[ "${FULL_PRECISION_ONLY}" == "yes" ]]; then
  echo -e "\n=== [2/3] Optimizing (full-precision only, no quantization) ===\n"
else
  echo -e "\n=== [2/3] Optimizing (quantization) ===\n"
fi

# Build optimize command
OPTIMIZE_ARGS=(
  --hw-arch "${HW_ARCH}"
  --calib-set-path "${CALIB_FILE}"
  --output-har-path "${HAR_OPT}"
)

if [[ "${FULL_PRECISION_ONLY}" == "yes" ]]; then
  OPTIMIZE_ARGS+=(--full-precision-only)
fi

hailo optimize "${HAR_NATIVE}" "${OPTIMIZE_ARGS[@]}"

echo "[OK] Optimized HAR: ${HAR_OPT}"

# ============ 3) Compile ============
if [[ "${FULL_PRECISION_ONLY}" == "yes" ]]; then
  echo -e "\n=== [3/3] Skipping compilation (full-precision mode) ===\n"
  echo "[INFO] Full-precision HAR cannot be compiled for Hailo hardware."
  echo "[INFO] This HAR is for CPU/GPU inference only."
  echo "[OK] Optimized HAR (full-precision): ${HAR_OPT}"
else
  echo -e "\n=== [3/3] Compiling to HEF ===\n"
  hailo compiler "${HAR_OPT}" \
    --hw-arch "${HW_ARCH}" \
    --output-dir "${HEF_DIR}" \
    --output-har-path "${HAR_COMP}"
  
  echo "[OK] Compiled HAR: ${HAR_COMP}"
  echo "[OK] HEF dir: ${HEF_DIR}"
fi
echo -e "\n=== Done. ===\n"