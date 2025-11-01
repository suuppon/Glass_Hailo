# Glass_Hailo

Anomaly detection with GLASS model on Hailo8 AI accelerator.

## Overview

This project provides tools for:
1. **Model Conversion**: ONNX → HEF (Hailo Executable Format)
2. **Inference**: Hailo-accelerated backbone + ONNX head
3. **Evaluation**: Full dataset benchmarking

## Prerequisites

### For Model Conversion (ONNX → HEF)
- Docker (for running Hailo converter)
- Hailo8 AI Software Suite SDK

### For Inference
- **HailoRT installed** on the target device (e.g., Raspberry Pi 5, Hailo PCIe)
- Hailo8 hardware or compatible accelerator
- Python 3.8+ with dependencies

> **⚠️ Important**: Hailo inference must run on a device with **HailoRT** installed. Please refer to the [Hailo Developer Zone](https://hailo.ai/developer-zone/) for HailoRT installation instructions for your platform.

## Setup

### 1. Hailo8 AI Software Suite

Place `hailo8_ai_sw_suite_2025-10.tar.gz` in the project root directory.

> **Note**: Please refer to the [Hailo Developer Zone](https://hailo.ai/developer-zone/) for the latest setup instructions.

## Usage

### Model Conversion: ONNX → HEF

#### Step 1: Start Docker Container

```bash
./run_hailo.sh --override  # Create new container (or use --resume for existing)
```

#### Step 2: Enter Container Shell

```bash
docker exec -it hailo8_ai_sw_suite_2025-10_container /bin/bash
```

#### Step 3: Navigate to Shared Directory

```bash
cd /local/shared_with_docker
```

#### Step 4 (Optional): Create Calibration Dataset

If you need quantization calibration:

```bash
python3 make_calib_npy.py \
  --src-dir /path/to/calibration/images \
  --out calib_npy_288 \
  --imagesize 288 \
  --max-samples 219 \
  --ext "*.png" "*.jpg" "*.jpeg" \
  --recursive
```

**Arguments**:
- `--src-dir`: Directory containing calibration images
- `--out`: Output NPY path (saved in `shared_with_docker/`)
- `--imagesize`: Image size (default: 288)
- `--max-samples`: Number of samples (default: 219)
- `--ext`: Image extensions (default: png, jpg, jpeg, bmp, tif, tiff)
- `--recursive`: Search recursively

#### Step 5: Build HEF

```bash
./build_hailo.sh /path/to/model.onnx
```

**Environment Variables** (optional):
```bash
HW_ARCH=hailo8              # hailo8 | hailo8r | hailo8l
INPUT_FORMAT=NCHW           # NCHW | NHWC
BATCH=1
CHANNELS=3
IMAGESIZE=288
CALIB_PATH=/local/shared_with_docker/calib_npy_288
FULL_PRECISION_ONLY=no      # yes | no
```

**Example**:
```bash
CALIB_PATH=/local/shared_with_docker/calib_npy_288 ./build_hailo.sh /local/shared_with_docker/glass_backbone.onnx
```

The conversion process executes:
1. **Parse** → Hailo Archive (.har)
2. **Optimize** → Quantized HAR (requires calibration)
3. **Compile** → HEF executable

**Output**: `{model_dir}/hef/{model_name}/` directory containing the HEF file.

**Note**: Conversion takes several minutes depending on model complexity.

### Inference

#### Single Image Inference

```bash
python inference.py \
  --hef /path/to/glass_backbone.hef \
  --head /path/to/glass_head.onnx \
  --image /path/to/image.jpg \
  --save-mask output_heatmap.npy  # optional
```

**Requirements**:
- `data/` directory with test images
- **Device with HailoRT installed** (see Prerequisites section)
- Hailo hardware connected

#### Batch Evaluation

```bash
python eval_hailo_with_norm.py \
  --data_path /path/to/data \
  --dataset mvtec \
  --subdatasets bottle \
  --subdatasets cable \
  --backbone_hef /path/to/glass_backbone.hef \
  --head_onnx /path/to/glass_head.onnx \
  --save_vis
```

**Supported Datasets**:
- `mvtec`: MVTec AD format
- `visa`: VisA format
- `mpdd`: MPDD format
- `wfdd`: WFDD format

**Key Arguments**:
- `--data_path`: Dataset root directory
- `--dataset`: Dataset type
- `--subdatasets`: Class names (repeat `-d` for multiple)
- `--backbone_hef`: HEF backbone path
- `--head_onnx`: ONNX head path
- `--save_vis`: Save heatmap visualizations

**Dataset Structure (MVTec)**:
```
data/
  bottle/
    test/
      good/
        *.png
      defect_001/
        *.png
    ground_truth/
      defect_001/
        *_mask.png
```

**Alternative Modes**:

1. **Full ONNX** (no Hailo):
```bash
python eval_hailo_with_norm.py \
  --data_path ... \
  --full_onnx /path/to/full_model.onnx \
  --subdatasets bottle
```

2. **Split ONNX** (backbone + head):
```bash
python eval_hailo_with_norm.py \
  --data_path ... \
  --backbone_onnx_with_norm /path/to/backbone.onnx \
  --head_onnx /path/to/head.onnx \
  --subdatasets bottle
```

## Directory Structure

```
Glass_Hailo/
├── README.md
├── run_hailo.sh                      # Docker container launcher
├── inference.py                       # Single image inference
├── eval_hailo_with_norm.py           # Batch evaluation
├── eval.sh                            # Evaluation wrapper script
├── shared_with_docker/
│   ├── make_calib_npy.py             # Calibration dataset generator
│   └── build_hailo.sh                # ONNX → HEF converter
├── hailo8_ai_sw_suite_2025-10.tar.gz # Hailo SDK (required)
└── data/                              # Dataset directory
```

## Outputs

### Conversion Outputs
- `{model_dir}/hef/{model_name}/` - HEF executable
- `{model_dir}/{model_name}_parse_report.json` - Parse report
- `{model_dir}/{model_name}_augmented.onnx` - Augmented ONNX
- `{model_dir}/{model_name}_optimized.har` - Optimized HAR
- `{model_dir}/{model_name}_compiled.har` - Compiled HAR

### Evaluation Outputs
- `results_eval_onnx/project/group/{run_name}/` - Metrics CSV
- `results_eval_onnx/project/group/{run_name}/vis/` - Visualizations (if `--save_vis`)

## Troubleshooting

### Docker Issues
- If the container won't start: `./run_hailo.sh --override --gpu`
- For GPU support: Use `--gpu` flag
- To resume existing container: `./run_hailo.sh --resume`

### Conversion Issues
- **Missing calibration**: Create NPY with `make_calib_npy.py`
- **Format errors**: Check input ONNX with `onnx.checker.check_model()`
- **Shape mismatches**: Verify `IMAGESIZE` matches training

### Runtime Issues
- **HEF not found**: Check path in conversion output
- **VDevice errors**: 
  - Ensure Hailo hardware is connected
  - Verify HailoRT is installed and configured properly
  - Refer to [Hailo Developer Zone](https://hailo.ai/developer-zone/)
- **"hailo_platform not found"**: Install HailoRT on target device
- **Head ONNX errors**: Verify layer2/layer3 output shapes

## License

See LICENSE file for details.

## References

- [Hailo Developer Zone](https://hailo.ai/developer-zone/) - Official Hailo documentation, downloads, and resources
- GLASS Model: [Original paper/repository]

