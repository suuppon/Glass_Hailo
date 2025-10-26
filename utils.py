from __future__ import annotations

import csv
import logging
import os
import random
import shutil
from typing import Iterable, List, Optional, Sequence

import cv2
import numpy as np

LOGGER = logging.getLogger(__name__)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


# --------------------------- Device / Seeds (no torch) ---------------------------

def set_torch_device(gpu_ids: Sequence[int] | tuple[int, ...] | list[int]) -> str:
    """
    Torch 미사용 호환용 더미 함수.
    기존 코드와의 시그니처 호환을 위해 유지하되 항상 'cpu' 문자열을 반환.
    """
    return "cpu"


def fix_seeds(seed: int, with_torch: bool = True, with_cuda: bool = True) -> None:
    """
    재현성 고정 (NumPy/표준 random만 사용).
    torch 미사용 환경을 위해 torch 관련 부분 제거.
    """
    random.seed(seed)
    np.random.seed(seed)
    # torch 관련 동작은 의도적으로 제거 (미사용)


# --------------------------- FS Helpers ---------------------------

def create_storage_folder(
    main_folder_path: str,
    project_folder: str,
    group_folder: str,
    run_name: str,
    mode: str = "iterate",
) -> str:
    """
    Create/prepare a storage root. (현재 레포에선 단순 루트만 보장)
    """
    os.makedirs(main_folder_path, exist_ok=True)
    return main_folder_path


def del_remake_dir(path: str, del_flag: bool = True) -> None:
    """Delete and remake directory (or just make if not exists)."""
    if os.path.exists(path):
        if del_flag:
            shutil.rmtree(path, ignore_errors=True)
        os.makedirs(path, exist_ok=True)
    else:
        os.makedirs(path, exist_ok=True)


# --------------------------- Vision Utils ---------------------------

def torch_format_2_numpy_img(img: np.ndarray) -> np.ndarray:
    """
    CHW(torch-like) -> HWC(BGR uint8)
    - 3채널이면 ImageNet 정규화 복원 + RGB->BGR 변환
    - 1채널/기타는 3채널 반복
    * PyTorch 미사용. 입력은 단순 NumPy 배열(CHW)로 가정.
    """
    if img.ndim != 3:
        raise ValueError("Expected CHW ndarray.")

    if img.shape[0] == 3:
        img = img.transpose(1, 2, 0)  # CHW -> HWC
        img = img * np.array(IMAGENET_STD, dtype=np.float32) + np.array(IMAGENET_MEAN, dtype=np.float32)
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
        # RGB -> BGR
        img = img[:, :, ::-1]
    else:
        img = img.transpose(1, 2, 0)
        img = np.repeat(img, 3, axis=-1)
        img = np.clip(img, 0.0, 1.0)
        img = (img * 255.0).astype(np.uint8)
    return img


def distribution_judge(img: np.ndarray, name: str) -> int:
    """
    Judge the distribution of specific category by FFT pattern.
    Returns:
        1 (HyperSphere) or 0 (Manifold)
    """
    img_ = cv2.resize(img, (289, 289))
    gray = cv2.cvtColor(img_, cv2.COLOR_BGR2GRAY)
    gray = cv2.blur(gray, (39, 39))

    dft = cv2.dft(np.float32(gray), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = 20 * np.log(cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]) + 1e-12)
    magnitude[magnitude > 170] = 255
    magnitude[magnitude <= 170] = 0

    h, w = magnitude.shape
    center = (h // 2, w // 2)
    y_idx, x_idx = np.where(magnitude == 255)
    y_all, x_all = np.indices((2 * h, 2 * w))

    l1_dist_x = np.abs(x_idx - center[1])
    l1_dist_y = np.abs(y_idx - center[0])

    dist = np.sqrt((x_idx - center[1]) ** 2 + (y_idx - center[0]) ** 2)
    l2_dist_all = np.sqrt((x_all - center[1]) ** 2 + (y_all - center[0]) ** 2)

    side_x = np.max(l1_dist_x) if l1_dist_x.size else 1
    side_y = np.max(l1_dist_y) if l1_dist_y.size else 1
    radius = np.max(dist) if dist.size else 0
    points_num = len(dist)

    l1_density = points_num / (4 * max(side_x, 1) * max(side_y, 1))
    l2_density = points_num / (np.sum(l2_dist_all <= radius) + 1e-10)
    flag = 1 if (l1_density > 0.21 or l2_density > 0.21) and radius > 12 and points_num > 60 else 0
    dtype = "HyperSphere" if flag == 1 else "Manifold"
    print(f"Distribution: {flag} / {dtype}.")

    out_dir = os.path.join("./results/judge/fft", str(flag))
    os.makedirs(out_dir, exist_ok=True)
    vis = np.hstack([img_, np.repeat(magnitude, 3).reshape((h, w, 3))])
    cv2.imwrite(os.path.join(out_dir, f"{name}.png"), vis)
    return flag


# --------------------------- Results CSV ---------------------------

def compute_and_store_final_results(
    results_path: str,
    results: List[List[float]],
    column_names: List[str],
    row_names: Optional[List[str]] = None,
    filename: str = "results.csv",
    mode: str = "auto",  # "auto" | "append" | "overwrite"
) -> dict:
    """
    Store/append results to a CSV and keep a running Mean row.

    Args:
        results_path: directory to save csv
        results: List of rows, e.g. [[i_auc, i_ap, p_auc, p_ap, p_pro, latency, fps], ...]
        column_names: e.g. ["image_auroc","image_ap","pixel_auroc","pixel_ap","pixel_pro","latency_sec","fps"]
        row_names: optional row names (same length as 'results')
        filename: csv file name
        mode:
          - "auto"     : append if header matches; otherwise overwrite
          - "append"   : force append (header mismatch -> overwrite)
          - "overwrite": always overwrite
    Returns:
        dict of mean_{col}: value
    """
    if row_names is not None:
        assert len(row_names) == len(results), "#Rownames != #Result-rows."

    os.makedirs(results_path, exist_ok=True)
    savename = os.path.join(results_path, filename)

    # load existing (drop trailing Mean)
    existing_rows: list[list[str]] = []
    existing_header: Optional[list[str]] = None
    if os.path.exists(savename) and mode in ("auto", "append"):
        with open(savename, "r") as f:
            reader = list(csv.reader(f))
        if reader:
            existing_header = reader[0]
            body = reader[1:]
            if body and body[-1] and str(body[-1][0]).strip().lower() == "mean":
                body = body[:-1]
            existing_rows = body

    # header
    header = column_names[:]
    use_row_names = row_names is not None or (
        existing_header is not None and len(existing_header) > 0 and existing_header[0].strip().lower() == "row names"
    )
    if use_row_names:
        header = ["Row Names"] + header

    def _norm(h: Optional[list[str]]) -> list[str]:
        return [x.strip().lower() for x in (h or [])]

    overwrite = (mode == "overwrite") or (existing_header is None) or (_norm(existing_header) != _norm(header))

    # rows to write
    rows_to_write: list[list[str | float]] = []
    if not overwrite:
        rows_to_write.extend(existing_rows)

    for i, res in enumerate(results):
        row_name = row_names[i] if row_names is not None else f"Row{len(rows_to_write)+1}"
        row = [row_name] + list(res) if use_row_names else list(res)
        rows_to_write.append(row)

    # mean
    data_only = [r[1:] if use_row_names else r for r in rows_to_write]
    data_arr = np.array([[_to_float(x) for x in r] for r in data_only], dtype=float)
    col_means = np.nanmean(data_arr, axis=0).tolist()
    mean_row = (["Mean"] + col_means) if use_row_names else col_means

    # write
    with open(savename, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows_to_write:
            w.writerow(r)
        w.writerow(mean_row)

    mean_metrics = {f"mean_{key}": val for key, val in zip(column_names, col_means)}
    return mean_metrics


def _to_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")