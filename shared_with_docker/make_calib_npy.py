#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, sys, argparse, glob
import numpy as np
import cv2
from typing import List

def _to_uint8_rgb(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if img.dtype == np.uint8:
        return img
    if img.dtype == np.uint16:
        return (img / 257.0).astype(np.uint8)
    if np.issubdtype(img.dtype, np.floating):
        return np.clip(np.rint(img), 0, 255).astype(np.uint8)
    return img.astype(np.uint8, copy=False)

def load_image_rgb(path: str, imagesize: int) -> np.ndarray:
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise RuntimeError(f"cannot read: {path}")
    img = _to_uint8_rgb(img)
    img = cv2.resize(img, (imagesize, imagesize), interpolation=cv2.INTER_LINEAR)
    return img  # (H,W,3) uint8

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-dir", "--src_dir", dest="src_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--imagesize", type=int, default=288)
    ap.add_argument("--max-samples", type=int, default=219)
    ap.add_argument("--ext", nargs="+", default=["*.png","*.jpg","*.jpeg","*.bmp","*.tif","*.tiff"])
    ap.add_argument("--recursive", action="store_true")
    args = ap.parse_args()

    # 이미지 파일 수집
    paths: List[str] = []
    for pat in args.ext:
        pattern = os.path.join(args.src_dir, pat)
        paths.extend(glob.glob(pattern, recursive=args.recursive))
        if args.recursive and "**" not in pat:
            pattern_recursive = os.path.join(args.src_dir, "**", pat)
            paths.extend(glob.glob(pattern_recursive, recursive=True))
    paths = sorted(set(paths))

    # 첫 차원 강제: 부족하면 반복, 많으면 자르기
    N = args.max_samples
    if len(paths) == 0:
        print(f"[ERR] no images under: {args.src_dir}", file=sys.stderr)
        sys.exit(1)
    elif len(paths) > N:
        paths = paths[:N]
    elif len(paths) < N:
        repeat_count = N - len(paths)
        extra = (paths * ((repeat_count // len(paths)) + 1))[:repeat_count]
        paths.extend(extra)
    print(f"[INFO] total paths after repeat padding = {len(paths)}")

    # 이미지 로드
    good_imgs: List[np.ndarray] = []
    used_paths: List[str] = []
    for p in paths:
        try:
            img = load_image_rgb(p, args.imagesize)
            good_imgs.append(img)
            used_paths.append(p)
        except Exception as e:
            print(f"[WARN] skip {p}: {e}", file=sys.stderr)

    arr = np.stack(good_imgs, axis=0).astype(np.uint8, copy=False)  # (N,H,W,3)
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    np.save(args.out, arr)

    # 검증 출력
    test = np.load(args.out, mmap_mode="r")
    vmin, vmax = int(test.min()), int(test.max())
    print(f"[OK] saved: {args.out}")
    print(f" shape={test.shape} dtype={test.dtype} range=({vmin},{vmax})")
    print(f" used={len(used_paths)} (padded={len(paths)-len(set(paths))})")

if __name__ == "__main__":
    main()