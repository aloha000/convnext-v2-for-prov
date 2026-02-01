#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
for _v in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS",
           "BLIS_NUM_THREADS","NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
    
import json
import math
import numpy as np
from typing import List, Tuple, Dict, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

# =========================
# Config
# =========================
PROCESSED_DIR = "/inspire/ssd/project/sais-mtm/public/qlz/linan/dl/ConvNeXt-V2/self_supervised/processed_wind_1.2_2"
OUT_NPZ = "/inspire/ssd/project/sais-mtm/public/qlz/linan/dl/ConvNeXt-V2/Guangdong_spatial_mean_std_fast_1.2_2.npz"
OUT_SUMMARY_JSON = "/inspire/ssd/project/sais-mtm/public/qlz/linan/dl/ConvNeXt-V2/Guangdong_spatial_mean_std_fast_1.2_2.json"

C, F, H, W = 2, 8, 60, 92

# 并行设置：固定 42 核，粗粒度任务
MAX_WORKERS: Optional[int] = 42      # 你机器是 42 核，就填 42；也可留 None 用亲和度
FILES_PER_TASK = 32                  # 先用 32；CPU 不满再试 64
DTYPE_ACC = np.float64

IGNORE_NAN = False
ASSUME_FLOAT32_INPUT = True

# =========================
# Helpers
# =========================
def list_npz_files(root: str) -> List[str]:
    files = []
    for farm_id in sorted(os.listdir(root)):
        if farm_id == 'Guangdong':
            farm_dir = os.path.join(root, farm_id)
            if not os.path.isdir(farm_dir):
                continue
            for fname in sorted(os.listdir(farm_dir)):
                if fname.endswith(".npz"):
                    files.append(os.path.join(farm_dir, fname))
    return files

def chunk_list(lst: List[str], k: int) -> List[List[str]]:
    if k <= 0:
        return [lst]
    return [lst[i:i+k] for i in range(0, len(lst), k)]



# =========================
# Helpers（仅在 np.load 处小改）
# =========================
def reduce_file_list(file_list: List[str]) -> Dict[str, np.ndarray]:
    S = np.zeros((C, F, H, W), dtype=DTYPE_ACC)
    Q = np.zeros((C, F, H, W), dtype=DTYPE_ACC)
    if IGNORE_NAN:
        Cnt = np.zeros((C, F, H, W), dtype=np.int64)
    else:
        N_tot = 0

    for path in file_list:
        try:
            # 关闭 pickle，npz 无法 memmap，保持解压并行
            with np.load(path, allow_pickle=False) as data:
                pwd = data["nwp"]  # (B, 3, 11, 8, 8)
                x = pwd if (ASSUME_FLOAT32_INPUT and pwd.dtype == np.float32) else pwd.astype(np.float32, copy=False)

                if IGNORE_NAN:
                    S += np.nan_to_num(x, nan=0.0).sum(axis=0, dtype=DTYPE_ACC)
                    Q += np.nan_to_num(x * x, nan=0.0).sum(axis=0, dtype=DTYPE_ACC)
                    Cnt += np.sum(~np.isnan(x), axis=0, dtype=np.int64)
                else:
                    S += x.sum(axis=0, dtype=DTYPE_ACC)
                    Q += (x * x).sum(axis=0, dtype=DTYPE_ACC)
                    N_tot += x.shape[0]
        except Exception as e:
            print(f"[warn] skip {path}: {e}")
            continue

    return {"S": S, "Q": Q, ("C" if IGNORE_NAN else "N"): (Cnt if IGNORE_NAN else N_tot)}



def merge_partials(acc: Dict[str, np.ndarray], part: Dict[str, np.ndarray]):
    acc["S"] += part["S"]
    acc["Q"] += part["Q"]
    if IGNORE_NAN:
        acc["C"] += part["C"]
    else:
        acc["N"] += part["N"]

# =========================
# Main
# =========================
if __name__ == "__main__":
    files = list_npz_files(PROCESSED_DIR)
    if not files:
        raise SystemExit(f"No .npz files found in {PROCESSED_DIR}")
    tasks = chunk_list(files, FILES_PER_TASK)

    # 亲和度优先（容器/调度器里更准确）
    try:
        effective_cpu = len(os.sched_getaffinity(0))
    except Exception:
        effective_cpu = os.cpu_count() or 4

    if MAX_WORKERS is None:
        max_workers = max(1, min(effective_cpu, len(tasks)))
    else:
        max_workers = max(1, min(int(MAX_WORKERS), effective_cpu, len(tasks)))

    print(f"Total files: {len(files)} | Tasks: {len(tasks)} | Workers: {max_workers}")
    
    # 初始化全局部分和
    total = {
        "S": np.zeros((C, F, H, W), dtype=DTYPE_ACC),
        "Q": np.zeros((C, F, H, W), dtype=DTYPE_ACC),
        ("C" if IGNORE_NAN else "N"): (np.zeros((C, F, H, W), dtype=np.int64) if IGNORE_NAN else 0),
    }
    
    for ti in tasks:
        reduce_file_list(ti)
    # 并行处理
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(reduce_file_list, t) for t in tasks]
        for i, fut in enumerate(as_completed(futs), 1):
            part = fut.result()
            merge_partials(total, part)
            if i % max(1, len(tasks)//20) == 0 or i == len(tasks):
                if IGNORE_NAN:
                    valid = total["C"].min()
                    print(f"[{i}/{len(tasks)}] merged. min_valid_count_per_cell={int(valid)}")
                else:
                    print(f"[{i}/{len(tasks)}] merged. N={int(total['N'])}")

    # 计算 mean / std
    if IGNORE_NAN:
        # 每个格点的有效计数
        Ceff = total["C"].astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            mean = np.where(Ceff > 0, total["S"] / Ceff, np.nan)
            pop_var = np.where(Ceff > 0, total["Q"] / Ceff - mean * mean, np.nan)
            # 样本方差（无偏）： M2 = Q - N*mean^2 -> var = M2/(N-1)
            M2 = total["Q"] - Ceff * (mean * mean)
            samp_var = np.where(Ceff > 1, M2 / (Ceff - 1.0), np.nan)
    else:
        N = float(total["N"])
        mean = total["S"] / N
        pop_var = total["Q"] / N - mean * mean


    # 8x8 视作整体
    mean_hw = mean.mean(axis=(-2, -1))                 # (C, F)
    pop_std_hw = np.sqrt(np.mean(pop_var, axis=(-2, -1)))  # (C, F)

    # 落盘
    np.savez_compressed(
        OUT_NPZ,
        mean_hw=mean_hw.astype(np.float32),
        pop_std_hw=pop_std_hw.astype(np.float32),
        mean=mean.astype(np.float32),
        pop_std=np.sqrt(np.maximum(pop_var, 0.0)).astype(np.float32),
        total_S=total["S"].astype(np.float64),
        total_Q=total["Q"].astype(np.float64),
        count=np.array(total["N"], dtype=np.int64),
        meta=np.array(["mean/std over axis=batch; 8x8 aggregated"], dtype=object),
    )
    summary = {
        "files": len(files),
        "tasks": len(tasks),
        "workers": max_workers,
        "shape": [C, F, H, W],
        "ignore_nan": IGNORE_NAN,
        "dtype_acc": str(DTYPE_ACC),
        "mean_minmax_hw": [float(np.nanmin(mean_hw)), float(np.nanmax(mean_hw))],
        "std_minmax_hw": [float(np.nanmin(pop_std_hw)), float(np.nanmax(pop_std_hw))],
        "mean_minmax_spatial": [float(np.nanmin(mean)), float(np.nanmax(mean))],
        "std_minmax_spatial": [float(np.nanmin(np.sqrt(np.maximum(pop_var, 0.0)))),
                            float(np.nanmax(np.sqrt(np.maximum(pop_var, 0.0))))],
        "N_or_min_valid": (int(total["N"]) if not IGNORE_NAN else int(np.nanmin(total["C"]))),
    }

    with open(OUT_SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print(f"Saved to: {OUT_NPZ}")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
