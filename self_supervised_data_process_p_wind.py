import os
import json
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Any, Optional

# ========================
# Configuration
# ========================
META_PATH = "./meta_data_wind.json"
DATA_X_DIR = "/inspire/ssd/project/sais-mtm/public/qlz/linan/dl/data"
OUTPUT_DIR = "/inspire/ssd/project/sais-mtm/public/qlz/linan/dl/ConvNeXt-V2/self_supervised/processed_wind"
NWP_LIST = ["NWP_1", "NWP_3"]

BATCH_SIZE = 16
TEST_LIMIT: Optional[int] = None   # 例如调试用 100；None 表示不限制
MAX_WORKERS: Optional[int] = None  # None 则使用 min(物理核数, 电站数)
MMAP_MODE: Optional[str] = None    # 可设为 "r" 以降低内存压力

# ========================
# Utilities
# ========================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def load_json(file_path: str):
    with open(file_path, "r") as f:
        return json.load(f)

def save_npz_nwp(output_path: str, nwp: np.ndarray, times: List[str]):
    """
    保存仅包含:
      - nwp: (B, 3, H, W, ...) 由 NWP_1/2/3 堆叠
      - time: (B,) 字符串时间戳（从文件名解析）
    """
    np.savez_compressed(output_path, nwp=nwp, time=np.array(times))

def fname_to_time_id(npy_file_name: str) -> str:
    """
    将文件名 YYYYMMDDHH.npy -> 'YYYY-MM-DD HH:00:00'
    （若原本就是无扩展名字符串，同样适配）
    """
    base = os.path.splitext(npy_file_name)[0]
    return f"{base[:4]}-{base[4:6]}-{base[6:8]} {base[8:10]}:00:00"

# ========================
# Worker: per farm
# ========================
def process_one_prov(args: Tuple[str, Dict[str, Any], str, str, List[str]]) -> Dict[str, Any]:
    farm_id, info, data_x_dir, output_dir, nwp_list = args

    farm_output_dir = os.path.join(output_dir, farm_id)
    ensure_dir(farm_output_dir)

    npy_files: List[str] = info.get("nwp_data", [])
    if not npy_files:
        return {"farm": farm_id, "ok": 0, "skip": 0, "saved_batches": 0, "reason": "no_npy"}

    # 可复现实验的本地打乱
    rng = np.random.default_rng(seed=hash(farm_id) & 0xFFFFFFFF)
    rng.shuffle(npy_files)
    if TEST_LIMIT is not None:
        npy_files = npy_files[: int(TEST_LIMIT)]

    ok_count = 0
    skip_count = 0
    batch_id = 0
    i = 0

    batch_nwp_list: List[np.ndarray] = []
    batch_time_list: List[str] = []

    for data_id in npy_files:
        # 读取三路 NWP
        stack_list: List[np.ndarray] = []
        missing = False
        for nwp_name in nwp_list:
            npy_path = os.path.join(data_x_dir, farm_id, nwp_name, data_id)
            if not os.path.exists(npy_path):
                missing = True
                break
            try:
                arr = np.load(npy_path, mmap_mode=MMAP_MODE)
                stack_list.append(arr)
            except Exception:
                missing = True
                break
        if missing or len(stack_list) != len(nwp_list):
            skip_count += 1
            continue

        # (3, H, W, ...)
        try:
            nwp = np.stack(stack_list, axis=0)
        except Exception:
            skip_count += 1
            continue

        # 记录
        batch_nwp_list.append(nwp)
        batch_time_list.append(fname_to_time_id(data_id))
        i += 1
        ok_count += 1

        # 满 BATCH_SIZE 落盘
        if i == BATCH_SIZE:
            try:
                batch_nwp = np.stack(batch_nwp_list, axis=0)  # (B, 3, ...)
                out_path = os.path.join(farm_output_dir, f"batch_{batch_id}.npz")
                save_npz_nwp(out_path, batch_nwp, batch_time_list)
            except Exception:
                # 若失败丢弃这一包
                skip_count += i
            else:
                batch_id += 1
            # reset
            batch_nwp_list.clear()
            batch_time_list.clear()
            i = 0

    # 余数包（可按需关闭）
    if i > 0:
        try:
            batch_nwp = np.stack(batch_nwp_list, axis=0)
            out_path = os.path.join(farm_output_dir, f"batch_{batch_id}.npz")
            save_npz_nwp(out_path, batch_nwp, batch_time_list)
            batch_id += 1
        except Exception:
            skip_count += i

    return {"farm": farm_id, "ok": ok_count, "skip": skip_count, "saved_batches": batch_id, "reason": "ok"}

# ========================
# Execution
# ========================
if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)
    meta = load_json(META_PATH)

    farm_ids = list(meta.keys())
    if MAX_WORKERS is None:
        cpu = os.cpu_count() or 4
        max_workers = max(1, min(cpu, len(farm_ids)))
    else:
        max_workers = int(MAX_WORKERS)

    print(f"Total farms: {len(farm_ids)} | Using {max_workers} workers")
    tasks = []
    results = []

    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        for fid in farm_ids:
            args = (fid, meta[fid], DATA_X_DIR, OUTPUT_DIR, NWP_LIST)
            tasks.append(ex.submit(process_one_prov, args))
            # tasks.append(process_one_prov(args))

        for fut in as_completed(tasks):
            try:
                res = fut.result()
                results.append(res)
                print(f"[{res['farm']}] ok={res['ok']} skip={res['skip']} batches={res['saved_batches']} reason={res['reason']}")
            except Exception as e:
                print(f"[worker error] {e}")

    total_ok = sum(r.get("ok", 0) for r in results)
    total_skip = sum(r.get("skip", 0) for r in results)
    total_batches = sum(r.get("saved_batches", 0) for r in results)
    print("=" * 60)
    print(f"Summary: ok={total_ok} | skip={total_skip} | batches={total_batches} | farms={len(results)}")
