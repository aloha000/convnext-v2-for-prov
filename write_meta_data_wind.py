import os
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = Path("../data")
NWP1 = "NWP_1"
NWP2 = "NWP_2"
NWP3 = "NWP_3"
MAX_WORKERS = min(32, (os.cpu_count() or 4) * 5)  # I/O 密集，开多些线程

def safe_int_stem(fname: str):
    """把文件名（不含扩展名）转为 int；失败则返回 None，用于过滤异常文件名。"""
    stem = Path(fname).stem
    try:
        return int(stem)
    except ValueError:
        return None

def list_dir_files(d: Path):
    """安全列目录，目录不存在则返回空列表。"""
    if not d.exists() or not d.is_dir():
        return []
    try:
        return os.listdir(d)
    except Exception:
        return []

def process_prov(prov: str):
    """处理单个电站，返回 (prov, meta, n_count)；若无有效数据，返回 None。"""
    meta = {"ID": prov}
    base = ROOT / prov

    # 1) 列 NWP_1 的 .npy 文件并按时间排序（按整数化的文件名排序）
    nwp1_dir = base / NWP1
    nwp1_files = [f for f in list_dir_files(nwp1_dir) if f.endswith(".npy")]
    # 过滤掉无法转为整数时间戳的文件
    nwp1_files = [f for f in nwp1_files if safe_int_stem(f) is not None]
    nwp1_files.sort(key=lambda x: safe_int_stem(x))

    if len(nwp1_files) == 0:
        print(f"Warning: No NWP data found for prov {prov}")
        return None

    # 2) 校验 NWP_2 与 NWP_3 是否具备相同文件
    nwp2_dir = base / NWP2
    nwp3_dir = base / NWP3
    nwp2_set = set([f for f in list_dir_files(nwp2_dir) if f.endswith(".npy")])
    nwp3_set = set([f for f in list_dir_files(nwp3_dir) if f.endswith(".npy")])

    filtered = []
    for f in nwp1_files:
        if f in nwp2_set and f in nwp3_set:
            filtered.append(f)
        else:
            print(f"Warning: NWP data mismatch for farm {prov}, file {f} missing in NWP_2 or NWP_3")

    if len(filtered) == 0:
        print(f"Warning: All NWP files filtered out for farm {prov}")
        return None

    meta["nwp_data"] = filtered
    meta["nwp_data_num"] = len(filtered)
    meta["start_time"] = Path(filtered[0]).stem
    meta["end_time"]   = Path(filtered[-1]).stem
    return (prov, meta, len(filtered))

def main():
    prov_list = [d.name for d in ROOT.iterdir() if d.is_dir()]
    prov_list.sort()

    meta_data = {}
    total_training_data = 0
    total_provs = 0

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(process_prov, prov): prov for prov in prov_list}
        for fut in as_completed(futures):
            res = fut.result()
            if res is None:
                continue
            prov, meta, cnt = res
            meta_data[prov] = meta
            total_training_data += cnt
            total_provs += 1

    print(f"Total number of training data: {total_training_data}")
    print(f"Total number of farms: {total_provs}")

    with open("meta_data_wind.json", "w") as f:
        json.dump(meta_data, f, indent=4)

if __name__ == "__main__":
    main()
