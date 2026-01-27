# dataloader_ss.py
import os
from glob import glob
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler, SequentialSampler, Subset

# ========================
# Stats & Transforms
# ========================
def load_stats_npz(stats_path: str):
    """
    读取 (C,F) 的 mean/std；若没有 (C,F)，则从 (C,F,H,W) 聚合为 (C,F)。
    需要你在保存时写入 mean_hw 与 pop_std_hw；否则回退到 mean/pop_std 的 H,W 聚合。
    """
    z = np.load(stats_path, allow_pickle=False)
    if "mean_hw" in z and "pop_std_hw" in z:
        mean_hw = z["mean_hw"]              # (C, F)
        std_hw  = z["pop_std_hw"]           # (C, F)
    else:
        mean_hw = z["mean"].mean(axis=(-2, -1))  # (C, F)
        pop_var = z["pop_std"]**2 if "pop_std" in z else z["pop_var"]
        std_hw  = np.sqrt(np.mean(pop_var, axis=(-2, -1)))     # (C, F)
    return mean_hw.astype(np.float32), std_hw.astype(np.float32)


class NormalizeCF(nn.Module):
    """
    用 (C,F) 的 mean/std 对 (C,F,H,W) 做标准化。
    注意：此处 buffer 形状为 (C,F,1,1)，与输入 (C,F,H,W) 广播匹配。
    """
    def __init__(self, mean_cf: np.ndarray, std_cf: np.ndarray, eps: float = 1e-6):
        super().__init__()
        assert mean_cf.shape == std_cf.shape and mean_cf.ndim == 2  # (C,F)
        mean = torch.from_numpy(mean_cf.astype(np.float32))  # (C,F)
        std  = torch.from_numpy(std_cf.astype(np.float32))   # (C,F)
        std = torch.clamp(std, min=eps)
        self.register_buffer("mean", mean.view(mean.size(0), mean.size(1), 1, 1))
        self.register_buffer("std",  std.view(std.size(0),  std.size(1),  1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C,F,H,W)
        return (x - self.mean) / self.std

    @torch.no_grad()
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean


# ========================
# Preprocessors
# ========================
@torch.no_grad()
def preprocess_solar(x_raw: torch.Tensor) -> torch.Tensor:
    """
    x_raw: (3, 8, H, W) for each NWP, stacked along first dim -> (NWP, F, H, W)
    Select 3 features per NWP: (ghi, poai, tcc) ==> indices [0,1,4]
    """
    x = []
    # nwp1: ['ghi', 'poai', 'sp', 't2m', 'tcc', 'tp', 'u100', 'v100']
    x.extend([x_raw[0, 0:1], x_raw[0, 1:2], x_raw[0, 4:5]])
    # nwp2: ['ghi', 'msl', 'poai', 't2m', 'tcc', 'tp', 'u100', 'v100']
    x.extend([x_raw[1, 0:1], x_raw[1, 2:3], x_raw[1, 4:5]])
    # nwp3: ['ghi', 'poai', 'sp', 't2m', 'tcc', 'tp', 'u100', 'v100']
    x.extend([x_raw[2, 0:1], x_raw[2, 1:2], x_raw[2, 4:5]])
    return torch.cat(x, dim=0)  # (9, H, W)


@torch.no_grad()
def preprocess_wind(x_raw: torch.Tensor) -> torch.Tensor:
    """
    取风相关特征：u100, v100, sp/msl, t2m, tp, ghi, poai （每个 NWP 各取一份）
    修正：原代码对 nwp2/nwp3 误用了 index=0；此处分别用 0/1/2。
    """
    x = []
    # nwp1
    x.extend([x_raw[0, 6:7], x_raw[0, 7:8], x_raw[0, 2:3], x_raw[0, 3:4], x_raw[0, 5:6], x_raw[0, 0:1], x_raw[0, 1:2]])
    # nwp2
    # x.extend([x_raw[1, 6:7], x_raw[1, 7:8], x_raw[1, 1:2], x_raw[1, 3:4], x_raw[1, 5:6], x_raw[1, 0:1], x_raw[1, 2:3]])
    # nwp3
    x.extend([x_raw[1, 6:7], x_raw[1, 7:8], x_raw[1, 2:3], x_raw[1, 3:4], x_raw[1, 5:6], x_raw[1, 0:1], x_raw[1, 1:2]])
    return torch.cat(x, dim=0)  # (21, H, W)

@torch.no_grad()
def preprocess_all(x_raw: torch.Tensor) -> torch.Tensor:
    """
    取全部相关特征：
    修正：原代码对 nwp2/nwp3 误用了 index=0；此处分别用 0/1/2。
    """
    x = []
    # nwp1
    x.extend([x_raw[0, 6:7], x_raw[0, 7:8], x_raw[0, 2:3], x_raw[0, 3:4], x_raw[0, 5:6], x_raw[0, 0:1], x_raw[0, 1:2], x_raw[0, 4:5]])
    # nwp2
    x.extend([x_raw[1, 6:7], x_raw[1, 7:8], x_raw[1, 1:2], x_raw[1, 3:4], x_raw[1, 5:6], x_raw[1, 0:1], x_raw[1, 2:3], x_raw[1, 4:5]])
    # nwp3  
    x.extend([x_raw[2, 6:7], x_raw[2, 7:8], x_raw[2, 2:3], x_raw[2, 3:4], x_raw[2, 5:6], x_raw[2, 0:1], x_raw[2, 1:2], x_raw[2, 4:5]])
    return torch.cat(x, dim=0)  # (24, H, W)


# ========================
# Dataset
# ========================
class NWPDataset(Dataset):
    """
    Dataset for processed NWP-only data.
    Each __getitem__ returns a *list* of samples decoded from one .npz batch file.
    Item structure: {"nwp": Tensor(C,F?,H,W) after preprocess, "time": str, "farm": str}
    """

    def __init__(self, root_dir, preprocessor=None, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.preprocessor = preprocessor
        self.samples = []

        farm_dirs = [d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))]
        for farm_id in farm_dirs:
            npz_files = sorted(glob(os.path.join(root_dir, farm_id, "batch_*.npz")))
            for path in npz_files:
                self.samples.append({"path": path, "farm": farm_id})

        if not self.samples:
            raise RuntimeError(f"No .npz files found under {root_dir}")

        print(f"[NWPDataset] Found {len(self.samples)} batch files from {len(farm_dirs)} farms.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        npz_data = np.load(sample_info["path"], allow_pickle=True)
        nwp = npz_data["nwp"]      # (B, 3, 8, H, W) or (B, F, H, W) depending on your packing
        times = npz_data["time"]   # (B,)

        out = []
        for i in range(nwp.shape[0]):
            x = torch.from_numpy(nwp[i]).float()  # expected (3, 8, H, W) -> 3 NWPs, 8 features
            if self.transform:
                x = self.transform(x)
            if self.preprocessor:
                x = self.preprocessor(x)
            out.append({
                "nwp": x,                   # (C_sel, H, W)
                "time": str(times[i]),
                "farm": sample_info["farm"]
            })
        return out


def collate_nwp(batch_list):
    """
    batch_list: list of lists (each inner list is samples decoded from one .npz file)
    Flattens, stacks nwp tensors, and keeps time/farm as python lists.
    """
    flat = []
    for sub in batch_list:
        if isinstance(sub, list):
            flat.extend(sub)
        else:
            flat.append(sub)

    if len(flat) == 0:
        # very defensive; should not happen unless an npz has no items
        return {"nwp": torch.empty(0), "time": [], "farm": []}

    nwps = torch.stack([s["nwp"] for s in flat], dim=0)
    times = [s["time"] for s in flat]
    farms = [s["farm"] for s in flat]
    return {"nwp": nwps, "time": times, "farm": farms}


# ========================
# DDP-safe Loader Builder
# ========================
def _seed_worker(worker_id):
    # Make dataloader workers deterministically seeded per rank/worker
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)

def _build_loader(dataset, batch_size, num_workers, pin_mem, sampler, drop_last):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,                 # never shuffle when sampler is given
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
        collate_fn=collate_nwp,
        worker_init_fn=_seed_worker,
        persistent_workers=(num_workers > 0),
        prefetch_factor=(2 if num_workers > 0 else None),
    )


def create_dataloader(
    dataset_type: str,
    data_dir: str,
    data_stat_dir: str,
    batch_size: int = 8,
    num_workers: int = 4,
    distributed: bool = False,
    pin_memory: bool = True,
    seed: int = 42,
    train_ratio: float = 0.8,
):
    """
    Returns: train_loader, test_loader, train_sampler, test_sampler
    - In DDP, use DistributedSampler; caller MUST call train_sampler.set_epoch(epoch).
    - We use a deterministic Subset split so that every rank sees the same partition.
    """
    mean_hw, std_hw = load_stats_npz(data_stat_dir)
    tfm = NormalizeCF(mean_hw, std_hw, eps=1e-6)

    if dataset_type == "solar":
        full_dataset = NWPDataset(data_dir, preprocessor=preprocess_solar, transform=tfm)
    elif dataset_type == "wind":
        full_dataset = NWPDataset(data_dir, preprocessor=preprocess_wind, transform=tfm)
    elif dataset_type == "all":
        full_dataset = NWPDataset(data_dir, preprocessor=preprocess_all, transform=tfm)
    else:
        raise ValueError(f"Unknown dataset_type: {dataset_type}")

    # ----- Deterministic split (shared across ranks)
    n_total = len(full_dataset)
    n_train = int(round(n_total * train_ratio))
    indices = np.arange(n_total)
    rng = np.random.default_rng(seed)  # same seed across ranks => same split
    rng.shuffle(indices)
    train_idx = indices[:n_train]
    test_idx  = indices[n_train:]

    train_set = Subset(full_dataset, train_idx.tolist())
    test_set  = Subset(full_dataset, test_idx.tolist())

    # ----- Samplers
    if distributed:
        train_sampler = DistributedSampler(train_set, shuffle=True, drop_last=True)
        test_sampler  = DistributedSampler(test_set,  shuffle=False, drop_last=False)
    else:
        train_sampler = RandomSampler(train_set)
        test_sampler  = SequentialSampler(test_set)

    # ----- Loaders
    train_loader = _build_loader(
        train_set, batch_size, num_workers, pin_memory, train_sampler, drop_last=True
    )
    test_loader = _build_loader(
        test_set, batch_size, num_workers, pin_memory, test_sampler, drop_last=False
    )

    # Helpful print once (rank 0 suggested outside)
    # print(f"[create_dataloader] train files: {len(train_set)} | test files: {len(test_set)}")

    return train_loader, test_loader, train_sampler, test_sampler


# ========================
# Example (local run)
# ========================
if __name__ == "__main__":
    data_dir = "/inspire/ssd/project/sais-mtm/public/qlz/data/PowerEstimateData/gongjia_processed_data/self_supervised/processed_wind"
    data_statistic_dir = "/inspire/ssd/project/sais-mtm/public/qlz/data/PowerEstimateData/gongjia_processed_data/global_spatial_mean_std_fast.npz"
    train_loader, test_loader, train_sampler, test_sampler = create_dataloader(
        "wind", data_dir, data_statistic_dir, batch_size=2, num_workers=2, distributed=False
    )
    print(f"train iters: {len(train_loader)} | test iters: {len(test_loader)}")
    for i, batch in enumerate(train_loader):
        import pdb; pdb.set_trace()
        print(f"Batch {i}: {batch['nwp'].shape}, farms={set(batch['farm'])}")
        print(f"Times: {batch['time'][:3]}")
        if i == 1:
            break
