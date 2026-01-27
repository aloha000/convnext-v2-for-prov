import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from glob import glob

import numpy as np

def load_stats_npz(stats_path: str):
    """
    从你的 np.savez_compressed 结果里读出 (C,F) 的 mean/std。
    需要你在保存时写入了 mean_hw 与 pop_std_hw。
    """
    z = np.load(stats_path, allow_pickle=False)
    # 若你只保存了 mean/pop_std (C,F,H,W)，也可在这里再聚合一下：
    if "mean_hw" in z and "pop_std_hw" in z:
        mean_hw = z["mean_hw"]          # (C, F)
        std_hw  = z["pop_std_hw"]       # (C, F)  —— 用总体标准差
    else:
        # 退化路径：从 (C,F,H,W) 聚合成 (C,F)
        mean_hw = z["mean"].mean(axis=(-2, -1))                # (C, F)
        pop_var = z["pop_std"]**2 if "pop_std" in z else z["pop_var"]
        std_hw  = np.sqrt(np.mean(pop_var, axis=(-2, -1)))     # (C, F)
    return mean_hw.astype(np.float32), std_hw.astype(np.float32)

class NormalizeCF(nn.Module):
    """
    用 (C,F) 的 mean/std 按 8×8 整体对 (C,F,H,W) 做标准化。
    会把 mean/std 注册为 buffer，方便随模型保存/迁移到 GPU。
    """
    def __init__(self, mean_cf: np.ndarray, std_cf: np.ndarray, eps: float = 1e-6):
        super().__init__()
        assert mean_cf.shape == std_cf.shape and mean_cf.ndim == 2  # (C,F)
        mean = torch.from_numpy(mean_cf.astype(np.float32))  # (C,F)
        std  = torch.from_numpy(std_cf.astype(np.float32))   # (C,F)
        std = torch.clamp(std, min=eps)

        # 变成 (1,C,F,1,1) 方便广播
        self.register_buffer("mean", mean.view(mean.size(0), mean.size(1), 1, 1))
        self.register_buffer("std",  std.view(std.size(0),  std.size(1),  1, 1))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (C,F,H,W)
        """
        return (x - self.mean) / self.std

    @torch.no_grad()
    def inverse(self, x: torch.Tensor) -> torch.Tensor:
        """反归一化（可选）"""
        return x * self.std + self.mean
    
    
def preprocess_solar(x_raw):
    x = []
    # nwp1: ['ghi', 'poai', 'sp', 't2m', 'tcc', 'tp', 'u100', 'v100']
    x.extend([x_raw[0,0:1,:,:], x_raw[0,1:2,:,:], x_raw[0,4:5,:,:]])
    # nwp2: ['ghi', 'msl', 'poai', 't2m', 'tcc', 'tp', 'u100', 'v100']
    x.extend([x_raw[1,0:1,:,:], x_raw[1,2:3,:,:], x_raw[1,4:5,:,:]])
    # nwp3: ['ghi', 'poai', 'sp', 't2m', 'tcc', 'tp', 'u100', 'v100']
    x.extend([x_raw[2,0:1,:,:], x_raw[2,1:2,:,:], x_raw[2,4:5,:,:]])
    
    return torch.concatenate(x, dim=0)


class SolarNWPDataset_PerFarm(Dataset):
    """
    Dataset for processed solar PWD data (only NWP stacks).
    Each sample: {'nwp': tensor(8, H, W, ...), 'time': str, 'farm': str}
    """

    def __init__(self, root_dir, farm_id, transform=None):
        """
        Args:
            root_dir (str): 例如 './paired_data/processed_solar_pwd_only'
            transform (callable, optional): 对 PWD 数据的变换函数
        """
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []

        # 遍历电站目录
        npz_files = sorted(glob(os.path.join(root_dir, farm_id, "batch_*.npz")))
        for path in npz_files:
            self.samples.append({"path": path, "farm": farm_id})

        if not self.samples:
            raise RuntimeError(f"No .npz files found under {root_dir}")

        print(f"Loaded {len(self.samples)} batch files from {len(farm_id)} farms.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        npz_data = np.load(sample_info["path"], allow_pickle=True)
        pwd = npz_data["nwp"]      # (B, 3, H, W, ...)
        times = npz_data["time"]   # (B,)
        power = npz_data["power"][:,1]
        # 返回为 list of samples
        batch = []
        for i in range(pwd.shape[0]):
            x_raw = pwd[i]
            x_raw = torch.from_numpy(x_raw).float()
            
            
            if self.transform:
                x_raw = self.transform(x_raw)
                
            x = preprocess_solar(x_raw)
            
            y = power[i:i+1]
            y = torch.from_numpy(y).float()
            
            
            batch.append({
                "pwd": x,
                "power": y,
                "time": str(times[i]),
                "farm": sample_info["farm"]
            })
        return batch


def collate_nwp(batch_list):
    """
    自定义 collate_fn，用于拼接 batch 列表。
    batch_list 是一个 list，每个元素是 Dataset.__getitem__ 返回的 list。
    """
    flat_samples = [item for sublist in batch_list for item in sublist]
    pwds = torch.stack([s["pwd"] for s in flat_samples])
    powers = torch.stack([s["power"] for s in flat_samples])
    times = [s["time"] for s in flat_samples]
    farms = [s["farm"] for s in flat_samples]
    return {"pwd": pwds, "power": powers, "time": times, "farm": farms}


def create_dataloader(dataset, batch_size=8, num_workers=4, shuffle=True):
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_nwp,
        pin_memory=True,
    )
    return loader


# ========================
# Example Usage
# ========================
if __name__ == "__main__":
    data_dir = "./paired_data/processed_solar"
    data_statistic_data = "./global_spatial_mean_std_fast.npz"
    farm_id = "1296"
    
    mean_hw, std_hw = load_stats_npz(data_statistic_data)
    tfm = NormalizeCF(mean_hw, std_hw, eps=1e-6)
    
    dataset = SolarNWPDataset_PerFarm(data_dir, farm_id, tfm)


    loader = create_dataloader(dataset, farm_id, batch_size=2, num_workers=2)

    for i, batch in enumerate(loader):
        print(f"Batch {i}: {batch['pwd'].shape} {batch['power'].shape}, farms={set(batch['farm'])}")
        # print(batch['pwd'][-3])   
        # print(batch['power'][-3])       
        if i == 1:
            break
