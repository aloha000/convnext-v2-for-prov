import numpy as np
import pandas as pd
import os
import torch.utils.data as data
import torch
from torch.utils.data import DataLoader
from loguru import logger


###### util ######
# IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif']
IMG_EXTENSIONS = ['.jpg', '.JPG', '.jpeg', '.JPEG', '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tif', 'npy']


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def get_image_paths(dataroot):
    paths = None  # return None if dataroot is None
    if isinstance(dataroot, str):
        paths = sorted(_get_paths_from_images(dataroot))
    elif isinstance(dataroot, list):
        paths = []
        for i in dataroot:
            paths += sorted(_get_paths_from_images(i))
    return paths


def _get_paths_from_images(path):
    assert os.path.isdir(path), '{:s} is not a valid directory'.format(path)
    images = []
    for dirpath, _, fnames in sorted(os.walk(path)):
        for fname in sorted(fnames):
            if is_image_file(fname):
                img_path = os.path.join(dirpath, fname)
                images.append(img_path)
    assert images, '{:s} has no valid image file'.format(path)
    return images


# convert single (HxWxC) to 3-dimensional torch tensor
def single2tensor3(img):
    img = img.copy()
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()


class DatasetSthPowerSolar(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetSthPowerSolar, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.fuxi_size = self.patch_size // self.sf

        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_nwp_1 = get_image_paths(os.path.join(opt['dataroot_nwp'], "NWP_1"))
        self.paths_nwp_2 = get_image_paths(os.path.join(opt['dataroot_nwp'], "NWP_2"))
        self.paths_nwp_3 = get_image_paths(os.path.join(opt['dataroot_nwp'], "NWP_3"))

        self.tabel_gt = self._csv_to_dict(opt['dataroot_gt'])
        
        # 初始化mean,std
        self.const_path = opt['const_path']
        self.NWP1_GHI_MEAN = np.load(os.path.join(self.const_path, 'NWP_1_ghi_mean.npy'))
        self.NWP1_GHI_STD = np.load(os.path.join(self.const_path, 'NWP_1_ghi_std.npy'))
        self.NWP1_POAI_MEAN = np.load(os.path.join(self.const_path, 'NWP_1_poai_mean.npy'))
        self.NWP1_POAI_STD = np.load(os.path.join(self.const_path, 'NWP_1_poai_std.npy'))
        self.NWP1_TCC_MEAN = np.load(os.path.join(self.const_path, 'NWP_1_tcc_mean.npy'))
        self.NWP1_TCC_STD = np.load(os.path.join(self.const_path, 'NWP_1_tcc_std.npy'))

        self.NWP2_GHI_MEAN = np.load(os.path.join(self.const_path, 'NWP_2_ghi_mean.npy'))
        self.NWP2_GHI_STD = np.load(os.path.join(self.const_path, 'NWP_2_ghi_std.npy'))
        self.NWP2_POAI_MEAN = np.load(os.path.join(self.const_path, 'NWP_2_poai_mean.npy'))
        self.NWP2_POAI_STD = np.load(os.path.join(self.const_path, 'NWP_2_poai_std.npy'))
        self.NWP2_TCC_MEAN = np.load(os.path.join(self.const_path, 'NWP_2_tcc_mean.npy'))
        self.NWP2_TCC_STD = np.load(os.path.join(self.const_path, 'NWP_2_tcc_std.npy'))

        self.NWP3_GHI_MEAN = np.load(os.path.join(self.const_path, 'NWP_3_ghi_mean.npy'))
        self.NWP3_GHI_STD = np.load(os.path.join(self.const_path, 'NWP_3_ghi_std.npy'))
        self.NWP3_POAI_MEAN = np.load(os.path.join(self.const_path, 'NWP_3_poai_mean.npy'))
        self.NWP3_POAI_STD = np.load(os.path.join(self.const_path, 'NWP_3_poai_std.npy'))
        self.NWP3_TCC_MEAN = np.load(os.path.join(self.const_path, 'NWP_3_tcc_mean.npy'))
        self.NWP3_TCC_STD = np.load(os.path.join(self.const_path, 'NWP_3_tcc_std.npy'))
        
        assert self.paths_nwp_1, 'Error: H path is empty.'

    def _csv_to_dict(self, file_path):
        """
        读取gt的csv文件
        返回table = {"202401020000"：value, ...}
        """
        # 读取 CSV 文件
        df = pd.read_csv(file_path)

        # 确保第一列是时间，第二列是功率
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0]).dt.strftime("%Y%m%d%H%M")  # 转换时间列为 datetime 格式

        # 转换 DataFrame 为字典
        data_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

        return data_dict

    def normalization_nwp_1(self, fuxi_np):
        # channle排列顺序为：array(['ghi', 'poai', 'sp', 't2m', 'tcc', 'tp', 'u100', 'v100'], dtype=object)
        # c,h,w -> h,w,c
        fuxi_np = np.transpose(fuxi_np, axes=(1, 2, 0))
        # 选择channel
        # ghi & poai & tcc
        # 11,11,2
        fuxi_np = np.stack((fuxi_np[..., 0], fuxi_np[..., 1], fuxi_np[..., 4]), axis=-1)

        mean_nwp_1 = np.array([self.NWP1_GHI_MEAN, self.NWP1_POAI_MEAN, self.NWP1_TCC_MEAN])
        std_nwp_1 = np.array([self.NWP1_GHI_STD, self.NWP1_POAI_STD, self.NWP1_TCC_STD])

        fuxi_np = (fuxi_np - mean_nwp_1) / std_nwp_1
        return fuxi_np

    def normalization_nwp_2(self, fuxi_np):
        # channel:array(['ghi', 'msl', 'poai', 't2m', 'tcc', 'tp', 'u100', 'v100'], dtype=object)
        # c,h,w -> h,w,c
        fuxi_np = np.transpose(fuxi_np, axes=(1, 2, 0))
        # 选择channel
        # ghi & poai
        fuxi_np = np.stack((fuxi_np[..., 0], fuxi_np[..., 2], fuxi_np[..., 4]), axis=-1)

        mean_nwp_2 = np.array([self.NWP2_GHI_MEAN, self.NWP2_POAI_MEAN, self.NWP2_TCC_MEAN])
        std_nwp_2 = np.array([self.NWP2_GHI_STD, self.NWP2_POAI_STD, self.NWP2_TCC_STD])

        fuxi_np = (fuxi_np - mean_nwp_2) / std_nwp_2
        return fuxi_np

    def normalization_nwp_3(self, fuxi_np):
        # channel:array(['ghi', 'poai', 'sp', 't2m', 'tcc', 'tp', 'u100', 'v100'], dtype=object)
        # c,h,w -> h,w,c
        fuxi_np = np.transpose(fuxi_np, axes=(1, 2, 0))
        # 选择channel
        # u & v & sp & t2m & tp
        fuxi_np = np.stack((fuxi_np[..., 0], fuxi_np[..., 1], fuxi_np[..., 4]), axis=-1)

        mean_nwp_3 = np.array([self.NWP3_GHI_MEAN, self.NWP3_POAI_MEAN, self.NWP3_TCC_MEAN])
        std_nwp_3 = np.array([self.NWP3_GHI_STD, self.NWP3_POAI_STD, self.NWP3_TCC_STD])

        fuxi_np = (fuxi_np - mean_nwp_3) / std_nwp_3
        return fuxi_np

    def normalization_gt(self, cra_np):
        cra_np = np.maximum(cra_np, 0)
        return cra_np

    def rev_normalization_gt(self, gt_np):
        """
        将结果clip到[0,1]
        """
        gt_np = np.clip(gt_np, 0, 1)
        return gt_np
    
    
    def __getitem__(self, index):
        # nwp_1_path:/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/tanruxin/sth_power/data/train_np/nwp_data_train/1/20241228.nc
        nwp_1_path = self.paths_nwp_1[index]
        img_nwp_1 = np.load(nwp_1_path)

        # 11,11,6
        img_nwp_1 = self.normalization_nwp_1(img_nwp_1)

        nwp_2_path = self.paths_nwp_2[index]
        img_nwp_2 = np.load(nwp_2_path)
        img_nwp_2 = self.normalization_nwp_2(img_nwp_2)

        nwp_3_path = self.paths_nwp_3[index]
        img_nwp_3 = np.load(nwp_3_path)
        img_nwp_3 = self.normalization_nwp_3(img_nwp_3)

        # 11,11,x
        img_nwp = np.concatenate((img_nwp_1, img_nwp_2, img_nwp_3), axis=2)

        # 2024123123
        now_time_hour = os.path.basename(nwp_1_path).split('.')[0]
        # 精确到分钟
        now_time_min = now_time_hour + '00'

        img_gt = self.tabel_gt[now_time_min]
        img_gt = self.normalization_gt(img_gt)

        # 时序信息
        # 转换为 datetime
        dt = pd.to_datetime(now_time_hour, format='%Y%m%d%H')
        # 提取时间特征
        hour = dt.hour
        day_of_year = dt.dayofyear

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_nwp = single2tensor3(img_nwp)
        img_gt = torch.tensor([img_gt], dtype=torch.float32)

        hour = torch.tensor(hour, dtype=torch.float32)
        day_of_year = torch.tensor(day_of_year, dtype=torch.float32)

        # return {'L': img_nwp, 'H': img_gt, 'hour': hour, 'day_of_year': day_of_year, 'nwp_1_path': nwp_1_path}
        return img_nwp, img_gt #compatable with fcame train_one_epoch()

    def __len__(self):
        return len(self.paths_nwp_1)


class DatasetSthPowerWind(data.Dataset):
    '''
    # -----------------------------------------
    # Get L/H for SISR.
    # If only "paths_H" is provided, sythesize bicubicly downsampled L on-the-fly.
    # -----------------------------------------
    # e.g., SRResNet
    # -----------------------------------------
    '''

    def __init__(self, opt):
        super(DatasetSthPowerWind, self).__init__()
        self.opt = opt
        self.n_channels = opt['n_channels'] if opt['n_channels'] else 3
        self.sf = opt['scale'] if opt['scale'] else 4
        self.patch_size = self.opt['H_size'] if self.opt['H_size'] else 96
        self.fuxi_size = self.patch_size // self.sf

        # ------------------------------------
        # get paths of L/H
        # ------------------------------------
        self.paths_nwp_1 = get_image_paths(os.path.join(opt['dataroot_nwp'], "NWP_1"))
        self.paths_nwp_2 = get_image_paths(os.path.join(opt['dataroot_nwp'], "NWP_2"))
        self.paths_nwp_3 = get_image_paths(os.path.join(opt['dataroot_nwp'], "NWP_3"))

        self.tabel_gt = self._csv_to_dict(opt['dataroot_gt'])

        # 初始化mean,std
        self.const_path = opt['const_path']

        self.NWP1_U100_MEAN = np.load(os.path.join(self.const_path, 'NWP_1_u100_mean.npy'))
        self.NWP1_U100_STD = np.load(os.path.join(self.const_path, 'NWP_1_u100_std.npy'))
        self.NWP1_V100_MEAN = np.load(os.path.join(self.const_path, 'NWP_1_v100_mean.npy'))
        self.NWP1_V100_STD = np.load(os.path.join(self.const_path, 'NWP_1_v100_std.npy'))
        self.NWP1_SP_MEAN = np.load(os.path.join(self.const_path, 'NWP_1_sp_mean.npy'))
        self.NWP1_SP_STD = np.load(os.path.join(self.const_path, 'NWP_1_sp_std.npy'))
        self.NWP1_T2M_MEAN = np.load(os.path.join(self.const_path, 'NWP_1_t2m_mean.npy'))
        self.NWP1_T2M_STD = np.load(os.path.join(self.const_path, 'NWP_1_t2m_std.npy'))
        self.NWP1_TP_MEAN = np.load(os.path.join(self.const_path, 'NWP_1_tp_mean.npy'))
        self.NWP1_TP_STD = np.load(os.path.join(self.const_path, 'NWP_1_tp_std.npy'))

        self.NWP2_U100_MEAN = np.load(os.path.join(self.const_path, 'NWP_2_u100_mean.npy'))
        self.NWP2_U100_STD = np.load(os.path.join(self.const_path, 'NWP_2_u100_std.npy'))
        self.NWP2_V100_MEAN = np.load(os.path.join(self.const_path, 'NWP_2_v100_mean.npy'))
        self.NWP2_V100_STD = np.load(os.path.join(self.const_path, 'NWP_2_v100_std.npy'))
        self.NWP2_MSL_MEAN = np.load(os.path.join(self.const_path, 'NWP_2_msl_mean.npy'))
        self.NWP2_MSL_STD = np.load(os.path.join(self.const_path, 'NWP_2_msl_std.npy'))
        self.NWP2_T2M_MEAN = np.load(os.path.join(self.const_path, 'NWP_2_t2m_mean.npy'))
        self.NWP2_T2M_STD = np.load(os.path.join(self.const_path, 'NWP_2_t2m_std.npy'))
        self.NWP2_TP_MEAN = np.load(os.path.join(self.const_path, 'NWP_2_tp_mean.npy'))
        self.NWP2_TP_STD = np.load(os.path.join(self.const_path, 'NWP_2_tp_std.npy'))

        self.NWP3_U100_MEAN = np.load(os.path.join(self.const_path, 'NWP_3_u100_mean.npy'))
        self.NWP3_U100_STD = np.load(os.path.join(self.const_path, 'NWP_3_u100_std.npy'))
        self.NWP3_V100_MEAN = np.load(os.path.join(self.const_path, 'NWP_3_v100_mean.npy'))
        self.NWP3_V100_STD = np.load(os.path.join(self.const_path, 'NWP_3_v100_std.npy'))
        self.NWP3_SP_MEAN = np.load(os.path.join(self.const_path, 'NWP_3_sp_mean.npy'))
        self.NWP3_SP_STD = np.load(os.path.join(self.const_path, 'NWP_3_sp_std.npy'))
        self.NWP3_T2M_MEAN = np.load(os.path.join(self.const_path, 'NWP_3_t2m_mean.npy'))
        self.NWP3_T2M_STD = np.load(os.path.join(self.const_path, 'NWP_3_t2m_std.npy'))
        self.NWP3_TP_MEAN = np.load(os.path.join(self.const_path, 'NWP_3_tp_mean.npy'))
        self.NWP3_TP_STD = np.load(os.path.join(self.const_path, 'NWP_3_tp_std.npy'))

        assert self.paths_nwp_1, 'Error: H path is empty.'

    def _csv_to_dict(self, file_path):
        """
        读取gt的csv文件
        返回table = {"202401020000"：value, ...}
        """
        # 读取 CSV 文件
        df = pd.read_csv(file_path)

        # 确保第一列是时间，第二列是功率
        df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0]).dt.strftime("%Y%m%d%H%M")  # 转换时间列为 datetime 格式

        # 转换 DataFrame 为字典
        data_dict = dict(zip(df.iloc[:, 0], df.iloc[:, 1]))

        return data_dict

    def normalization_nwp_1(self, fuxi_np):
        # channle排列顺序为：array(['ghi', 'poai', 'sp', 't2m', 'tcc', 'tp', 'u100', 'v100'], dtype=object)
        # c,h,w -> h,w,c
        fuxi_np = np.transpose(fuxi_np, axes=(1, 2, 0))
        # 选择channel
        # u & v & sp & t2m & tp
        # 11,11,5
        fuxi_np = np.stack((fuxi_np[..., 6], fuxi_np[..., 7], fuxi_np[..., 2], fuxi_np[..., 3], fuxi_np[..., 5]),
                           axis=-1)

        mean_nwp_1 = np.array(
            [self.NWP1_U100_MEAN, self.NWP1_V100_MEAN, self.NWP1_SP_MEAN, self.NWP1_T2M_MEAN, self.NWP1_TP_MEAN])
        std_nwp_1 = np.array(
            [self.NWP1_U100_STD, self.NWP1_V100_STD, self.NWP1_SP_STD, self.NWP1_T2M_STD, self.NWP1_TP_STD])

        fuxi_np = (fuxi_np - mean_nwp_1) / std_nwp_1

        # 计算ws
        u_fuxi_np = fuxi_np[..., 0:1]
        v_fuxi_np = fuxi_np[..., 1:2]
        ws_fuxi_np = np.sqrt(u_fuxi_np ** 2 + v_fuxi_np ** 2)

        # 11,11,6
        fuxi_np = np.concatenate((fuxi_np, ws_fuxi_np), axis=-1)
        return fuxi_np

    def normalization_nwp_2(self, fuxi_np):
        # channel:array(['ghi', 'msl', 'poai', 't2m', 'tcc', 'tp', 'u100', 'v100'], dtype=object)
        # c,h,w -> h,w,c
        fuxi_np = np.transpose(fuxi_np, axes=(1, 2, 0))
        # 选择channel
        # u & v & msl & t2m & tp
        fuxi_np = np.stack((fuxi_np[..., 6], fuxi_np[..., 7], fuxi_np[..., 1], fuxi_np[..., 3], fuxi_np[..., 5]),
                           axis=-1)

        mean_nwp_2 = np.array(
            [self.NWP2_U100_MEAN, self.NWP2_V100_MEAN, self.NWP2_MSL_MEAN, self.NWP2_T2M_MEAN, self.NWP2_TP_MEAN])
        std_nwp_2 = np.array(
            [self.NWP2_U100_STD, self.NWP2_V100_STD, self.NWP2_MSL_STD, self.NWP2_T2M_STD, self.NWP2_TP_STD])

        fuxi_np = (fuxi_np - mean_nwp_2) / std_nwp_2

        # 计算ws
        u_fuxi_np = fuxi_np[..., 0:1]
        v_fuxi_np = fuxi_np[..., 1:2]
        ws_fuxi_np = np.sqrt(u_fuxi_np ** 2 + v_fuxi_np ** 2)

        # 11,11,6
        fuxi_np = np.concatenate((fuxi_np, ws_fuxi_np), axis=-1)
        return fuxi_np

    def normalization_nwp_3(self, fuxi_np):
        # channel:array(['ghi', 'poai', 'sp', 't2m', 'tcc', 'tp', 'u100', 'v100'], dtype=object)
        # c,h,w -> h,w,c
        fuxi_np = np.transpose(fuxi_np, axes=(1, 2, 0))
        # 选择channel
        # u & v & sp & t2m & tp
        fuxi_np = np.stack((fuxi_np[..., 6], fuxi_np[..., 7], fuxi_np[..., 2], fuxi_np[..., 3], fuxi_np[..., 5]),
                           axis=-1)

        mean_nwp_3 = np.array(
            [self.NWP3_U100_MEAN, self.NWP3_V100_MEAN, self.NWP3_SP_MEAN, self.NWP3_T2M_MEAN, self.NWP3_TP_MEAN])
        std_nwp_3 = np.array(
            [self.NWP3_U100_STD, self.NWP3_V100_STD, self.NWP3_SP_STD, self.NWP3_T2M_STD, self.NWP3_TP_STD])

        fuxi_np = (fuxi_np - mean_nwp_3) / std_nwp_3

        # 计算ws
        u_fuxi_np = fuxi_np[..., 0:1]
        v_fuxi_np = fuxi_np[..., 1:2]
        ws_fuxi_np = np.sqrt(u_fuxi_np ** 2 + v_fuxi_np ** 2)

        # 11,11,6
        fuxi_np = np.concatenate((fuxi_np, ws_fuxi_np), axis=-1)
        return fuxi_np

    def normalization_gt(self, cra_np):
        cra_np = np.maximum(cra_np, 0)
        return cra_np

    def rev_normalization_gt(self, gt_np):
        """
        将结果clip到[0,1]
        """
        gt_np = np.clip(gt_np, 0, 1)
        return gt_np

    def __getitem__(self, index):
        # nwp_1_path:/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/tanruxin/sth_power/data/train_np/nwp_data_train/1/20241228.nc
        nwp_1_path = self.paths_nwp_1[index]
        img_nwp_1 = np.load(nwp_1_path)

        # 11,11,6
        img_nwp_1 = self.normalization_nwp_1(img_nwp_1)

        nwp_2_path = self.paths_nwp_2[index]
        img_nwp_2 = np.load(nwp_2_path)
        img_nwp_2 = self.normalization_nwp_2(img_nwp_2)

        nwp_3_path = self.paths_nwp_3[index]
        img_nwp_3 = np.load(nwp_3_path)
        img_nwp_3 = self.normalization_nwp_3(img_nwp_3)

        # 11,11,x
        img_nwp = np.concatenate((img_nwp_1, img_nwp_2, img_nwp_3), axis=2)

        now_time = os.path.basename(nwp_1_path).split('.')[0]
        # 精确到分钟
        now_time = now_time + '00'

        img_gt = self.tabel_gt[now_time]
        img_gt = self.normalization_gt(img_gt)

        # ------------------------------------
        # L/H pairs, HWC to CHW, numpy to tensor
        # ------------------------------------
        img_nwp = single2tensor3(img_nwp)
        img_gt = torch.tensor([img_gt], dtype=torch.float32)

        #return {'L': img_nwp, 'H': img_gt, 'path': nwp_1_path}
        return img_nwp, img_gt #compatable with fcame train_one_epoch()
    
    def __len__(self):
        return len(self.paths_nwp_1)



def get_x_y_solar(opt, batch_size=1024):
    dataset = DatasetSthPowerSolar(opt)
    dataset_size = len(dataset)
    logger.info(f"数据集大小: {dataset_size}")

    # 使用较小的批量大小，避免内存溢出
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    X_list = []
    Y_list = []

    logger.info("开始批量加载数据...")
    total_batches = len(loader)

    for batch_idx, batch_data in enumerate(loader):
        if batch_idx % 10 == 0:
            logger.info(f"处理批次 {batch_idx + 1}/{total_batches}")

        # 处理批次数据
        B = batch_data['L'].shape[0]
        X_batch = batch_data['L'].numpy().reshape(B, -1).astype(np.float32)
        Y_batch = batch_data['H'].numpy().reshape(B).astype(np.float32)

        X_list.append(X_batch)
        Y_list.append(Y_batch)

    # 合并所有批次
    logger.info("合并批次数据...")
    X_final = np.vstack(X_list)
    Y_final = np.hstack(Y_list)

    logger.info(f"数据加载完成: X shape={X_final.shape}, Y shape={Y_final.shape}")
    return X_final, Y_final



def get_x_y_wind(opt, batch_size=1024):
    dataset = D_wind(opt)
    dataset_size = len(dataset)
    logger.info(f"数据集大小: {dataset_size}")

    # 使用较小的批量大小，避免内存溢出
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)

    X_list = []
    Y_list = []

    logger.info("开始批量加载数据...")
    total_batches = len(loader)

    for batch_idx, batch_data in enumerate(loader):
        # 处理批次数据
        B = batch_data['L'].shape[0]
        X_batch = batch_data['L'].numpy().reshape(B, -1).astype(np.float32)
        Y_batch = batch_data['H'].numpy().reshape(B).astype(np.float32)

        X_list.append(X_batch)
        Y_list.append(Y_batch)

    # 合并所有批次
    logger.info("合并批次数据...")
    X_final = np.vstack(X_list)
    Y_final = np.hstack(Y_list)

    logger.info(f"数据加载完成: X shape={X_final.shape}, Y shape={Y_final.shape}")
    return X_final, Y_final



######### ############################3

dataroot_gt_dir = '/inspire/ssd/project/sais-mtm/public/qlz/data/PowerEstimateData/PowerEstimateData-Per-Farm-Debug/fact_data'
dataroot_nwp = '/inspire/ssd/project/sais-mtm/public/qlz/data/PowerEstimateData/PowerEstimateData-Per-Farm-Debug'
const_dir = '/inspire/ssd/project/sais-mtm/public/qlz/data/PowerEstimateData/PowerEstimateData-Per-Farm-Debug/const'

farm_id_solar = '1329'
farm_id_wind = '1001'

# get solar data
# opt_solar = {
#     'dataroot_gt': os.path.join(dataroot_gt_dir, f'{farm_id_solar}_norm.csv'),
#     'dataroot_nwp': os.path.join(dataroot_nwp, f'solar_dataset_np_filter_match/nwp_data_train/{farm_id_solar}'),
#     'n_channels': 1,
#     'scale': 1,
#     'H_size': 11,
#     'const_path': os.path.join(const_dir, farm_id_solar)
# }
# X_list_solar, Y_list_solar = get_x_y_solar(opt_solar)
# print("x list solar", X_list_solar)
# print("y list solar", Y_list_solar)


# get wind data
# opt_wind = {
#     'dataroot_gt': os.path.join(dataroot_gt_dir, f'{farm_id_wind}_norm.csv'),
#     'dataroot_nwp': os.path.join(dataroot_nwp, f'wind_dataset_np_match/nwp_data_train/{farm_id_wind}'),
#     'n_channels': 1,
#     'scale': 1,
#     'H_size': 11,
#     'const_path': os.path.join(const_dir, farm_id_wind)
# }
# X_list_wind, Y_list_wind = get_x_y_wind(opt_wind)
# print("x list wind", X_list_wind)
# print("y list wind", Y_list_wind)