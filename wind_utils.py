import os
import numpy as np
import glob
import re
import pandas as pd
import xarray as xr
import csv
from loguru import logger


def wind_get_normalization_func(const_path):
    NWP1_U100_MEAN = np.load(os.path.join(const_path, 'NWP_1_u100_mean.npy'))
    NWP1_U100_STD = np.load(os.path.join(const_path, 'NWP_1_u100_std.npy'))
    NWP1_V100_MEAN = np.load(os.path.join(const_path, 'NWP_1_v100_mean.npy'))
    NWP1_V100_STD = np.load(os.path.join(const_path, 'NWP_1_v100_std.npy'))
    NWP1_SP_MEAN = np.load(os.path.join(const_path, 'NWP_1_sp_mean.npy'))
    NWP1_SP_STD = np.load(os.path.join(const_path, 'NWP_1_sp_std.npy'))
    NWP1_T2M_MEAN = np.load(os.path.join(const_path, 'NWP_1_t2m_mean.npy'))
    NWP1_T2M_STD = np.load(os.path.join(const_path, 'NWP_1_t2m_std.npy'))
    NWP1_TP_MEAN = np.load(os.path.join(const_path, 'NWP_1_tp_mean.npy'))
    NWP1_TP_STD = np.load(os.path.join(const_path, 'NWP_1_tp_std.npy'))

    NWP2_U100_MEAN = np.load(os.path.join(const_path, 'NWP_2_u100_mean.npy'))
    NWP2_U100_STD = np.load(os.path.join(const_path, 'NWP_2_u100_std.npy'))
    NWP2_V100_MEAN = np.load(os.path.join(const_path, 'NWP_2_v100_mean.npy'))
    NWP2_V100_STD = np.load(os.path.join(const_path, 'NWP_2_v100_std.npy'))
    NWP2_MSL_MEAN = np.load(os.path.join(const_path, 'NWP_2_msl_mean.npy'))
    NWP2_MSL_STD = np.load(os.path.join(const_path, 'NWP_2_msl_std.npy'))
    NWP2_T2M_MEAN = np.load(os.path.join(const_path, 'NWP_2_t2m_mean.npy'))
    NWP2_T2M_STD = np.load(os.path.join(const_path, 'NWP_2_t2m_std.npy'))
    NWP2_TP_MEAN = np.load(os.path.join(const_path, 'NWP_2_tp_mean.npy'))
    NWP2_TP_STD = np.load(os.path.join(const_path, 'NWP_2_tp_std.npy'))

    NWP3_U100_MEAN = np.load(os.path.join(const_path, 'NWP_3_u100_mean.npy'))
    NWP3_U100_STD = np.load(os.path.join(const_path, 'NWP_3_u100_std.npy'))
    NWP3_V100_MEAN = np.load(os.path.join(const_path, 'NWP_3_v100_mean.npy'))
    NWP3_V100_STD = np.load(os.path.join(const_path, 'NWP_3_v100_std.npy'))
    NWP3_SP_MEAN = np.load(os.path.join(const_path, 'NWP_3_sp_mean.npy'))
    NWP3_SP_STD = np.load(os.path.join(const_path, 'NWP_3_sp_std.npy'))
    NWP3_T2M_MEAN = np.load(os.path.join(const_path, 'NWP_3_t2m_mean.npy'))
    NWP3_T2M_STD = np.load(os.path.join(const_path, 'NWP_3_t2m_std.npy'))
    NWP3_TP_MEAN = np.load(os.path.join(const_path, 'NWP_3_tp_mean.npy'))
    NWP3_TP_STD = np.load(os.path.join(const_path, 'NWP_3_tp_std.npy'))

    def normalization_nwp_1(fuxi_np):
        # channle排列顺序为：array(['ghi', 'poai', 'sp', 't2m', 'tcc', 'tp', 'u100', 'v100'], dtype=object)
        # c,h,w -> h,w,c
        fuxi_np = np.transpose(fuxi_np, axes=(1, 2, 0))
        # 选择channel
        # u & v & sp & t2m & tp
        # 11,11,5
        fuxi_np = np.stack((fuxi_np[..., 6], fuxi_np[..., 7], fuxi_np[..., 2], fuxi_np[..., 3], fuxi_np[..., 5]),
                           axis=-1)

        mean_nwp_1 = np.array([NWP1_U100_MEAN, NWP1_V100_MEAN, NWP1_SP_MEAN, NWP1_T2M_MEAN, NWP1_TP_MEAN])
        std_nwp_1 = np.array([NWP1_U100_STD, NWP1_V100_STD, NWP1_SP_STD, NWP1_T2M_STD, NWP1_TP_STD])

        fuxi_np = (fuxi_np - mean_nwp_1) / std_nwp_1

        # 计算ws
        u_fuxi_np = fuxi_np[..., 0:1]
        v_fuxi_np = fuxi_np[..., 1:2]
        ws_fuxi_np = np.sqrt(u_fuxi_np ** 2 + v_fuxi_np ** 2)

        # 11,11,6
        fuxi_np = np.concatenate((fuxi_np, ws_fuxi_np), axis=-1)
        return fuxi_np

    def normalization_nwp_2(fuxi_np):
        # channel:array(['ghi', 'msl', 'poai', 't2m', 'tcc', 'tp', 'u100', 'v100'], dtype=object)
        # c,h,w -> h,w,c
        fuxi_np = np.transpose(fuxi_np, axes=(1, 2, 0))
        # 选择channel
        # u & v & msl & t2m & tp
        fuxi_np = np.stack((fuxi_np[..., 6], fuxi_np[..., 7], fuxi_np[..., 1], fuxi_np[..., 3], fuxi_np[..., 5]),
                           axis=-1)

        mean_nwp_2 = np.array([NWP2_U100_MEAN, NWP2_V100_MEAN, NWP2_MSL_MEAN, NWP2_T2M_MEAN, NWP2_TP_MEAN])
        std_nwp_2 = np.array([NWP2_U100_STD, NWP2_V100_STD, NWP2_MSL_STD, NWP2_T2M_STD, NWP2_TP_STD])

        fuxi_np = (fuxi_np - mean_nwp_2) / std_nwp_2

        # 计算ws
        u_fuxi_np = fuxi_np[..., 0:1]
        v_fuxi_np = fuxi_np[..., 1:2]
        ws_fuxi_np = np.sqrt(u_fuxi_np ** 2 + v_fuxi_np ** 2)

        # 11,11,6
        fuxi_np = np.concatenate((fuxi_np, ws_fuxi_np), axis=-1)
        return fuxi_np

    def normalization_nwp_3(fuxi_np):
        # channel:array(['ghi', 'poai', 'sp', 't2m', 'tcc', 'tp', 'u100', 'v100'], dtype=object)
        # c,h,w -> h,w,c
        fuxi_np = np.transpose(fuxi_np, axes=(1, 2, 0))
        # 选择channel
        # u & v & sp & t2m & tp
        fuxi_np = np.stack((fuxi_np[..., 6], fuxi_np[..., 7], fuxi_np[..., 2], fuxi_np[..., 3], fuxi_np[..., 5]),
                           axis=-1)

        mean_nwp_3 = np.array([NWP3_U100_MEAN, NWP3_V100_MEAN, NWP3_SP_MEAN, NWP3_T2M_MEAN, NWP3_TP_MEAN])
        std_nwp_3 = np.array([NWP3_U100_STD, NWP3_V100_STD, NWP3_SP_STD, NWP3_T2M_STD, NWP3_TP_STD])

        fuxi_np = (fuxi_np - mean_nwp_3) / std_nwp_3

        # 计算ws
        u_fuxi_np = fuxi_np[..., 0:1]
        v_fuxi_np = fuxi_np[..., 1:2]
        ws_fuxi_np = np.sqrt(u_fuxi_np ** 2 + v_fuxi_np ** 2)

        # 11,11,6
        fuxi_np = np.concatenate((fuxi_np, ws_fuxi_np), axis=-1)
        return fuxi_np

    def rev_normalization_gt(gt_np):
        """
        将结果clip到[0,1]
        """
        gt_np = np.clip(gt_np, 0, 1)
        return gt_np

    return rev_normalization_gt, normalization_nwp_1, normalization_nwp_2, normalization_nwp_3


def solar_get_normalization_func(const_path):
    NWP1_GHI_MEAN = np.load(os.path.join(const_path, 'NWP_1_ghi_mean.npy'))
    NWP1_GHI_STD = np.load(os.path.join(const_path, 'NWP_1_ghi_std.npy'))
    NWP1_POAI_MEAN = np.load(os.path.join(const_path, 'NWP_1_poai_mean.npy'))
    NWP1_POAI_STD = np.load(os.path.join(const_path, 'NWP_1_poai_std.npy'))
    NWP1_TCC_MEAN = np.load(os.path.join(const_path, 'NWP_1_tcc_mean.npy'))
    NWP1_TCC_STD = np.load(os.path.join(const_path, 'NWP_1_tcc_std.npy'))

    NWP2_GHI_MEAN = np.load(os.path.join(const_path, 'NWP_2_ghi_mean.npy'))
    NWP2_GHI_STD = np.load(os.path.join(const_path, 'NWP_2_ghi_std.npy'))
    NWP2_POAI_MEAN = np.load(os.path.join(const_path, 'NWP_2_poai_mean.npy'))
    NWP2_POAI_STD = np.load(os.path.join(const_path, 'NWP_2_poai_std.npy'))
    NWP2_TCC_MEAN = np.load(os.path.join(const_path, 'NWP_2_tcc_mean.npy'))
    NWP2_TCC_STD = np.load(os.path.join(const_path, 'NWP_2_tcc_std.npy'))

    NWP3_GHI_MEAN = np.load(os.path.join(const_path, 'NWP_3_ghi_mean.npy'))
    NWP3_GHI_STD = np.load(os.path.join(const_path, 'NWP_3_ghi_std.npy'))
    NWP3_POAI_MEAN = np.load(os.path.join(const_path, 'NWP_3_poai_mean.npy'))
    NWP3_POAI_STD = np.load(os.path.join(const_path, 'NWP_3_poai_std.npy'))
    NWP3_TCC_MEAN = np.load(os.path.join(const_path, 'NWP_3_tcc_mean.npy'))
    NWP3_TCC_STD = np.load(os.path.join(const_path, 'NWP_3_tcc_std.npy'))

    def normalization_nwp_1(fuxi_np):
        # channle排列顺序为：array(['ghi', 'poai', 'sp', 't2m', 'tcc', 'tp', 'u100', 'v100'], dtype=object)
        # c,h,w -> h,w,c
        fuxi_np = np.transpose(fuxi_np, axes=(1, 2, 0))
        # 选择channel
        # ghi & poai & tcc
        # 11,11,2
        fuxi_np = np.stack((fuxi_np[..., 0], fuxi_np[..., 1], fuxi_np[..., 4]), axis=-1)

        mean_nwp_1 = np.array([NWP1_GHI_MEAN, NWP1_POAI_MEAN, NWP1_TCC_MEAN])
        std_nwp_1 = np.array([NWP1_GHI_STD, NWP1_POAI_STD, NWP1_TCC_STD])

        fuxi_np = (fuxi_np - mean_nwp_1) / std_nwp_1
        return fuxi_np

    def normalization_nwp_2(fuxi_np):
        # channel:array(['ghi', 'msl', 'poai', 't2m', 'tcc', 'tp', 'u100', 'v100'], dtype=object)
        # c,h,w -> h,w,c
        fuxi_np = np.transpose(fuxi_np, axes=(1, 2, 0))
        # 选择channel
        # ghi & poai
        fuxi_np = np.stack((fuxi_np[..., 0], fuxi_np[..., 2], fuxi_np[..., 4]), axis=-1)

        mean_nwp_2 = np.array([NWP2_GHI_MEAN, NWP2_POAI_MEAN, NWP2_TCC_MEAN])
        std_nwp_2 = np.array([NWP2_GHI_STD, NWP2_POAI_STD, NWP2_TCC_STD])

        fuxi_np = (fuxi_np - mean_nwp_2) / std_nwp_2
        return fuxi_np

    def normalization_nwp_3(fuxi_np):
        # channel:array(['ghi', 'poai', 'sp', 't2m', 'tcc', 'tp', 'u100', 'v100'], dtype=object)
        # c,h,w -> h,w,c
        fuxi_np = np.transpose(fuxi_np, axes=(1, 2, 0))
        # 选择channel
        # u & v & sp & t2m & tp
        fuxi_np = np.stack((fuxi_np[..., 0], fuxi_np[..., 1], fuxi_np[..., 4]), axis=-1)

        mean_nwp_3 = np.array([NWP3_GHI_MEAN, NWP3_POAI_MEAN, NWP3_TCC_MEAN])
        std_nwp_3 = np.array([NWP3_GHI_STD, NWP3_POAI_STD, NWP3_TCC_STD])

        fuxi_np = (fuxi_np - mean_nwp_3) / std_nwp_3
        return fuxi_np

    def rev_normalization_gt(gt_np):
        """
        将结果clip到[0,1]
        """
        gt_np = np.clip(gt_np, 0, 1)
        return gt_np

    return rev_normalization_gt, normalization_nwp_1, normalization_nwp_2, normalization_nwp_3


def find_latest_ckpt(save_dir, net_type='G'):
    file_list = glob.glob(os.path.join(save_dir, '*_{}.pth'.format(net_type)))
    assert file_list
    iter_exist = []
    for file_ in file_list:
        iter_current = re.findall(r"(\d+)_{}.pth".format(net_type), file_)
        iter_exist.append(int(iter_current[0]))
    init_iter = max(iter_exist)
    init_path = os.path.join(save_dir, '{}_{}.pth'.format(init_iter, net_type))
    return init_path


def renorm_csv(out_path, max_path, out_renorm_path):
    """
    读取csv文件，进行反归一化操作
    """
    # 读取 CSV 文件
    df = pd.read_csv(out_path)
    # 读取归一化值
    max_value = np.load(max_path)
    # 反归一化
    df.iloc[:, 1] = df.iloc[:, 1] * max_value
    # 保存到新的 CSV 文件
    df.to_csv(out_renorm_path, index=False)
    return df

def handle_csv(power_path, train_power_path, test_power_path, power_const_max_path):
    """
    1. 划分训练 & 验证集
    2. 格式 & 时区转换 & nan值填充
    3. 数值归一化（max-norm）
    """
    # 读取 CSV 文件
    df = pd.read_csv(power_path)

    # 假设第一列为时间列，格式为字符串 'YYYY/MM/DD/HH'
    # 将其转换为 datetime 类型
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0]).dt.tz_localize(None)

    # 时区转换为北京时
    df.iloc[:, 0] = df.iloc[:, 0] + pd.Timedelta(hours=8)

    # 填充时间序列
    # 假设你知道统一时间范围和频率
    full_index = pd.date_range(start='2024-04-09 00:00:00', end='2025-05-12 23:45:00', freq='15min')

    df = df.set_index(df.columns[0])
    df = df.reindex(full_index)  # 强行按统一时间轴对齐

    df = df.reset_index()
    df = df.rename(columns={"index": "time"})  # 可选：重命名为 'time'

    # 格式调整
    # 添加新的 label 列（可以先设置为默认值，如 0 或 None）
    df['label'] = 1

    # 填充nan值
    df.iloc[:, 1] = df.iloc[:, 1].ffill().bfill()  # 向前&后填充

    # max归一化
    # 最大值
    max_val = df.iloc[:, 1].max()
    # Max归一化
    df.iloc[:, 1] = df.iloc[:, 1] / max_val
    # 保存max值，供后续反归一化
    np.save(os.path.join(power_const_max_path, 'power_max.npy'), max_val)

    # 训练测试集划分
    # 筛选 2024/12/01 00 点 到 2024/12/31 23 点之间的数据
    train_start_time = pd.to_datetime('2024/04/09/00', format='%Y/%m/%d/%H')
    train_end_time = pd.to_datetime('2025/04/12/00', format='%Y/%m/%d/%H')

    df_train = df[(df.iloc[:, 0] >= train_start_time) & (df.iloc[:, 0] < train_end_time)]

    # 保存到新的 CSV 文件
    df_train.to_csv(train_power_path, index=False)

    test_start_time = pd.to_datetime('2025/04/12/00', format='%Y/%m/%d/%H')
    test_end_time = pd.to_datetime('2025/05/13/00', format='%Y/%m/%d/%H')

    df_test = df[(df.iloc[:, 0] >= test_start_time) & (df.iloc[:, 0] < test_end_time)]

    # 保存到新的 CSV 文件
    df_test.to_csv(test_power_path, index=False)

    # 保存未归一化的测试csv数据，供后续评测使用
    df_test.iloc[:, 1] = df_test.iloc[:, 1] * max_val
    # test_power_path:"/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/tanruxin/sth_power/data/online/fact_data/{}_normalization_test.csv".format(
    #         farm)
    test_power_renorm_path = test_power_path.replace("_normalization_", "_")
    df_test.to_csv(test_power_renorm_path, index=False)


def online_nc2np(nwp_nc_root_path, save_root_root_path, farm='506', time='20250225'):
    """
    farm:场站id
    time:测试的起报时间
    """
    name2nwp = {'HRES': 'NWP_1', 'FUXI': 'NWP_2', 'ENS_AVG': 'NWP_3'}
    NWP = ('HRES', 'FUXI', 'ENS_AVG')
    for nwp in NWP:
        nwp_nc_path = os.path.join(nwp_nc_root_path, farm, nwp)
        save_root_path = os.path.join(save_root_root_path, time, farm, name2nwp[nwp])
        if not os.path.exists(save_root_path):
            os.makedirs(save_root_path)

        nc_file = time + '.nc'
        assert nc_file in os.listdir(nwp_nc_path)
        # 20240101
        now_time = os.path.splitext(nc_file)[0]
        pd_now_time = pd.to_datetime(now_time, format="%Y%m%d")
        # 预报时刻 20240102
        pd_start_time = pd_now_time + pd.Timedelta(days=1)
        # 2024010200
        str_start_time = pd_start_time.strftime("%Y%m%d") + '00'

        nc_file_path = os.path.join(nwp_nc_path, nc_file)
        nc = xr.open_dataset(nc_file_path)
        # d+1--->d+4
        for lead_time in range(0, 97):
            hour_nc = nc.sel(lead_time=lead_time)
            # 8,11,11
            hour_np = hour_nc.to_array().to_numpy().squeeze()
            save_name = (pd_start_time + pd.Timedelta(hours=lead_time)).strftime("%Y%m%d%H")
            save_path = os.path.join(save_root_path, save_name)
            np.save(save_path, hour_np)
            logger.info(f"save {save_name}")
        nc.close()



def offline_nc2np(nwp_nc_root_path, save_root_root_path, farm='506', pred_time='2025', start_time='20250101', end_time='20250228'):
    """
    farm:场站id
    time:测试的起报时间
    """
    name2nwp = {'HRES': 'NWP_1', 'FUXI': 'NWP_2', 'ENS_AVG': 'NWP_3'}
    NWP = ('HRES', 'FUXI', 'ENS_AVG')

    # 生成日期范围（闭区间）
    date_range = pd.date_range(start=start_time, end=end_time, freq='D')

    # 转换为 "YYYYMMDD" 字符串格式
    # 时间-1天
    date_str_list = (date_range + pd.Timedelta(days=-1)).strftime('%Y%m%d').tolist()

    # 示例打印
    for time in date_str_list:
        print(time)
        for nwp in NWP:
            nwp_nc_path = os.path.join(nwp_nc_root_path, farm, nwp)
            save_root_path = os.path.join(save_root_root_path, pred_time, farm, name2nwp[nwp])
            if not os.path.exists(save_root_path):
                os.makedirs(save_root_path)

            nc_file = time + '.nc'
            assert nc_file in os.listdir(nwp_nc_path)
            # 20240101
            now_time = os.path.splitext(nc_file)[0]
            pd_now_time = pd.to_datetime(now_time, format="%Y%m%d")
            # 预报时刻 20240102
            pd_start_time = pd_now_time + pd.Timedelta(days=1)
            # 2024010200
            str_start_time = pd_start_time.strftime("%Y%m%d") + '00'

            nc_file_path = os.path.join(nwp_nc_path, nc_file)
            nc = xr.open_dataset(nc_file_path)
            # d+1--->d+4
            for lead_time in range(0, 24):
                hour_nc = nc.sel(lead_time=lead_time)
                # 8,11,11
                hour_np = hour_nc.to_array().to_numpy().squeeze()
                save_name = (pd_start_time + pd.Timedelta(hours=lead_time)).strftime("%Y%m%d%H")
                save_path = os.path.join(save_root_path, save_name)
                np.save(save_path, hour_np)
                logger.info(f"save {save_name}")
            nc.close()


def get_farm_info(csv_path):
    """
    读取csv文件，第一列为farm_id,第二列第三列为纬度、经度，第四列为是光场 or 风场
    """
    # 读取 CSV
    df = pd.read_csv(csv_path)

    # 转换为字典：key 为 farm_id，value 为其余三列的元组或列表
    result = {
        str(row['farm_id']): (row['latitude'], row['longitude'], row['field'])
        for _, row in df.iterrows()
    }

    return result


def write_farm_id(csv_file, farm_id):
    # 判断文件是否存在
    file_exists = os.path.isfile(csv_file)

    # 以追加模式打开，如果文件不存在则会创建
    with open(csv_file, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)

        # 如果文件不存在，则写入表头
        if not file_exists:
            writer.writerow(['station'])

        # 写入新的 farm_id 行
        writer.writerow([farm_id])

def set_adjacent_neighbors_zero(s: pd.Series, k: int = 2, min_len: int = 20) -> pd.Series:
    """
    对每个连续为 0 的区间，若长度 >= min_len，则将该区间左右各 k 个相邻值（如果存在）也置为 0。
    """
    s = s.copy()
    n = len(s)
    if n == 0:
        return s

    # 是否为0
    is_zero = s.eq(0).to_numpy(dtype=bool)

    # 连续段分组
    change = np.empty(n, dtype=bool)
    change[0] = True
    change[1:] = is_zero[1:] != is_zero[:-1]
    grp = np.cumsum(change)

    # 掩码
    mask = np.zeros(n, dtype=bool)

    for label in np.unique(grp):
        pos = np.flatnonzero(grp == label)
        if is_zero[pos[0]] and len(pos) >= min_len:  # 连续零段 且 长度满足要求
            left = max(0, pos[0] - k)
            right = min(n - 1, pos[-1] + k)
            mask[left:right+1] = True

    s.iloc[mask] = 0
    return s



