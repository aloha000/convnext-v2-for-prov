import torch
import numpy as np
from loguru import logger
import os
import pandas as pd

from wind_utils import wind_get_normalization_func, renorm_csv
from models.convnextv2 import ConvNeXtV2ForPowerEstimate, ConvNeXtV2
from models.mlp import PowerEstimatorMLP

def create_pred_series(data: np.ndarray, start_time, end_time, freq='h'):
    time_index = pd.date_range(start=start_time, end=end_time, freq=freq)
    assert len(data) == len(time_index), "Length of data and time index must match"
    return pd.Series(data, index=time_index)


def get_test_x(nwp_1_path, nwp_2_path, nwp_3_path, guiyihua_nwp_1, guiyihua_nwp_2, guiyihua_nwp_3):
    outputs = []
    for idx, img_name in enumerate(sorted(os.listdir(nwp_1_path))):
        img_nwp_1 = np.load(os.path.join(nwp_1_path, img_name))
        img_nwp_2 = np.load(os.path.join(nwp_2_path, img_name))
        img_nwp_3 = np.load(os.path.join(nwp_3_path, img_name))

        img_nwp_1 = guiyihua_nwp_1(img_nwp_1)
        img_nwp_2 = guiyihua_nwp_2(img_nwp_2)
        img_nwp_3 = guiyihua_nwp_3(img_nwp_3)

        # 11,11,x
        img_nwp = np.concatenate((img_nwp_1, img_nwp_2, img_nwp_3), axis=2)

        # 11,11,x -> x,11,11
        img_nwp = img_nwp.transpose(2, 0, 1)

        outputs.append(img_nwp)
    B = len(outputs)
    C, H, W = outputs[0].shape
    outputs = np.array(outputs).reshape(B, C, H, W)
    return outputs


def parse_time_from_filename(filename: str) -> pd.Timestamp:
    """
    从文件名中解析时间

    Args:
        filename: 如 '2024010200.npy'

    Returns:
        pandas Timestamp对象
    """
    # 去掉.npy后缀
    time_str = filename.replace('.npy', '')
    # 解析时间：YYYYMMDDHH
    return pd.to_datetime(time_str, format='%Y%m%d%H')


def main(dataroot_nwp, model_path, save_dir, farm, save_const_path, pred_time, max_power_path, save_renorm_dir):
    # 加载模型
    model = ConvNeXtV2(
        in_chans = 18,
        num_classes=1000, # useless
        depths=[2, 2, 6, 2],
        dims=[9,18,36,72],
        drop_path_rate=0.,
        head_init_scale=0.001,
    ).to("cuda")
    mlp = PowerEstimatorMLP(input_dim=72).to("cuda")
    model = ConvNeXtV2ForPowerEstimate(model, mlp)
    model_state_dict = torch.load(model_path)
    model.load_state_dict(model_state_dict["model"])

    logger.info("start load test data")
    # 6. 模型推理
    nwp_1_path = os.path.join(dataroot_nwp, 'NWP_1')
    nwp_2_path = os.path.join(dataroot_nwp, 'NWP_2')
    nwp_3_path = os.path.join(dataroot_nwp, 'NWP_3')

    # 根据farm确定rev_normalization_gt, normalization_nwp_1, normalization_nwp_2, normalization_nwp_3
    const_path = os.path.join(save_const_path, farm)
    rev_normalization_gt, normalization_nwp_1, normalization_nwp_2, normalization_nwp_3 = wind_get_normalization_func(const_path)
    X_test = get_test_x(nwp_1_path, nwp_2_path, nwp_3_path, normalization_nwp_1, normalization_nwp_2, normalization_nwp_3)

    logger.info(f"✅ 成功加载样本数: {X_test.shape[0]}，特征维度: {X_test.shape[1]}")
    X_test = torch.tensor(X_test).to(torch.float).to("cuda")
    with torch.no_grad():
        y_pred = model(X_test)
        y_pred = y_pred.detach().cpu().numpy().flatten()

    logger.info(f"🎯 预测结果: {y_pred}")

    # ----------------------------
    # 6. 保存预测结果（可选）
    # 保存到csv
    pred_pw = np.array(list(map(rev_normalization_gt, y_pred)))

    os.makedirs(os.path.dirname(save_dir), exist_ok=True)

    # pred_time:"20250225"
    # 确定推理的时间范围：d+1->d+4
    pd_start = pd.to_datetime(pred_time, format="%Y%m%d")
    pd_start = pd_start + pd.Timedelta(days=1)
    pd_end = pd_start + pd.Timedelta(days=4)
    # start = datetime(2025, 1, 1, 0)  # 2025-04-07 00:00
    # end = datetime(2025, 2, 28, 23)  # 2025-04-07 03:00
    pred = create_pred_series(pred_pw, pd_start, pd_end, freq='h')
    res = pred.resample('15T').interpolate(method='linear')
    # 去除最后一个时刻
    res = res.iloc[:-1]
    res.to_csv(save_dir)
    logger.info("finish save output")

    # 将csv的结果返归一化
    renorm_csv(save_dir, max_power_path, save_renorm_dir)
    res = res.to_frame().reset_index()
    return res

if __name__ == '__main__':

    farms = ['1000']
    
    # 离线多步预测
    beg_str = '20250801'
    end_str = '20250904'
    
    npy_dir = '/inspire/ssd/project/sais-mtm/public/qlz/data/PowerEstimateData/PowerEstimateData-Per-Farm-Test'
    const_dir = '/inspire/ssd/project/sais-mtm/public/qlz/data/PowerEstimateData/const'
    inference_output_dir = './qlz_test'
    model_path = "/inspire/ssd/project/sais-mtm/public/qlz/code/PowerEstimate/ConvNeXt-V2/checkpoints/gj-mlp-wind-1000-72/checkpoint-5.pth"
    
    pred_times = [itime.strftime('%Y%m%d') for itime in pd.date_range(start=beg_str, end=end_str, freq='24H')]
    for pred_time in pred_times:
        for farm in farms:

            ## 2.lgb推理
            dataroot_nwp = os.path.join(npy_dir, farm, pred_time)
            
            save_const_path = const_dir
            
            # 功率数据的归一化值保存路径
            max_power_path = os.path.join(const_dir, farm, 'power_max.npy')
            
            save_dir = os.path.join(inference_output_dir, pred_time, farm, 'norm_output.csv')
            
            save_renorm_dir = os.path.join(inference_output_dir, pred_time, farm, 'output.csv')

            main(dataroot_nwp, model_path, save_dir, farm, save_const_path, pred_time, max_power_path, save_renorm_dir)