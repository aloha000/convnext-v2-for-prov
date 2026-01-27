"""
对南网线上运行的结果进行批量评估
读取每个时刻的预测文件，选取d+1的结果，拼接成连续的d+1预测结果
读取每个时刻的预测文件，选取d+4的结果，拼接成连续的d+1预测结果
re_norm后的结果，考虑容量
"""
import os

import pandas as pd
from loguru import logger

start_date = '20250801'
end_date = '20250904'

farm_output_path = './qlz_test'
save_root_path = farm_output_path

farms = ['1000']


pd_start_date = pd.to_datetime(start_date, format='%Y%m%d')
pd_end_date = pd.to_datetime(end_date, format='%Y%m%d')

for farm in farms:
    logger.info(farm)
    output_all = None
    save_path = os.path.join(save_root_path, farm)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for pd_date in pd.date_range(pd_start_date, pd_end_date, freq='D'):
        date = pd_date.strftime('%Y%m%d')
        logger.info(date)
        csv_path = os.path.join(farm_output_path, date, farm, 'output.csv')
        output = pd.read_csv(csv_path)
        # 选取d+1预测结果
        output_d1 = output.iloc[:96, :]
        # 选取d+4预测结果
        # output_d1 = output.iloc[-97:-1, :]
        # 时间进行拼接
        if output_all is None:
            output_all = output_d1
        else:
            output_all = pd.concat((output_all, output_d1))
    # 保存
    output_all.to_csv(os.path.join(save_path, 'output.csv'), index=False)

