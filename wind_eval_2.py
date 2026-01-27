
# coding=utf-8
import json
import os.path
import sys
import numpy as np
import pandas as pd
import time
from decimal import Decimal
from loguru import logger
from datetime import timedelta


def get_cap(txt_path):
    result = {}
    lines = open(txt_path, 'r').readlines()
    info = lines[1].split('查询所有场站信息：')[1].strip()
    # 安全解析字符串为 Python 对象
    # 注意 Decimal 是一个对象，这里通过定义全局环境使其能被识别
    # 替换 Decimal 表达式，使其可以用 literal_eval 正确处理

    string_data = info.replace("Decimal", "Decimal")  # 保留是为了 eval 能识别 Decimal
    data = eval(string_data, {"Decimal": Decimal})  # 注意安全性，若来自不信任源请改用 ast + 自定义解析

    # 提取所需字段并转换
    for item in data:
        farm = str(item['id'])  # or item['code'] if you'd prefer
        cap = item['capacity']
        result[farm] = cap
    return result


def cal_farm_cr(power, pred, cap):
    cr = []
    start_time = pred.index[0].date()
    for date in pd.date_range(start_time, periods=59, freq='D'):
        # for date in pd.date_range(start_time, periods=31, freq='D'):
        y_true = power['power'].loc[date:date + timedelta(minutes=15 * 95)]
        y_pred = pred.loc[date:date + timedelta(minutes=15 * 95)]
        label = power['label'].loc[date:date + timedelta(minutes=15 * 95)]
        mask = y_true.notna() & y_pred.notna() & (label == 1)
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        y_true_clean.index = range(len(y_true_clean))
        y_pred_clean.index = range(len(y_pred_clean))
        # gt中有些label==0,此时不计算
        if len(y_pred_clean) == 0:
            continue
        cr_tmp = 1 - np.sqrt(
            np.sum(np.square(
                (y_pred_clean - y_true_clean) / y_true_clean.where(y_true_clean > (0.2 * cap), (0.2 * cap)))) / len(
                y_pred_clean))
        cr_tmp = max(0, cr_tmp)
        cr.append(cr_tmp)
    cr = pd.Series(cr)
    cr[cr < 0] = 0
    return cr


def cal_score(pred_path, fact_path, cap=1.0):
    # 每个场站算精度
    all_farm = []
    power = pd.read_csv(fact_path, index_col=0)
    power.columns = ['power', 'label']
    power.index = pd.to_datetime(power.index)
    pred = pd.read_csv(pred_path, index_col=0)
    pred.columns = ['power']
    pred = pred['power']
    pred.index = pd.to_datetime(pred.index)
    cr_farm = cal_farm_cr(power, pred, cap)
    cr_farm_mean = cr_farm.mean()
    all_farm.append(cr_farm_mean)
    score = np.mean(all_farm)
    return score


if __name__ == "__main__":
    
    # 获取场站的容量
    txt_path = '/inspire/ssd/project/sais-mtm/public/sunyuqing/weather/all_farms0910.txt'
    cap_dict = get_cap(txt_path)

    #pred_root_path = '/cpfs01/projects-HDD/cfff-4a8d9af84f66_HDD/public/tanruxin/sth_power/eval/d1/v202509_2/mse_loss'
    pred_root_path = './qlz_test1'

    # 标识位 ensemble / lgb/ dl
    tag = 'd1_wind_lgb'
    fact_root_path = '/inspire/ssd/project/sais-mtm/public/qlz/data/PowerEstimateData/test_fact_data'
    save_path = f'./qlz_test1/score_{tag}.csv'

    # farms = ("1639",)
    farms = ['1000']
    #farms = ['1640', '1641', '1642', '1643', '4985', '5405', '5515']

    result = {'farm': [], 'score': []}
    for farm in farms:
        pred_path = os.path.join(pred_root_path, '{}'.format(farm), 'output.csv')
        fact_path = os.path.join(fact_root_path, '{}_eval.csv'.format(farm))
        # 存在部分场站暂无推理结果
        if not os.path.exists(pred_path):
            continue
        cap = cap_dict[farm]
        score = cal_score(pred_path, fact_path, cap)
        # 转换为百分数
        score = score * 100
        logger.info("farm:{}, tag:{}, score:{}".format(farm, tag, score))
        result['farm'].append(farm)
        result['score'].append(score)
        # break

    # 转换为 DataFrame 并写入 CSV
    df = pd.DataFrame(result)
    df.to_csv(save_path, index=False)  # index=False 避免写入行索引
