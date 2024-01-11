# -*- coding: utf-8 -*-
"""
@Time:      2023/10/18 14:42
@Author:    MS28175"zhifu huang""hzf125521@163.com"
@Project:   AnomalyDetection
@File       get_data
"""
import pandas as pd
import os
from sklearn.model_selection import train_test_split


def get_data0(file_path,
              type=0
              ):
    """
    数据源：宝钢，2#，电机非连轴端
    传感器：3-axis速度，水平加速度，温度，轴承冲击SK
    特征选择：3-axis速度每个axis的有效值，水平加速度的有效值，温度测量值，轴承冲击SK的测量值 => 共6个输入特征
    数据选择:下载2023-05-25 16:50:00 ~ 2023 10:16:50的数据
            速度阈值报警时段为 05-18 11:00:00 ~ 05-28 01:30:05
            剔除 05-28 01:30:05 之前的数据
            根据每个传感器的情况,剔除一些很大、很小的明显有问题的数据
            最后，传感器数据按相同时间戳进行拼接，组成多特征序列
    最终样本情况：共16699个样本，全是正常样本，每个样本有6个特征
    划分情况:训练13359个,测试3340个
    """

    # ------------
    dfSpeed_shuiping = pd.read_excel(os.path.join(file_path, '电机非连轴端_水平_速度.xlsx')
                                     )[['采集时间', '有效值']].rename(columns={'有效值': '速度_水平_有效值'})
    dfSpeed_shuiping['采集时间'] = pd.to_datetime(dfSpeed_shuiping['采集时间'])
    dfSpeed_shuiping = dfSpeed_shuiping.sort_values('采集时间', ascending=True).reset_index().drop(labels='index',
                                                                                                   axis=1)
    specific_timestamp = pd.to_datetime('2023-05-28 01:30:05')
    dfSpeed_shuiping = dfSpeed_shuiping[dfSpeed_shuiping['采集时间'] > specific_timestamp]
    dfSpeed_shuiping = dfSpeed_shuiping[
        (dfSpeed_shuiping['速度_水平_有效值'] > 0.25) & (dfSpeed_shuiping['速度_水平_有效值'] < 3.5)]

    # ------------
    dfSpeed_zhouxiang = pd.read_excel(os.path.join(file_path, '电机非连轴端_轴向_速度.xlsx')
                                      )[['采集时间', '有效值']].rename(columns={'有效值': '速度_轴向_有效值'})
    dfSpeed_zhouxiang['采集时间'] = pd.to_datetime(dfSpeed_zhouxiang['采集时间'])
    dfSpeed_zhouxiang = dfSpeed_zhouxiang.sort_values('采集时间', ascending=True).reset_index().drop(labels='index',
                                                                                                     axis=1)
    specific_timestamp = pd.to_datetime('2023-05-28 01:30:05')
    dfSpeed_zhouxiang = dfSpeed_zhouxiang[dfSpeed_zhouxiang['采集时间'] > specific_timestamp]
    dfSpeed_zhouxiang = dfSpeed_zhouxiang[dfSpeed_zhouxiang['速度_轴向_有效值'] > 0.5]

    # ------------
    dfSpeed_chuizhi = pd.read_excel(os.path.join(file_path, '电机非连轴端_垂直_速度(转速).xlsx')
                                    )[['采集时间', '有效值']].rename(columns={'有效值': '速度_垂直_有效值'})
    dfSpeed_chuizhi['采集时间'] = pd.to_datetime(dfSpeed_chuizhi['采集时间'])
    dfSpeed_chuizhi = dfSpeed_chuizhi.sort_values('采集时间', ascending=True).reset_index().drop(labels='index', axis=1)
    specific_timestamp = pd.to_datetime('2023-05-28 01:30:05')
    dfSpeed_chuizhi = dfSpeed_chuizhi[dfSpeed_chuizhi['采集时间'] > specific_timestamp]
    dfSpeed_chuizhi = dfSpeed_chuizhi[dfSpeed_chuizhi['速度_垂直_有效值'] > 0.5]

    # ------------
    dfAcc = pd.read_excel(os.path.join(file_path, '电机非连轴端_水平_加速度(润滑)(启停).xlsx')
                          )[['采集时间', '有效值']].rename(columns={'有效值': '加速度_有效值'})
    dfAcc['采集时间'] = pd.to_datetime(dfAcc['采集时间'])
    dfAcc = dfAcc.sort_values('采集时间', ascending=True).reset_index().drop(labels='index', axis=1)
    specific_timestamp = pd.to_datetime('2023-05-28 01:30:05')
    dfAcc = dfAcc[dfAcc['采集时间'] > specific_timestamp]
    dfAcc = dfAcc[dfAcc['加速度_有效值'] > 1.5]

    # ------------
    dfT = pd.read_excel(os.path.join(file_path, '电机非连轴端_温度.xlsx')
                        )[['采集时间', '测量值']].rename(columns={'测量值': '温度_测量值'})
    dfT['采集时间'] = pd.to_datetime(dfT['采集时间'])
    dfT = dfT.sort_values('采集时间', ascending=True).reset_index().drop(labels='index', axis=1)
    specific_timestamp = pd.to_datetime('2023-05-28 01:30:05')
    dfT = dfT[dfT['采集时间'] > specific_timestamp]

    # ------------
    dfSK = pd.read_excel(os.path.join(file_path, '电机非连轴端_轴承冲击SK.xlsx')
                         )[['采集时间', '测量值']].rename(columns={'测量值': 'SK_测量值'})
    dfSK['采集时间'] = pd.to_datetime(dfSK['采集时间'])
    dfSK = dfSK.sort_values('采集时间', ascending=True).reset_index().drop(labels='index', axis=1)
    specific_timestamp = pd.to_datetime('2023-05-28 01:30:05')
    dfSK = dfSK[dfSK['采集时间'] > specific_timestamp]
    dfSK = dfSK[dfSK['SK_测量值'] > 2]

    # ------------
    dfAll = pd.merge(dfSpeed_shuiping, dfSpeed_zhouxiang, on='采集时间', how='inner')
    dfAll = pd.merge(dfAll, dfSpeed_chuizhi, on='采集时间', how='inner')
    dfAll = pd.merge(dfAll, dfAcc, on='采集时间', how='inner')
    dfAll = pd.merge(dfAll, dfT, on='采集时间', how='inner')
    dfAll = pd.merge(dfAll, dfSK, on='采集时间', how='inner')

    train_df, test_df = train_test_split(dfAll, test_size=0.2, random_state=0)
    train_x = train_df.drop('采集时间', axis=1).values

    if type == 0:
        return train_x
    elif type == 1:
        test_x = test_df.drop('采集时间', axis=1).values
        return test_x
    elif type == 2:
        test_x = test_df.drop('采集时间', axis=1).values
        return train_x, test_x
    else:
        print("No matching 'data division form' for the value of 'a'.")


def get_data1(file_path,
              type=0
              ):
    # ------------
    dfSpeed_shuiping = pd.read_excel(os.path.join(file_path, '电机连轴端_水平_速度.xlsx')
                                     )[['采集时间', '有效值']].rename(columns={'有效值': '速度_水平_有效值'})
    dfSpeed_shuiping['采集时间'] = pd.to_datetime(dfSpeed_shuiping['采集时间'])
    dfSpeed_shuiping = dfSpeed_shuiping.sort_values('采集时间', ascending=True).reset_index().drop(labels='index',
                                                                                                   axis=1)
    specific_timestamp = pd.to_datetime('2023-05-28 23:00:05')
    dfSpeed_shuiping = dfSpeed_shuiping[dfSpeed_shuiping['采集时间'] > specific_timestamp]
    dfSpeed_shuiping = dfSpeed_shuiping[(dfSpeed_shuiping['速度_水平_有效值'] > 0.25)]

    # ------------
    dfSpeed_zhouxiang = pd.read_excel(os.path.join(file_path, '电机连轴端_轴向_速度.xlsx')
                                      )[['采集时间', '有效值']].rename(columns={'有效值': '速度_轴向_有效值'})
    dfSpeed_zhouxiang['采集时间'] = pd.to_datetime(dfSpeed_zhouxiang['采集时间'])
    dfSpeed_zhouxiang = dfSpeed_zhouxiang.sort_values('采集时间', ascending=True).reset_index().drop(labels='index',
                                                                                                     axis=1)
    specific_timestamp = pd.to_datetime('2023-05-28 23:00:05')
    dfSpeed_zhouxiang = dfSpeed_zhouxiang[dfSpeed_zhouxiang['采集时间'] > specific_timestamp]
    dfSpeed_zhouxiang = dfSpeed_zhouxiang[dfSpeed_zhouxiang['速度_轴向_有效值'] >= 0.5]

    # ------------
    dfSpeed_chuizhi = pd.read_excel(os.path.join(file_path, '电机连轴端_垂直_速度.xlsx')
                                    )[['采集时间', '有效值']].rename(columns={'有效值': '速度_垂直_有效值'})
    dfSpeed_chuizhi['采集时间'] = pd.to_datetime(dfSpeed_chuizhi['采集时间'])
    dfSpeed_chuizhi = dfSpeed_chuizhi.sort_values('采集时间', ascending=True).reset_index().drop(labels='index', axis=1)
    specific_timestamp = pd.to_datetime('2023-05-28 23:00:05')
    dfSpeed_chuizhi = dfSpeed_chuizhi[dfSpeed_chuizhi['采集时间'] > specific_timestamp]
    dfSpeed_chuizhi = dfSpeed_chuizhi[dfSpeed_chuizhi['速度_垂直_有效值'] > 0.5]

    # ------------
    dfAcc = pd.read_excel(os.path.join(file_path, '电机连轴端_水平_加速度(润滑).xlsx')
                          )[['采集时间', '有效值']].rename(columns={'有效值': '加速度_有效值'})
    dfAcc['采集时间'] = pd.to_datetime(dfAcc['采集时间'])
    dfAcc = dfAcc.sort_values('采集时间', ascending=True).reset_index().drop(labels='index', axis=1)
    specific_timestamp = pd.to_datetime('2023-05-28 23:00:05')
    dfAcc = dfAcc[dfAcc['采集时间'] > specific_timestamp]
    dfAcc = dfAcc[dfAcc['加速度_有效值'] > 1]

    # ------------
    dfT = pd.read_excel(os.path.join(file_path, '电机连轴端_温度.xlsx')
                        )[['采集时间', '测量值']].rename(columns={'测量值': '温度_测量值'})
    dfT['采集时间'] = pd.to_datetime(dfT['采集时间'])
    dfT = dfT.sort_values('采集时间', ascending=True).reset_index().drop(labels='index', axis=1)
    specific_timestamp = pd.to_datetime('2023-05-28 23:00:05')
    dfT = dfT[dfT['采集时间'] > specific_timestamp]

    # ------------
    dfSK = pd.read_excel(os.path.join(file_path, '电机连轴端_轴承冲击SK.xlsx')
                         )[['采集时间', '测量值']].rename(columns={'测量值': 'SK_测量值'})
    dfSK['采集时间'] = pd.to_datetime(dfSK['采集时间'])
    dfSK = dfSK.sort_values('采集时间', ascending=True).reset_index().drop(labels='index', axis=1)
    specific_timestamp = pd.to_datetime('2023-05-28 23:00:05')
    dfSK = dfSK[dfSK['采集时间'] > specific_timestamp]
    dfSK = dfSK[dfSK['SK_测量值'] > 2]

    # ------------
    dfAll = pd.merge(dfSpeed_shuiping, dfSpeed_zhouxiang, on='采集时间', how='inner')
    dfAll = pd.merge(dfAll, dfSpeed_chuizhi, on='采集时间', how='inner')
    dfAll = pd.merge(dfAll, dfAcc, on='采集时间', how='inner')
    dfAll = pd.merge(dfAll, dfT, on='采集时间', how='inner')
    dfAll = pd.merge(dfAll, dfSK, on='采集时间', how='inner')

    train_df, test_df = train_test_split(dfAll, test_size=0.2, random_state=0)
    train_x = train_df.drop('采集时间', axis=1).values

    if type == 0:
        return train_x
    elif type == 1:
        test_x = test_df.drop('采集时间', axis=1).values
        return test_x
    elif type == 2:
        test_x = test_df.drop('采集时间', axis=1).values
        return train_x, test_x
    else:
        print("No matching 'data division form' for the value of 'a'.")


def get_data2(file_path,
              type=0
              ):
    # ------------
    dfSpeed_shuiping = pd.read_excel(os.path.join(file_path, '风机连轴端_水平_速度.xlsx')
                                     )[['采集时间', '有效值']].rename(columns={'有效值': '速度_水平_有效值'})
    dfSpeed_shuiping['采集时间'] = pd.to_datetime(dfSpeed_shuiping['采集时间'])
    dfSpeed_shuiping = dfSpeed_shuiping.sort_values('采集时间', ascending=True).reset_index().drop(labels='index',
                                                                                                   axis=1)
    specific_timestamp = pd.to_datetime('2023-06-25 15:30:05')
    dfSpeed_shuiping = dfSpeed_shuiping[dfSpeed_shuiping['采集时间'] > specific_timestamp]
    dfSpeed_shuiping = dfSpeed_shuiping[(dfSpeed_shuiping['速度_水平_有效值'] > 0.5)]

    # ------------
    dfSpeed_zhouxiang = pd.read_excel(os.path.join(file_path, '风机连轴端_轴向_速度.xlsx')
                                      )[['采集时间', '有效值']].rename(columns={'有效值': '速度_轴向_有效值'})
    dfSpeed_zhouxiang['采集时间'] = pd.to_datetime(dfSpeed_zhouxiang['采集时间'])
    dfSpeed_zhouxiang = dfSpeed_zhouxiang.sort_values('采集时间', ascending=True).reset_index().drop(labels='index',
                                                                                                     axis=1)
    specific_timestamp = pd.to_datetime('2023-06-25 15:30:05')
    dfSpeed_zhouxiang = dfSpeed_zhouxiang[dfSpeed_zhouxiang['采集时间'] > specific_timestamp]
    dfSpeed_zhouxiang = dfSpeed_zhouxiang[dfSpeed_zhouxiang['速度_轴向_有效值'] >= 0.5]

    # ------------
    dfSpeed_chuizhi = pd.read_excel(os.path.join(file_path, '风机连轴端_垂直_速度.xlsx')
                                    )[['采集时间', '有效值']].rename(columns={'有效值': '速度_垂直_有效值'})
    dfSpeed_chuizhi['采集时间'] = pd.to_datetime(dfSpeed_chuizhi['采集时间'])
    dfSpeed_chuizhi = dfSpeed_chuizhi.sort_values('采集时间', ascending=True).reset_index().drop(labels='index', axis=1)
    specific_timestamp = pd.to_datetime('2023-06-25 15:30:05')
    dfSpeed_chuizhi = dfSpeed_chuizhi[dfSpeed_chuizhi['采集时间'] > specific_timestamp]
    dfSpeed_chuizhi = dfSpeed_chuizhi[dfSpeed_chuizhi['速度_垂直_有效值'] > 0.6]

    # ------------
    dfAcc = pd.read_excel(os.path.join(file_path, '风机连轴端_水平_加速度(润滑).xlsx')
                          )[['采集时间', '有效值']].rename(columns={'有效值': '加速度_有效值'})
    dfAcc['采集时间'] = pd.to_datetime(dfAcc['采集时间'])
    dfAcc = dfAcc.sort_values('采集时间', ascending=True).reset_index().drop(labels='index', axis=1)
    specific_timestamp = pd.to_datetime('2023-06-25 15:30:05')
    dfAcc = dfAcc[dfAcc['采集时间'] > specific_timestamp]
    dfAcc = dfAcc[dfAcc['加速度_有效值'] > 1]

    # ------------
    dfT = pd.read_excel(os.path.join(file_path, '风机连轴端_温度.xlsx')
                        )[['采集时间', '测量值']].rename(columns={'测量值': '温度_测量值'})
    dfT['采集时间'] = pd.to_datetime(dfT['采集时间'])
    dfT = dfT.sort_values('采集时间', ascending=True).reset_index().drop(labels='index', axis=1)
    specific_timestamp = pd.to_datetime('2023-06-25 15:30:05')
    dfT = dfT[dfT['采集时间'] > specific_timestamp]

    # ------------
    dfSK = pd.read_excel(os.path.join(file_path, '风机连轴端_轴承冲击SK.xlsx')
                         )[['采集时间', '测量值']].rename(columns={'测量值': 'SK_测量值'})
    dfSK['采集时间'] = pd.to_datetime(dfSK['采集时间'])
    dfSK = dfSK.sort_values('采集时间', ascending=True).reset_index().drop(labels='index', axis=1)
    specific_timestamp = pd.to_datetime('2023-06-25 15:30:05')
    dfSK = dfSK[dfSK['采集时间'] > specific_timestamp]
    dfSK = dfSK[dfSK['SK_测量值'] > 2]

    # ------------
    dfAll = pd.merge(dfSpeed_shuiping, dfSpeed_zhouxiang, on='采集时间', how='inner')
    dfAll = pd.merge(dfAll, dfSpeed_chuizhi, on='采集时间', how='inner')
    dfAll = pd.merge(dfAll, dfAcc, on='采集时间', how='inner')
    dfAll = pd.merge(dfAll, dfT, on='采集时间', how='inner')
    dfAll = pd.merge(dfAll, dfSK, on='采集时间', how='inner')

    train_df, test_df = train_test_split(dfAll, test_size=0.2, random_state=0)
    train_x = train_df.drop('采集时间', axis=1).values

    if type == 0:
        return train_x
    elif type == 1:
        test_x = test_df.drop('采集时间', axis=1).values
        return test_x
    elif type == 2:
        test_x = test_df.drop('采集时间', axis=1).values
        return train_x, test_x
    else:
        print("No matching 'data division form' for the value of 'a'.")


def get_data3(file_path,
              type=0
              ):
    # ------------
    dfSpeed_shuiping = pd.read_excel(os.path.join(file_path, '风机非连轴端_水平_速度.xlsx')
                                     )[['采集时间', '有效值']].rename(columns={'有效值': '速度_水平_有效值'})
    dfSpeed_shuiping['采集时间'] = pd.to_datetime(dfSpeed_shuiping['采集时间'])
    dfSpeed_shuiping = dfSpeed_shuiping.sort_values('采集时间', ascending=True).reset_index().drop(labels='index',
                                                                                                   axis=1)
    specific_timestamp = pd.to_datetime('2023-06-25 15:30:05')
    dfSpeed_shuiping = dfSpeed_shuiping[dfSpeed_shuiping['采集时间'] > specific_timestamp]

    specific_timestamp1 = pd.to_datetime('2023-09-24 06:50:00')
    specific_timestamp2 = pd.to_datetime('2023-10-16 15:50:00')
    dfSpeed_shuiping = dfSpeed_shuiping[
        (dfSpeed_shuiping['采集时间'] < specific_timestamp1) | (dfSpeed_shuiping['采集时间'] > specific_timestamp2)]
    dfSpeed_shuiping = dfSpeed_shuiping[(dfSpeed_shuiping['速度_水平_有效值'] > 0.5)]

    # ------------
    dfSpeed_zhouxiang = pd.read_excel(os.path.join(file_path, '风机非连轴端_轴向_速度.xlsx')
                                      )[['采集时间', '有效值']].rename(columns={'有效值': '速度_轴向_有效值'})
    dfSpeed_zhouxiang['采集时间'] = pd.to_datetime(dfSpeed_zhouxiang['采集时间'])
    dfSpeed_zhouxiang = dfSpeed_zhouxiang.sort_values('采集时间', ascending=True).reset_index().drop(labels='index',
                                                                                                     axis=1)
    specific_timestamp = pd.to_datetime('2023-06-25 15:30:05')
    dfSpeed_zhouxiang = dfSpeed_zhouxiang[dfSpeed_zhouxiang['采集时间'] > specific_timestamp]

    specific_timestamp1 = pd.to_datetime('2023-09-24 06:50:00')
    specific_timestamp2 = pd.to_datetime('2023-10-16 15:50:00')
    dfSpeed_zhouxiang = dfSpeed_zhouxiang[
        (dfSpeed_zhouxiang['采集时间'] < specific_timestamp1) | (dfSpeed_zhouxiang['采集时间'] > specific_timestamp2)]
    dfSpeed_zhouxiang = dfSpeed_zhouxiang[dfSpeed_zhouxiang['速度_轴向_有效值'] >= 0.5]

    # ------------
    dfSpeed_chuizhi = pd.read_excel(os.path.join(file_path, '风机非连轴端_垂直_速度.xlsx')
                                    )[['采集时间', '有效值']].rename(columns={'有效值': '速度_垂直_有效值'})
    dfSpeed_chuizhi['采集时间'] = pd.to_datetime(dfSpeed_chuizhi['采集时间'])
    dfSpeed_chuizhi = dfSpeed_chuizhi.sort_values('采集时间', ascending=True).reset_index().drop(labels='index', axis=1)
    specific_timestamp = pd.to_datetime('2023-06-25 15:30:05')
    dfSpeed_chuizhi = dfSpeed_chuizhi[dfSpeed_chuizhi['采集时间'] > specific_timestamp]

    specific_timestamp1 = pd.to_datetime('2023-09-24 06:50:00')
    specific_timestamp2 = pd.to_datetime('2023-10-16 15:50:00')
    dfSpeed_chuizhi = dfSpeed_chuizhi[
        (dfSpeed_chuizhi['采集时间'] < specific_timestamp1) | (dfSpeed_chuizhi['采集时间'] > specific_timestamp2)]
    dfSpeed_chuizhi = dfSpeed_chuizhi[dfSpeed_chuizhi['速度_垂直_有效值'] > 0.6]

    # ------------
    dfAcc = pd.read_excel(os.path.join(file_path, '风机非连轴端_水平_加速度(润滑).xlsx')
                          )[['采集时间', '有效值']].rename(columns={'有效值': '加速度_有效值'})
    dfAcc['采集时间'] = pd.to_datetime(dfAcc['采集时间'])
    dfAcc = dfAcc.sort_values('采集时间', ascending=True).reset_index().drop(labels='index', axis=1)
    specific_timestamp = pd.to_datetime('2023-06-25 15:30:05')
    dfAcc = dfAcc[dfAcc['采集时间'] > specific_timestamp]

    specific_timestamp1 = pd.to_datetime('2023-09-24 06:50:00')
    specific_timestamp2 = pd.to_datetime('2023-10-16 15:50:00')
    dfAcc = dfAcc[(dfAcc['采集时间'] < specific_timestamp1) | (dfSpeed_chuizhi['采集时间'] > specific_timestamp2)]
    dfAcc = dfAcc[dfAcc['加速度_有效值'] > 1.5]

    # ------------
    dfT = pd.read_excel(os.path.join(file_path, '风机非连轴端_温度.xlsx')
                        )[['采集时间', '测量值']].rename(columns={'测量值': '温度_测量值'})
    dfT['采集时间'] = pd.to_datetime(dfT['采集时间'])
    dfT = dfT.sort_values('采集时间', ascending=True).reset_index().drop(labels='index', axis=1)
    specific_timestamp = pd.to_datetime('2023-06-25 15:30:05')
    dfT = dfT[dfT['采集时间'] > specific_timestamp]

    specific_timestamp1 = pd.to_datetime('2023-09-24 06:50:00')
    specific_timestamp2 = pd.to_datetime('2023-10-16 15:50:00')
    dfT = dfT[(dfT['采集时间'] < specific_timestamp1) | (dfSpeed_chuizhi['采集时间'] > specific_timestamp2)]

    # ------------
    dfSK = pd.read_excel(os.path.join(file_path, '风机非连轴端_轴承冲击SK.xlsx')
                         )[['采集时间', '测量值']].rename(columns={'测量值': 'SK_测量值'})
    dfSK['采集时间'] = pd.to_datetime(dfSK['采集时间'])
    dfSK = dfSK.sort_values('采集时间', ascending=True).reset_index().drop(labels='index', axis=1)
    specific_timestamp = pd.to_datetime('2023-06-25 15:30:05')
    dfSK = dfSK[dfSK['采集时间'] > specific_timestamp]

    specific_timestamp1 = pd.to_datetime('2023-09-24 06:50:00')
    specific_timestamp2 = pd.to_datetime('2023-10-16 15:50:00')
    dfSK = dfSK[(dfSK['采集时间'] < specific_timestamp1) | (dfSpeed_chuizhi['采集时间'] > specific_timestamp2)]
    dfSK = dfSK[dfSK['SK_测量值'] > 2.5]

    # ------------
    dfAll = pd.merge(dfSpeed_shuiping, dfSpeed_zhouxiang, on='采集时间', how='inner')
    dfAll = pd.merge(dfAll, dfSpeed_chuizhi, on='采集时间', how='inner')
    dfAll = pd.merge(dfAll, dfAcc, on='采集时间', how='inner')
    dfAll = pd.merge(dfAll, dfT, on='采集时间', how='inner')
    dfAll = pd.merge(dfAll, dfSK, on='采集时间', how='inner')

    train_df, test_df = train_test_split(dfAll, test_size=0.2, random_state=0)
    train_x = train_df.drop('采集时间', axis=1).values

    if type == 0:
        return train_x
    elif type == 1:
        test_x = test_df.drop('采集时间', axis=1).values
        return test_x
    elif type == 2:
        test_x = test_df.drop('采集时间', axis=1).values
        return train_x, test_x
    else:
        print("No matching 'data division form' for the value of 'a'.")


if __name__ == "__main__":
    print('--------- 1.电机非连轴端 ---------')
    file_path1 = r'D:\company\data\trueBaogang\2#\电机非连轴端'
    train_x, test_x = get_data0(file_path1, type=2)
    print(train_x, train_x.shape)
    print(test_x, test_x.shape)

    print('--------- 2.电机连轴端 ---------')
    file_path2 = r'D:\company\data\trueBaogang\2#\电机连轴端'
    train_x, test_x = get_data1(file_path2, type=2)
    print(train_x, train_x.shape)
    print(test_x, test_x.shape)

    print('--------- 3.风机连轴端 ---------')
    file_path3 = r'D:\company\data\trueBaogang\2#\风机连轴端'
    train_x, test_x = get_data2(file_path3, type=2)
    print(train_x, train_x.shape)
    print(test_x, test_x.shape)

    print('--------- 4.风机非连轴端 ---------')
    file_path4 = r'D:\company\data\trueBaogang\2#\风机非连轴端'
    train_x, test_x = get_data3(file_path4, type=2)
    print(train_x, train_x.shape)
    print(test_x, test_x.shape)
