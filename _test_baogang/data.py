# -*- coding: utf-8 -*-
"""
@Time:      2023/10/27 16:01
@Author:    MS28175"zhifu huang""hzf125521@163.com"
@Project:   AnomalyDetection
@File       data
"""
from get_data import get_data0, get_data1, get_data2, get_data3


def select_data_func(folder,
                     mapping=None
                     ):
    """
       Parameters
       ----------
       folder : str
        name of measuring point(folder name)
        e.g.,  '电机连轴端'
       mapping : dict
        correspondence between measuring points and data processing functions,
        e.g., {
            '电机非连轴端': get_data0,
            '电机连轴端': get_data1,
            '风机连轴端': get_data2,
            '风机非连轴端': get_data3
            }

       Returns
       -------
        data processing function (one of 'get_data0', 'get_data1', 'get_data2', 'get_data3')

       """

    if mapping is None:
        mapping = {
            '电机非连轴端': get_data0,
            '电机连轴端': get_data1,
            '风机连轴端': get_data2,
            '风机非连轴端': get_data3
        }
    selected_func = mapping.get(folder, None)
    if selected_func is not None:
        return selected_func
    else:
        raise ValueError(f"No function found for '{folder}'")


if __name__ == '__main__':
    folder_name = '电机连轴端'
    print(select_data_func(folder_name))
