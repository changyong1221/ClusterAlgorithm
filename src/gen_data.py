# encoding=utf8
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

'''
parameters
'''

# 划分簇个数
n_clusters = 20
# 特征数量
n_features = 4
# 每批数据集大小，总数据集大小为 n_records*5
n_records = 1000
# 特征名称
# 各项指标分别为：cpu利用率，内存利用率，磁盘利用率，排网络带宽利用率，系统平均负载，队延迟，优先级（即数据标签）
features = ['MIPS', 'ram', 'bandwidth', 'disk']
# 每个特征的取值范围
ranges = [3000, 16000, 1000, 100]


# 计算每个特征的取值区间
range_set = []
range_num = n_clusters
for i in range(0, len(ranges)):
    step = ranges[i] / n_clusters
    init_val = 0
    range_set.append([])
    for j in range(0, n_clusters):
        range_set[i].append((init_val, round(init_val + step, 2)))
        init_val = round(init_val + step, 2)
print(range_set)


def create_dataset(shuffle=True):
    mips_range = ranges[0]
    mips_scale = mips_range * 0.4
    normal_numbers = np.random.normal(loc=mips_range, scale=mips_scale, size=n_records + 10)
    normal_numbers = normal_numbers.astype(int)
    normal_numbers.sort()
    normal_numbers = normal_numbers[10:]
    print("normal_numbers: ", normal_numbers)
    file_name = '../dataset/computer_information.csv'

    with open(file_name, 'w') as f:
        for id, number in enumerate(normal_numbers):
            rate = number / mips_range
            f.write(str(id) + ',' + str(number) + ',' + str(ranges[1] * rate) + ',' + str(ranges[2] * rate) + ',' +
                    str(ranges[3] * rate) + '\n')


# 创建数据集
create_dataset(shuffle=True)


# 解决控制台输出省略号的问题
pd.set_option('display.max_columns',1000)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth',1000)

# 从csv文件中读取数据集
dataset_location = '../dataset/computer_information.csv'
df = pd.read_csv(dataset_location, names=features)
print('number of records in dataset: ', df.shape[0])
print(df[:10])
# plot_data(df)
