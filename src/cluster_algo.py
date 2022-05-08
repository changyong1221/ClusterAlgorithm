import torch
import time
import numpy as np
import pandas as pd
import os

class KMEANS:
    def __init__(self, n_clusters=None, max_iter=None, verbose=True,device = torch.device("cpu")):
        self.n_cluster = n_clusters
        self.n_clusters = n_clusters
        self.labels = None
        self.dists = None  # shape: [x.shape[0],n_cluster]
        self.centers = None
        self.variation = torch.Tensor([float("Inf")]).to(device)
        self.verbose = verbose
        self.started = False
        self.representative_samples = None
        self.max_iter = max_iter
        self.count = 0
        self.device = device

    def fit(self, x):
        # 随机选择初始中心点，想更快的收敛速度可以借鉴sklearn中的kmeans++初始化方法
        init_row = torch.randint(0, x.shape[0], (self.n_clusters,)).to(self.device)
        init_points = x[init_row]
        self.centers = init_points
        while True:
            # 聚类标记
            self.nearest_center(x)
            # 更新中心点
            self.update_center(x)
            if self.verbose:
                print(self.variation, torch.argmin(self.dists, (0)))
            if torch.abs(self.variation) < 1e-3 and self.max_iter is None:
                break
            elif self.max_iter is not None and self.count == self.max_iter:
                break
            self.count += 1
        self.representative_sample()

    def nearest_center(self, x):
        labels = torch.empty((x.shape[0],)).long().to(self.device)
        dists = torch.empty((0, self.n_clusters)).to(self.device)
        for i, sample in enumerate(x):
            dist = torch.sum(torch.mul(sample - self.centers, sample - self.centers), (1))
            labels[i] = torch.argmin(dist)
            dists = torch.cat([dists, dist.unsqueeze(0)], (0))
        self.labels = labels
        if self.started:
            self.variation = torch.sum(self.dists - dists)
        self.dists = dists
        self.started = True

    def update_center(self, x):
        centers = torch.empty((0, x.shape[1])).to(self.device)
        total_num = 0
        for i in range(self.n_clusters):
            mask = self.labels == i
            cluster_samples = x[mask]
            total_num += len(cluster_samples)
            centers = torch.cat([centers, torch.mean(cluster_samples, (0)).unsqueeze(0)], (0))
        self.centers = centers

    def representative_sample(self):
        # 查找距离中心点最近的样本，作为聚类的代表样本，更加直观
        self.representative_samples = torch.argmin(self.dists, (0))


def choose_device(cuda=False):
    if cuda:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    return device


if __name__ == "__main__":
    device = choose_device(False)
    # 创建特征
    # tmp_list = [
    #     [10, 20, 30, 40, 50],
    #     [20, 30, 40, 50, 60],
    #     [30, 40, 50, 60, 70],
    #     [100, 200, 300, 400, 500],
    #     [200, 300, 400, 500, 600],
    #     [300, 400, 500, 600, 700],
    #     [1000, 2000, 3000, 4000, 5000],
    #     [2000, 3000, 4000, 5000, 6000],
    #     [3000, 4000, 5000, 6000, 7000],
    #     [10, 20, 30, 40, 50],
    #     [20, 30, 40, 50, 60],
    #     [30, 40, 50, 60, 70],
    #     [100, 200, 300, 400, 500],
    #     [200, 300, 400, 500, 600],
    #     [300, 400, 500, 600, 700],
    #     [1000, 2000, 3000, 4000, 5000],
    #     [2000, 3000, 4000, 5000, 6000],
    #     [3000, 4000, 5000, 6000, 7000],
    # ]

    df = pd.read_csv("../dataset/computer_information.csv", header=None, delimiter=',')
    column_names = ['id', 'mips', 'ram', 'bandwidth', 'disk']
    df.columns = column_names
    df = df[['mips', 'ram', 'bandwidth', 'disk']]
    tmp_arr = df.values
    # print(df[:100])
    # print(type(df))

    # tmp_arr = np.array(tmp_list)
    matrix = torch.Tensor(tmp_arr)

    # print(f"matrix: {matrix}")
    # print(f"matrix.shape: {matrix.shape}")

    # 特征标准化
    tmp_mean = torch.mean(matrix, dim=0)
    matrix = matrix - tmp_mean
    tmp_std = torch.std(matrix, dim=0)
    matrix = matrix / tmp_std
    # print(f"matrix: {matrix}")
    # print(f"matrix.shape: {matrix.shape}")

    # 开始聚类
    n_clusters = 20
    k = KMEANS(n_clusters=n_clusters, max_iter=20, verbose=True, device=device)
    k.fit(matrix)

    # 将聚类结果写入文件，包括每个簇的簇中心和簇内所有节点
    print(f"k.labels: {k.labels}")
    cluster_num_list = []
    result_dir = "../result/test1/"
    os.makedirs(result_dir)
    center_file_dir = result_dir + "center.txt"
    df_center = pd.DataFrame()
    for i in range(n_clusters):
        mask = k.labels == i
        cur_cluster = tmp_arr[mask]
        print(f"cluster({i}):\n {cur_cluster}")
        cluster_num_list.append(len(tmp_arr[mask]))
        file_dir = result_dir + f"{i}.txt"
        file_df = pd.DataFrame(cur_cluster)
        file_df.to_csv(file_dir, header=False, index=True, sep=',')
        df_center = df_center.append(file_df.mean(axis=0), ignore_index=True)
    df_center.to_csv(center_file_dir, header=False, index=True, sep=',', mode='a+')

    for i, elem in enumerate(cluster_num_list):
        print(f"cluster({i}): {elem}")
    print(f"sum of all clusters: {sum(cluster_num_list)}")
