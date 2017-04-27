# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np

if __name__ == '__main__':
    ## step 1: 加载数据
    print("step 1: load data...")

    dataSet = []
    fileIn = open('./data.txt')
    for line in fileIn.readlines():
        lineArr = line.strip().split(' ')
        dataSet.append([float(lineArr[0]), float(lineArr[1])])

    numSamples = len(dataSet)
    X = np.array(dataSet)

    bandwidth = estimate_bandwidth(X, quantile=0.2, n_samples=500)
    clf = MeanShift(bandwidth=bandwidth, bin_seeding=True, cluster_all=True).fit(X)

    centroids = clf.labels_
    # 显示每一个点的聚类归属
    print(centroids, type(centroids))
    # 计算其自动生成的k，并将聚类数量小于3的排除
    arr_flag = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in clf.labels_:
        arr_flag[i] += 1
    k = 0
    for i in arr_flag:
        if (i > 3):
            k += 1
    print(k)

    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    # 画出所有样例点 属于同一分类的绘制同样的颜色
    for i in range(numSamples):
        plt.plot(dataSet[i][0], dataSet[i][1], mark[clf.labels_[i]])
    mark = ['Dr', 'Db', 'Dg', 'Dk', '^b', '+b', 'sb', 'db', '<b', 'pb']
    # 画出质点，用特殊图型
    centroids = clf.cluster_centers_
    for i in range(k):
        plt.plot(centroids[i][0], centroids[i][1], mark[i], markersize=12)
    print(centroids)
    plt.show()
