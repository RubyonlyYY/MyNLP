'''
Author: 21308004-Yao Yuan
Date: 2023-3-30
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def plotStatistic(pos,neg):

    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 中文负号
    plt.rcParams['axes.unicode_minus'] = False

    # 设置分别率 为100
    plt.rcParams['figure.dpi'] = 100

    # 设置大小
    plt.rcParams['figure.figsize'] = (5, 3)
    # ================确定距离左侧========

    # 设置线条高度
    height = 0.2

    # 绘制图形:
    movie=['Location', 'Amenity', 'Rating', 'Cuisine', 'Restaurant_Name', 'Hours', 'Dish', 'Price']
    col1 = np.array([pos['Location'], pos['Amenity'], pos['Rating'], pos['Cuisine'], pos['Restaurant_Name'], pos['Hours'], pos['Dish'], pos['Price']])
    plt.barh(movie, col1, height=height, color='blue', label='正面')  # 第一天图形
    plt.legend()

    col2 = np.array([neg['Location'], neg['Amenity'], neg['Rating'], neg['Cuisine'], neg['Restaurant_Name'], neg['Hours'], neg['Dish'], neg['Price']])
    plt.barh(movie, col2, left=col1, height=height, color='red', label='负面')  # 第二天图形
    plt.legend()
    plt.show()

df =  pd.read_csv('sentimentResult.csv')
pos = df.to_dict(orient='records')[0]
neg = df.to_dict(orient='records')[1]
print(pos)
plotStatistic(pos, neg)