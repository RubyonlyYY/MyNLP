'''
Author: 21308004-Yao Yuan
Date: 2023-3-30
'''

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def plot(pos):
    # 定义饼的标签，
    labels = ['Location', 'Amenity', 'Rating', 'Cuisine', 'Restaurant_Name', 'Hours', 'Dish', 'Price']

    # 每个标签所占的数量
    x = np.array([pos['Location'], pos['Amenity'], pos['Rating'], pos['Cuisine'], pos['Restaurant_Name'], pos['Hours'], pos['Dish'], pos['Price']])

    # 饼图分离
    explode = (0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08)

    # 设置阴影效果
    # plt.pie(x,labels=labels,autopct='%3.2f%%',explode=explode,shadow=True)

    plt.pie(x, labels=labels, autopct='%3.2f%%', explode=explode)
    plt.legend()

    plt.show()
df =  pd.read_csv('sentimentResult.csv')
pos = df.to_dict(orient='records')[0]
print(pos)
plot(pos)