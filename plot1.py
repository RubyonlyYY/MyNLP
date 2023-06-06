'''
Author: 21308004-Yao Yuan
Date: 2023-3-30
'''

# 引入Matplotlib引入
from matplotlib import pyplot as plt
import numpy as np

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 中文负号
plt.rcParams['axes.unicode_minus'] = False

# 设置分别率 为100
plt.rcParams['figure.dpi'] = 100

# 设置大小
plt.rcParams['figure.figsize'] = (5,3)

# 由于牵扯计算,因此将数据转numpy数组
movie = ['quality', 'dryingspeed', 'airflow', 'scenario', 'usability', 'power', 'appearance', 'noisy', 'attachment', 'temperature', 'price', 'expectation', 'comfort']
# 第一天
real_day1 = np.array( [9410, 9291, 7917])

# 第二天
real_day2 = np.array([7840, 4013, 2421])

# 第三天
real_day3 = np.array([8080, 3673, 1342])

real_day4 = np.array([6059, 1588, 5428])

real_day5 = np.array([6944, 8814, 2644])

real_day6 = np.array([8323, 4714, 6009])

real_day7 = np.array([7643, 9834, 6404])

real_day8 = np.array([6248, 4970, 9547])

real_day9 = np.array([5966, 6726, 2894])

real_day10 = np.array([2277, 7018, 4096])

real_day11 = np.array([5956, 2717, 6218])

real_day12 = np.array([9830, 9252, 5935])

real_day13 = np.array([4053, 2548, 1543])

# ================确定距离左侧========

left_day2 = real_day1 # 第二天距离左侧的为第一天的数值

left_day3 = real_day1 + real_day2  # 第三天距离左侧为 第一天+第二天的数据

# 设置线条高度
height = 0.2

# 绘制图形:
col1=np.array([real_day1[0], real_day2[0], real_day3[0], real_day4[0], real_day5[0], real_day6[0], real_day7[0], real_day8[0], real_day9[0], real_day10[0], real_day11[0], real_day12[0], real_day13[0]])
plt.barh(movie, col1, height=height, color='blue', label='正面')      # 第一天图形
plt.legend()

col2=np.array([real_day1[1], real_day2[1], real_day3[1], real_day4[1], real_day5[1], real_day6[1], real_day7[1], real_day8[1], real_day9[1], real_day10[1], real_day11[1], real_day12[1], real_day13[1]])
plt.barh(movie, col2, left=col1, height=height, color='red', label='负面')  # 第二天图形
plt.legend()

col3=np.array([real_day1[2], real_day2[2], real_day3[2], real_day4[2], real_day5[2], real_day6[2], real_day7[2], real_day8[2], real_day9[2], real_day10[2], real_day11[2], real_day12[2], real_day13[2]])
plt.barh(movie, col3, left=col1+col2, height=height, color='yellow', label='中性') # 第三天图形
plt.legend()



# 设置数值文本:  计算宽度值和y轴为值

sum_data = real_day1 + real_day2 +real_day3
# horizontalalignment控制文本的x位置参数表示文本边界框的左边，中间或右边。---->ha
# verticalalignment控制文本的y位置参数表示文本边界框的底部，中心或顶部 ---- va
#for i in range(len(movie)):
    #plt.text(sum_data[i], movie[i], sum_data[i],va="center" , ha="left")
#plt.xlim(0,sum_data.max()+2000)

plt.show()

#定义饼的标签，
labels = ['quality', 'dryingspeed', 'airflow', 'scenario', 'usability', 'power', 'appearance', 'noisy', 'attachment', 'temperature', 'price', 'expectation', 'comfort']

#每个标签所占的数量
x = [9410,840,80,59,944,323,643,248,966,277,956,830,53]

#饼图分离
explode = (0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.11,0.12,0.13)

#设置阴影效果
#plt.pie(x,labels=labels,autopct='%3.2f%%',explode=explode,shadow=True)

plt.pie(x,labels=labels,autopct='%3.2f%%',explode=explode)
plt.legend()

plt.show()