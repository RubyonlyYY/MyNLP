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

# 使用numpy随机生成300个随机数据
x_value = np.random.randint(140, 180, 300)
plt.hist(x_value, bins=10, edgecolor='white')
# plt.hist(x_value, bins=20, edgecolor='white')

plt.title("数据统计")
plt.xlabel("身高")
plt.ylabel("比率")

num,bins_limit,patches = plt.hist(x_value, bins=10, edgecolor='white')
plt.grid(ls="--")
num
bins_limit
for i in patches:
    print(i)
    print(i.get_x())
    print(i.get_y())
    print(i.get_height())
    print(i.get_width())
patches[0].get_width()

# 绘制直方图返回元组,元组中有三个元素
num,bins_limit,patches = plt.hist(x_value, bins=10, edgecolor='white')
print("n 是分组区间对应的频率：",num,end="\n\n")
print("bins_limit 是分组时的分隔值：",bins_limit,end="\n\n")
print("patches 指的是是直方图中列表对象",type(patches),end="\n\n")
#plt.xticks(bins_limit)

x_limit_value = []
height_value = []
for item in patches:
    print(item)
    x_limit_value.append(item.get_x())
    height_value.append(item.get_height())

print(x_limit_value)
print(height_value)

plt.show()

# 创建一个画布
fig, ax = plt.subplots()

# 绘制直方图
num,bins_limit,patches = ax.hist(x_value, bins=10, edgecolor='white')

# 注意num返回的个数是10,bins_limit返回的个数为11,需要截取
print(bins_limit[:-1])
# 曲线图
ax.plot(bins_limit[:10], num, '--',marker="o")
#ax.set_xticks(bins_limit)
# 需要单独设置x轴的旋转
plt.xticks(bins_limit,rotation=45)

plt.show()

fig, ax = plt.subplots()
x = np.random.normal(100,20,100) # 均值和标准差
bins = [50, 60, 70, 90, 100,110, 140, 150]
ax.hist(x, bins, color="g",rwidth=0.5)
ax.set_title('不等距分组')

plt.show()

# 指定分组个数
n_bins=10

fig,ax=plt.subplots(figsize=(8,5))

# 分别生成10000 ， 5000 ， 2000 个值
x_multi = [np.random.randn(n) for n in [10000, 5000, 2000]]


# 实际绘图代码与单类型直方图差异不大，只是增加了一个图例项
# 在 ax.hist 函数中先指定图例 label 名称
ax.hist(x_multi, n_bins, histtype='bar',label=list("ABC"))

ax.set_title('多类型直方图')

# 通过 ax.legend 函数来添加图例
ax.legend()

plt.show()

x_value = np.random.randint(140,180,200)
x2_value = np.random.randint(140,180,200)
#plt.hist([x_value,x2_value],bins=10,stacked=True)
plt.hist([x_value,x2_value],bins=10, stacked=True)
plt.show()