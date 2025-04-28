import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
from scipy.stats import gamma

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# 读取数据
data = pd.read_csv('arival.csv', header=None, sep=' ').values.flatten()
data = data[~np.isnan(data)]

# 第一个图：只显示原始数据的直方图
plt.figure(figsize=(10, 6))
sns.histplot(data, bins=15, stat='density', alpha=0.7, color='skyblue')
plt.title('到达时间间隔分布直方图')
plt.xlabel('时间 (分钟)')
plt.ylabel('密度')
plt.grid(True, alpha=0.3)
plt.savefig('histogram_raw.png', dpi=300, bbox_inches='tight')
plt.close()

# 第二个图：显示原始数据直方图和伽马分布拟合曲线
plt.figure(figsize=(10, 6))
sns.histplot(data, bins=15, stat='density', alpha=0.7, color='skyblue', label='实际数据')

# 拟合伽马分布
shape, loc, scale = gamma.fit(data)
x = np.linspace(min(data), max(data), 100)
pdf = gamma.pdf(x, shape, loc=loc, scale=scale)
plt.plot(x, pdf, 'r-', lw=2, label='伽马分布拟合')

plt.title('到达时间间隔分布与伽马分布拟合')
plt.xlabel('时间 (分钟)')
plt.ylabel('密度')
plt.legend()
plt.grid(True, alpha=0.3)

# 添加参数说明文本
param_text = f'伽马分布参数:\nα (shape) = {shape:.4f}\nβ (scale) = {scale:.4f}\nlocation = {loc:.4f}'
plt.text(0.8, 0.98, param_text,
         transform=plt.gca().transAxes,
         verticalalignment='top',
         horizontalalignment='right',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.ylim(0, 0.25)
plt.savefig('histogram_with_fit.png', dpi=300, bbox_inches='tight')
plt.close()