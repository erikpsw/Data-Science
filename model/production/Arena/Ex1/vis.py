import matplotlib.pyplot as plt
import numpy as np
#支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False     # 用来正常显示负号

# 数据准备
cases = ["① U(1,2)", "② U(0.5,2.5)", "③ N(1.5,0.3)", "④ N(1.5,0.6)"]
total_time = [1.0637, 1.2537, 1.0760, 1.3092]
wip = [0.7084, 0.8343, 0.7167, 0.8706]
idle_rate = [0.2989, 0.2994, 0.2989, 0.2997]

x = np.arange(len(cases))  # x轴位置
width = 0.25  # 柱状图宽度

# 绘图
fig, ax = plt.subplots(figsize=(10, 6))
rects1 = ax.bar(x - width, total_time, width, label='平均系统时间 Total Time', color='skyblue')
rects2 = ax.bar(x, wip, width, label='平均零件数量 WIP', color='lightgreen')
rects3 = ax.bar(x + width, idle_rate, width, label='机器闲置率 Idle Rate', color='salmon')

# 添加标签和标题
ax.set_ylabel('数值')
ax.set_title('四种到达/加工时间分布下的系统性能对比')
ax.set_xticks(x)
ax.set_xticklabels(cases)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

# 数值标签
def add_labels(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

add_labels(rects1)
add_labels(rects2)
add_labels(rects3)

plt.tight_layout()
plt.show()

