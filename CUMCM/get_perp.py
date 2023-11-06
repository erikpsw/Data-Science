def get_perp_point(l1,l2):
    point_A = l1[0]
    point_B = l1[1]
    # 第二条线段的端点
    point_C = l2[0]
    point_D = l2[1]
    x1=point_A [0]
    x2=point_B [0]
    x3=point_C [0]
    x4=point_D [0]
    y1=point_A [1]
    y2=point_B [1]
    y3=point_C [1]
    y4=point_D [1]
    
    m1 = (y2 - y1) / (x2 - x1)
    m2 = (y4 - y3) / (x4 - x3)
    # print(m1, m2)

    # 判断两条线段是否相交
    if m1 != m2:
        # 计算交点的坐标
        intersection_x = (m1 * x1 - y1 - m2 * x3 + y3) / (m1 - m2)
        intersection_y = m1 * (intersection_x - x1) + y1
        # 输出交点的坐标
        return [intersection_x, intersection_y]

        



import numpy as np
import matplotlib.pyplot as plt

# 生成随机散点数据
x = np.random.randn(1000)
y = np.random.randn(1000)

# 创建密度图
heatmap, xedges, yedges = np.histogram2d(x, y, bins=50)
extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

# 绘制密度图
plt.imshow(heatmap.T, extent=extent, origin='lower', cmap='hot')
plt.colorbar()

# 添加标题和标签
plt.title('Scatter Density Plot')
plt.xlabel('X')
plt.ylabel('Y')

# 显示图形
plt.show()