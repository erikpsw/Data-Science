import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import cKDTree

# 定义两个点的坐标
x = [1, 4]  # x坐标
y = [2, 6]  # y坐标

# 绘制两点之间的线
# plt.Rectangle()
rectangle = plt.Rectangle((0, 0), 4, 5, edgecolor='r', facecolor='none')
NM=1852
# 创建一个图形并添加矩形
# fig, ax = plt.subplots()
# fig.dpi=150
# ax.set_xlim(-6, 5)
# ax.set_ylim(-1, 6)
# ax.add_patch(rectangle)
import pandas as pd


# 设置标题和坐标轴标签
# plt.title('Line Between Two Points')
# plt.xlabel('X')
# plt.ylabel('Y')
beta=45
beta=np.radians(beta)
height=5*NM
width=4*NM

d=1*NM

x1=-height/(np.tan(beta))
dx=d/np.sin(beta) #x轴间距
# ax.plot([x1,0],[0,height])

def get_start_end_points(xi,beta,width,height):
    ans=[]
    if(xi>0 and xi<width):#下边交点
        ans.append(np.array([xi,0]))
    y=-np.tan(beta)*xi
    if(y>0 and y<height):#左边交点
        ans.append(np.array([0,y]))
    y=np.tan(beta)*(width-xi)
    if(y>0 and y<=height):#右边
        ans.append(np.array([width,y]))
    x=xi+(height/np.tan(beta))
    if(x>0 and x<width):#下边交点
        ans.append(np.array([x,height]))
    return ans


# cur_x=x1
# while(cur_x<width):
#     cur_x+=d
#     cur_ans=get_start_end_points(cur_x,beta)
#     if(len(cur_ans)==2):
#         p1=np.array(cur_ans)[:,0]
#         p2=np.array(cur_ans)[:,1]
#         ax.plot(p1,p2)
# plt.show()
df=pd.read_csv("CUMCM/data/data.csv")
x=df.columns[1:].astype(float).values
y=df.iloc[:,0].values
Z1=[]
Z2=[]
for index, row in df.iloc[:, 1:].iterrows():
    for column, value in row.iteritems():
        Z1.append([NM*float(column),NM*df.iloc[index, 0],-value])
        # Z2.append([5*NM-NM*df.iloc[index, 0],NM*float(column),-value])

# points = np.array(Z2)  # 曲面上的离散点
points = np.array(Z1)
kdtree = cKDTree(points)
# 离散点数据
def get_intersect_point(line_point,line_direction):
    # 构建离散点数据的KD树，用于快速查找最近点
    # 定义迭代参数
    max_iterations = 20  # 最大迭代次数
    distance_threshold =10  # 距离阈值，用于判断迭代终止条件
    t = 0 # 直线参数初始值
    step=10
    # 迭代计算交点
    min_distance=np.Inf
    for i in range(max_iterations):
        # 根据当前直线参数 t 计算直线上的点
        line_point_current = line_point + step*i * line_direction
        # 在离散点数据中查找最近点
        _, nearest_point_index = kdtree.query(line_point_current)

        # 获取最近点的坐标
        nearest_point = points[nearest_point_index]
        # 计算当前直线点与最近点之间的距离
        distance = np.linalg.norm(line_point_current - nearest_point)
        if(distance<min_distance):
            min_distance=distance
            t=step*i
        # 判断迭代终止条件
        if distance < distance_threshold:
            break
    
    best_j=0
    for j in np.linspace(-10,10,5):
        line_point_current = line_point + (t+j) * line_direction
        _, nearest_point_index = kdtree.query(line_point_current)

        nearest_point = points[nearest_point_index]
        distance = np.linalg.norm(line_point_current - nearest_point)
        if(distance<min_distance):
            min_distance=distance
            best_j=j
        if distance < distance_threshold:
            break
    best_k=0 
    for k in np.linspace(-1,1,5):
        line_point_current = line_point + (t+j+k) * line_direction
        _, nearest_point_index = kdtree.query(line_point_current)

        nearest_point = points[nearest_point_index]
        distance = np.linalg.norm(line_point_current - nearest_point)
        if(distance<min_distance):
            min_distance=distance
            best_k=k
        if distance < distance_threshold:
            break

    # 输出交点坐标
    intersection_point = line_point + (t+best_j+best_k) * line_direction
    return intersection_point

# print(get_intersect_point(np.array([3,3,0]),np.array([0,0,1])))

left_dir=np.array([np.cos(beta+np.pi/2),np.sin(beta+np.pi/2),0])*np.sqrt(3)/2+np.array([0,0,-1])/2
right_dir=np.array([np.cos(beta-np.pi/2),np.sin(beta-np.pi/2),0])*np.sqrt(3)/2+np.array([0,0,-1])/2

def get_path_array(begin,end,N=60):
    path=[]
    line_dir=(end-begin)
    line_dis=np.linalg.norm(line_dir)
    line_dir/=line_dis

    dl=line_dis/N
    left_dir=np.array([np.cos(beta+np.pi/2),np.sin(beta+np.pi/2),0])*np.sqrt(3)/2+np.array([0,0,-1])/2
    right_dir=np.array([np.cos(beta-np.pi/2),np.sin(beta-np.pi/2),0])*np.sqrt(3)/2+np.array([0,0,-1])/2

    left_list=[]
    right_list=[]
    for i in range(N+1):
        cur_pos=begin+i*dl*line_dir
        path.append(cur_pos)
        cur_pos3=np.zeros(3)
        cur_pos3[0]=cur_pos[0]
        cur_pos3[1]=cur_pos[1]
        left_list.append(get_intersect_point(cur_pos3,left_dir))
        right_list.append(get_intersect_point(cur_pos3,right_dir))
    left_path_x=np.array(left_list)[:,0]
    left_path_y=np.array(left_list)[:,1]
    right_path_x=np.array(right_list)[:,0]
    right_path_y=np.array(right_list)[:,1]
    path_x=np.array(path)[:,0]
    path_y=np.array(path)[:,1]
    return left_path_x,left_path_y,right_path_x,right_path_y,path_x,path_y

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