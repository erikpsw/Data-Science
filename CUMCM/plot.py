import matplotlib.pyplot as plt
import numpy as np

# 定义两个点的坐标
x = [1, 4]  # x坐标
y = [2, 6]  # y坐标
NM=1852

# 绘制两点之间的线
# plt.Rectangle()
rectangle = plt.Rectangle((0, 0), 4*NM, 5*NM, edgecolor='r', facecolor='none')

# 创建一个图形并添加矩形
fig, ax = plt.subplots()
fig.dpi=150
ax.set_xlim(-3*NM, 6*NM)
ax.set_ylim(-0.5*NM, 5.5*NM)
ax.add_patch(rectangle)
import pandas as pd

# 设置标题和坐标轴标签
plt.rc("font",family='SimSun')
plt.title('海平面俯视图', fontsize=14)
plt.xlabel('纵向坐标/m', fontsize=14)
plt.ylabel('横向坐标/m', fontsize=14)
plt.tick_params(axis='both', labelsize=14)
NM=1852
beta=30
beta=np.radians(beta)
width=4*NM
height=5*NM
d=0.5*NM

x1=-height/(np.tan(beta))
dx=d/np.sin(beta) #x轴间距
# ax.plot([x1,0],[0,height])

def get_start_end_points(xi,beta):
    ans=[]
    if(xi>0 and xi<=width):#下边交点
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

cur_x=x1+dx
point_list1=[]
while(cur_x<width):
    cur_ans=get_start_end_points(cur_x,beta)
    point_list1.append(cur_ans)
    if(len(cur_ans)==2):
        p1=np.array(cur_ans)[:,0]
        p2=np.array(cur_ans)[:,1]
        ax.plot(p1,p2)
    cur_x+=dx
dx=0.5*NM
cur_x=dx
ans=get_start_end_points(cur_x,beta+np.pi/2)
point_list=[]

while(len(ans)!=0):
    point_list.append(ans)
    print(cur_x)
    cur_x+=dx   
    if(len(ans)==2):
        p1=np.array(ans)[:,0]
        p2=np.array(ans)[:,1]
        ax.plot(p1,p2,linestyle='--')
    ans=get_start_end_points(cur_x,beta+np.pi/2)
plt.show()
