import numpy as np

NM=1852
beta=45
beta=np.radians(beta)
width=4*NM
height=5*NM
d=1*NM

x1=-height/(np.tan(beta))
dx=d/np.sin(beta) #x轴间距
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
    
    # 判断两条线段是否相交
    if (x3 - x1) * (x4 - x2) < 0 and (y3 - y1) * (y4 - y2) < 0:
        # 计算交点的坐标
        intersection_x = (x3 * (y2 - y1) * (x2 - x1) - x4 * (y2 - y1) * (x2 - x1) - x1 * (y4 - y3) * (x4 - x3) + x2 * (y4 - y3) * (x4 - x3)) / ((y2 - y1) * (x4 - x3) - (y4 - y3) * (x2 - x1))
        intersection_y = (y1 * (x4 - x3) * (y4 - y3) - y3 * (x4 - x3) * (y4 - y3) - y4 * (x2 - x1) * (y2 - y1) + y2 * (x2 - x1) * (y2 - y1)) / ((y2 - y1) * (x4 - x3) - (y4 - y3) * (x2 - x1))

    # 输出交点的坐标
        print("交点的坐标：", intersection_x, intersection_y)
line_list1=[]
cur_x=x1
while(cur_x<width):
    cur_x+=dx
    cur_ans=get_start_end_points(cur_x,beta)
    line_list1.append(cur_ans)
    if(len(cur_ans)==2):
        p1=np.array(cur_ans)[:,0]
        p2=np.array(cur_ans)[:,1]

dx=0.5*NM
cur_x=dx
ans=get_start_end_points(cur_x,beta+np.pi/2)
#ans是两个点的坐标
line_list2=[]

while(len(ans)!=0):
    line_list2.append(ans)
    cur_x+=dx   
    if(len(ans)==2):
        p1=np.array(ans)[:,0]
        p2=np.array(ans)[:,1]
    ans=get_start_end_points(cur_x,beta+np.pi/2)
for l2 in line_list2:
    for l1 in line_list1:
        print(l2,l1)
        
# intersection_list=[[[1,5],[3,3]],[[2,4],[4,2]],[[3,3],[5,1]]]
# overlap_list=
# if(len(intersection_list)>1):
    
# print(get_perp_point(line_list1[3],line_list2[3]))