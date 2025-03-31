import B_ques_four_func as func
import numpy as np
import matplotlib.pyplot as plt

NM=1852
width=4*NM
height=5*NM


def get_all_res(beta,d1,d2,d3,d4):
    NM=1852
    N=20
    beta=np.radians(beta)
    dx1=d1/np.sin(beta) #x轴间距
    dx2=d2/np.sin(beta) #x轴间距
    dx3=d3/np.sin(beta) #x轴间距
    dx4=d4/np.sin(beta) #x轴间距
    x1=-height/(np.tan(beta))

    line_list1=[]
    cur_x=x1+dx1
    length=0
    n=0
    while(cur_x<width):
        n+=1
        cur_ans=func.get_start_end_points(cur_x,beta,width,height)
        if(len(cur_ans)==2):
            line_list1.append(cur_ans)
            length+=np.linalg.norm(cur_ans[1]-cur_ans[0])
      
        if(n<N):
            cur_x+=dx1
        elif(n>=N and n<2*N):     
            cur_x+=dx2
        elif(n>=2*N and n<3*N):     
            cur_x+=dx3
        elif(n>=3*N):     
            cur_x+=dx4

    dt=0.4*NM
    cur_x=dt
    ans=func.get_start_end_points(cur_x,beta+np.pi/2,width,height)
    left_dir=np.array([np.cos(beta+np.pi/2),np.sin(beta+np.pi/2),0])*np.sqrt(3)/2+np.array([0,0,-1])/2
    right_dir=np.array([np.cos(beta-np.pi/2),np.sin(beta-np.pi/2),0])*np.sqrt(3)/2+np.array([0,0,-1])/2
    line_list2=[]

  
    while(len(ans)!=0):
        cur_x+=dt   
        
        if(len(ans)==2):
            line_list2.append(ans)
        ans=func.get_start_end_points(cur_x,beta+np.pi/2,width,height)

    # print("list1",line_list1)
    # print("list2",line_list2)

    iteration_arr=[]
    for l2 in line_list2:
        cur_arr=[]
        for l1 in line_list1:
            p=func.get_perp_point(l2,l1)
            if(p[0]<width and p[0]>0 and p[1]<height and p[1]>0):
                cur_arr.append(p)
        iteration_arr.append(cur_arr) 
    sum=0
    for i in iteration_arr:
        sum+=len(i) 
    # print(sum)
    path_list=[]
    for points in iteration_arr:
        tmp_list=[]
        if(len(points)>1):
            for point in points:
                cur_p=np.array([point[0],point[1],0])
                tmp_list.append([func.get_intersect_point(cur_p,left_dir),func.get_intersect_point(cur_p,right_dir)])
        path_list.append(tmp_list)
    overlap_point_list=[]
    all_area=0
    lost_area=0#漏测
    lost_points=[]
    for index in range(len(path_list)):
        points=path_list[index]
        overlap_arr=[]
        width_list=[]
        if(len(points)>1):
            all_area+=points[-1][0][0]-points[0][1][0]
         
            for i in range(len(points)-1):
                tmp_overlap=points[i+1][0][0]-points[i][1][0]
                overlap_arr.append(tmp_overlap)
                if(tmp_overlap>0):
                    lost_area+=tmp_overlap
                    lost_points.append(points[i+1][0])
                    # lost_points.append(points[i][1])

            for point in points:
                width_list.append(point[1][0]-point[0][0])
            if(overlap_arr[0]/width_list[0]<-0.2 ):
                overlap_point_list.append(iteration_arr[index][0])
            if(overlap_arr[-1]/width_list[-1]<-0.2 ):
                overlap_point_list.append(iteration_arr[index][-1])
            for j in range(1,len(points)-1):
                cur_overlap_left=overlap_arr[j-1]
                cur_overlap_right=overlap_arr[j]
                cur_width=width_list[j]
                if(cur_overlap_left/cur_width<-0.2 or cur_overlap_right/cur_width<-0.2):
                    overlap_point_list.append(iteration_arr[index][j])
            # print(all_area)
    overlap_point_arr=np.array(overlap_point_list)
    point_list=[]
    for i in iteration_arr:
        for j in i:
            point_list.append(j)
    #重叠超20%的区域，总采样点，总长，未覆盖区域占比，重叠超20%的区域占比，未覆盖区域坐标

    if(len(point_list)!=0 and all_area!=0):
        print(length,lost_area/all_area,len(overlap_point_arr)/len(point_list))
        return overlap_point_arr,np.array(point_list),length,lost_area/all_area,len(overlap_point_arr)/len(point_list),np.array(lost_points),np.array(line_list1)


def get_graph(beta,d1,d2,d3,d4):

    res=get_all_res(beta,d1,d2,d3,d4)
    overlap_point_arr=res[0]
    lost_point_arr=res[-2]
    line_list=res[-1]
    print(res[2],res[3],res[4])
    plt.rc("font",family='SimSun')
    rectangle = plt.Rectangle((0, 0), width, height, edgecolor='r', facecolor='none')
    fig, ax = plt.subplots(1,2)
    fig.dpi=150
    # ax.set_xlim(-1*NM, 5*NM)
    # ax.set_ylim(-1*NM, 6*NM)

    ax[0].set_xlim(-1*NM, width+NM)
    ax[0].set_ylim(-1*NM, height+NM)
    ax[1].set_xlim(-1*NM, width+NM)
    ax[1].set_ylim(-1*NM, height+NM)
    ax[0].add_patch(rectangle)
    ax[1].add_patch(plt.Rectangle((0, 0), width, height, edgecolor='r', facecolor='none'))

    # 设置标题和坐标轴标签
    ax[1].set_title('重叠率超过20%的区域',fontsize=14)
    ax[0].set_title('漏测海区',fontsize=14)
    # plt.xlabel('横向坐标/m',fontsize=14)
    # plt.ylabel('纵向坐标/m',fontsize=14)
    ax[0].tick_params(axis='both', labelsize=14)
    ax[1].tick_params(axis='both', labelsize=14)
    ax[0].scatter(overlap_point_arr[:,0],overlap_point_arr[:,1])
    ax[1].scatter(lost_point_arr[:,0],lost_point_arr[:,1])
    for line in line_list:
        ax[0].plot(line[:,0],line[:,1])
    plt.show()

get_graph(55.159231751885955, 192.58871262329208, 249.38190919857394, 149.60031469687584, 54.4771774712838)


#实验
NM=1852
width=5*NM
height=4*NM
d=300
beta=np.radians(45)
dx=d/np.sin(beta) #x轴间距
x1=-height/(np.tan(beta))
ans=func.get_start_end_points(-3*NM,beta,width,height)
print(ans)
start_p=ans[0]
end_p=ans[1]
x1=-height/(np.tan(beta))
cur_x=x1
cur_x+=4*dx
plt.figure()
i=0
while(cur_x<x1+13*dx):
    cur_x+=dx
    i+=1
    cur_ans=func.get_start_end_points(cur_x,beta,width,height)
    if(len(cur_ans)==2):
        start_p=cur_ans[0]
        end_p=cur_ans[1]
        left_path_x,left_path_y,right_path_x,right_path_y,path_x,path_y=func.get_path_array(start_p,end_p,100)
        if(i%2==0):
            plt.plot(left_path_x,left_path_y,linestyle = "-")
            plt.plot(right_path_x,right_path_y,linestyle = "-")
        else:
            plt.plot(left_path_x,left_path_y,linestyle = "-.")
            plt.plot(right_path_x,right_path_y,linestyle = "-.")
        plt.plot(path_x,path_y)

plt.rc("font",family='SimSun')
plt.title('轨迹和覆盖边界', fontsize=12)
plt.xlabel('纵向坐标/m', fontsize=12)
plt.ylabel('横向坐标/m', fontsize=12)
plt.tick_params(axis='both', labelsize=12)
plt.show()