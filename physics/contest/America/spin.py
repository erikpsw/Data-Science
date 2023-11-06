import numpy as np
from scipy.spatial.transform import Rotation
import copy
import matplotlib.pyplot as plt

theta=np.radians(30)
v=5
vel=np.array([v*np.cos(theta),0,v*np.sin(theta)])

N=1000
t_max=10
dt=t_max/N
G=6.67e-11
m=190
M=5.965e24
R=6371.393e3
# h=400e3
h=40e3
pos=np.array([0.,0.,h])
pos_list=[]

for i in range(N):
    tmp=np.cross(vel,(np.cross(vel,np.array([0,0,1]))))
    axis=0.0001*tmp*dt*np.linalg.norm(vel)**2/np.linalg.norm(tmp)
    rotation = Rotation.from_rotvec(axis)
    vel=rotation.apply(vel)
    print(pos)
    d=1.5*np.e**(-h/6450)
    a=G*M*(1-d*v**2/(47**2))/(h+R)**2
    vel[2]-=a*dt
    pos+=vel*dt
    pos_list.append(copy.deepcopy(pos))

res=np.array(pos_list)
x=res[:,0]
y=res[:,1]
z=res[:,2]
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim(0,18)
ax.set_ylim(-18,0)
ax.plot(x, y, z)

# 设置图形参数
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()