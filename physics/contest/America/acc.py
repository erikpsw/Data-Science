import numpy as np
import matplotlib.pyplot as plt
N=1000
t_max=300
dt=t_max/N

def get_max_acc(h):
    vel_s=[]
    height=[]
    a_list=[]
    G=6.67e-11
    M=5.965e24
    R=6371.393e3
    m=190
    v=0
    for i in range(N):
        if(h<0):
            break
        d=1.5*np.e**(-h/6450)
        a=G*M/(h+R)**2-0.712*d*v**2/m
        a_list.append(a)
        v+=a*dt
        h-=v*dt
        
        vel_s.append(v)
        height.append(h)
    return a_list

plt.rcParams['text.usetex'] = True
plt.rc("font",family='Times New Roman')
fig,ax = plt.subplots(1,1)
H=np.linspace(50,200,6)
time_list=np.linspace(0,t_max,N)

for h in H:
    acc_list=get_max_acc(h*1e3)
    ax.plot(time_list,acc_list,label=f'{h:.0f}km')
plt.axhline(0, color='red', linestyle='--')
ax.set_xlabel("$t$/s",fontsize=14)
ax.set_title("height-acceleration graph",fontsize=14)
plt.ylabel("$a$/m$\cdot$s$^{-2}$",fontsize=14)
plt.legend()
plt.show()