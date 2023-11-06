import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Data of time and height
time1 = pd.Series([
    0, 9.85977, 19.9387, 30.0175, 39.6582, 44.6976, 49.7371, 60.0351, 70.1139, 79.7546,
    89.8335, 99.9124, 109.991, 120.289, 129.93, 140.009, 150.088, 160.386, 170.245,
    180.324, 190.403, 200.482, 210.561, 220.421, 230.28, 240.14, 250, 260.079
])

height0 = pd.Series([
    39.0506, 38.6051, 36.8304, 34.6127, 31.4457, 29.6089, 27.7722, 24.0354, 21.3747,
    19.0937, 17.2557, 15.6709, 14.4025, 13.1974, 12.119, 11.1671, 10.2785, 9.26319,
    8.43791, 7.61258, 6.85053, 6.02519, 5.51631, 4.88091, 4.37208, 3.48351, 3.16456,
    2.52909
])

# Data of time and velocity
time2 = pd.Series([
    0, 9.91282, 19.825, 30.3646, 40.6698, 50.5062, 60.4788, 70.2028, 80.3866, 90.3765,
    99.9355, 110.152, 120.37, 130.603, 140.193, 150.21, 160.22, 170.448, 180.043,
    190.06, 200.708, 210.087, 220.108, 230.342, 240.148, 250.168, 260.395
])

velocity = pd.Series([
    0, 98.2226, 195.843, 282.019, 348.315, 374.851, 329.098, 250.211, 202.652, 173.164,
    139.458, 122.621, 106.989, 105.213, 101.025, 96.838, 86.6272, 80.6339, 80.0604,
    75.8737, 63.8576, 61.4762, 61.5064, 60.9348, 58.5547, 57.9824, 50.7843
])

R=6371.393e3 # radius of the earth
gamma=1.4 # ratio of the molar heat capacity

v=0

def get_T(h):
    h/=1e3
    h_p=6357.766*h/(6357.766+h)
    if(0<=h_p<=11):
        return 288.15 - 6.5*h_p
    elif(11<h_p<=20):
        return 216.65
    elif(20<h_p<=32):
        return 216.65+(h_p-20)
    elif(32<h_p<=47):
        return 228.65 + 2.8*(h_p-32)
    elif(47<h_p<=51):
        return 270.65 
    elif(51<h_p<=71):
        return 270.65 - 2.8 *(h_p-51)
    elif( 71 <h_p <= 84.852):
        return 214.65 - 2.0 *(h_p-71) 
    elif(86 <= h <= 91):
        return 186.8673
    elif(91 < h <= 100):
        return 263.1905 - 76.3232 * (1 - ((h - 91) / 19.9429) ** 2) ** 0.5
    else:
        return 195.07
get_v_s=lambda h:np.sqrt(1.4*8.314*get_T(h)/0.029)

# pressure in hPa

def get_P(h):
    h/=1e3
    h_p=6357.766*h/(6357.766+h)
    if 0 <= h_p <= 11:
        return 1013.25 * (288.15 / (288.15 - 6.5 * h_p)) ** (-34.1632 / 6.5)
    elif 11 < h_p <= 20:
        return 226.3226 * np.exp(-34.1632 * (h_p - 11) / 216.65)
    elif 20 < h_p <= 32:
        return 54.74980 * (216.65 / (216.65 + (h_p - 20))) ** 34.1632
    elif 32 < h_p <= 47:
        return 8.680422 * (228.65 / (228.65 + 2.8 * (h_p - 32))) ** (34.1632 / 2.8)
    elif 47 < h_p <= 51:
        return 1.109106 * np.exp(-34.1632 * (h_p - 47) / 270.65)
    elif 51 < h_p <= 71:
        return 0.6694167 * (270.65 / (270.65 - 2.8 * (h_p - 51))) ** (-34.1632 / 2.8)
    elif 71 < h_p <= 84.852:
        return 0.03956649 * (214.65 / (214.65 - 2.0 * (h_p - 71))) ** (-34.1632 / 2.0)
    elif 86 < h :
        return np.exp(95.571 - 4.012 * h)
    else: return 1013

# t_max=260

N=10000

# initial height and simulation time

h=39.0506e3
t_max=260

dt=t_max/N
g=9.6
G=6.67e-11
m=190
M=5.965e24

def get_res():
    vel_s=[] # list of simulated velocity
    height=[]
    a_list=[]
    P_list=[] #list of pressure
    sos_list=[] #list of sound in an specific height
    global h,v
    for i in range(N):
        
        if(h<0):
            break
        else:
            d=1.5*np.e**(-h/6450) # density
            a=G*M/(h+R)**2-0.712*d*v**2/m # acceleration
            a_list.append(a) 
            v+=a*dt
            h-=v*dt
            
            sos=get_v_s(h) #speed of sound
            Mach=v/sos # Mach Number
            P0=get_P(h) 
            P=(2*gamma*Mach**2-(gamma-1))*P0/(gamma+1)
            P_list.append(P*100) # hPa to Pa
            sos_list.append(sos)
            
            vel_s.append(v)
            height.append(h/1e3)
    return vel_s,height,a_list,P_list,sos_list

ans=get_res()
(vel_s,height,a_list,P_list,sos_list)=ans

plt.rcParams['text.usetex'] = True
plt.rc("font",family='Times New Roman')

fig = plt.figure(num=1,figsize=(7,5), facecolor='white',dpi=150)
time_list=np.linspace(0,t_max,N)


# velocity
fig = plt.figure(num=1,figsize=(7,5), facecolor='white',dpi=150)
plt.plot(time2,velocity,label=r"recorded")
plt.plot(time_list,vel_s,label=r"simulation")
plt.xlabel("$t$/s")
plt.title("Velocity-Time Relationship During the Skydiving Process")
plt.ylabel("$v$/m$\cdot$s$^{-1}$")
plt.legend()
plt.show()

# height

fig = plt.figure(num=1,figsize=(7,5), facecolor='white',dpi=150)
plt.plot(time1,height0,label=r"recorded")
plt.plot(time_list,height,label=r"simulation")
plt.xlabel("$t$/s")
plt.ylabel("$h$/km")
plt.title("Height-Time Relationship During the Skydiving Process")
plt.show()

# plot speed of sound

fig = plt.figure(num=1,figsize=(7,5), facecolor='white',dpi=150)
mask = np.array(sos_list)<np.array(vel_s)

fig = plt.figure(num=1,figsize=(7,5), facecolor='white',dpi=150)
plt.plot(time_list,sos_list,label=r"speed of sound")
plt.plot(time_list,vel_s,label=r"simulation")
plt.fill_between(time_list, sos_list, vel_s, where=mask, facecolor="red", alpha=1)
x=34.9
y=305.53
plt.plot([x, x], [y, 0], 'k--')  # 绘制垂线
x=67.736
y=296.233
plt.plot([x, x], [y, 0], 'k--')  # 绘制垂线
plt.xlabel("$t$/s")
plt.ylabel("$v$/m$\cdot$s$^{-1}$")
plt.legend()
plt.show()

# plot acceleration

fig = plt.figure(num=1,figsize=(7,5), facecolor='white',dpi=150)
plt.plot(time_list,a_list)
plt.xlabel("$t$/s",fontsize=14)
plt.ylabel("$a$/m$\cdot$s$^{-2}$",fontsize=14)
plt.title("Acceleration-Time Relationship During the Skydiving Process",fontsize=14)
plt.show()

# plot pressure h=100km

fig = plt.figure(num=1,figsize=(7,5), facecolor='white',dpi=150)
H=np.linspace(40,200,5)
for cur_h in H:
    h,v=cur_h*1e3,0
    res=get_res()
    p_list=res[3]
    plt.plot(res[1],p_list,label=f'{cur_h:.0f}km')
print(f'max over pressure is {max(res[3])}')
# plt.axhline(0, color='red', linestyle='--')
plt.ylim(-1000,30000)
plt.xlabel("$h$/km",fontsize=14)
plt.title("overPressue-height graph",fontsize=14)
plt.ylabel("$p$/Pa",fontsize=14)
plt.legend()
plt.show()

fig = plt.figure(num=1,figsize=(7,5), facecolor='white',dpi=150)
H=np.linspace(50,200,6)
for cur_h in H:
    h,v=cur_h*1e3,0
    acc_list=get_res()[2]

    plt.plot(time_list,acc_list,label=f'{cur_h:.0f}km')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("$t$/s",fontsize=14)
plt.title("acceleration-time graph",fontsize=14)
plt.ylabel("$a$/m$\cdot$s$^{-2}$",fontsize=14)
plt.legend()
plt.show()