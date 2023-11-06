import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

time=pd.Series([0,
9.91282,
19.825,
30.3646,
40.6698,
50.5062,
60.4788,
70.2028,
80.3866,
90.3765,
99.9355,
110.152,
120.37,
130.603,
140.193,
150.21,
160.22,
170.448,
180.043,
190.06,
200.708,
210.087,
220.108,
230.342,
240.148,
250.168,
260.395
])

vel=pd.Series(
    [
        0,
98.2226,
195.843,
282.019,
348.315,
374.851,
329.098,
250.211,
202.652,
173.164,
139.458,
122.621,
106.989,
105.213,
101.025,
96.838,
86.6272,
80.6339,
80.0604,
75.8737,
63.8576,
61.4762,
61.5064,
60.9348,
58.5547,
57.9824,
50.7843,

    ]
)

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

degree = 9  # 拟合多项式的次数
coefficients = np.polyfit(time, vel, degree)
poly = np.poly1d(coefficients)

# 绘制拟合曲线
fit_time = np.linspace(min(time), max(time), 100)  # 用于拟合的时间点
fit_vel = np.polyval(poly, fit_time)  # 拟合曲线上的速度

degree = 9  # 拟合多项式的次数
coefficients = np.polyfit(time1, height0, degree)
poly = np.poly1d(coefficients)

# 绘制拟合曲线
fit_time2 = np.linspace(min(time1), max(time1), 100)  # 用于拟合的时间点
fit_height = np.polyval(poly, fit_time2)  # 拟合曲线上的速度
plt.rcParams['text.usetex'] = True
plt.rc("font",family='Times New Roman')

fig,ax = plt.subplots(1,1)

# ax.plot(fit_time,fit_vel)
# ax.set_xlabel("$t$/s",fontsize=14)
# ax.set_title("velocity-time graph",fontsize=14)
# ax.set_ylabel("$v$/m/s",fontsize=14)

ax.plot(fit_time2,fit_height)
ax.set_xlabel("$t$/s",fontsize=14)
ax.set_title("height-time graph",fontsize=14)
ax.set_ylabel("$h$/km",fontsize=14)
plt.show()