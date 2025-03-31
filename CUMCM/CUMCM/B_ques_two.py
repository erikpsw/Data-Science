import numpy as np
import math
import pandas as pd

DO=120

beta_list=np.linspace(0,315,8)
dis_list=np.linspace(0,2.1,8)
theta = 120
alpha=np.radians(1.5)

def caculate_width(depth,alpha):
    width_left = math.sin(math.radians(theta / 2)) * depth / math.sin(math.radians(90 - theta / 2 - alpha))
    width_right = math.sin(math.radians(theta / 2)) * depth / math.sin(math.radians(90 - theta / 2 + alpha))
    width = width_left + width_right
    width = width * math.cos(math.radians(alpha))
    return width

def get_depth(beta,dis):
    return DO+dis*np.tan(alpha)*np.cos(np.radians(beta))
    

def get_width(beta,dis):
    dis*=1852
    D=get_depth(beta,dis)
    alpha_1=np.arctan(np.tan(alpha)*np.sin(beta))
    alpha_1=np.degrees(alpha_1)
    return caculate_width(D,alpha_1)

df=pd.DataFrame(index=beta_list,columns=dis_list)
for dis in dis_list:
    for beta in beta_list:
        df[dis][beta]=get_width(beta,dis)

print(df)
df.to_csv("questwo.csv")