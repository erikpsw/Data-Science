import math

central_depth = 70
alpha = 1.5 #用度数表示
theta = 120

#计算距中心点某距离下的海水深度
#参数为测线距中心点处的距离 即为到central-depth的中心线的距离
#返回计算结果，即当前距离下的海水深度
#计算过程大致为使用距离*tan（alpha）即为相对深度，符号能表示方向
def caculate_depth(central_distance):
    #中心深度 +/- 距离*tan 
    #tan必须使用弧度制
    depth = central_depth - central_distance * math.tan(math.radians(alpha)) 
    # print(f"{math.tan(math.radians(alpha)) }")
    # print(f"{math.radians(alpha) }")
    return depth


#计算距中心点某距离下的覆盖面积
#参数1为测线距中心点处的距离 即为到central-depth的中心线的距离
#参数2为此时海水深度，传入参数避免再次调用函数
#返回计算结果，即当前距离下的覆盖距离
#逻辑即为左边长度 + 右边长度
#说明：覆盖宽度指的是与海底交点在水平面投影上的宽度
def caculate_width(depth):
    width_left = math.sin(math.radians(theta / 2)) * depth / math.sin(math.radians(90 - theta / 2 - alpha))
    width_right = math.sin(math.radians(theta / 2)) * depth / math.sin(math.radians(90 - theta / 2 + alpha))
    width = width_left + width_right
    width = width * math.cos(math.radians(alpha))

    # width = 2 * depth * math.tan(math.radians(theta / 2))

    return width


#参数1 2分别为前后（左右）的深度
#参数3表示两次航迹之间的距离
#参数4表示此次覆盖宽度，减少函数调用
#返回重叠率
#逻辑为前的右边 + 后的左边 - 两者在海底的距离
def caculate_overlap_rate(depth_before, depth_later, gap, width):
    width_left = math.sin(math.radians(theta / 2)) * depth_before / math.sin(math.radians(90 - theta / 2 + alpha))
    width_right = math.sin(math.radians(theta / 2)) * depth_later / math.sin(math.radians(90 - theta / 2 - alpha))
    width_all = gap / math.cos(math.radians(alpha))

    overlap = width_left + width_right - width_all
    print(f"{width} {width_left + width_right}")
    overlap_rate = overlap * math.cos(math.radians(alpha)) / width   # 
    return overlap_rate


def main():
    before_distance = float(input("前一个到中心距离："))
    central_distance = float(input("到中心距离："))
    depth = caculate_depth(central_distance)
    depth_before = caculate_depth(before_distance)
    width = caculate_width(depth)
    overlap_rate = caculate_overlap_rate(depth_before, depth_before
                                         , abs(before_distance - central_distance), caculate_width(depth_before))
    
    print(f"前一个x： {before_distance}, 以及当前x：{central_distance}")
    print(f"海水深度：{depth}")
    print(f"覆盖宽度：{width}")
    print(f"重叠率{overlap_rate}")
    print(f"海水深度：{depth:.2f}")
    print(f"覆盖宽度：{width:.2f}")
    print(f"重叠率{overlap_rate * 100 :.2f}%")

    return


if __name__ == "__main__":
    main()
