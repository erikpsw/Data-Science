import simpy
import random
import statistics

# 定义模拟参数
RANDOM_SEED = 42
NUM_MACHINES = 2  # 生产线上的机器数量
PROCESS_TIME = [5, 3]  # 每台机器的处理时间（分钟）
PROCESS_TIME_DEVIATION = [1, 0.5]  # 处理时间的随机偏差
ARRIVAL_RATE = 2  # 原料到达的平均时间间隔（分钟）
SIM_TIME = 480  # 模拟时间（分钟）（相当于一个8小时工作日）

class ProductionLine:
    def __init__(self, env, num_machines, process_time, process_deviation):
        self.env = env
        self.machine = [simpy.Resource(env, capacity=1) for _ in range(num_machines)]
        self.process_time = process_time
        self.process_deviation = process_deviation
        
        # 统计数据
        self.products_completed = 0
        self.waiting_times = []
        self.production_times = []
    
    def process(self, product_id, step=0):
        """表示产品在某一步骤的处理过程"""
        # 到达当前步骤的时间
        arrival_time = self.env.now
        
        # 请求使用机器
        with self.machine[step].request() as request:
            print(f"{self.env.now:.2f}: 产品{product_id}在步骤{step+1}等待处理")
            yield request
            
            # 计算等待时间
            waiting_time = self.env.now - arrival_time
            self.waiting_times.append(waiting_time)
            
            print(f"{self.env.now:.2f}: 产品{product_id}开始在步骤{step+1}处理")
            
            # 处理时间有随机偏差
            proc_time = random.normalvariate(
                self.process_time[step], 
                self.process_deviation[step]
            )
            proc_time = max(0.1, proc_time)  # 确保处理时间为正
            
            yield self.env.timeout(proc_time)
            
            print(f"{self.env.now:.2f}: 产品{product_id}在步骤{step+1}处理完成")
            
            # 如果还有下一步
            if step < len(self.machine) - 1:
                yield self.env.process(self.process(product_id, step+1))
            else:
                # 完成全部生产过程
                production_time = self.env.now - arrival_time
                self.production_times.append(production_time)
                self.products_completed += 1
                print(f"{self.env.now:.2f}: 产品{product_id}完成所有处理步骤")

def product_generator(env, production_line):
    """生成产品到达事件"""
    product_id = 1
    while True:
        # 产品到达时间间隔服从指数分布
        inter_arrival = random.expovariate(1.0 / ARRIVAL_RATE)
        yield env.timeout(inter_arrival)
        
        print(f"{env.now:.2f}: 产品{product_id}到达生产线")
        env.process(production_line.process(product_id))
        product_id += 1

def setup(env, num_machines, process_time, process_deviation):
    """设置并启动模拟"""
    # 创建生产线
    production_line = ProductionLine(env, num_machines, process_time, process_deviation)
    
    # 启动产品生成器
    env.process(product_generator(env, production_line))
    
    # 返回生产线对象以便收集统计数据
    return production_line

def print_statistics(production_line):
    """打印模拟结果统计"""
    print("\n生产线模拟统计:")
    print(f"完成产品数量: {production_line.products_completed}")
    
    if production_line.waiting_times:
        avg_wait = statistics.mean(production_line.waiting_times)
        print(f"平均等待时间: {avg_wait:.2f}分钟")
    
    if production_line.production_times:
        avg_prod = statistics.mean(production_line.production_times)
        print(f"平均生产周期: {avg_prod:.2f}分钟")

# 设置随机种子以便结果可复现
random.seed(RANDOM_SEED)

# 创建环境
env = simpy.Environment()

# 设置并运行模拟
line = setup(env, NUM_MACHINES, PROCESS_TIME, PROCESS_TIME_DEVIATION)
env.run(until=SIM_TIME)

# 打印统计结果
print_statistics(line)