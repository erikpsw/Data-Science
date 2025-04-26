import simpy
import random
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats

plt.rcParams['text.usetex'] = False
plt.rc("font",family='MicroSoft YaHei')

# 生产计划循环
tv_schedule = [
    ('Small', 20),
    ('Medium', 30),
    ('Large', 40),
    ('Flat', 20)
]

# 维修工人
class Maintenance:
    def __init__(self, env):
        self.resource = simpy.Resource(env, capacity=1)


def operation(env, name, in_queue, out_queue, cycle_time,
              setup_dist=None, breakdown=False, mttf=None, repair_dist=None,
              maintenance=None, tv_type=None, current_type=None, repair_log=None):

    def breakdown_process():
        while True:
            failure_time = random.expovariate(1.0 / mttf)
            yield env.timeout(failure_time)
            with maintenance.resource.request() as req:
                yield req
                repair_time = repair_dist()
                repair_log.append((env.now, name, repair_time))
                yield env.timeout(repair_time)

    # 如果启用故障，创建子进程
    if breakdown and mttf and repair_dist and maintenance:
        env.process(breakdown_process())

    while True:
        if len(in_queue.items) == 0:
            yield env.timeout(0.1)
            continue

        pallet = yield in_queue.get()

        # 是否需要更换 TV 类型（需要 setup）
        if setup_dist and (pallet['type'] != current_type[0]):
            setup_time = setup_dist()
            yield env.timeout(setup_time)
            current_type[0] = pallet['type']

        # 加工时间支持常数或正态分布
        if isinstance(cycle_time, tuple):
            ct = random.normalvariate(*cycle_time)
        else:
            ct = cycle_time

        yield env.timeout(ct)
        yield out_queue.put(pallet)


def test_station(env, in_queue, out_queue, rework_queue, failure_rate,
                 cycle_time, setup_dist, repair_dist, maintenance,
                 current_type, repair_log):

    def breakdown_process():
        while True:
            failure_time = random.expovariate(1.0 / 250)  # Test MTTF 固定为 250
            yield env.timeout(failure_time)
            with maintenance.resource.request() as req:
                yield req
                repair_time = repair_dist()
                repair_log.append((env.now, 'Test', repair_time))
                yield env.timeout(repair_time)

    env.process(breakdown_process())

    while True:
        if len(in_queue.items) == 0:
            yield env.timeout(0.1)
            continue

        pallet = yield in_queue.get()

        if pallet['type'] != current_type[0]:
            setup_time = setup_dist()
            yield env.timeout(setup_time)
            current_type[0] = pallet['type']

        yield env.timeout(cycle_time)

        if random.random() < failure_rate:
            yield rework_queue.put(pallet)
        else:
            yield out_queue.put(pallet)

# 返工站
def rework_station(env, in_queue, out_queue, mean_time):
    while True:
        # 检查队列中是否有零件等待加工
        if len(in_queue.items) == 0:
            # 无零件则等待一段时间
            yield env.timeout(0.1)
            continue
            
        pallet = yield in_queue.get()
        rework_time = random.expovariate(1.0 / mean_time)
        yield env.timeout(rework_time)
        yield out_queue.put(pallet)

# 生产投入过程 - 新增的函数
def part_generator(env, out_queue, tv_schedule):
    """根据生产计划循环投入不同类型的零件"""
    schedule_index = 0
    while True:
        current_type, batch_size = tv_schedule[schedule_index]
        # 将当前批次的零件投入首站队列
        for _ in range(batch_size):
            yield out_queue.put({'type': current_type})
            # 引入随机间隔时间，模拟现实中的投料不均匀性
            yield env.timeout(random.uniform(0.8, 1.2))
        
        # 循环使用生产计划
        schedule_index = (schedule_index + 1) % len(tv_schedule)

def run_experiments(runs=50, sim_days=1):
    daily_throughput = []

    for run in range(runs):
        
        env = simpy.Environment()
        maintenance = Maintenance(env)
        repair_log = []
        
        # 创建有限容量的队列
        queues = {f"q{i}": simpy.Store(env, capacity=5) for i in range(1, 8)}
        rework_q = simpy.Store(env)
        done_q = simpy.Store(env)

        current_type = ['Small']  # 初始产品类型
        
        # 启动零件生成器
        env.process(part_generator(env, queues['q1'], tv_schedule))
        
        # 启动各工序
        env.process(operation(env, 'OP10', queues['q1'], queues['q2'], (1.9, 0.19), tv_type='type', current_type=current_type))
        env.process(operation(env, 'OP20', queues['q2'], queues['q3'], 2.1,
                        setup_dist=lambda: random.normalvariate(5.0, 0.5),
                        breakdown=True, mttf=300,
                        repair_dist=lambda: random.triangular(5, 25, 60),
                        maintenance=maintenance, current_type=current_type, repair_log=repair_log))

        env.process(operation(env, 'OP30', queues['q3'], queues['q4'], 2.0,
                              setup_dist=lambda: random.normalvariate(5.0, 0.5),
                              breakdown=True, mttf=450,
                              repair_dist=lambda: np.random.gamma(3, 35/3),
                              maintenance=maintenance, current_type=current_type, repair_log=repair_log))
        for i in range(5):
            env.process(operation(env, f'OP40{i}', queues['q4'], queues['q5'], 2.0))  # OP40 五个工位
        dem = 98
        env.process(test_station(env, queues['q5'], queues['q6'], rework_q, 0.05, 1.5,
                 setup_dist=lambda: random.normalvariate(3.0, 0.3),
                 repair_dist=lambda: np.random.choice(
                 [5, 15, 25, 35, 45, 55, 65],
                 p=np.array([10, 25, 20, 7, 5, 17, 14])/dem  # 修正概率使其总和为1
                 ),
                 maintenance=maintenance, current_type=current_type, repair_log=repair_log))
        env.process(rework_station(env, rework_q, queues['q6'], mean_time=35))
        
        env.process(operation(env, 'OP50', queues['q6'], queues['q7'], 2.1,
                              setup_dist=lambda: random.normalvariate(5.0, 0.5),
                              breakdown=True, mttf=370,
                              repair_dist=lambda: random.triangular(10, 30, 80),
                              maintenance=maintenance, current_type=current_type, repair_log=repair_log))
        
        env.process(operation(env, 'OP60', queues['q7'], done_q, (1.9, 0.19)))

        # 模拟运行
        sim_time = sim_days * 21 * 60  # 转换为分钟
        env.run(until=sim_time)
        
        # 记录完成数量
        daily_throughput.append(len(done_q.items))

    # 输出统计
    mean_throughput = np.mean(daily_throughput)
    std_throughput = np.std(daily_throughput)
    min_throughput = np.min(daily_throughput)
    max_throughput = np.max(daily_throughput)

    print("\n模拟结果（{}次运行，每次{}天）".format(runs, sim_days))
    print(f"平均每日吞吐量: {mean_throughput:.2f}")
    print(f"标准差: {std_throughput:.2f}")
    print(f"最大值: {max_throughput}")
    print(f"最小值: {min_throughput}")

    # 可视化直方图
    # plt.figure(figsize=(10, 6))
    # plt.hist(daily_throughput, bins=10, alpha=0.7, label="模拟吞吐量")
    # plt.xlabel("每日吞吐量")
    # plt.ylabel("频率")
    # plt.title("每日吞吐量分布（模拟）")
    # plt.legend()
    # plt.grid(True)
    # plt.show()

    return daily_throughput

sim_data = run_experiments(runs=50, sim_days=1)

# 历史数据
historic_data = [
    432, 411, 447, 447, 389, 396, 453, 356, 407, 392,
    333, 428, 387, 462, 424, 431, 459, 420, 439, 433,
    396, 386, 433, 485, 485, 435, 395, 458, 384, 380,
    385, 402, 427, 437, 442, 472, 433, 489, 394, 421,
    390, 381, 416, 401, 393, 449, 409, 398, 397, 351
]

# 直方图对比
plt.figure(figsize=(12, 7))
bins = np.linspace(min(min(sim_data), min(historic_data)), max(max(sim_data), max(historic_data)), 15)
plt.hist(sim_data, bins=bins, alpha=0.7, label="模拟吞吐量", color='blue')
plt.hist(historic_data, bins=bins, alpha=0.7, label="历史吞吐量", color='green')
plt.xlabel("每日吞吐量")
plt.ylabel("频率")
plt.title("模拟与历史数据吞吐量对比")
plt.legend()
plt.grid(True)
plt.show()

# 置信区间
diff = np.array(sim_data) - np.array(historic_data)
conf_int = stats.t.interval(0.95, len(diff)-1, loc=np.mean(diff), scale=stats.sem(diff))
print(f"\n95% 置信区间（模型与真实平均每日吞吐量差值）: {conf_int}")

# P-P图
sim_sorted = np.sort(sim_data)
hist_sorted = np.sort(historic_data)
plt.plot(sim_sorted, hist_sorted, marker='o', linestyle='')
plt.plot([min(sim_sorted), max(sim_sorted)], [min(sim_sorted), max(sim_sorted)], 'r--')
plt.xlabel("模拟值（P）")
plt.ylabel("历史值（P）")
plt.title("P-P概率图")
plt.grid(True)
plt.show()
