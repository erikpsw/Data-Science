from numpy.random import *
from simpy import *
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# 设置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用黑体显示中文
matplotlib.rcParams['axes.unicode_minus'] = False   # 正常显示负号

# 控制是否打印详细仿真过程
PRINT_SIMULATION_DETAILS = False

def rng(dis,param):
    """random number generator"""
    def generate():
        return dis(lam=param,size=1)[0]
    return generate

def erlang(k):
    """由k个指数分布拟合"""
    def exp2erlang(lam,size):
        res=[]
        for n in range(size):
            k_poisson= exponential(lam/k,size=k)
            total=0
            for x in k_poisson:
                total = total + x
            res.append(total)
        return res
    return exp2erlang

x=rng(erlang(10),10)
total=0
for i in range(10000):
    total= total+x()
print(total/10000)

def plot_restaurant_analysis(stats, table_config, title_prefix=""):
    """绘制餐厅分析图表"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'{title_prefix}餐厅排队服务分析', fontsize=16, fontweight='bold')
    
    # 1. 各桌位类型服务人数对比
    table_types = list(table_config.keys())
    served_counts = [stats[t]['SUCC'] for t in table_types]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    bars1 = ax1.bar(table_types, served_counts, color=colors, alpha=0.8)
    ax1.set_title(f'{title_prefix}各桌位类型服务人数', fontweight='bold')
    ax1.set_ylabel('服务组数')
    ax1.set_xlabel('桌位类型')
    
    # 在柱子上添加数值标签
    for bar, count in zip(bars1, served_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{count}组', ha='center', va='bottom', fontweight='bold')
    
    # 增加y轴上方空间
    ax1.set_ylim(0, max(served_counts) * 1.15)
    
    # 2. 平均等待时间对比
    avg_wait_times = []
    for t in table_types:
        if stats[t]['SUCC'] > 0:
            avg_wait_times.append(stats[t]['WAIT']/stats[t]['SUCC'])
        else:
            avg_wait_times.append(0)
    
    bars2 = ax2.bar(table_types, avg_wait_times, color=colors, alpha=0.8)
    ax2.set_title(f'{title_prefix}各桌位类型平均等待时间', fontweight='bold')
    ax2.set_ylabel('等待时间 (分钟)')
    ax2.set_xlabel('桌位类型')
    
    for bar, time in zip(bars2, avg_wait_times):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                f'{time:.1f}分钟', ha='center', va='bottom', fontweight='bold')
    
    # 增加y轴上方空间
    ax2.set_ylim(0, max(avg_wait_times) * 1.15 if avg_wait_times else 1)
    
    # 3. 桌位利用率分析
    utilization_rates = []
    for t in table_types:
        seats, count = table_config[t]
        if stats[t]['SUCC'] > 0:
            avg_stay = stats[t]['STAY']/stats[t]['SUCC']
            # 简化的利用率计算
            utilization = (stats[t]['SUCC'] * avg_stay) / (count * 500) * 100  # 假设总营业时间500分钟
            utilization_rates.append(min(utilization, 100))
        else:
            utilization_rates.append(0)
    
    bars3 = ax3.bar(table_types, utilization_rates, color=colors, alpha=0.8)
    ax3.set_title(f'{title_prefix}桌位利用率', fontweight='bold')
    ax3.set_ylabel('利用率 (%)')
    ax3.set_xlabel('桌位类型')
    ax3.set_ylim(0, 110)  # 增加上方空间
    
    for bar, rate in zip(bars3, utilization_rates):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 4. 桌位配置饼图
    table_counts = [table_config[t][1] for t in table_types]
    table_labels = [f'{t}类桌位\n({table_config[t][0]}人桌)\n{table_config[t][1]}张' 
                   for t in table_types]
    
    wedges, texts, autotexts = ax4.pie(table_counts, labels=table_labels, colors=colors, 
                                      autopct='%1.1f%%', startangle=90)
    ax4.set_title(f'{title_prefix}餐厅桌位配置分布', fontweight='bold')
    
    plt.subplots_adjust(hspace=0.3, wspace=0.3)  # 增加子图间距
    plt.tight_layout()
    plt.show()

def restaurantSample(X, Y, table_config, A, B, party_size_dist, config_name=""):
    """
    餐厅排队服务例子

    情景:
      餐厅有不同座位配置，客户根据聚餐人数选择相应桌位排队
        X: 时间间隔分布
        Y: 用餐时间分布
        table_config: 餐桌配置 {'A': (2, 5), 'B': (3, 3), 'C': (4, 4)} 
                     格式为 {桌位类型: (座位数, 桌子数量)}
        A: 客户耐心时间分布
        B: 客户总数
        party_size_dist: 聚餐人数分布函数
        config_name: 配置名称
    """
    seed(2)
    
    # 统计变量，按桌位类型分类
    stats = {}
    for table_type in table_config:
        stats[table_type] = {
            'SUCC': 0, 'WAIT': 0, 'STAY': 0, 'QUEUE_NUM': 0
        }
    
    def get_table_type(party_size):
        """根据聚餐人数确定桌位类型"""
        if party_size <= 2:
            return 'A'
        elif party_size <= 3:
            return 'B'
        else:
            return 'C'
    
    def source(env, number, interval, counters):
        """生成客户"""
        for i in range(number):
            party_size = max(1, int(party_size_dist()))
            table_type = get_table_type(party_size)
            stats[table_type]['QUEUE_NUM'] += 1
            queue_num = f"{table_type}{stats[table_type]['QUEUE_NUM']:02d}"
            
            c = customer(env, queue_num, counters[table_type], 
                        time_in_restaurant=Y(), party_size=party_size, table_type=table_type)
            env.process(c)
            yield env.timeout(interval)

    def customer(env, name, counter, time_in_restaurant, party_size, table_type):
        """顾客服务与离开仿真"""
        arrive = env.now
        if PRINT_SIMULATION_DETAILS:
            print(f'{arrive:7.4f} {name}: {party_size}人聚餐到达，排队等{table_config[table_type][0]}人桌')
        
        with counter.request() as req:
            patience = A()
            results = yield req | env.timeout(patience)
            wait = env.now - arrive
        
            if req in results:
                # 成功入座
                stats[table_type]['WAIT'] += wait
                stats[table_type]['STAY'] += time_in_restaurant
                if PRINT_SIMULATION_DETAILS:
                    print(f'{env.now:7.4f} {name}: 等待{wait:6.3f}分钟后入座')
                yield env.timeout(time_in_restaurant)
                stats[table_type]['SUCC'] += 1
                if PRINT_SIMULATION_DETAILS:
                    print(f'{env.now:7.4f} {name}: 用餐完成，离开餐厅')
            else:
                # 失去耐心离开
                if PRINT_SIMULATION_DETAILS:
                    print(f'{env.now:7.4f} {name}: 等待{wait:6.3f}分钟后离开')

    # 初始化环境
    print('餐厅排队问题仿真')
    print('桌位配置:')
    for table_type, (seats, count) in table_config.items():
        print(f'  {table_type}类桌位: {seats}人桌 x {count}张')
    
    env = Environment()

    # 为每种桌位类型创建资源
    counters = {}
    for table_type, (seats, count) in table_config.items():
        counters[table_type] = Resource(env, capacity=count)
    
    env.process(source(env, B, X(), counters))
    env.run()
    
    # 输出统计结果
    print("\n=== 餐厅服务统计 ===")
    total_served = sum(stats[t]['SUCC'] for t in stats) 
    total_lost = B - total_served
    
    for table_type in table_config:
        seats, count = table_config[table_type]
        s = stats[table_type]
        if s['SUCC'] > 0:
            print(f"\n{table_type}类桌位({seats}人桌):")
            print(f"  总服务人数：{s['SUCC']}组客人")
            print(f"  平均等待时间：{s['WAIT']/s['SUCC']:.2f}分钟")
            print(f"  平均用餐时间：{s['STAY']/s['SUCC']:.2f}分钟")
    
    print(f"\n餐厅总体情况:")
    print(f"总服务组数：{total_served}组")
    print(f"总计失去：{total_lost}组客人")
    print(f"流失率：{total_lost/B*100:.1f}%")
    
    # 添加可视化
    title_prefix = f"{config_name}-" if config_name else "基础配置-"
    plot_restaurant_analysis(stats, table_config, title_prefix)
    
    return stats

# 餐厅配置和测试
table_config = {
    'A': (2, 5),  # 2人桌，5张
    'B': (3, 3),  # 3人桌，3张  
    'C': (4, 4)   # 4人桌，4张
}

# 时间间隔分布（客人到达间隔）
X = rng(erlang(3), 5)
# 用餐时间分布
Y = rng(erlang(3), 60)
# 耐心时间分布（愿意等待的时间）
A = rng(erlang(3), 30)
# 聚餐人数分布
def party_size_distribution():
    # 简单分布：1-2人(40%), 3人(30%), 4-6人(30%)
    r = random()
    if r < 0.4:
        return choice([1, 2])
    elif r < 0.7:
        return 3
    else:
        return choice([4, 5, 6])

normal_stats = restaurantSample(X, Y, table_config, A, 200, party_size_distribution, "原始配置")

print("\n" + "="*50)

# 测试不同配置的餐厅
def test_different_configurations():
    """测试不同的餐厅配置"""
    
    # 配置1：原始配置
    config1 = {
        'A': (2, 5),  # 2人桌，5张
        'B': (3, 3),  # 3人桌，3张  
        'C': (4, 4)   # 4人桌，4张
    }
    
    # 配置2：增加小桌位
    config2 = {
        'A': (2, 8),  # 2人桌，8张
        'B': (3, 3),  # 3人桌，3张  
        'C': (4, 3)   # 4人桌，3张
    }
    
    # 配置3：平衡配置
    config3 = {
        'A': (2, 6),  # 2人桌，6张
        'B': (3, 5),  # 3人桌，5张  
        'C': (4, 3)   # 4人桌，3张
    }
    
    # 配置4：大桌位为主
    config4 = {
        'A': (2, 3),  # 2人桌，3张
        'B': (3, 4),  # 3人桌，4张  
        'C': (4, 6)   # 4人桌，6张
    }
    
    configs = [
        ("原始配置", config1),
        ("小桌位优先", config2),
        ("均衡配置", config3),
        ("大桌位为主", config4)
    ]
    
    results = []
    
    for config_name, config in configs:
        print(f"\n{'='*60}")
        print(f"测试配置: {config_name}")
        print(f"{'='*60}")
        
        # 将配置名称传递给restaurantSample函数
        stats = restaurantSample(X, Y, config, A, 200, party_size_distribution, config_name)
        
        # 计算总体指标
        total_served = sum(stats[t]['SUCC'] for t in stats)
        total_lost = 200 - total_served
        loss_rate = total_lost / 200 * 100
        
        # 计算总桌数和平均利用率
        total_tables = sum(config[t][1] for t in config)
        avg_utilization = 0
        table_count = 0
        
        for t in config:
            seats, count = config[t]
            if stats[t]['SUCC'] > 0:
                avg_stay = stats[t]['STAY'] / stats[t]['SUCC']
                utilization = (stats[t]['SUCC'] * avg_stay) / (count * 500) * 100
                avg_utilization += min(utilization, 100) * count
                table_count += count
        
        if table_count > 0:
            avg_utilization = avg_utilization / table_count
        
        # 计算加权平均等待时间
        total_wait = sum(stats[t]['WAIT'] for t in stats)
        avg_wait = total_wait / total_served if total_served > 0 else 0
        
        results.append({
            'name': config_name,
            'config': config,
            'served': total_served,
            'lost': total_lost,
            'loss_rate': loss_rate,
            'total_tables': total_tables,
            'avg_utilization': avg_utilization,
            'avg_wait': avg_wait,
            'stats': stats
        })
    
    # 输出对比总结
    print(f"\n{'='*80}")
    print("配置对比总结")
    print(f"{'='*80}")
    print(f"{'配置名称':<15} {'总桌数':<8} {'服务组数':<10} {'流失率':<10} {'平均等待':<12} {'平均利用率':<10}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['name']:<15} {result['total_tables']:<8} {result['served']:<10} "
              f"{result['loss_rate']:<9.1f}% {result['avg_wait']:<11.1f}分 {result['avg_utilization']:<9.1f}%")
    
    # 找出最佳配置
    best_by_served = max(results, key=lambda x: x['served'])
    best_by_loss_rate = min(results, key=lambda x: x['loss_rate'])
    best_by_wait_time = min(results, key=lambda x: x['avg_wait'] if x['avg_wait'] > 0 else float('inf'))
    
    print(f"\n{'='*60}")
    print("最佳配置分析:")
    print(f"服务组数最多: {best_by_served['name']} ({best_by_served['served']}组)")
    print(f"流失率最低: {best_by_loss_rate['name']} ({best_by_loss_rate['loss_rate']:.1f}%)")
    print(f"等待时间最短: {best_by_wait_time['name']} ({best_by_wait_time['avg_wait']:.1f}分钟)")
    
    return results

# 运行配置测试
test_results = test_different_configurations()

# 可视化配置对比
def plot_configuration_comparison(results):
    """绘制不同配置的对比图"""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('不同餐厅配置对比分析', fontsize=16, fontweight='bold')
    
    config_names = [r['name'] for r in results]
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
    
    # 1. 服务组数对比
    served_counts = [r['served'] for r in results]
    bars1 = ax1.bar(config_names, served_counts, color=colors, alpha=0.8)
    ax1.set_title('各配置服务组数对比', fontweight='bold')
    ax1.set_ylabel('服务组数')
    ax1.tick_params(axis='x', rotation=45)
    ax1.set_ylim(0, max(served_counts) * 1.15)  # 增加15%的上方空间
    
    for bar, count in zip(bars1, served_counts):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{count}', ha='center', va='bottom', fontweight='bold')
    
    # 2. 流失率对比
    loss_rates = [r['loss_rate'] for r in results]
    bars2 = ax2.bar(config_names, loss_rates, color=colors, alpha=0.8)
    ax2.set_title('各配置流失率对比', fontweight='bold')
    ax2.set_ylabel('流失率 (%)')
    ax2.tick_params(axis='x', rotation=45)
    ax2.set_ylim(0, max(loss_rates) * 1.15)  # 增加15%的上方空间
    
    for bar, rate in zip(bars2, loss_rates):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 3. 平均等待时间对比
    wait_times = [r['avg_wait'] for r in results]
    bars3 = ax3.bar(config_names, wait_times, color=colors, alpha=0.8)
    ax3.set_title('各配置平均等待时间对比', fontweight='bold')
    ax3.set_ylabel('等待时间 (分钟)')
    ax3.tick_params(axis='x', rotation=45)
    ax3.set_ylim(0, max(wait_times) * 1.15)  # 增加15%的上方空间
    
    for bar, time in zip(bars3, wait_times):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.3,
                f'{time:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. 桌位利用率对比
    utilizations = [r['avg_utilization'] for r in results]
    bars4 = ax4.bar(config_names, utilizations, color=colors, alpha=0.8)
    ax4.set_title('各配置平均桌位利用率对比', fontweight='bold')
    ax4.set_ylabel('利用率 (%)')
    ax4.tick_params(axis='x', rotation=45)
    ax4.set_ylim(0, 110)  # 利用率图表留出10%的上方空间
    
    for bar, util in zip(bars4, utilizations):
        height = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{util:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()

# 绘制配置对比图
plot_configuration_comparison(test_results)