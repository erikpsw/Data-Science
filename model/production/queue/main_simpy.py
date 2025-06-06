import simpy
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pandas as pd
import random
from dataclasses import dataclass
from typing import List, Dict, Tuple
import math
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

@dataclass
class Customer:
    """顾客类"""
    id: int
    arrival_time: float
    service_time: float
    start_service_time: float = 0
    finish_time: float = 0
    
    @property
    def wait_time(self) -> float:
        return self.start_service_time - self.arrival_time
    
    @property
    def system_time(self) -> float:
        return self.finish_time - self.arrival_time

@dataclass
class PerformanceMetrics:
    """性能指标类"""
    avg_wait_time: float
    avg_service_time: float
    avg_system_time: float
    avg_queue_length: float
    avg_system_length: float
    utilization: float
    throughput: float
    total_customers: int = 0
    served_customers: int = 0
    wait_probability: float = 0

class CafeteriaQueueSimPy:
    """基于SimPy的食堂排队系统仿真类"""
    
    def __init__(self, num_servers: int, arrival_rate: float, service_rate: float):
        self.num_servers = num_servers
        self.arrival_rate = arrival_rate  # 人/分钟
        self.service_rate = service_rate  # 人/分钟/台
        self.reset()
    
    def reset(self):
        """重置仿真状态"""
        self.completed_customers = []
        self.total_customers = 0
        self.queue_length_history = []
        self.system_length_history = []
        self.server_busy_time = [0] * self.num_servers
    
    def generate_interarrival_time(self) -> float:
        """生成顾客间到达时间（指数分布）"""
        return np.random.exponential(1 / self.arrival_rate)
    
    def generate_service_time(self) -> float:
        """生成服务时间（指数分布）"""
        return np.random.exponential(1 / self.service_rate)
    
    def customer_arrival_process(self, env: simpy.Environment, servers: simpy.Resource):
        """顾客到达过程"""
        customer_id = 0
        while True:
            # 等待下一个顾客到达
            yield env.timeout(self.generate_interarrival_time())
            
            customer_id += 1
            self.total_customers += 1
            
            # 创建顾客并启动服务过程
            customer = Customer(
                id=customer_id,
                arrival_time=env.now,
                service_time=self.generate_service_time()
            )
            
            # 启动顾客服务过程
            env.process(self.customer_service_process(env, customer, servers))
    
    def customer_service_process(self, env: simpy.Environment, customer: Customer, servers: simpy.Resource):
        """顾客服务过程"""
        # 请求服务台
        request = servers.request()
        yield request
        
        # 开始服务
        customer.start_service_time = env.now
        
        # 服务时间
        yield env.timeout(customer.service_time)
        
        # 服务完成
        customer.finish_time = env.now
        self.completed_customers.append(customer)
        
        # 释放服务台
        servers.release(request)
    
    def monitor_process(self, env: simpy.Environment, servers: simpy.Resource):
        """监控系统状态过程"""
        while True:
            # 记录队列长度和系统长度
            queue_length = len(servers.queue)
            system_length = len(servers.queue) + servers.count
            
            self.queue_length_history.append((env.now, queue_length))
            self.system_length_history.append((env.now, system_length))
            
            # 每分钟记录一次
            yield env.timeout(1.0)
    
    def run_simulation(self, simulation_time: float = 480) -> PerformanceMetrics:
        """运行SimPy仿真"""
        self.reset()
        
        # 创建仿真环境
        env = simpy.Environment()
        
        # 创建服务台资源
        servers = simpy.Resource(env, capacity=self.num_servers)
        
        # 启动顾客到达过程
        env.process(self.customer_arrival_process(env, servers))
        
        # 启动监控过程
        env.process(self.monitor_process(env, servers))
        
        # 运行仿真
        env.run(until=simulation_time)
        
        return self._calculate_metrics(simulation_time)
    
    def _calculate_metrics(self, simulation_time: float) -> PerformanceMetrics:
        """计算性能指标"""
        if not self.completed_customers:
            return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0)
        
        wait_times = [c.wait_time for c in self.completed_customers]
        service_times = [c.service_time for c in self.completed_customers]
        system_times = [c.system_time for c in self.completed_customers]
        
        avg_wait_time = np.mean(wait_times)
        avg_service_time = np.mean(service_times)
        avg_system_time = np.mean(system_times)
        
        # 计算平均队列长度和系统长度（时间加权平均）
        if self.queue_length_history:
            avg_queue_length = self._calculate_time_weighted_average(self.queue_length_history, simulation_time)
            avg_system_length = self._calculate_time_weighted_average(self.system_length_history, simulation_time)
        else:
            avg_queue_length = 0
            avg_system_length = 0
        
        # 计算利用率
        total_service_time = sum(service_times)
        utilization = (total_service_time / self.num_servers) / simulation_time * 100
        
        # 计算吞吐量
        throughput = len(self.completed_customers) / simulation_time
        
        # 计算等待概率
        wait_probability = sum(1 for wt in wait_times if wt > 0.01) / len(wait_times) * 100 if wait_times else 0
        
        return PerformanceMetrics(
            avg_wait_time=avg_wait_time,
            avg_service_time=avg_service_time,
            avg_system_time=avg_system_time,
            avg_queue_length=avg_queue_length,
            avg_system_length=avg_system_length,
            utilization=utilization,
            throughput=throughput,
            total_customers=self.total_customers,
            served_customers=len(self.completed_customers),
            wait_probability=wait_probability
        )
    
    def _calculate_time_weighted_average(self, time_series: List[Tuple[float, float]], total_time: float) -> float:
        """计算时间加权平均值"""
        if not time_series:
            return 0
        
        weighted_sum = 0
        prev_time = 0
        prev_value = 0
        
        for time, value in time_series:
            if time > prev_time:
                weighted_sum += prev_value * (time - prev_time)
            prev_time = time
            prev_value = value
        
        # 处理最后一段时间
        if prev_time < total_time:
            weighted_sum += prev_value * (total_time - prev_time)
        
        return weighted_sum / total_time if total_time > 0 else 0

class QueueingTheoryAnalysis:
    """排队论分析类"""
    
    @staticmethod
    def factorial(n: int) -> float:
        """计算阶乘"""
        if n <= 1:
            return 1.0
        return math.factorial(n)
    
    @staticmethod
    def analyze_mm_c_queue(arrival_rate: float, service_rate: float, num_servers: int) -> PerformanceMetrics:
        """M/M/c排队系统分析"""
        lambda_rate = arrival_rate
        mu = service_rate
        c = num_servers
        
        # 计算流量强度
        rho = lambda_rate / mu  # 每个服务台的流量强度
        total_rho = lambda_rate / (c * mu)  # 系统流量强度
        
        if total_rho >= 1:
            print(f"警告: 系统不稳定! 流量强度 = {total_rho:.3f} ≥ 1")
            return PerformanceMetrics(float('inf'), 1/mu, float('inf'), 
                                    float('inf'), float('inf'), 
                                    total_rho * 100, lambda_rate)
        
        # 计算P0 (系统空闲概率)
        sum1 = sum(rho**n / QueueingTheoryAnalysis.factorial(n) for n in range(c))
        sum2 = (rho**c / QueueingTheoryAnalysis.factorial(c)) * (1 / (1 - total_rho))
        P0 = 1 / (sum1 + sum2)
        
        # 计算Pc (所有服务台都忙的概率)
        Pc = (rho**c / QueueingTheoryAnalysis.factorial(c)) * P0 / (1 - total_rho)
        
        # 计算性能指标
        Lq = Pc * total_rho / (1 - total_rho)  # 平均排队长度
        L = Lq + rho  # 平均系统长度
        Wq = Lq / lambda_rate  # 平均等待时间
        W = Wq + 1/mu  # 平均系统时间
        utilization = total_rho * 100  # 利用率
        
        return PerformanceMetrics(
            avg_wait_time=Wq,
            avg_service_time=1/mu,
            avg_system_time=W,
            avg_queue_length=Lq,
            avg_system_length=L,
            utilization=utilization,
            throughput=lambda_rate,
            wait_probability=Pc * 100
        )

class QueueAnalyzer:
    """排队系统分析器"""
    
    def __init__(self):
        self.results = {}
    
    def compare_methods(self, scenarios: List[Dict]) -> pd.DataFrame:
        """比较不同方法的结果"""
        results = []
        
        for scenario in scenarios:
            name = scenario['name']
            servers = scenario['servers']
            arrival_rate = scenario['arrival_rate']
            service_rate = scenario['service_rate']
            
            print(f"\n分析场景: {name}")
            print(f"参数: {servers}个服务台, 到达率{arrival_rate}人/分钟, 服务率{service_rate}人/分钟/台")
            
            # 排队论分析
            analytical = QueueingTheoryAnalysis.analyze_mm_c_queue(
                arrival_rate, service_rate, servers
            )
            
            # SimPy仿真分析
            sim = CafeteriaQueueSimPy(servers, arrival_rate, service_rate)
            
            # 多次仿真求平均
            sim_results = []
            for _ in range(10):
                sim_result = sim.run_simulation(simulation_time=480)
                sim_results.append(sim_result)
            
            # 计算仿真平均值
            wait_times = [r.avg_wait_time for r in sim_results]
            system_times = [r.avg_system_time for r in sim_results]
            queue_lengths = [r.avg_queue_length for r in sim_results]
            utilizations = [r.utilization for r in sim_results]
            
            sim_avg = PerformanceMetrics(
                avg_wait_time=np.mean(wait_times),
                avg_service_time=np.mean([r.avg_service_time for r in sim_results]),
                avg_system_time=np.mean(system_times),
                avg_queue_length=np.mean(queue_lengths),
                avg_system_length=np.mean([r.avg_system_length for r in sim_results]),
                utilization=np.mean(utilizations),
                throughput=np.mean([r.throughput for r in sim_results]),
                wait_probability=np.mean([r.wait_probability for r in sim_results])
            )
            
            # 计算误差
            wait_error = abs(analytical.avg_wait_time - sim_avg.avg_wait_time) / analytical.avg_wait_time * 100 if analytical.avg_wait_time > 0 else 0
            system_error = abs(analytical.avg_system_time - sim_avg.avg_system_time) / analytical.avg_system_time * 100
            queue_error = abs(analytical.avg_queue_length - sim_avg.avg_queue_length) / analytical.avg_queue_length * 100 if analytical.avg_queue_length > 0 else 0
            
            results.append({
                '场景': name,
                '服务台数': servers,
                '到达率': arrival_rate,
                '服务率': service_rate,
                '理论等待时间': analytical.avg_wait_time,
                'SimPy等待时间': sim_avg.avg_wait_time,
                '等待时间误差(%)': wait_error,
                '理论系统时间': analytical.avg_system_time,
                'SimPy系统时间': sim_avg.avg_system_time,
                '系统时间误差(%)': system_error,
                '理论队列长度': analytical.avg_queue_length,
                'SimPy队列长度': sim_avg.avg_queue_length,
                '队列长度误差(%)': queue_error,
                '理论利用率': analytical.utilization,
                'SimPy利用率': sim_avg.utilization,
                '理论等待概率(%)': analytical.wait_probability,
                'SimPy等待概率(%)': sim_avg.wait_probability
            })
            
            print(f"等待时间: 理论{analytical.avg_wait_time:.2f} vs SimPy{sim_avg.avg_wait_time:.2f} (误差{wait_error:.1f}%)")
            print(f"队列长度: 理论{analytical.avg_queue_length:.2f} vs SimPy{sim_avg.avg_queue_length:.2f} (误差{queue_error:.1f}%)")
        
        return pd.DataFrame(results)
    
    def sensitivity_analysis(self, base_scenario: Dict) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """敏感性分析"""
        print("\n=== 敏感性分析 ===")
        
        # 分析服务台数量的影响
        servers_range = range(1, 8)
        server_results = []
        
        for servers in servers_range:
            if base_scenario['arrival_rate'] / (servers * base_scenario['service_rate']) >= 1:
                continue  # 跳过不稳定的系统
                
            analytical = QueueingTheoryAnalysis.analyze_mm_c_queue(
                base_scenario['arrival_rate'], base_scenario['service_rate'], servers
            )
            
            server_results.append({
                '服务台数': servers,
                '平均等待时间': analytical.avg_wait_time,
                '平均队列长度': analytical.avg_queue_length,
                '利用率': analytical.utilization,
                '等待概率': analytical.wait_probability
            })
        
        # 分析到达率的影响
        arrival_rates = np.arange(0.5, 4.0, 0.2)
        arrival_results = []
        
        for rate in arrival_rates:
            if rate / (base_scenario['servers'] * base_scenario['service_rate']) >= 1:
                continue
                
            analytical = QueueingTheoryAnalysis.analyze_mm_c_queue(
                rate, base_scenario['service_rate'], base_scenario['servers']
            )
            
            arrival_results.append({
                '到达率': rate,
                '平均等待时间': analytical.avg_wait_time,
                '平均队列长度': analytical.avg_queue_length,
                '利用率': analytical.utilization,
                '等待概率': analytical.wait_probability
            })
        
        return pd.DataFrame(server_results), pd.DataFrame(arrival_results)
    
    def plot_comparison(self, df: pd.DataFrame):
        """绘制对比图表"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('排队论分析 vs SimPy仿真分析对比', fontsize=16, fontweight='bold')
        
        # 等待时间对比
        axes[0,0].scatter(df['理论等待时间'], df['SimPy等待时间'], s=100, alpha=0.7)
        axes[0,0].plot([0, df['理论等待时间'].max()], [0, df['理论等待时间'].max()], 'r--', alpha=0.5)
        axes[0,0].set_xlabel('理论等待时间 (分钟)')
        axes[0,0].set_ylabel('SimPy等待时间 (分钟)')
        axes[0,0].set_title('等待时间对比')
        axes[0,0].grid(True, alpha=0.3)
        
        # 队列长度对比
        axes[0,1].scatter(df['理论队列长度'], df['SimPy队列长度'], s=100, alpha=0.7, color='green')
        axes[0,1].plot([0, df['理论队列长度'].max()], [0, df['理论队列长度'].max()], 'r--', alpha=0.5)
        axes[0,1].set_xlabel('理论队列长度 (人)')
        axes[0,1].set_ylabel('SimPy队列长度 (人)')
        axes[0,1].set_title('队列长度对比')
        axes[0,1].grid(True, alpha=0.3)
        
        # 误差分析
        scenarios = df['场景'].tolist()
        wait_errors = df['等待时间误差(%)'].tolist()
        queue_errors = df['队列长度误差(%)'].tolist()
        
        x = np.arange(len(scenarios))
        width = 0.35
        
        axes[1,0].bar(x - width/2, wait_errors, width, label='等待时间误差', alpha=0.8)
        axes[1,0].bar(x + width/2, queue_errors, width, label='队列长度误差', alpha=0.8)
        axes[1,0].set_xlabel('场景')
        axes[1,0].set_ylabel('相对误差 (%)')
        axes[1,0].set_title('理论值与SimPy仿真值误差对比')
        axes[1,0].set_xticks(x)
        axes[1,0].set_xticklabels(scenarios, rotation=45)
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # 利用率对比
        axes[1,1].scatter(df['理论利用率'], df['SimPy利用率'], s=100, alpha=0.7, color='orange')
        axes[1,1].plot([0, 100], [0, 100], 'r--', alpha=0.5)
        axes[1,1].set_xlabel('理论利用率 (%)')
        axes[1,1].set_ylabel('SimPy利用率 (%)')
        axes[1,1].set_title('利用率对比')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_sensitivity(self, server_df: pd.DataFrame, arrival_df: pd.DataFrame):
        """绘制敏感性分析图表"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('系统性能敏感性分析', fontsize=16, fontweight='bold')
        
        # 服务台数量对等待时间的影响
        axes[0,0].plot(server_df['服务台数'], server_df['平均等待时间'], 'bo-', linewidth=2, markersize=8)
        axes[0,0].set_xlabel('服务台数量')
        axes[0,0].set_ylabel('平均等待时间 (分钟)')
        axes[0,0].set_title('服务台数量对等待时间的影响')
        axes[0,0].grid(True, alpha=0.3)
        
        # 服务台数量对利用率的影响
        axes[0,1].plot(server_df['服务台数'], server_df['利用率'], 'ro-', linewidth=2, markersize=8)
        axes[0,1].set_xlabel('服务台数量')
        axes[0,1].set_ylabel('利用率 (%)')
        axes[0,1].set_title('服务台数量对利用率的影响')
        axes[0,1].grid(True, alpha=0.3)
        
        # 到达率对等待时间的影响
        axes[1,0].plot(arrival_df['到达率'], arrival_df['平均等待时间'], 'go-', linewidth=2, markersize=8)
        axes[1,0].set_xlabel('到达率 (人/分钟)')
        axes[1,0].set_ylabel('平均等待时间 (分钟)')
        axes[1,0].set_title('到达率对等待时间的影响')
        axes[1,0].grid(True, alpha=0.3)
        
        # 到达率对队列长度的影响
        axes[1,1].plot(arrival_df['到达率'], arrival_df['平均队列长度'], 'mo-', linewidth=2, markersize=8)
        axes[1,1].set_xlabel('到达率 (人/分钟)')
        axes[1,1].set_ylabel('平均队列长度 (人)')
        axes[1,1].set_title('到达率对队列长度的影响')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def main():
    """主函数"""
    print("=" * 60)
    print("大学食堂排队系统仿真分析 - SimPy版本")
    print("=" * 60)
    
    # 定义分析场景
    scenarios = [
        {'name': '早餐时段', 'servers': 2, 'arrival_rate': 1.5, 'service_rate': 1.0},
        {'name': '午餐高峰', 'servers': 4, 'arrival_rate': 3.5, 'service_rate': 1.2},
        {'name': '晚餐时段', 'servers': 3, 'arrival_rate': 2.8, 'service_rate': 1.1},
        {'name': '夜宵时段', 'servers': 1, 'arrival_rate': 0.8, 'service_rate': 0.9}
    ]
    
    # 创建分析器
    analyzer = QueueAnalyzer()
    
    # 方法对比分析
    print("\n1. 排队论分析 vs SimPy仿真分析对比")
    comparison_df = analyzer.compare_methods(scenarios)
    print("\n详细对比结果:")
    print(comparison_df.round(3))
    
    # 绘制对比图表
    analyzer.plot_comparison(comparison_df)
    
    # 敏感性分析
    base_scenario = {'servers': 3, 'arrival_rate': 2.5, 'service_rate': 1.2}
    server_sensitivity, arrival_sensitivity = analyzer.sensitivity_analysis(base_scenario)
    
    print("\n2. 服务台数量敏感性分析:")
    print(server_sensitivity.round(3))
    
    print("\n3. 到达率敏感性分析:")
    print(arrival_sensitivity.round(3))
    
    # 绘制敏感性分析图表
    analyzer.plot_sensitivity(server_sensitivity, arrival_sensitivity)
    
    # 方法论总结
    print("\n" + "=" * 60)
    print("SimPy仿真方法论总结与建议")
    print("=" * 60)
    
    print("\n【SimPy离散事件仿真】")
    print("优势:")
    print("• 基于进程的建模方式，更直观地描述系统行为")
    print("• 自动处理事件调度和时间管理")
    print("• 资源管理功能强大，支持复杂的资源约束")
    print("• 代码结构清晰，易于扩展和维护")
    print("• 支持实时监控和统计收集")
    
    print("\n应用场景:")
    print("• 复杂排队网络建模")
    print("• 多种资源类型的系统仿真")
    print("• 需要详细观察系统动态行为的场景")
    print("• 大规模仿真实验")
    
    print("\n【与传统仿真方法对比】")
    print("• 代码可读性更高，更易于理解和调试")
    print("• 自动管理事件时间，减少编程错误")
    print("• 内置统计收集机制，方便性能分析")
    print("• 更好的可扩展性，支持复杂系统建模")

if __name__ == "__main__":
    # 设置随机种子以便结果可重现
    np.random.seed(42)
    random.seed(42)
    
    main()
