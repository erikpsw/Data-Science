import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple
plt.rcParams['text.usetex'] = False
plt.rc("font",family='MicroSoft YaHei')

# 产品
class Product:
    def __init__(self):
        self.initial_inventory = 30
        self.holding_cost = 0.5
        self.purchase_cost = 10
        self.min_batch = 10

# 生产中心
class WorkCenter:
    def __init__(self):
        self.setup_time = 5
        self.setup_cost = 10
        self.process_time = 1
        self.available_hours = 30
        self.overtime_cost = 2
        self.lead_time = 1

def calculate_costs(mps_plan): 
    product = Product()
    work_center = WorkCenter()
    
    inventory = product.initial_inventory
    total_cost = 0

    gross_req = [20, 25, 20, 25]
    
    for period in range(4):
        if mps_plan[period] > 0:
            # 准备时间 + 加工时间
            hours_needed = work_center.setup_time + work_center.process_time * mps_plan[period]
            
            overtime = max(0, hours_needed - work_center.available_hours)
            regular_hours = min(hours_needed, work_center.available_hours)
            
            total_cost += work_center.setup_cost
            total_cost += overtime * work_center.overtime_cost
            
        # 计算库存
        if period > 0: 
            # 上一阶段的计划投入量 = 本阶段计划产出量
            inventory += mps_plan[period-1]
        
        # 毛需求多余库存
        if inventory < gross_req[period]:
            # 购买量
            purchase_qty = gross_req[period] - inventory
            total_cost += purchase_qty * product.purchase_cost
            inventory = 0
        else: 
            # 使用库存满足毛需求
            inventory -= gross_req[period]
        print("inventory", inventory)
        total_cost += inventory * product.holding_cost

    return total_cost

def random_search(n_iterations: int = 1000) -> Tuple[List[int], float, List[float], List[float]]:
    best_plan = None
    best_cost = float('inf')
    cost_history = []
    best_cost_history = []
    
    for i in range(n_iterations):
        # 生成随机方案，考虑最小批量约束
        plan = []
        for _ in range(4):
            # 随机生成0或10的倍数
            batch = np.random.randint(0, 6) * 10  # 0到50之间，步长为10
            plan.append(batch)
            
        cost = calculate_costs(plan)
        cost_history.append(cost)
        
        if cost < best_cost:
            best_cost = cost
            best_plan = plan.copy()
        
        best_cost_history.append(best_cost)
            
    return best_plan, best_cost, cost_history, best_cost_history

def plot_search_results(cost_history: List[float], best_cost_history: List[float], 
                       direct_cost: float, fixed_cost: float):
    plt.figure(figsize=(12, 6))
    
    # 左侧子图：所有解
    plt.subplot(1, 2, 1)
    plt.plot(cost_history, 'b.', alpha=0.3, label='随机搜索解')
    plt.axhline(y=direct_cost, color='r', linestyle='--', label='直接批量法')
    plt.axhline(y=fixed_cost, color='g', linestyle='--', label='固定批量法')
    plt.xlabel('迭代次数')
    plt.ylabel('总成本')
    plt.title('随机搜索所有解的分布')
    plt.legend()
    plt.grid(True)

    # 右侧子图：最优解演化
    plt.subplot(1, 2, 2)
    plt.plot(best_cost_history, 'r-', label='当前最优解')
    plt.axhline(y=direct_cost, color='r', linestyle='--', label='直接批量法')
    plt.axhline(y=fixed_cost, color='g', linestyle='--', label='固定批量法')
    plt.xlabel('迭代次数')
    plt.ylabel('最优总成本')
    plt.title('最优解演化过程')
    plt.legend()
    plt.grid(True)
    
    plt.suptitle('MPS成本优化 - 随机搜索', fontsize=12)
    plt.tight_layout()
    plt.show()

def main():
    # 计算直接批量法和固定批量法的成本
    direct_batch = [20, 20, 20, 0]
    direct_cost = calculate_costs(direct_batch)
    print(f"Direct batch method cost: {direct_cost:.2f}")
    
    fixed_batch = [50, 0, 50, 0]
    fixed_cost = calculate_costs(fixed_batch)
    print(f"Fixed batch method cost: {fixed_cost:.2f}")
    
    # 运行随机搜索
    best_plan, best_cost, cost_history, best_cost_history = random_search(10000)
    print(f"\nBest plan found: {best_plan}")
    print(f"Best cost found: {best_cost:.2f}")
    
    # 可视化搜索结果
    plot_search_results(cost_history, best_cost_history, direct_cost, fixed_cost)

if __name__ == "__main__":
    main()
