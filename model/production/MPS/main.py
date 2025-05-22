import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
plt.rcParams['text.usetex'] = False
plt.rc("font",family='MicroSoft YaHei')

class Product:
    def __init__(self, id: int, initial_inventory: int, holding_cost: float, 
                 purchase_cost: float, min_batch: int, routing: List[Dict]):
        self.id = id
        self.initial_inventory = initial_inventory
        self.holding_cost = holding_cost
        self.purchase_cost = purchase_cost
        self.min_batch = min_batch
        self.routing = routing 

class WorkCenter:
    def __init__(self, name: str, available_hours: float, overtime_cost: float):
        self.name = name
        self.available_hours = available_hours
        self.overtime_cost = overtime_cost

def create_products_and_centers():
    work_centers = {
        'A': WorkCenter('A', 112, 2),
        'B': WorkCenter('B', 95, 2)
    }
    
    products = [
        Product(1, 30, 0.5, 20, 5, [
            {'center': 'A', 'setup_time': 5, 'setup_cost': 10, 'process_time': 2},
            {'center': 'B', 'setup_time': 3, 'setup_cost': 6, 'process_time': 3}
        ]),
        Product(2, 30, 0.8, 15, 5, [
            {'center': 'A', 'setup_time': 8, 'setup_cost': 12, 'process_time': 2},
            {'center': 'B', 'setup_time': 5, 'setup_cost': 6, 'process_time': 1}
        ])
    ]
    return products, work_centers

def calculate_costs(mps_plans: List[List[int]]) -> float:
    products, work_centers = create_products_and_centers()
    total_cost = 0
    gross_reqs = [
        [25, 40, 30, 15],  # 产品 1
        [15, 20, 50, 60]   # 产品 2
    ]
    
    center_hours = {center: [0]*4 for center in work_centers}
    
    for p_idx, product in enumerate(products):
        inventory = product.initial_inventory
        
        for period in range(4):
            # 计算工作中心负荷
            if mps_plans[p_idx][period] > 0:
                for op in product.routing:
                    center = op['center']
                    hours = op['setup_time'] + op['process_time'] * mps_plans[p_idx][period]
                    center_hours[center][period] += hours
            
            # 更新库存并计算持有/采购成本
            if period > 0:
                inventory += mps_plans[p_idx][period-1]
                
            if inventory < gross_reqs[p_idx][period]:
                purchase_qty = gross_reqs[p_idx][period] - inventory
                total_cost += purchase_qty * product.purchase_cost
                inventory = 0
            else:
                inventory -= gross_reqs[p_idx][period]
            
            total_cost += inventory * product.holding_cost
    
    for p_idx, product in enumerate(products):
        for period in range(4):
            if mps_plans[p_idx][period] > 0:
                for op in product.routing:
                    total_cost += op['setup_cost']
    
    for center_name, loads in center_hours.items():
        for load in loads:
            overtime = max(0, load - work_centers[center_name].available_hours)
            total_cost += overtime * work_centers[center_name].overtime_cost
    
    return total_cost

def random_search(n_iterations: int = 1000) -> Tuple[List[List[int]], float, List[float], List[float]]:
    best_plans = None
    best_cost = float('inf')
    cost_history = []
    best_cost_history = []
    
    for i in range(n_iterations):
        plans = []
        for _ in range(2):  # 两个产品
            plan = []
            for _ in range(4):
                # 由于最小批量为5 直接生成0-10的整数乘以5
                batch = np.random.randint(0, 11) * 5
                plan.append(batch)
            plans.append(plan)
            
        cost = calculate_costs(plans)
        cost_history.append(cost)
        
        if cost < best_cost:
            best_cost = cost
            best_plans = [plan.copy() for plan in plans]
        
        best_cost_history.append(best_cost)
            
    return best_plans, best_cost, cost_history, best_cost_history

def generate_neighbor(current_plans: List[List[int]]) -> List[List[int]]:
    """Generate a neighboring solution by randomly modifying one batch quantity"""
    neighbor = [plan.copy() for plan in current_plans]
    prod = np.random.randint(0, 2)
    period = np.random.randint(0, 4)
    change = np.random.choice([-5, 5])
    neighbor[prod][period] = max(0, neighbor[prod][period] + change)
    return neighbor

def local_search(n_iterations: int = 1000, restart_threshold: int = 100) -> Tuple[List[List[int]], float, List[float], List[float]]:
    best_plans = None
    best_cost = float('inf')
    cost_history = []
    best_cost_history = []
    
    current_plans = []
    for _ in range(2):
        plan = [np.random.randint(0, 11) * 5 for _ in range(4)]
        current_plans.append(plan)
    
    current_cost = calculate_costs(current_plans)
    restart_count = 0
    
    for i in range(n_iterations):
        neighbor_plans = generate_neighbor(current_plans)
        neighbor_cost = calculate_costs(neighbor_plans)
        
        # 更好
        if neighbor_cost < current_cost:
            current_plans = neighbor_plans
            current_cost = neighbor_cost
            restart_count = 0
        else:
            restart_count += 1
        
        # 最好
        if current_cost < best_cost:
            best_cost = current_cost
            best_plans = [plan.copy() for plan in current_plans]
        
        # 重启
        if restart_count >= restart_threshold:
            current_plans = []
            for _ in range(2):
                plan = [np.random.randint(0, 11) * 5 for _ in range(4)]
                current_plans.append(plan)
            current_cost = calculate_costs(current_plans)
            restart_count = 0
        
        cost_history.append(current_cost)
        best_cost_history.append(best_cost)
    
    return best_plans, best_cost, cost_history, best_cost_history

def calculate_detailed_metrics(mps_plans: List[List[int]]) -> Dict:
    products, work_centers = create_products_and_centers()
    metrics = {
        'gross_reqs': [
            [25, 40, 30, 15],  # 产品1
            [15, 20, 50, 60]   # 产品2
        ],
        'planned_input': mps_plans,
        'planned_output': [[0]*4, [0]*4],  # 两个产品的初始化输出
        'purchase': [[0]*4, [0]*4],
        'inventory': [[0]*4, [0]*4],
        'process_hours': [[0]*4, [0]*4],
        'overtime_hours': {'A': [0]*4, 'B': [0]*4},
        'costs': {
            'holding': [[0]*4, [0]*4],
            'setup': [[0]*4, [0]*4],
            'overtime': {'A': [0]*4, 'B': [0]*4},
            'purchase': [[0]*4, [0]*4]
        }
    }
    
    center_hours = {center: [0]*4 for center in work_centers}
    
    for p_idx, product in enumerate(products):
        inventory = product.initial_inventory
        metrics['inventory'][p_idx][0] = inventory
        
        for period in range(4):
            if mps_plans[p_idx][period] > 0:
                process_hours = 0
                for op in product.routing:
                    center = op['center']
                    hours = op['setup_time'] + op['process_time'] * mps_plans[p_idx][period]
                    center_hours[center][period] += hours
                    process_hours += hours
                    metrics['costs']['setup'][p_idx][period] += op['setup_cost']
                metrics['process_hours'][p_idx][period] = process_hours
            
            if period > 0:
                metrics['planned_output'][p_idx][period] = mps_plans[p_idx][period-1]
                inventory += mps_plans[p_idx][period-1]
            
            if inventory < metrics['gross_reqs'][p_idx][period]:
                purchase_qty = metrics['gross_reqs'][p_idx][period] - inventory
                metrics['purchase'][p_idx][period] = purchase_qty
                metrics['costs']['purchase'][p_idx][period] = purchase_qty * product.purchase_cost
                inventory = 0
            else:
                inventory -= metrics['gross_reqs'][p_idx][period]
            
            metrics['inventory'][p_idx][period] = inventory
            metrics['costs']['holding'][p_idx][period] = inventory * product.holding_cost
    
    for center_name, loads in center_hours.items():
        for period, load in enumerate(loads):
            overtime = max(0, load - work_centers[center_name].available_hours)
            metrics['overtime_hours'][center_name][period] = overtime
            metrics['costs']['overtime'][center_name][period] = overtime * work_centers[center_name].overtime_cost
    
    return metrics

def print_detailed_metrics(metrics: Dict):
    products = ['Product 1', 'Product 2']
    periods = ['Period 1', 'Period 2', 'Period 3', 'Period 4']
    
    print("\n=== MPS Detailed Metrics ===")
    
    for p_idx, prod in enumerate(products):
        print(f"\n{prod}")
        print("-" * 50)
        print("Parameter      | " + " | ".join(f"{p:10}" for p in periods))
        print("-" * 50)
        print(f"Gross Req.    | " + " | ".join(f"{x:10}" for x in metrics['gross_reqs'][p_idx]))
        print(f"Plan Input    | " + " | ".join(f"{x:10}" for x in metrics['planned_input'][p_idx]))
        print(f"Plan Output   | " + " | ".join(f"{x:10}" for x in metrics['planned_output'][p_idx]))
        print(f"Purchase      | " + " | ".join(f"{x:10}" for x in metrics['purchase'][p_idx]))
        print(f"Inventory     | " + " | ".join(f"{x:10}" for x in metrics['inventory'][p_idx]))
        print(f"Process Hours | " + " | ".join(f"{x:10.1f}" for x in metrics['process_hours'][p_idx]))
    
    print("\nWork Center Overtime Hours:")
    for center in ['A', 'B']:
        print(f"Center {center}    | " + " | ".join(f"{x:10.1f}" for x in metrics['overtime_hours'][center]))
    
    print("\nCosts Breakdown:")
    for p_idx, prod in enumerate(products):
        print(f"\n{prod}")
        print(f"Holding Cost  | " + " | ".join(f"{x:10.1f}" for x in metrics['costs']['holding'][p_idx]))
        print(f"Setup Cost    | " + " | ".join(f"{x:10.1f}" for x in metrics['costs']['setup'][p_idx]))
        print(f"Purchase Cost | " + " | ".join(f"{x:10.1f}" for x in metrics['costs']['purchase'][p_idx]))
    
    print("\nOvertime Costs:")
    for center in ['A', 'B']:
        print(f"Center {center}    | " + " | ".join(f"{x:10.1f}" for x in metrics['costs']['overtime'][center]))

def main():
    given_plans = [
        [40, 40, 30, 0],  # 产品 1
        [50, 50, 50, 0]   # 产品 2
    ]
    given_cost = calculate_costs(given_plans)
    print(f"Given MPS total cost: {given_cost:.2f}")
    
    # 运行两种优化方法
    rs_best_plans, rs_best_cost, rs_cost_history, rs_best_cost_history = random_search(100000)
    ls_best_plans, ls_best_cost, ls_cost_history, ls_best_cost_history = local_search(100000)
    
    # 输出结果
    print("\n随机搜索结果:")
    print(f"产品1计划: {rs_best_plans[0]}")
    print(f"产品2计划: {rs_best_plans[1]}")
    print(f"最优成本: {rs_best_cost:.2f}")
    
    print("\n局部搜索结果:")
    print(f"产品1计划: {ls_best_plans[0]}")
    print(f"产品2计划: {ls_best_plans[1]}")
    print(f"最优成本: {ls_best_cost:.2f}")
    
    # 绘制随机搜索结果
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(rs_cost_history, 'b.', alpha=0.3, label='搜索过程')
    plt.plot(rs_best_cost_history, 'r-', label='历史最优')
    plt.axhline(y=given_cost, color='g', linestyle='--', label='初始方案')
    plt.xlabel('迭代次数')
    plt.ylabel('总成本')
    plt.title('随机搜索优化过程')
    plt.legend()
    plt.grid(True)

    # 绘制局部搜索结果
    plt.subplot(1, 2, 2)
    plt.plot(ls_cost_history, 'b.', alpha=0.3, label='搜索过程')
    plt.plot(ls_best_cost_history, 'r-', label='历史最优')
    plt.axhline(y=given_cost, color='g', linestyle='--', label='初始方案')
    plt.xlabel('迭代次数')
    plt.ylabel('总成本')
    plt.title('局部搜索优化过程')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
