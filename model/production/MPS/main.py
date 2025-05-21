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
        self.routing = routing  # List of operations with work center and times

class WorkCenter:
    def __init__(self, name: str, available_hours: float, overtime_cost: float):
        self.name = name
        self.available_hours = available_hours
        self.overtime_cost = overtime_cost

def create_products_and_centers():
    # Work centers
    work_centers = {
        'A': WorkCenter('A', 112, 2),
        'B': WorkCenter('B', 95, 2)
    }
    
    # Products with their routings
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
        [25, 40, 30, 15],  # Product 1
        [15, 20, 50, 60]   # Product 2
    ]
    
    # Calculate work center loads and costs
    center_hours = {center: [0]*4 for center in work_centers}
    
    for p_idx, product in enumerate(products):
        inventory = product.initial_inventory
        
        for period in range(4):
            # Calculate work center loads
            if mps_plans[p_idx][period] > 0:
                for op in product.routing:
                    center = op['center']
                    hours = op['setup_time'] + op['process_time'] * mps_plans[p_idx][period]
                    center_hours[center][period] += hours
            
            # Update inventory and calculate holding/purchase costs
            if period > 0:
                inventory += mps_plans[p_idx][period-1]
                
            if inventory < gross_reqs[p_idx][period]:
                purchase_qty = gross_reqs[p_idx][period] - inventory
                total_cost += purchase_qty * product.purchase_cost
                inventory = 0
            else:
                inventory -= gross_reqs[p_idx][period]
            
            total_cost += inventory * product.holding_cost
    
    # Calculate overtime costs and setup costs
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
        # Generate random plans for both products
        plans = []
        for _ in range(2):  # Two products
            plan = []
            for _ in range(4):
                # Random multiple of 5 between 0 and 50
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

def calculate_detailed_metrics(mps_plans: List[List[int]]) -> Dict:
    products, work_centers = create_products_and_centers()
    metrics = {
        'gross_reqs': [
            [25, 40, 30, 15],  # Product 1
            [15, 20, 50, 60]   # Product 2
        ],
        'planned_input': mps_plans,
        'planned_output': [[0]*4, [0]*4],  # Initialize with zeros for both products
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
            # Calculate work center loads and process hours
            if mps_plans[p_idx][period] > 0:
                process_hours = 0
                for op in product.routing:
                    center = op['center']
                    hours = op['setup_time'] + op['process_time'] * mps_plans[p_idx][period]
                    center_hours[center][period] += hours
                    process_hours += hours
                    metrics['costs']['setup'][p_idx][period] += op['setup_cost']
                metrics['process_hours'][p_idx][period] = process_hours
            
            # Calculate planned output (considering lead time)
            if period > 0:
                metrics['planned_output'][p_idx][period] = mps_plans[p_idx][period-1]
                inventory += mps_plans[p_idx][period-1]
            
            # Calculate purchase and inventory
            if inventory < metrics['gross_reqs'][p_idx][period]:
                purchase_qty = metrics['gross_reqs'][p_idx][period] - inventory
                metrics['purchase'][p_idx][period] = purchase_qty
                metrics['costs']['purchase'][p_idx][period] = purchase_qty * product.purchase_cost
                inventory = 0
            else:
                inventory -= metrics['gross_reqs'][p_idx][period]
            
            metrics['inventory'][p_idx][period] = inventory
            metrics['costs']['holding'][p_idx][period] = inventory * product.holding_cost
    
    # Calculate overtime hours and costs
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
    # Evaluate given MPS
    given_plans = [
        [40, 40, 30, 0],  # Product 1
        [50, 50, 50, 0]   # Product 2
    ]
    given_cost = calculate_costs(given_plans)
    print(f"Given MPS total cost: {given_cost:.2f}")
    
    # Print detailed metrics
    metrics = calculate_detailed_metrics(given_plans)
    print_detailed_metrics(metrics)
    
    # # Run optimization
    # best_plans, best_cost, cost_history, best_cost_history = random_search(10000)
    # print(f"\nBest plans found:")
    # print(f"Product 1: {best_plans[0]}")
    # print(f"Product 2: {best_plans[1]}")
    # print(f"Best cost found: {best_cost:.2f}")
    
    # # Plot results
    # plt.figure(figsize=(12, 6))
    # plt.plot(cost_history, 'b.', alpha=0.3, label='Random solutions')
    # plt.plot(best_cost_history, 'r-', label='Best solution')
    # plt.axhline(y=given_cost, color='g', linestyle='--', label='Given MPS')
    # plt.xlabel('Iteration')
    # plt.ylabel('Total Cost')
    # plt.title('MPS Cost Optimization - Random Search')
    # plt.legend()
    # plt.grid(True)
    # plt.show()

if __name__ == "__main__":
    main()
