import pandas as pd
import numpy as np
import random
import math
import matplotlib.pyplot as plt

plt.rcParams['text.usetex'] = False
plt.rc("font",family='MicroSoft YaHei')

df = pd.read_csv('test_data1.csv', sep=' ')
df = pd.read_csv('data.csv', sep=' ')

df['任务时间'] = df['任务时间'].astype(str).str.extract('(\d+)').astype(float)
def process_predecessors(pred_str):
    if pd.isna(pred_str):
        return []
    else:
        return [int(x.strip()) for x in str(pred_str).split(',')]

predecessors = {task: process_predecessors(pred) for task, pred in zip(df['任务编号'], df['紧前任务'])}
successors = {task: [] for task in df['任务编号']}
for task, preds in predecessors.items():
    for pred in preds:
        successors[pred].append(task)

task_times = dict(zip(df['任务编号'], df['任务时间']))

 
def get_rpw(task_times, successors):
    all_successors = {task: [] for task in task_times.keys()}
    rpw_values = {}
    for task in task_times:
        if not successors[task]: 
            rpw_values[task] = task_times[task] 
    while len(rpw_values) < len(task_times):
        print(f"rpw_values: {rpw_values}")
        for task in task_times:
            if task not in rpw_values:
                if all(succ in rpw_values for succ in successors[task]):
                    prev_all_successors = [all_successors[succ] for succ in successors[task]]
                    # print(prev_all_successors, successors[task])
                    all_successors[task] = list(set(np.concatenate((*prev_all_successors, successors[task]))))
                    # print(all_successors[task])
                    rpw_values[task] = task_times[task] + sum(task_times[succ] for succ in all_successors[task])
                    # print(f"task {task}, rpw {rpw_values[task]}")
                    del task
    return rpw_values, all_successors
rpw_values, all_successors = get_rpw(task_times, successors)
df['RPW'] = df['任务编号'].map(rpw_values)

def allocate_workstations(df, predecessors, cycle_time=10):
    workstations = {1: []}  
    station_times = {1: 0}  
    allocated_tasks = set()
    
    sorted_tasks = df.sort_values('RPW', ascending=False)
    
    current_station = 1
    while len(allocated_tasks) < len(df):
        tasks_allocated_this_round = False
        # print(f"current_station: {current_station}, allocated_tasks: {allocated_tasks}")
        for _, task in sorted_tasks.iterrows():
            task_id = task['任务编号']
            if task_id in allocated_tasks:
                continue
                
            cur_predecessors = predecessors[task_id]
            if not all(pred in allocated_tasks for pred in cur_predecessors):
                continue
                
            if station_times[current_station] + task['任务时间'] <= cycle_time:
                workstations[current_station].append(task_id)
                station_times[current_station] += task['任务时间']
                allocated_tasks.add(task_id)
                tasks_allocated_this_round = True
        
        if not tasks_allocated_this_round:
            current_station += 1
            workstations[current_station] = []
            station_times[current_station] = 0
            
    return workstations

def print_workstation_details(workstations, task_times):
    print("\n工位分配详情:")
    for i, station in enumerate(workstations, 1):
        total_time = sum(task_times[task] for task in station)
        print(f"\n工位 {i}:")
        print(f"任务: {station}")
        print(f"总时间: {total_time:.1f}")
        print(f"任务时间: {[task_times[task] for task in station]}")

cycle_time = 53
# Use the function
workstation_assignments = allocate_workstations(df, predecessors, cycle_time)
print_workstation_details(workstation_assignments.values(), task_times)
for station, tasks in workstation_assignments.items():
    print(f"Workstation {station}: Tasks {tasks}")

def find_available_tasks(predecessors, assigned_tasks):
    #  找到没分配且无前序任务
    available = []
    for task, preds in predecessors.items():
        if task not in assigned_tasks and all(p in assigned_tasks for p in preds):
            available.append(task)
    return available

def allocate_tasks(df, cycle_time):
    workstations = []
    current_station = []
    current_time = 0
    assigned_tasks = set()
    
    while len(assigned_tasks) < len(df):
        available = find_available_tasks(predecessors, assigned_tasks)
        if not available:
            break
            
        # 排序
        available.sort(key=lambda x: task_times[x], reverse=True)
        
        tasks_added = False
        for task in available:
            if current_time + task_times[task] <= cycle_time:
                current_station.append(task)
                current_time += task_times[task]
                assigned_tasks.add(task)
                tasks_added = True
        
        # 不能再添加任务了
        if not tasks_added and current_station:
            workstations.append(current_station)
            current_station = []
            current_time = 0
        
    if current_station:
        workstations.append(current_station)
        
    return workstations

workstations = allocate_tasks(df, cycle_time)
print_workstation_details(workstations, task_times)

def check_cycle_time_constraint(workstation_tasks, task_times, cycle_time):
    """检查工位时间是否满足循环时间约束"""
    station_time = sum(task_times[t] for t in workstation_tasks)
    print(f"工位时间: {station_time}, 循环时间: {cycle_time}")
    if station_time > cycle_time:
        print(f"超出循环时间约束")
        return False
    return True

def check_predecessor_constraints(task, ws_idx, temp_workstations, predecessors_dict):
    """检查紧前任务约束"""
    print(f"检查任务 {task} 的紧前任务: {predecessors_dict[task]}")
    for pred in predecessors_dict[task]:
        pred_assigned = False
        print(f"检查紧前任务 {pred} 是否在工位 {ws_idx+1} 之前")
        for i in range(ws_idx + 1):
            if pred in temp_workstations[i]:
                print(f"任务 {pred} 在工位 {i+1} 中")
                pred_assigned = True
                break
        if not pred_assigned:
            print(f"任务 {task} 的紧前任务 {pred} 未满足约束")
            return False
    return True

def check_successor_constraints(task, ws_idx, temp_workstations, successors_dict):
    """检查紧后任务约束"""
    print(f"检查任务 {task} 的紧后任务: {successors_dict[task]}")
    for succ in successors_dict[task]:
        if succ in [t for ws in temp_workstations[:ws_idx] for t in ws]:
            print(f"任务 {task} 的紧后任务 {succ} 在工位 {ws_idx+1} 之前，违反约束")
            return False
        print(f"任务 {task} 的紧后任务 {succ} 满足约束")
    return True

def can_move_task(task, from_ws_idx, to_ws_idx, workstations, predecessors_dict, successors_dict, task_times, cycle_time):
    """检查任务是否可以从一个工位移动到另一个工位"""
    temp_workstations = [ws.copy() for ws in workstations]
    temp_workstations[from_ws_idx].remove(task)
    temp_workstations[to_ws_idx].append(task)
    print(f"移动后的临时工位分配: {temp_workstations}")
    
    # 检查循环时间约束
    if not check_cycle_time_constraint(temp_workstations[to_ws_idx], task_times, cycle_time):
        return False
        
    # 检查紧前任务约束
    if not check_predecessor_constraints(task, to_ws_idx, temp_workstations, predecessors_dict):
        return False
        
    # 检查紧后任务约束
    if not check_successor_constraints(task, to_ws_idx, temp_workstations, successors_dict):
        return False
    
    print("所有约束检查通过，任务可以移动")
    return True

def can_swap_tasks(task1, ws1_idx, task2, ws2_idx, workstations, predecessors_dict, successors_dict, task_times, cycle_time):
    """检查两个任务是否可以交换"""
    temp_workstations = [ws.copy() for ws in workstations]
    temp_workstations[ws1_idx].remove(task1)
    temp_workstations[ws2_idx].remove(task2)
    temp_workstations[ws1_idx].append(task2)
    temp_workstations[ws2_idx].append(task1)
    print(f"交换后的临时工位分配: {temp_workstations}")
    
    # 检查循环时间约束
    if not check_cycle_time_constraint(temp_workstations[ws1_idx], task_times, cycle_time):
        return False
    if not check_cycle_time_constraint(temp_workstations[ws2_idx], task_times, cycle_time):
        return False
    
    # 检查task1的约束
    if not check_predecessor_constraints(task1, ws2_idx, temp_workstations, predecessors_dict):
        return False
    if not check_successor_constraints(task1, ws2_idx, temp_workstations, successors_dict):
        return False
    
    # 检查task2的约束
    if not check_predecessor_constraints(task2, ws1_idx, temp_workstations, predecessors_dict):
        return False
    if not check_successor_constraints(task2, ws1_idx, temp_workstations, successors_dict):
        return False
    
    print("所有约束检查通过，任务可以交换")
    return True

def trade_and_transfer(workstations, predecessors_dict, successors_dict, task_times, cycle_time):
    """执行Trade and Transfer阶段均衡分析"""
    improved = True
    iteration = 0
    workstations = [ws.copy() for ws in workstations]  # 深拷贝工位分配
    while improved:
        iteration += 1
        print(f"\n迭代 {iteration}:")
        improved = False
        
        # 计算每个工位的工作时间
        station_times = [sum(task_times[task] for task in station) for station in workstations]
        print(f"工位时间: {station_times}")
        # 步骤1: 确定最大时间工位和最小时间工位
        max_time = max(station_times)
        min_time = min(station_times)
        max_ws_candidates = [idx for idx, time in enumerate(station_times) if time == max_time]
        min_ws_candidates = [idx for idx, time in enumerate(station_times) if time == min_time]
        max_ws_idx = random.choice(max_ws_candidates)
        min_ws_idx = random.choice(min_ws_candidates)
        
        print(f"最大时间工位: {max_ws_idx+1}, 时间: {max_time}")
        print(f"最小时间工位: {min_ws_idx+1}, 时间: {min_time}")
        
        # 步骤2: 计算目标值G
        G = (max_time - min_time) / 2
        print(f"目标值G: {G}")
        
        if G <= 0.01:  # 如果差距很小，认为已经均衡
            print("工位已经基本均衡，算法终止")
            break
        
        # 步骤3: 初始化候选集合C
        candidates = []
        
        # 步骤4: Transfer - 考虑从最大工位转移任务到最小工位
        for task in workstations[max_ws_idx]:
            # 检查任务时间是否小于2G
            if task_times[task] <= 2 * G:
                # 检查是否可行  
                print("-"*30)
                print(f"检查任务 {task} 从工位 {max_ws_idx+1} 转移到工位 {min_ws_idx+1}")
                if can_move_task(task, max_ws_idx, min_ws_idx, workstations, predecessors_dict, successors_dict, task_times, cycle_time):
                    print(f"任务 {task} 可以转移")
                    # 计算转移后的工位时间
                    new_max_time = max_time - task_times[task]
                    new_min_time = min_time + task_times[task]
                    # 计算不平衡度 (用最大工位和最小工位时间差评估)
                    imbalance = abs(new_max_time - new_min_time)
                    candidates.append({
                        'type': 'transfer',
                        'task': task,
                        'from_ws': max_ws_idx,
                        'to_ws': min_ws_idx,
                        'imbalance': imbalance
                    })
        
        # 步骤5: Trade - 考虑交换任务
        for task1 in workstations[max_ws_idx]:
            for task2 in workstations[min_ws_idx]:
                # 检查交换后工位时间的变化是否满足条件
                time_decrease = task_times[task1] - task_times[task2]
                if time_decrease <= 2 * G:  # 确保最大工位时间减少且减少量不超过2G
                    print("-"*30)
                    print(f"检查任务 {task1} (工位 {max_ws_idx+1}) 与任务 {task2} (工位 {min_ws_idx+1}) 交换")
                    # 检查交换是否可行
                    if can_swap_tasks(task1, max_ws_idx, task2, min_ws_idx, workstations, predecessors_dict, successors_dict, task_times, cycle_time):
                        # 计算交换后的工位时间
                        new_max_time = max_time - time_decrease
                        new_min_time = min_time + time_decrease
                        # 计算不平衡度
                        imbalance = abs(new_max_time - new_min_time)
                        candidates.append({
                            'type': 'trade',
                            'task1': task1,
                            'ws1': max_ws_idx,
                            'task2': task2,
                            'ws2': min_ws_idx,
                            'imbalance': imbalance
                        })
        
        # 步骤6: 执行最优的移动或交换
        if candidates:
            # 按不平衡度排序，选择使工位最平衡的操作
            best_move = min(candidates, key=lambda x: x['imbalance'])
            print("-"*30)
            if best_move['type'] == 'transfer':
                task = best_move['task']
                from_ws = best_move['from_ws']
                to_ws = best_move['to_ws']
                workstations[from_ws].remove(task)
                workstations[to_ws].append(task)
                print(f"执行Transfer: 任务 {task} 从工位 {from_ws+1} 移动到工位 {to_ws+1}")
                improved = True
            else:  # trade
                task1 = best_move['task1']
                ws1 = best_move['ws1']
                task2 = best_move['task2']
                ws2 = best_move['ws2']
                workstations[ws1].remove(task1)
                workstations[ws2].remove(task2)
                workstations[ws1].append(task2)
                workstations[ws2].append(task1)
                print(f"执行Trade: 任务 {task1} (工位 {ws1+1}) 与任务 {task2} (工位 {ws2+1}) 交换")
                improved = True
        else:
            print("无可行的移动或交换，算法终止")
            break
    
    return workstations

def calculate_smoothness_index(workstations, task_times):
    """
    计算平滑指数(SI)
    参数:
        workstations: 工位分配方案
        task_times: 任务时间字典
    返回:
        smoothness_index: 平滑指数
    """
    # 计算每个工位的时间
    station_times = [sum(task_times[task] for task in station) for station in workstations]
    
    # 找出最大工位时间
    max_station_time = max(station_times)
    
    # 计算平滑指数
    squared_diff_sum = sum((max_station_time - time) ** 2 for time in station_times)
    smoothness_index = math.sqrt(squared_diff_sum)
    
    return smoothness_index

# 在初始分配之后执行Trade and Transfer阶段
print("\n开始执行Trade and Transfer阶段均衡分析...")
balanced_workstations = trade_and_transfer(workstations, predecessors, successors, task_times, cycle_time)

# 显示优化后的结果
print("\n优化后的工位分配:")
for i, station in enumerate(balanced_workstations):
    station_time = sum(task_times[task] for task in station)
    print(f"工位 {i+1}:")
    print(f"  任务: {station}")
    print(f"  总时间: {station_time}")
    print(f"  闲置时间: {cycle_time - station_time}")

# RPW值和任务时间的对比
si = calculate_smoothness_index(workstation_assignments.values(), task_times)
print(f"\nRPW的平滑指数(SI): {si:.2f}")
# 在显示优化结果时添加平滑指数计算
print("\n计算优化前的平滑指数...")
si = calculate_smoothness_index(workstations, task_times)
print(f"平滑指数(SI): {si:.2f}")
print("\n计算优化后的平滑指数...")
si = calculate_smoothness_index(balanced_workstations, task_times)
print(f"平滑指数(SI): {si:.2f}")

# 探索莫迪和杨法作业元素分配阶段中，不同任务排序规则对所得到工位数量的影响
def sorting_rule_rpw(task_id):
    """根据RPW值降序排序"""
    return -df.loc[df['任务编号'] == task_id, 'RPW'].values[0]

def sorting_rule_task_time(task_id):
    """根据任务时间降序排序"""
    return -task_times[task_id]

def sorting_rule_num_followers(task_id):
    """根据后继任务数量降序排序"""
    return -len(all_successors[task_id])

def sorting_rule_num_predecessors(task_id):
    """根据前驱任务数量降序排序"""
    return -len(predecessors[task_id])

def sorting_rule_combined(task_id):
    """综合规则：RPW值 + 任务时间"""
    rpw = df.loc[df['任务编号'] == task_id, 'RPW'].values[0]
    time = task_times[task_id]
    return -(rpw + time)

def allocate_tasks_with_rule(df, cycle_time, sorting_rule):
    """使用指定排序规则分配任务"""
    workstations = []
    current_station = []
    current_time = 0
    assigned_tasks = set()
    
    while len(assigned_tasks) < len(df):
        available = find_available_tasks(predecessors, assigned_tasks)
        if not available:
            break
            
        # 使用指定的排序规则
        available.sort(key=sorting_rule)
        
        tasks_added = False
        for task in available:
            if current_time + task_times[task] <= cycle_time:
                current_station.append(task)
                current_time += task_times[task]
                assigned_tasks.add(task)
                tasks_added = True
        
        if not tasks_added and current_station:
            workstations.append(current_station)
            current_station = []
            current_time = 0
        
    if current_station:
        workstations.append(current_station)
        
    return workstations

# 探索不同排序规则的影响
print("\n\n=========================================================")
print("探索莫迪和杨法作业元素分配阶段中，不同任务排序规则对所得到工位数量的影响")
print("=========================================================")

sorting_rules = {
    "RPW降序": sorting_rule_rpw,
    "任务时间降序": sorting_rule_task_time,
    "后继任务数量降序": sorting_rule_num_followers,
    "前驱任务数量降序": sorting_rule_num_predecessors,
    "综合RPW和任务时间": sorting_rule_combined
}

results_allocation = {}

for rule_name, rule_func in sorting_rules.items():
    print(f"\n使用排序规则: {rule_name}")
    assigned_workstations = allocate_tasks_with_rule(df, cycle_time, rule_func)
    
    # 计算结果指标
    num_stations = len(assigned_workstations)
    station_times = [sum(task_times[task] for task in station) for station in assigned_workstations]
    max_station_time = max(station_times) if station_times else 0
    avg_station_time = sum(station_times) / len(station_times) if station_times else 0
    si = calculate_smoothness_index(assigned_workstations, task_times)
    
    # 存储结果
    results_allocation[rule_name] = {
        "工位数量": num_stations,
        "最大工位时间": max_station_time,
        "平均工位时间": avg_station_time,
        "平滑指数": si,
        "工位分配": assigned_workstations
    }
    
    # 打印详细结果
    print(f"工位数量: {num_stations}")
    print(f"最大工位时间: {max_station_time:.2f}")
    print(f"平均工位时间: {avg_station_time:.2f}")
    print(f"平滑指数(SI): {si:.2f}")
    print("工位分配详情:")
    for i, station in enumerate(assigned_workstations, 1):
        station_time = sum(task_times[task] for task in station)
        print(f"  工位 {i}: 任务 {station}, 时间: {station_time:.2f}")

# 结果对比表格
print("\n不同排序规则结果对比:")
print("-" * 80)
print(f"{'排序规则':<20} | {'工位数量':<10} | {'最大工位时间':<12} | {'平均工位时间':<12} | {'平滑指数':<10}")
print("-" * 80)
for rule, result in results_allocation.items():
    print(f"{rule:<20} | {result['工位数量']:<10} | {result['最大工位时间']:.2f}{' ':<5} | {result['平均工位时间']:.2f}{' ':<5} | {result['平滑指数']:.2f}")

# 探索莫迪和杨法Trade and Transfer阶段中，不同工位选择规则对所得到平衡指数的影响
def max_min_time_selection(station_times):
    """选择时间最大和最小的工位"""
    max_time = max(station_times)
    min_time = min(station_times)
    max_ws_candidates = [idx for idx, time in enumerate(station_times) if time == max_time]
    min_ws_candidates = [idx for idx, time in enumerate(station_times) if time == min_time]
    max_ws_idx = random.choice(max_ws_candidates)
    min_ws_idx = random.choice(min_ws_candidates)
    return max_ws_idx, min_ws_idx

def max_time_diff_selection(station_times):
    """选择使工位间时间差最大的工位对"""
    max_diff = -1
    selected_pair = (0, 0)
    
    for i in range(len(station_times)):
        for j in range(i + 1, len(station_times)):
            diff = abs(station_times[i] - station_times[j])
            if diff > max_diff:
                max_diff = diff
                if station_times[i] > station_times[j]:
                    selected_pair = (i, j)
                else:
                    selected_pair = (j, i)
    
    return selected_pair

def max_idle_time_selection(station_times, cycle_time):
    """选择闲置时间最大和最小的工位"""
    idle_times = [cycle_time - time for time in station_times]
    min_idle = min(idle_times)  # 对应最大工作时间
    max_idle = max(idle_times)  # 对应最小工作时间
    
    max_ws_candidates = [idx for idx, idle in enumerate(idle_times) if idle == min_idle]
    min_ws_candidates = [idx for idx, idle in enumerate(idle_times) if idle == max_idle]
    
    max_ws_idx = random.choice(max_ws_candidates)
    min_ws_idx = random.choice(min_ws_candidates)
    
    return max_ws_idx, min_ws_idx

def above_below_average_selection(station_times):
    """选择时间最高的超过平均值的工位和时间最低的低于平均值的工位"""
    avg_time = sum(station_times) / len(station_times)
    
    above_avg = [(idx, time) for idx, time in enumerate(station_times) if time > avg_time]
    below_avg = [(idx, time) for idx, time in enumerate(station_times) if time < avg_time]
    
    if not above_avg or not below_avg:
        return max_min_time_selection(station_times)
    
    max_ws_idx = max(above_avg, key=lambda x: x[1])[0]
    min_ws_idx = min(below_avg, key=lambda x: x[1])[0]
    
    return max_ws_idx, min_ws_idx

def trade_and_transfer_with_rule(workstations, predecessors_dict, successors_dict, task_times, cycle_time, selection_rule):
    """执行带有指定工位选择规则的Trade and Transfer阶段均衡分析"""
    improved = True
    iteration = 0
    workstations = [ws.copy() for ws in workstations]  # 深拷贝工位分配
    
    while improved and iteration < 50:  # 限制迭代次数避免无限循环
        iteration += 1
        improved = False
        
        # 计算每个工位的工作时间
        station_times = [sum(task_times[task] for task in station) for station in workstations]
        
        # 使用指定的选择规则选择工位
        if selection_rule == max_min_time_selection:
            max_ws_idx, min_ws_idx = max_min_time_selection(station_times)
        elif selection_rule == max_time_diff_selection:
            max_ws_idx, min_ws_idx = max_time_diff_selection(station_times)
        elif selection_rule == max_idle_time_selection:
            max_ws_idx, min_ws_idx = max_idle_time_selection(station_times, cycle_time)
        elif selection_rule == above_below_average_selection:
            max_ws_idx, min_ws_idx = above_below_average_selection(station_times)
        
        max_time = station_times[max_ws_idx]
        min_time = station_times[min_ws_idx]
        
        # 如果两个工位是同一个或时间几乎相等，则停止
        if max_ws_idx == min_ws_idx or abs(max_time - min_time) <= 0.01:
            break
        
        # 计算目标值G
        G = (max_time - min_time) / 2
        
        # 初始化候选集合
        candidates = []
        
        # Transfer - 考虑从最大工位转移任务到最小工位
        for task in workstations[max_ws_idx]:
            if task_times[task] <= 2 * G:
                # 简化检查逻辑以提高性能
                temp_workstations = [ws.copy() for ws in workstations]
                temp_workstations[max_ws_idx].remove(task)
                temp_workstations[min_ws_idx].append(task)
                
                # 检查循环时间约束
                to_ws_time = sum(task_times[t] for t in temp_workstations[min_ws_idx])
                if to_ws_time > cycle_time:
                    continue
                
                # 检查前后任务约束
                valid_move = True
                # 前驱任务约束
                for pred in predecessors_dict[task]:
                    pred_assigned = False
                    for i in range(min_ws_idx + 1):
                        if pred in temp_workstations[i]:
                            pred_assigned = True
                            break
                    if not pred_assigned:
                        valid_move = False
                        break
                
                if not valid_move:
                    continue
                
                # 后继任务约束
                for succ in successors_dict[task]:
                    if succ in [t for ws in temp_workstations[:min_ws_idx] for t in ws]:
                        valid_move = False
                        break
                
                if not valid_move:
                    continue
                
                # 计算不平衡度
                new_max_time = max_time - task_times[task]
                new_min_time = min_time + task_times[task]
                imbalance = abs(new_max_time - new_min_time)
                candidates.append({
                    'type': 'transfer',
                    'task': task,
                    'from_ws': max_ws_idx,
                    'to_ws': min_ws_idx,
                    'imbalance': imbalance
                })
        
        # Trade - 考虑交换任务
        for task1 in workstations[max_ws_idx]:
            for task2 in workstations[min_ws_idx]:
                time_decrease = task_times[task1] - task_times[task2]
                if time_decrease > 0:  # 确保最大工位时间减少
                    temp_workstations = [ws.copy() for ws in workstations]
                    temp_workstations[max_ws_idx].remove(task1)
                    temp_workstations[min_ws_idx].remove(task2)
                    temp_workstations[max_ws_idx].append(task2)
                    temp_workstations[min_ws_idx].append(task1)
                    
                    # 检查循环时间约束
                    ws1_time = sum(task_times[t] for t in temp_workstations[max_ws_idx])
                    ws2_time = sum(task_times[t] for t in temp_workstations[min_ws_idx])
                    if ws1_time > cycle_time or ws2_time > cycle_time:
                        continue
                    
                    # 简化约束检查
                    valid_swap = True
                    
                    # 检查task1的约束
                    for pred in predecessors_dict[task1]:
                        pred_assigned = False
                        for i in range(min_ws_idx + 1):
                            if pred in temp_workstations[i]:
                                pred_assigned = True
                                break
                        if not pred_assigned:
                            valid_swap = False
                            break
                    
                    if not valid_swap:
                        continue
                    
                    for succ in successors_dict[task1]:
                        if succ in [t for ws in temp_workstations[:min_ws_idx] for t in ws]:
                            valid_swap = False
                            break
                    
                    if not valid_swap:
                        continue
                    
                    # 检查task2的约束
                    for pred in predecessors_dict[task2]:
                        pred_assigned = False
                        for i in range(max_ws_idx + 1):
                            if pred in temp_workstations[i]:
                                pred_assigned = True
                                break
                        if not pred_assigned:
                            valid_swap = False
                            break
                    
                    if not valid_swap:
                        continue
                    
                    for succ in successors_dict[task2]:
                        if succ in [t for ws in temp_workstations[:max_ws_idx] for t in ws]:
                            valid_swap = False
                            break
                    
                    if not valid_swap:
                        continue
                    
                    # 计算不平衡度
                    new_max_time = max_time - time_decrease
                    new_min_time = min_time + time_decrease
                    imbalance = abs(new_max_time - new_min_time)
                    candidates.append({
                        'type': 'trade',
                        'task1': task1,
                        'ws1': max_ws_idx,
                        'task2': task2,
                        'ws2': min_ws_idx,
                        'imbalance': imbalance
                    })
        
        # 执行最优的移动或交换
        if candidates:
            best_move = min(candidates, key=lambda x: x['imbalance'])
            if best_move['type'] == 'transfer':
                task = best_move['task']
                from_ws = best_move['from_ws']
                to_ws = best_move['to_ws']
                workstations[from_ws].remove(task)
                workstations[to_ws].append(task)
                improved = True
            else:  # trade
                task1 = best_move['task1']
                ws1 = best_move['ws1']
                task2 = best_move['task2']
                ws2 = best_move['ws2']
                workstations[ws1].remove(task1)
                workstations[ws2].remove(task2)
                workstations[ws1].append(task2)
                workstations[ws2].append(task1)
                improved = True
    
    return workstations, iteration

print("\n\n=========================================================")
print("探索莫迪和杨法Trade and Transfer阶段中，不同工位选择规则对所得到平衡指数的影响")
print("=========================================================")

# 选择一个初始工位分配作为基准
initial_workstations = results_allocation["RPW降序"]["工位分配"]

selection_rules = {
    "最大最小时间工位": max_min_time_selection,
    "最大时间差工位对": max_time_diff_selection,
    "最大最小闲置时间": max_idle_time_selection,
    "高于/低于平均时间": above_below_average_selection
}

results_tt = {}

for rule_name, rule_func in selection_rules.items():
    print(f"\n使用工位选择规则: {rule_name}")
    balanced_ws, iterations = trade_and_transfer_with_rule(
        initial_workstations, predecessors, successors, task_times, cycle_time, rule_func
    )
    
    # 计算结果指标
    station_times = [sum(task_times[task] for task in station) for station in balanced_ws]
    max_station_time = max(station_times) if station_times else 0
    min_station_time = min(station_times) if station_times else 0
    avg_station_time = sum(station_times) / len(station_times) if station_times else 0
    si = calculate_smoothness_index(balanced_ws, task_times)
    
    # 存储结果
    results_tt[rule_name] = {
        "迭代次数": iterations,
        "最大工位时间": max_station_time,
        "最小工位时间": min_station_time,
        "时间差": max_station_time - min_station_time,
        "平均工位时间": avg_station_time,
        "平滑指数": si,
        "工位分配": balanced_ws
    }
    
    # 打印详细结果
    print(f"迭代次数: {iterations}")
    print(f"最大工位时间: {max_station_time:.2f}")
    print(f"最小工位时间: {min_station_time:.2f}")
    print(f"工位时间差: {max_station_time - min_station_time:.2f}")
    print(f"平均工位时间: {avg_station_time:.2f}")
    print(f"平滑指数(SI): {si:.2f}")
    print("工位分配详情:")
    for i, station in enumerate(balanced_ws, 1):
        station_time = sum(task_times[task] for task in station)
        print(f"  工位 {i}: 任务 {station}, 时间: {station_time:.2f}")

# 结果对比表格
print("\n不同工位选择规则结果对比:")
print("-" * 95)
print(f"{'选择规则':<20} | {'迭代次数':<10} | {'最大时间':<10} | {'最小时间':<10} | {'时间差':<8} | {'平滑指数':<10}")
print("-" * 95)
for rule, result in results_tt.items():
    print(f"{rule:<20} | {result['迭代次数']:<10} | {result['最大工位时间']:.2f}{' ':<3} | {result['最小工位时间']:.2f}{' ':<3} | {result['时间差']:.2f}{' ':<1} | {result['平滑指数']:.2f}")

# 绘制结果比较图表
try:

    # 排序规则比较图
    plt.figure(figsize=(10, 6))
    rules = list(results_allocation.keys())
    workstations_count = [results_allocation[rule]['工位数量'] for rule in rules]
    smoothness_indices = [results_allocation[rule]['平滑指数'] for rule in rules]
    
    x = range(len(rules))
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 工位数量柱状图
    bars = ax1.bar(x, workstations_count, width=0.4, alpha=0.7, color='blue', label='工位数量')
    ax1.set_ylabel('工位数量', fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(rules, rotation=45, ha='right', fontsize=10)
    
    # 在柱状图上标注数值
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom')
    
    # 平滑指数折线图
    ax2 = ax1.twinx()
    ax2.plot(x, smoothness_indices, 'r-o', linewidth=2, label='平滑指数')
    ax2.set_ylabel('平滑指数', fontsize=12, color='r')
    
    # 在折线上标注数值
    for i, si in enumerate(smoothness_indices):
        ax2.annotate(f'{si:.2f}', (x[i], si), xytext=(0, 10), 
                     textcoords='offset points', ha='center', va='bottom', color='r')
    
    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('莫迪和杨法分配阶段：不同排序规则对工位数量和平滑指数的影响', fontsize=14)
    plt.tight_layout()
    plt.savefig('allocation_rules_comparison.png')
    
    # Trade and Transfer规则比较图
    plt.figure(figsize=(12, 6))
    rules = list(results_tt.keys())
    iterations = [results_tt[rule]['迭代次数'] for rule in rules]
    time_diffs = [results_tt[rule]['时间差'] for rule in rules]
    si_values = [results_tt[rule]['平滑指数'] for rule in rules]
    
    x = range(len(rules))
    
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # 迭代次数柱状图
    bars = ax1.bar([i-0.2 for i in x], iterations, width=0.4, alpha=0.7, color='green', label='迭代次数')
    ax1.set_ylabel('迭代次数', fontsize=12)
    
    # 平滑指数柱状图
    ax2 = ax1.twinx()
    bars2 = ax2.bar([i+0.2 for i in x], si_values, width=0.4, alpha=0.7, color='orange', label='平滑指数')
    ax2.set_ylabel('平滑指数', fontsize=12)
    
    # 在柱状图上标注数值
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{int(height)}', ha='center', va='bottom', color='green')
    
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                 f'{height:.2f}', ha='center', va='bottom', color='orange')
    
    ax1.set_xticks(x)
    ax1.set_xticklabels(rules, rotation=45, ha='right', fontsize=10)
    
    # 图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.title('莫迪和杨法Trade and Transfer阶段：不同工位选择规则的影响', fontsize=14)
    plt.tight_layout()
    plt.savefig('tt_rules_comparison.png')
    
    print("\n已生成图表并保存为 'allocation_rules_comparison.png' 和 'tt_rules_comparison.png'")
except ImportError:
    print("\n无法生成图表，请安装matplotlib库")
except Exception as e:
    print(f"\n生成图表时出错: {e}")
