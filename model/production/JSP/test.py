import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import defaultdict, deque
import itertools
import random
import copy
from typing import List, Dict, Tuple, Set, Optional
import time

class Operation:
    """工序类 - 表示单个加工工序"""
    def __init__(self, job_id: int, op_index: int, machine: int, processing_time: int):
        self.job_id = job_id          # 所属工件ID
        self.op_index = op_index      # 在工件中的工序索引
        self.machine = machine        # 加工机器
        self.processing_time = processing_time  # 加工时间
        self.id = f"O_{job_id}{op_index}"      # 工序唯一标识
    
    def __str__(self):
        return f"{self.id}(J{self.job_id}, M{self.machine}, t={self.processing_time})"
    
    def __repr__(self):
        return self.__str__()

class Job:
    """工件类 - 表示完整的工件及其工艺路径"""
    def __init__(self, job_id: int, machines: List[int], processing_times: List[int]):
        self.job_id = job_id
        self.operations = []
        
        # 创建工序序列
        for i, (machine, time) in enumerate(zip(machines, processing_times)):
            op = Operation(job_id, i+1, machine, time)
            self.operations.append(op)
    
    def get_operation(self, op_index: int) -> Operation:
        """获取指定索引的工序"""
        return self.operations[op_index - 1]
    
    def __str__(self):
        return f"Job {self.job_id}: {' -> '.join([str(op) for op in self.operations])}"

class DisjunctiveGraph:
    """析取图类 - 核心数据结构"""
    def __init__(self, jobs: List[Job]):
        self.jobs = jobs
        self.operations = []
        self.conjunctive_arcs = []  # 连接弧（固定工艺顺序）
        self.disjunctive_arcs = []  # 析取弧（机器资源竞争）
        self.machine_operations = defaultdict(list)  # 每台机器上的工序
        
        self._build_graph()
    
    def _build_graph(self):
        """构建析取图的节点和边"""
        # 收集所有工序
        for job in self.jobs:
            self.operations.extend(job.operations)
            
        # 按机器分组工序
        for op in self.operations:
            self.machine_operations[op.machine].append(op)
        
        # 构建连接弧（工艺路径约束）
        self._build_conjunctive_arcs()
        
        # 构建析取弧（机器资源约束）
        self._build_disjunctive_arcs()
    
    def _build_conjunctive_arcs(self):
        """构建连接弧 - 表示工件内部工序的固定顺序"""
        for job in self.jobs:
            for i in range(len(job.operations) - 1):
                from_op = job.operations[i]
                to_op = job.operations[i + 1]
                self.conjunctive_arcs.append((from_op, to_op, from_op.processing_time))
    
    def _build_disjunctive_arcs(self):
        """构建析取弧 - 表示同一机器上工序间的竞争关系"""
        for machine, ops in self.machine_operations.items():
            # 对每台机器上的工序两两组合创建析取弧
            for i in range(len(ops)):
                for j in range(i + 1, len(ops)):
                    op1, op2 = ops[i], ops[j]
                    # 创建双向析取弧
                    self.disjunctive_arcs.append((op1, op2, op1.processing_time))
                    self.disjunctive_arcs.append((op2, op1, op2.processing_time))
    
    def is_acyclic(self, machine_orders: Dict[int, List[Operation]]) -> bool:
        """
        检查给定机器顺序是否形成无环图
        
        Args:
            machine_orders: 每台机器上工序的加工顺序
            
        Returns:
            bool: True表示无环，False表示有环
        """
        # 构建有向图的邻接表
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        # 初始化所有工序的入度
        for op in self.operations:
            in_degree[op.id] = 0
        
        # 添加工艺路径约束边（连接弧）
        for from_op, to_op, _ in self.conjunctive_arcs:
            graph[from_op.id].append(to_op.id)
            in_degree[to_op.id] += 1
        
        # 添加机器顺序约束边（选择的析取弧）
        for machine, ops in machine_orders.items():
            for i in range(len(ops) - 1):
                from_op = ops[i]
                to_op = ops[i + 1]
                graph[from_op.id].append(to_op.id)
                in_degree[to_op.id] += 1
        
        # 使用拓扑排序检测环
        queue = deque()
        
        # 找到所有入度为0的节点
        for op_id in in_degree:
            if in_degree[op_id] == 0:
                queue.append(op_id)
        
        processed_count = 0
        
        while queue:
            current = queue.popleft()
            processed_count += 1
            
            # 处理当前节点的所有邻居
            for neighbor in graph[current]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # 如果处理的节点数等于总节点数，则无环
        return processed_count == len(self.operations)

class ScheduleSolver:
    """调度求解器 - 实现多种优化算法"""
    def __init__(self, disjunctive_graph: DisjunctiveGraph):
        self.graph = disjunctive_graph
        self.best_makespan = float('inf')
        self.best_schedule = None
    
    def calculate_makespan(self, machine_orders: Dict[int, List[Operation]]) -> Tuple[int, Dict[str, int]]:
        """
        计算给定机器顺序下的最大完工时间（仅处理无环图）
        
        Args:
            machine_orders: 每台机器上工序的加工顺序
            
        Returns:
            (makespan, operation_start_times): 最大完工时间和各工序开始时间
        """
        # 检查是否为无环图，有环直接返回无穷大
        if not self.graph.is_acyclic(machine_orders):
            return float('inf'), {}
        
        # 构建有向无环图的邻接表
        graph = defaultdict(list)
        in_degree = defaultdict(int)
        
        # 初始化所有工序
        for op in self.graph.operations:
            in_degree[op.id] = 0
        
        # 添加工艺路径约束边
        for from_op, to_op, _ in self.graph.conjunctive_arcs:
            graph[from_op.id].append((to_op.id, from_op.processing_time))
            in_degree[to_op.id] += 1
        
        # 添加机器顺序约束边
        for machine, ops in machine_orders.items():
            for i in range(len(ops) - 1):
                from_op = ops[i]
                to_op = ops[i + 1]
                graph[from_op.id].append((to_op.id, from_op.processing_time))
                in_degree[to_op.id] += 1
        
        # 使用拓扑排序计算最早开始时间
        queue = deque()
        start_times = {}
        
        # 初始化所有工序的开始时间
        for op in self.graph.operations:
            start_times[op.id] = 0
        
        # 找到所有入度为0的节点
        for op_id in in_degree:
            if in_degree[op_id] == 0:
                queue.append(op_id)
        
        while queue:
            current_id = queue.popleft()
            
            # 处理当前节点的所有后继
            for next_id, processing_time in graph[current_id]:
                # 更新后继节点的最早开始时间
                start_times[next_id] = max(start_times[next_id], 
                                         start_times[current_id] + processing_time)
                
                in_degree[next_id] -= 1
                if in_degree[next_id] == 0:
                    queue.append(next_id)
        
        # 计算最大完工时间
        makespan = 0
        for op in self.graph.operations:
            finish_time = start_times[op.id] + op.processing_time
            makespan = max(makespan, finish_time)
        
        return makespan, start_times
    
    def enumerate_all_schedules(self) -> Tuple[int, Dict[int, List[Operation]]]:
        """
        枚举所有可能的无环调度方案
        
        Returns:
            (best_makespan, best_machine_orders): 最优解和对应的机器顺序
        """
        print("开始枚举所有可能的无环调度方案...")
        
        machines = list(self.graph.machine_operations.keys())
        machine_permutations = {}
        
        # 生成每台机器上所有可能的工序排列
        for machine in machines:
            ops = self.graph.machine_operations[machine]
            machine_permutations[machine] = list(itertools.permutations(ops))
        
        # 计算总的组合数
        total_combinations = 1
        for machine in machines:
            total_combinations *= len(machine_permutations[machine])
        
        print(f"总共需要评估 {total_combinations} 种调度方案")
        
        best_makespan = float('inf')
        best_orders = None
        evaluated = 0
        acyclic_count = 0
        
        # 枚举所有可能的组合
        for combination in itertools.product(*[machine_permutations[m] for m in machines]):
            machine_orders = {}
            for i, machine in enumerate(machines):
                machine_orders[machine] = list(combination[i])
            
            evaluated += 1
            
            # 计算makespan（内部会检查是否无环）
            makespan, _ = self.calculate_makespan(machine_orders)
            
            # 只处理无环方案
            if makespan != float('inf'):
                acyclic_count += 1
                
                if makespan < best_makespan:
                    best_makespan = makespan
                    best_orders = machine_orders.copy()
                    print(f"发现更优解: Makespan = {best_makespan} (第 {acyclic_count} 个无环方案)")
            
            if evaluated % 100 == 0:
                print(f"已评估 {evaluated}/{total_combinations} 个方案, "
                      f"无环方案: {acyclic_count}, 当前最优解: {best_makespan}")
        
        print(f"枚举完成! 无环方案总数: {acyclic_count}/{total_combinations}")
        if best_makespan == float('inf'):
            print("警告: 未找到任何无环方案!")
            return float('inf'), None
        
        print(f"最优 Makespan: {best_makespan}")
        return best_makespan, best_orders
    
    def genetic_algorithm(self, population_size: int = 50, generations: int = 100) -> Tuple[int, Dict[int, List[Operation]]]:
        """
        遗传算法求解调度问题（只处理无环方案）
        """
        print(f"开始遗传算法求解 (种群大小: {population_size}, 进化代数: {generations})...")
        
        def create_random_individual():
            """创建随机个体，跳过有环配置"""
            max_attempts = 100
            for _ in range(max_attempts):
                machine_orders = {}
                for machine, ops in self.graph.machine_operations.items():
                    shuffled_ops = list(ops)
                    random.shuffle(shuffled_ops)
                    machine_orders[machine] = shuffled_ops
                
                # 检查是否无环
                if self.graph.is_acyclic(machine_orders):
                    return machine_orders
            
            # 如果随机生成失败，使用确定性方法
            machine_orders = {}
            for machine, ops in self.graph.machine_operations.items():
                sorted_ops = sorted(ops, key=lambda x: (x.job_id, x.op_index))
                machine_orders[machine] = sorted_ops
            return machine_orders
        
        def mutate(individual, mutation_rate=0.2):
            """变异操作 - 跳过产生环的变异"""
            for _ in range(20):  # 增加尝试次数
                mutated = {}
                for machine in individual:
                    mutated[machine] = list(individual[machine])
                    
                    if random.random() < mutation_rate and len(mutated[machine]) > 1:
                        i, j = random.sample(range(len(mutated[machine])), 2)
                        mutated[machine][i], mutated[machine][j] = mutated[machine][j], mutated[machine][i]
                
                if self.graph.is_acyclic(mutated):
                    return mutated
            
            return individual
        
        def crossover(parent1, parent2):
            """交叉操作 - 跳过产生环的交叉"""
            for _ in range(20):  # 增加尝试次数
                child = {}
                for machine in parent1:
                    if random.random() < 0.5:
                        child[machine] = list(parent1[machine])
                    else:
                        child[machine] = list(parent2[machine])
                
                if self.graph.is_acyclic(child):
                    return child
            
            return copy.deepcopy(parent1)
        
        # 初始化种群
        population = []
        for _ in range(population_size):
            individual = create_random_individual()
            population.append(individual)
        
        best_makespan = float('inf')
        best_individual = None
        stagnation_counter = 0
        max_stagnation = 20
        
        for generation in range(generations):
            fitness_scores = []
            valid_count = 0
            
            for i, individual in enumerate(population):
                makespan, _ = self.calculate_makespan(individual)
                fitness_scores.append(makespan)
                
                if makespan != float('inf'):
                    valid_count += 1
                    
                    if makespan < best_makespan:
                        best_makespan = makespan
                        best_individual = copy.deepcopy(individual)
                        stagnation_counter = 0
                        if generation > 0:
                            print(f"第 {generation} 代发现更优解: Makespan = {best_makespan}")
                else:
                    # 替换无效个体
                    population[i] = create_random_individual()
                    fitness_scores[i], _ = self.calculate_makespan(population[i])
            
            stagnation_counter += 1
            
            if stagnation_counter >= max_stagnation:
                print(f"连续 {max_stagnation} 代无改进，提前停止")
                break
            
            # 锦标赛选择
            def tournament_selection(population, fitness_scores, tournament_size=3):
                selected = []
                for _ in range(len(population)):
                    valid_indices = [i for i, f in enumerate(fitness_scores) if f != float('inf')]
                    if len(valid_indices) < tournament_size:
                        tournament_indices = valid_indices
                    else:
                        tournament_indices = random.sample(valid_indices, tournament_size)
                    
                    if tournament_indices:
                        best_idx = min(tournament_indices, key=lambda i: fitness_scores[i])
                        selected.append(copy.deepcopy(population[best_idx]))
                    else:
                        selected.append(create_random_individual())
                return selected
            
            selected_population = tournament_selection(population, fitness_scores)
            
            # 保留精英
            valid_indices = [(i, fitness_scores[i]) for i in range(len(fitness_scores)) 
                           if fitness_scores[i] != float('inf')]
            valid_indices.sort(key=lambda x: x[1])
            
            elite_size = max(2, population_size // 10)
            elite_population = []
            for i, _ in valid_indices[:elite_size]:
                elite_population.append(copy.deepcopy(population[i]))
            
            if not elite_population:
                elite_population = [create_random_individual()]
            
            # 生成新一代
            new_population = elite_population.copy()
            
            while len(new_population) < population_size:
                parent1 = random.choice(selected_population)
                parent2 = random.choice(selected_population)
                
                child = crossover(parent1, parent2)
                child = mutate(child)
                new_population.append(child)
            
            population = new_population[:population_size]
            
            if generation % 10 == 0:
                valid_scores = [f for f in fitness_scores if f != float('inf')]
                if valid_scores:
                    avg_fitness = np.mean(valid_scores)
                    print(f"第 {generation} 代: 最优解 = {best_makespan}, 平均解 = {avg_fitness:.2f}, 有效个体 = {valid_count}")
        
        if best_individual is None:
            print("遗传算法未找到有效解")
            return float('inf'), None
        
        print(f"遗传算法完成! 最优 Makespan: {best_makespan}")
        return best_makespan, best_individual
    
    def priority_dispatch_rule(self, rule: str = 'SPT') -> Tuple[int, Dict[int, List[Operation]]]:
        """
        优先分派规则启发式算法（跳过有环配置）
        """
        print(f"应用优先分派规则: {rule}")
        
        machine_orders = {}
        
        for machine, ops in self.graph.machine_operations.items():
            if rule == 'SPT':
                sorted_ops = sorted(ops, key=lambda x: x.processing_time)
            elif rule == 'LPT':
                sorted_ops = sorted(ops, key=lambda x: x.processing_time, reverse=True)
            else:
                sorted_ops = sorted(ops, key=lambda x: (x.job_id, x.op_index))
            
            machine_orders[machine] = sorted_ops
        
        makespan, _ = self.calculate_makespan(machine_orders)
        
        if makespan == float('inf'):
            print(f"警告: {rule} 规则产生有环图，尝试其他排序")
            # 尝试按工件顺序排序
            for machine, ops in self.graph.machine_operations.items():
                sorted_ops = sorted(ops, key=lambda x: (x.job_id, x.op_index))
                machine_orders[machine] = sorted_ops
            makespan, _ = self.calculate_makespan(machine_orders)
        
        print(f"{rule} 规则结果: Makespan = {makespan}")
        return makespan, machine_orders

def visualize_schedule(machine_orders: Dict[int, List[Operation]], 
                      start_times: Dict[str, int], 
                      title: str = "作业车间调度甘特图"):
    """
    可视化调度方案的甘特图
    
    Args:
        machine_orders: 每台机器的工序顺序
        start_times: 各工序的开始时间
        title: 图表标题
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # 颜色映射（为不同工件分配不同颜色）
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow', 'lightpink', 'lightgray']
    job_colors = {}
    
    machines = sorted(machine_orders.keys())
    y_positions = {machine: i for i, machine in enumerate(machines)}
    
    # 绘制甘特图
    for machine, operations in machine_orders.items():
        y_pos = y_positions[machine]
        
        for op in operations:
            job_id = op.job_id
            if job_id not in job_colors:
                job_colors[job_id] = colors[len(job_colors) % len(colors)]
            
            start_time = start_times[op.id]
            duration = op.processing_time
            
            # 绘制工序矩形
            rect = patches.Rectangle((start_time, y_pos - 0.4), duration, 0.8,
                                   linewidth=1, edgecolor='black', 
                                   facecolor=job_colors[job_id], alpha=0.7)
            ax.add_patch(rect)
            
            # 添加工序标签
            ax.text(start_time + duration/2, y_pos, op.id, 
                   ha='center', va='center', fontsize=10, fontweight='bold')
    
    # 设置图表属性
    ax.set_xlabel('时间', fontsize=12)
    ax.set_ylabel('机器', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # 设置y轴
    ax.set_yticks(range(len(machines)))
    ax.set_yticklabels([f'M{m}' for m in machines])
    ax.set_ylim(-0.5, len(machines) - 0.5)
    
    # 设置x轴
    max_time = max(start_times[op.id] + op.processing_time 
                  for ops in machine_orders.values() for op in ops)
    ax.set_xlim(0, max_time + 1)
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    legend_elements = [patches.Patch(facecolor=job_colors[job_id], 
                                   label=f'工件 J{job_id}', alpha=0.7) 
                      for job_id in sorted(job_colors.keys())]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.show()

def print_schedule_details(machine_orders: Dict[int, List[Operation]], 
                          start_times: Dict[str, int], 
                          makespan: int):
    """
    打印调度方案的详细信息
    
    Args:
        machine_orders: 每台机器的工序顺序
        start_times: 各工序的开始时间
        makespan: 最大完工时间
    """
    print("\n" + "="*60)
    print("调度方案详细信息")
    print("="*60)
    
    for machine in sorted(machine_orders.keys()):
        print(f"\n机器 M{machine} 的加工顺序:")
        operations = machine_orders[machine]
        
        for i, op in enumerate(operations):
            start_time = start_times[op.id]
            finish_time = start_time + op.processing_time
            print(f"  {i+1}. {op.id} (工件J{op.job_id}) | "
                  f"开始时间: {start_time}, 结束时间: {finish_time}, "
                  f"加工时间: {op.processing_time}")
    
    print(f"\n总完工时间 (Makespan): {makespan}")
    print("="*60)

def main():
    """主函数 - 演示析取图求解作业车间调度问题"""
    print("基于析取图模型的作业车间调度问题求解系统")
    print("="*60)
    
    # 定义问题数据（用户提供的示例）
    job_data = [
        # 工件1: M1(2) -> M2(1) -> M3(3)
        ([1, 2, 3], [2, 1, 3]),
        # 工件2: M2(3) -> M1(3) -> M3(2)  
        ([2, 1, 3], [3, 3, 2]),
        # 工件3: M3(2) -> M1(2) -> M2(2)
        ([3, 1, 2], [2, 2, 2])
    ]
    
    # 创建工件对象
    jobs = []
    for i, (machines, times) in enumerate(job_data):
        job = Job(i + 1, machines, times)
        jobs.append(job)
        print(f"工件 {i+1}: {job}")
    
    print("\n" + "="*60)
    
    # 构建析取图
    graph = DisjunctiveGraph(jobs)
    
    print(f"析取图构建完成:")
    print(f"- 工序总数: {len(graph.operations)}")
    print(f"- 连接弧数量: {len(graph.conjunctive_arcs)}")
    print(f"- 析取弧数量: {len(graph.disjunctive_arcs)}")
    
    print(f"\n各机器上的工序分布:")
    for machine, ops in graph.machine_operations.items():
        print(f"- 机器 M{machine}: {[op.id for op in ops]}")
    
    # 创建求解器
    solver = ScheduleSolver(graph)
    
    print("\n" + "="*60)
    print("开始求解调度问题...")
    
    # 方法1: 完全枚举法
    start_time = time.time()
    best_makespan_enum, best_orders_enum = solver.enumerate_all_schedules()
    enum_time = time.time() - start_time
    
    print(f"\n枚举法求解时间: {enum_time:.3f} 秒")
    
    if best_orders_enum is not None:
        _, start_times_enum = solver.calculate_makespan(best_orders_enum)
    else:
        start_times_enum = {}
    
    # 方法2: 遗传算法
    start_time = time.time()
    best_makespan_ga, best_orders_ga = solver.genetic_algorithm(population_size=20, generations=30)
    ga_time = time.time() - start_time
    
    print(f"遗传算法求解时间: {ga_time:.3f} 秒")
    
    if best_orders_ga is not None:
        _, start_times_ga = solver.calculate_makespan(best_orders_ga)
    else:
        start_times_ga = {}
    
    # 方法3: 启发式规则
    start_time = time.time()
    makespan_spt, orders_spt = solver.priority_dispatch_rule('SPT')
    spt_time = time.time() - start_time
    
    if orders_spt is not None:
        _, start_times_spt = solver.calculate_makespan(orders_spt)
    else:
        start_times_spt = {}
    
    # 方法4: LPT规则
    makespan_lpt, orders_lpt = solver.priority_dispatch_rule('LPT')
    
    # 输出结果比较
    print("\n" + "="*60)
    print("求解结果比较:")
    print("="*60)
    print(f"完全枚举法    : Makespan = {best_makespan_enum}, 求解时间 = {enum_time:.3f}s")
    print(f"遗传算法      : Makespan = {best_makespan_ga}, 求解时间 = {ga_time:.3f}s")
    print(f"SPT启发式规则 : Makespan = {makespan_spt}, 求解时间 = {spt_time:.3f}s")
    print(f"LPT启发式规则 : Makespan = {makespan_lpt}, 求解时间 = {spt_time:.3f}s")
    
    # 显示最优解的详细信息
    if best_orders_enum is not None and best_makespan_enum != float('inf'):
        print_schedule_details(best_orders_enum, start_times_enum, best_makespan_enum)
        
        # 可视化最优调度方案
        print("\n正在生成甘特图...")
        visualize_schedule(best_orders_enum, start_times_enum, 
                          f"最优调度方案 (Makespan = {best_makespan_enum})")
        
        # 如果遗传算法结果不同，也显示其甘特图
        if (best_orders_ga is not None and best_makespan_ga != float('inf') and 
            best_makespan_ga != best_makespan_enum):
            visualize_schedule(best_orders_ga, start_times_ga, 
                              f"遗传算法结果 (Makespan = {best_makespan_ga})")
    else:
        print("未找到有效的调度方案")

if __name__ == "__main__":
    # 设置中文字体支持
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
    
    main()
