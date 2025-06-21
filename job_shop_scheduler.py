import random
from typing import List, Dict, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict

@dataclass
class Operation:
    job_id: int
    operation_id: int
    processing_time: int
    feasible_machines: List[int]
    
@dataclass
class Job:
    job_id: int
    operations: List[Operation]
    current_operation: int = 0  # 当前应该执行的工序
    
    def get_next_operation(self) -> Operation:
        """获取下一个需要执行的工序"""
        if self.current_operation < len(self.operations):
            return self.operations[self.current_operation]
        return None
    
    def complete_operation(self):
        """完成当前工序，移动到下一个"""
        self.current_operation += 1
    
    def is_completed(self) -> bool:
        """检查工件是否完成所有工序"""
        return self.current_operation >= len(self.operations)

class JobShopScheduler:
    def __init__(self, jobs: List[Job], num_machines: int):
        self.jobs = jobs
        self.num_machines = num_machines
        self.machine_schedules = defaultdict(list)  # machine_id -> [(start_time, end_time, job_id, operation_id)]
        self.machine_available_time = [0] * num_machines
        self.job_completion_time = [0] * len(jobs)
        
    def get_available_jobs(self) -> List[Job]:
        """获取可以执行下一工序的工件"""
        return [job for job in self.jobs if not job.is_completed()]
    
    def get_feasible_machines(self, operation: Operation) -> List[int]:
        """获取指定工序的可行机器"""
        return operation.feasible_machines
    
    def get_earliest_start_time(self, job: Job, machine_id: int) -> int:
        """计算在指定机器上的最早开始时间"""
        # 机器可用时间
        machine_ready_time = self.machine_available_time[machine_id]
        # 工件前序工序完成时间
        job_ready_time = self.job_completion_time[job.job_id]
        
        return max(machine_ready_time, job_ready_time)
    
    def schedule_operation(self, job: Job, operation: Operation, machine_id: int):
        """在指定机器上调度工序"""
        start_time = self.get_earliest_start_time(job, machine_id)
        end_time = start_time + operation.processing_time
        
        # 更新机器调度
        self.machine_schedules[machine_id].append(
            (start_time, end_time, job.job_id, operation.operation_id)
        )
        
        # 更新机器可用时间
        self.machine_available_time[machine_id] = end_time
        
        # 更新工件完成时间
        self.job_completion_time[job.job_id] = end_time
        
        # 完成当前工序
        job.complete_operation()
        
        print(f"调度: 工件{job.job_id} 工序{operation.operation_id} -> 机器{machine_id} "
              f"时间[{start_time}, {end_time}]")
    
    def random_schedule(self) -> Dict[int, List[Tuple]]:
        """随机调度算法"""
        print("开始随机调度...")
        
        while True:
            # 获取可执行的工件
            available_jobs = self.get_available_jobs()
            
            if not available_jobs:
                break
            
            # 随机选择一个工件
            selected_job = random.choice(available_jobs)
            
            # 获取该工件的下一个工序
            next_operation = selected_job.get_next_operation()
            
            if next_operation is None:
                continue
            
            # 获取该工序的可行机器
            feasible_machines = self.get_feasible_machines(next_operation)
            
            if not feasible_machines:
                print(f"警告: 工件{selected_job.job_id}工序{next_operation.operation_id}没有可行机器")
                continue
            
            # 随机选择一个可行机器
            selected_machine = random.choice(feasible_machines)
            
            # 调度该工序
            self.schedule_operation(selected_job, next_operation, selected_machine)
        
        print("调度完成!")
        return dict(self.machine_schedules)
    
    def get_makespan(self) -> int:
        """计算完工时间"""
        return max(self.machine_available_time) if self.machine_available_time else 0
    
    def print_schedule(self):
        """打印调度结果"""
        print("\n=== 调度结果 ===")
        for machine_id in range(self.num_machines):
            print(f"机器 {machine_id}:")
            schedule = self.machine_schedules[machine_id]
            if schedule:
                for start, end, job_id, op_id in schedule:
                    print(f"  工件{job_id}工序{op_id}: [{start}, {end}]")
            else:
                print("  空闲")
        print(f"\n总完工时间: {self.get_makespan()}")

# 示例使用
def create_example_jobs() -> List[Job]:
    """创建示例工件数据"""
    jobs = []
    
    # 工件0: 3个工序
    job0_ops = [
        Operation(0, 0, 3, [0, 1]),      # 工序0可在机器0,1上执行
        Operation(0, 1, 2, [1, 2]),      # 工序1可在机器1,2上执行  
        Operation(0, 2, 4, [0, 2])       # 工序2可在机器0,2上执行
    ]
    jobs.append(Job(0, job0_ops))
    
    # 工件1: 3个工序
    job1_ops = [
        Operation(1, 0, 2, [1, 2]),      # 工序0可在机器1,2上执行
        Operation(1, 1, 3, [0, 1]),      # 工序1可在机器0,1上执行
        Operation(1, 2, 1, [2])          # 工序2只能在机器2上执行
    ]
    jobs.append(Job(1, job1_ops))
    
    # 工件2: 2个工序
    job2_ops = [
        Operation(2, 0, 4, [0]),         # 工序0只能在机器0上执行
        Operation(2, 1, 2, [1, 2])       # 工序1可在机器1,2上执行
    ]
    jobs.append(Job(2, job2_ops))
    
    return jobs

if __name__ == "__main__":
    # 设置随机种子以便复现结果
    random.seed(42)
    
    # 创建示例数据
    jobs = create_example_jobs()
    num_machines = 3
    
    # 创建调度器
    scheduler = JobShopScheduler(jobs, num_machines)
    
    # 执行随机调度
    schedule = scheduler.random_schedule()
    
    # 打印结果
    scheduler.print_schedule()
