import B_ques_four_main as main
import matplotlib.pyplot as plt

import random
from deap import algorithms, base, creator, tools


# 定义多目标优化问题
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -1.0,-1.0))  # 定义多目标优化的权重
creator.create("Individual", list, fitness=creator.FitnessMin)  # 定义个体类型

# 初始化种群和个体
toolbox = base.Toolbox()

# 定义自变量 x 的属性生成函数和取值范围
toolbox.register("attr_float_beta", random.uniform, 20, 70)
# 定义自变量 y 的属性生成函数和取值范围
toolbox.register("attr_float_d1", random.uniform, 50,250)
toolbox.register("attr_float_d2", random.uniform, 50,250)
toolbox.register("attr_float_d3", random.uniform, 50,250)
toolbox.register("attr_float_d4", random.uniform, 50,250)
toolbox.register("attr_float_d5", random.uniform, 50,250)
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_float_beta, toolbox.attr_float_d1,toolbox.attr_float_d2,toolbox.attr_float_d3,toolbox.attr_float_d4), n=1)  # 定义个体的基因数量

toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # 定义种群
# 定义评估函数
def evaluate(individual):
    beta, d1,d2,d3,d4 = individual
    res=main.get_all_res(beta,d1,d2,d3,d4)
    return res[2], res[3], res[4]

toolbox.register("evaluate", evaluate)  # 注册评估函数

# 定义交叉和变异操作
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.1)
toolbox.register("select", tools.selNSGA2)  # 使用NSGA-II选择算法

# 设置算法参数
population_size = 30 # 种群大小
max_generations = 10  # 最大迭代次数
crossover_probability = 0.7  # 交叉概率
mutation_probability = 0.2  # 变异概率


# 创建种群
population = toolbox.population(n=population_size)

# 创建种群
population = toolbox.population(n=population_size)
best_fitness_values = []
# 运行多目标优化算法
for generation in range(max_generations):
    # 输出当前代数以及最优个体的适应度值
    best_individuals = tools.selBest(population, k=1)

    best_objectives = [individual.fitness.values for individual in best_individuals]
    best_fitness_values.append(best_objectives[0])
    print("Generation:", generation + 1)
    print(best_fitness_values)
    print("Best objectives:", best_objectives)
    print("Best parameters:",  best_individuals)
    # 运行一个迭代的算法步骤
    algorithms.eaSimple(population, toolbox, cxpb=crossover_probability, mutpb=mutation_probability,
                        ngen=1, verbose=False)
# 输出最优解
best_individuals = tools.selBest(population, k=1)
best_objectives = [individual.fitness.values for individual in best_individuals]
print("Best objectives:", best_objectives)
print("Best parameters:",  best_individuals)
print(best_fitness_values)
plt.plot(best_fitness_values)
plt.xlabel('Generation')
plt.ylabel('Fitness')
plt.title('Fitness Curve')
plt.show()