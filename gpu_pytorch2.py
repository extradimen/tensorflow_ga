import torch
import torch.nn as nn
import torch.optim as optim
import random
import math
from sklearn.metrics import pairwise_distances
import numpy as np
# 设置 GPU 设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Problem:
    def __init__(self, NOBJ, K, BOUND_LOW, BOUND_UP):
        self.NOBJ = NOBJ
        self.K = K
        self.NDIM = NOBJ + K - 1
        self.BOUND_LOW = BOUND_LOW
        self.BOUND_UP = BOUND_UP

    def evaluate(self, individual):
        raise NotImplementedError("evaluate() method is not implemented.")

    def calculate_pf(self, population):
        non_dominated_pop = []
        for i, ind in enumerate(population):
            dominated = False
            for j, other_ind in enumerate(population):
                if i != j and torch.all(ind <= other_ind):
                    dominated = True
                    break
            if not dominated:
                non_dominated_pop.append(ind)
        return non_dominated_pop

    def calculate_igd(self, pf, ref_points):
        distances = pairwise_distances(pf, ref_points, metric='euclidean')
        min_distances = torch.min(distances, dim=0).values
        igd = torch.mean(min_distances)
        return igd

class DTLZ(Problem):
    def __init__(self, NOBJ, K, BOUND_LOW, BOUND_UP, problem_id):
        super().__init__(NOBJ, K, BOUND_LOW, BOUND_UP)
        self.problem_id = problem_id

    def evaluate(self, individual):
        g = torch.sum(torch.square(individual[self.NOBJ-1:] - 0.5))
        f = 0.5 * torch.prod(individual[:self.NOBJ]) * (1 + g)
        return f

class CDTLZ(Problem):
    def __init__(self, NOBJ, K, BOUND_LOW, BOUND_UP, problem_id):
        super().__init__(NOBJ, K, BOUND_LOW, BOUND_UP)
        self.problem_id = problem_id

    def evaluate(self, individual):
        g = torch.sum(torch.square(individual[self.NOBJ-1:] - 0.5) - torch.cos(20 * math.pi * (individual[self.NOBJ-1:] - 0.5)))
        f = 0.5 * torch.prod(individual[:self.NOBJ]) * (1 + g)
        return f

# 定义 DTLZ1, DTLZ2, DTLZ3, DTLZ4 和 C1DTLZ1, C1DTLZ2, C1DTLZ3, C1DTLZ4 对象

class DTLZ1(DTLZ):
    def __init__(self, NOBJ, K):
        super().__init__(NOBJ, K, 0, 1, problem_id=1)

class DTLZ2(DTLZ):
    def __init__(self, NOBJ, K):
        super().__init__(NOBJ, K, 0, 1, problem_id=2)

class DTLZ3(DTLZ):
    def __init__(self, NOBJ, K):
        super().__init__(NOBJ, K, 0, 1, problem_id=3)

class DTLZ4(DTLZ):
    def __init__(self, NOBJ, K):
        super().__init__(NOBJ, K, 0, 1, problem_id=4)

class C1DTLZ1(CDTLZ):
    def __init__(self, NOBJ, K):
        super().__init__(NOBJ, K, 0, 1, problem_id=1)

class C1DTLZ2(CDTLZ):
    def __init__(self, NOBJ, K):
        super().__init__(NOBJ, K, 0, 1, problem_id=2)

class C1DTLZ3(CDTLZ):
    def __init__(self, NOBJ, K):
        super().__init__(NOBJ, K, 0, 1, problem_id=3)

class C1DTLZ4(CDTLZ):
    def __init__(self, NOBJ, K):
        super().__init__(NOBJ, K, 0, 1, problem_id=4)

# 设置参数
NOBJ = 3
K = 10
BOUND_LOW = 0
BOUND_UP = 1
P = [2, 1]
SCALES = [1, 0.5]
CXPB = 1.0
MUTPB = 1.0
eta = 35
iteration = 2000
pop = 100

# 定义问题对象
problem = C1DTLZ1(NOBJ, K)

# 计算参考点
ref_points = []
for p, s in zip(P, SCALES):
    ref_points.append(torch.from_numpy(np.random.uniform(size=(p, problem.NOBJ))) * s)
ref_points = torch.cat(ref_points, dim=0).to(device)

# 初始化种群并将其移动到GPU上
population = torch.rand(pop, problem.NDIM, dtype=torch.float32, device=device)

# NSGA-III算法迭代
for gen in range(iteration):
    # 计算适应度值并将其移动到GPU上
    fitness_values = torch.tensor([problem.evaluate(individual) for individual in population], dtype=torch.float32, device=device)

    # 计算排名和拥挤度距离
    ranked_indices = torch.argsort(fitness_values)
    ranked_pop = population[ranked_indices]
    fitness_ranks = torch.arange(pop, device=device)
    fitness_crowding_distances = torch.zeros(pop, dtype=torch.float32, device=device)

    for obj in range(problem.NOBJ):
        obj_fitness = torch.zeros(pop, dtype=torch.float32, device=device)
        for i in range(pop):
            obj_fitness[i] = fitness_values[ranked_indices[i]][obj]
        fitness_crowding_distances += (obj_fitness[2:] - obj_fitness[:-2]) / (obj_fitness[-1] - obj_fitness[0] + 1e-10)

    # 选择和交叉
    mating_pool = torch.zeros((pop, problem.NDIM), dtype=torch.float32, device=device)
    for i in range(pop):
        selected = torch.multinomial(torch.ones(pop), 2, replacement=False)
        a, b = ranked_pop[selected[0]], ranked_pop[selected[1]]
        child = torch.cat([a[:problem.NOBJ], b[problem.NOBJ:]], dim=0)
        mating_pool[i] = child

    # 变异
    mutated_pop = torch.zeros((pop, problem.NDIM), dtype=torch.float32, device=device)
    for i in range(pop):
        if random.random() < MUTPB:
            mutant = torch.clone(mating_pool[i])
            for j in range(problem.NDIM):
                if random.random() < 1.0 / problem.NDIM:
                    lower = max(problem.BOUND_LOW, mutant[j] - 0.1)
                    upper = min(problem.BOUND_UP, mutant[j] + 0.1)
                    mutant[j] = random.uniform(lower, upper)
            mutated_pop[i] = mutant
        else:
            mutated_pop[i] = mating_pool[i]

    population = torch.cat([ranked_pop[:pop//2], mutated_pop[pop//2:]], dim=0)

    # 计算pf和IGD
    pf = problem.calculate_pf(population)
    pf = torch.tensor(pf, dtype=torch.float32, device=device)
    igd = problem.calculate_igd(pf, ref_points)
    igd = torch.tensor(igd, dtype=torch.float32, device=device)

    print("Generation:", gen, "IGD:", igd)
