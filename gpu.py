import tensorflow as tf
import numpy as np
from sklearn.metrics import pairwise_distances

# 设置 GPU 设备
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

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
                if i != j and tf.reduce_all(ind <= other_ind):
                    dominated = True
                    break
            if not dominated:
                non_dominated_pop.append(ind)
        return non_dominated_pop

    def calculate_igd(self, pf, ref_points):
        distances = pairwise_distances(pf, ref_points, metric='euclidean')
        min_distances = tf.reduce_min(distances, axis=0)
        igd = tf.reduce_mean(min_distances)
        return igd

class DTLZ(Problem):
    def __init__(self, NOBJ, K, BOUND_LOW, BOUND_UP, problem_id):
        super().__init__(NOBJ, K, BOUND_LOW, BOUND_UP)
        self.problem_id = problem_id

    def evaluate(self, individual):
        g = tf.reduce_sum(tf.square(individual[self.NOBJ-1:] - 0.5))
        f = tf.multiply(0.5 * tf.reduce_prod(individual[:self.NOBJ]), (1 + g))
        return f

class CDTLZ(Problem):
    def __init__(self, NOBJ, K, BOUND_LOW, BOUND_UP, problem_id):
        super().__init__(NOBJ, K, BOUND_LOW, BOUND_UP)
        self.problem_id = problem_id

    def evaluate(self, individual):
        g = tf.reduce_sum(tf.square(individual[self.NOBJ-1:] - 0.5) - tf.cos(20 * np.pi * (individual[self.NOBJ-1:] - 0.5)))
        f = tf.multiply(0.5 * tf.reduce_prod(individual[:self.NOBJ]), (1 + g))
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

# NSGA-III算法迭代
for gen in range(iteration):
    # 初始化种群并将其移动到GPU上
    population = tf.random.uniform((pop, problem.NDIM), problem.BOUND_LOW, problem.BOUND_UP)
    population = tf.Variable(population, dtype=tf.float32)

    # 计算适应度值并将其移动到GPU上
    fitness_values = tf.map_fn(problem.evaluate, population)
    fitness_values = tf.Variable(fitness_values, dtype=tf.float32)

    # 计算排名和拥挤度距离
    ranked_indices = tf.argsort(fitness_values)
    ranked_pop = tf.gather(population, ranked_indices)
    fitness_ranks = tf.range(pop)
    fitness_crowding_distances = tf.zeros(pop, dtype=tf.float32)

    for obj in range(problem.NOBJ):
        obj_fitness = tf.gather(fitness_values[:, obj], ranked_indices)
        fitness_crowding_distances += (obj_fitness[2:] - obj_fitness[:-2]) / (obj_fitness[-1] - obj_fitness[0] + 1e-10)

    # 选择和交叉
    mating_pool = tf.zeros((pop, problem.NDIM), dtype=tf.float32)
    for i in range(pop):
        selected = tf.random.choice(pop, size=2, replace=False)
        a, b = ranked_pop[selected[0]], ranked_pop[selected[1]]
        child = tf.concat([a[:problem.NOBJ], b[problem.NOBJ:]], axis=0)
        mating_pool[i] = child

    # 变异
    mutated_pop = tf.zeros((pop, problem.NDIM), dtype=tf.float32)
    for i in range(pop):
        if tf.random.uniform(()) < MUTPB:
            mutant = tf.Variable(mating_pool[i])
            for j in range(problem.NDIM):
                if tf.random.uniform(()) < 1.0 / problem.NDIM:
                    lower = tf.maximum(problem.BOUND_LOW, mutant[j] - 0.1)
                    upper = tf.minimum(problem.BOUND_UP, mutant[j] + 0.1)
                    mutant[j].assign(tf.random.uniform((), lower, upper))
            mutated_pop[i] = mutant
        else:
            mutated_pop[i] = mating_pool[i]

    population = tf.concat([ranked_pop[:pop//2], mutated_pop[pop//2:]], axis=0)

    # 计算pf和IGD
    pf = problem.calculate_pf(population)
    pf = tf.Variable(pf, dtype=tf.float32)
    igd = problem.calculate_igd(pf, ref_points)
    igd = tf.Variable(igd, dtype=tf.float32)

    print("Generation:", gen, "IGD:", igd)
