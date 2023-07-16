import tensorflow as tf
import numpy as np
from sklearn.metrics import pairwise_distances

# 1. 定义问题参数
NOBJ = 3
K = 10
NDIM = NOBJ + K - 1
BOUND_LOW = 0
BOUND_UP = 1

# 2. 定义C1DTLZ1问题
def c1dtlz1(individual):
    g = tf.reduce_sum(tf.square(individual[NOBJ-1:NDIM] - 0.5) - tf.cos(20 * np.pi * (individual[NOBJ-1:NDIM] - 0.5)), axis=1)
    f = tf.multiply(0.5 * tf.reduce_prod(individual[:NOBJ]), 1 + g)
    return f

# 3. 定义NSGA-III算法参数
CXPB = 1.0
MUTPB = 1.0
eta = 35
iteration = 2000
pop = 100

# 4. 计算参考点 ref_points
P = [2, 1]
SCALES = [1, 0.5]
ref_points = []
for p, s in zip(P, SCALES):
    points = tf.random.uniform((pop, NOBJ - 1), 0, s)
    points = tf.concat([points, tf.zeros((pop, 1))], axis=1)
    ref_points.append(points)

# 5. 计算pareto frontier (pf)
def calculate_pf(population):
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

# 6. 计算IGD
def calculate_igd(pf, ref_points):
    distances = pairwise_distances(pf, ref_points, metric='euclidean')
    min_distances = tf.reduce_min(distances, axis=0)
    igd = tf.reduce_mean(min_distances)
    return igd

# 初始化种群
population = tf.random.uniform((pop, NDIM), BOUND_LOW, BOUND_UP)

# NSGA-III算法迭代
for gen in range(iteration):
    # 计算适应度值
    fitness_values = tf.map_fn(c1dtlz1, population)

    # 计算排名和拥挤度距离
    ranked_indices = tf.argsort(fitness_values)
    ranked_pop = tf.gather(population, ranked_indices)
    fitness_ranks = tf.range(pop)
    fitness_crowding_distances = tf.zeros(pop, dtype=tf.float32)

    for obj in range(NOBJ):
        obj_fitness = tf.gather(fitness_values[:, obj], ranked_indices)
        fitness_crowding_distances += (obj_fitness[2:] - obj_fitness[:-2]) / (obj_fitness[-1] - obj_fitness[0] + 1e-10)

    # 选择和交叉
    mating_pool = tf.zeros((pop, NDIM), dtype=tf.float32)
    for i in range(pop):
        selected = tf.random.choice(pop, size=2, replace=False)
        a, b = ranked_pop[selected[0]], ranked_pop[selected[1]]
        child = tf.concat([a[:NOBJ], b[NOBJ:]], axis=0)
        mating_pool[i] = child

    # 变异
    mutated_pop = tf.zeros((pop, NDIM), dtype=tf.float32)
    for i in range(pop):
        if tf.random.uniform(()) < MUTPB:
            mutant = tf.Variable(mating_pool[i])
            for j in range(NDIM):
                if tf.random.uniform(()) < 1.0 / NDIM:
                    lower = tf.maximum(BOUND_LOW, mutant[j] - 0.1)
                    upper = tf.minimum(BOUND_UP, mutant[j] + 0.1)
                    mutant[j].assign(tf.random.uniform((), lower, upper))
            mutated_pop[i] = mutant
        else:
            mutated_pop[i] = mating_pool[i]

    population = tf.concat([ranked_pop[:pop//2], mutated_pop[pop//2:]], axis=0)

    # 计算pf和IGD
    pf = calculate_pf(population)
    igd = calculate_igd(pf, ref_points)
    print("Generation:", gen, "IGD:", igd)
