import copy
import time
import numpy as np
import random
from Problem.JSP import JSP
from Test.optimizer.JSP_optimizer.JSP_Traditional_algorithm.Bassic_algorithm import Basic_Algorithm


class PSO(Basic_Algorithm):
    def __init__(self, config):
       self.config = config
       self.w = 0.8  # Inertia Weight
       self.c1 = 0.1  # Cognitive Weight
       self.c2 = 0.1  # Social Weight
       self.particals = 100  # Swarm Size
       self.cycle_ = 30  # Number of Iterations
       self.mesh_div = 10  # Number of Grid Divisions
       self.thresh = 300  # External Archiving Threshold
       self.min_ = np.array([0, 0])  # Minimum Particle Coordinate Value
       self.max_ = np.array([10, 10])  # Maximum Particle Coordinate Value
       self.num_iteration=1000
       self.mutation_rate = 0.1

    def generate_initial_population(self, num_jobs, num_machines):
        population = np.zeros((self.particals, int(num_jobs *num_machines)), dtype=np.int32)
        sequence = np.zeros(num_jobs *num_machines)
        start = 0
        for i in range(num_jobs):
            sequence[start:start + num_machines] = i
            start += num_machines

        for i in range(self.particals):
            np.random.shuffle(sequence)
            population[i] = sequence

        return population

    def update_velocity(self,velocity, particle_pos, global_pos, c1, c2):
        r1 = np.random.rand(*velocity.shape)
        r2 = np.random.rand(*velocity.shape)
        return velocity + c1 * r1 * (particle_pos - velocity) + c2 * r2 * (global_pos - velocity)

    def update_position(self,position, velocity):
        return position + velocity

    def mutate(self,individual,position,mutation_rate):
        if random.random() < mutation_rate:
           for len1 in range(len(position)):
              if position[len1] >= self.config.Pn_j:
                individual[0], individual[len1] = individual[len1], individual[0]


    # def run_episode(self, problem,dataset):
    #     config = self.config
    #     num_jobs = config.Pn_j
    #     num_machines = config.Pn_m
    #     num_particles=self.particals
    #     population = self.generate_initial_population(num_jobs, num_machines)
    #     position=np.zeros((num_particles, num_jobs*num_machines), dtype=int)
    #     best_fitness = float('inf')
    #     global_best_fitness = float('inf')
    #     global_best_individual = None
    #     min_makespan=99999
    #     velocities = np.zeros((num_particles, num_jobs*num_machines), dtype=int)
    #     problem = eval(config.problem_name)(copy.deepcopy(config.Pn_j), copy.deepcopy(config.Pn_m))
    #     for iteration in range(self.num_iteration):
    #         for i, individual in enumerate(population):
    #             fitness = problem.cal_objective(individual, dataset)
    #             if fitness < best_fitness:
    #                 best_fitness = fitness
    #                 best_individual = individual
    #
    #             if fitness < global_best_fitness:
    #                 global_best_fitness = fitness
    #                 global_best_individual = individual
    #
    #             velocities[i] = self.update_velocity(velocities[i], individual, global_best_individual, self.c1, self.c2)
    #             position[i] = self.update_position(individual, velocities[i])
    #             self.mutate(population[i], position[i],self.mutation_rate)
    #             if(min_makespan>fitness):
    #                 min_makespan=fitness
    #     print('best_fitness',min_makespan)
    #     return min_makespan

    def run_episode(self, problem,dataset):
        config = self.config
        num_jobs = config.Pn_j
        num_machines = config.Pn_m
        num_particles=self.particals
        position = np.zeros((num_particles, num_jobs * num_machines), dtype=int)
        best_fitness = float('inf')
        global_best_fitness = float('inf')
        global_best_individual = None
        best_minfiness=9999999
        velocities = np.zeros((num_particles, num_jobs * num_machines), dtype=int)
        problem = eval(config.problem_name)(copy.deepcopy(config.Pn_j), copy.deepcopy(config.Pn_m))
        max_time = config.Pn_j * config.Pn_m
        with open('./Result/JSP/'+self.config.optimizer + '-20 times runing solution.txt', 'a') as f:
            f.write(self.config.test_datas_type + str(self.config.Pn_j) + 'x' + str(self.config.Pn_m) + '\n')
            for repeat in range(10):
                repeat += 1
                min_makespan = 999999
                start_time = time.time()
                population = self.generate_initial_population(num_jobs, num_machines)
                for iteration in range(10000000):
                    for i, individual in enumerate(population):
                        fitness = problem.cal_objective(individual, dataset)
                        if fitness < best_fitness:
                            best_fitness = fitness
                            best_individual = individual
                        if fitness < global_best_fitness:
                            global_best_fitness = fitness
                            global_best_individual = individual

                        velocities[i] = self.update_velocity(velocities[i], individual, global_best_individual, self.c1, self.c2)
                        position[i] = self.update_position(individual, velocities[i])
                        self.mutate(population[i], position[i],self.mutation_rate)
                        if(min_makespan>fitness):
                            min_makespan=fitness
                        if best_minfiness>min_makespan:
                            best_minfiness = min_makespan
                    end_time = time.time()
                    if end_time - start_time > max_time:
                        break
                f.write(str(min_makespan) + '\n')
                print('best_fitness', min_makespan)
        return best_minfiness