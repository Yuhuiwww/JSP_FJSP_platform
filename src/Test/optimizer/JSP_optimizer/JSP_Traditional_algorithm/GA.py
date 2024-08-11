# -*- coding: utf-8 -*-
import copy
import time
import numpy as np
from Problem.JSP import JSP
from Test.optimizer.JSP_optimizer.JSP_Traditional_algorithm.Bassic_algorithm import Basic_Algorithm


class GA(Basic_Algorithm):
    def __init__(self, config):
        self.config = config
        self.population_size = 100
        self.crossover_rate = 1.0
        self.mutation_rate = 0.15
        self.mutation_selection_rate = 0.15
        self.num_mutation_jobs = round(config.Pn_j * config.Pn_m * self.mutation_selection_rate)
        self.num_iteration = 1000
        self.min_makespan_record = []
        self.avg_makespan_record = []
        self.min_makespan = 9999999

    def generate_init_pop(self,population_size, j, m):
        population_list = np.zeros((population_size, int(j * m)), dtype=np.int32)
        chromosome = np.zeros(j * m)
        start = 0
        for i in range(j):
            chromosome[start:start + m] = i
            start += m

        for i in range(population_size):
            np.random.shuffle(chromosome)
            population_list[i] = chromosome

        return population_list

    def two_point_crossover(populationlist, crossover_rate):
        parentlist = copy.deepcopy(populationlist)
        childlist = copy.deepcopy(populationlist)
        for i in range(len(parentlist), 2):
            sample_prob = np.random.rand()
            if sample_prob <= crossover_rate:
                cutpoint = np.random.choice(2, parentlist.shape[1], replace=False)
                cutpoint.sort()
                parent_1 = parentlist[i]
                parent_2 = parentlist[i + 1]
                child_1 = copy.deepcopy(parent_1)
                child_2 = copy.deepcopy(parent_2)
                child_1[cutpoint[0]:cutpoint[1]] = parent_2[cutpoint[0]:cutpoint[1]]
                child_2[cutpoint[0]:cutpoint[1]] = parent_1[cutpoint[0]:cutpoint[1]]
                childlist[i] = child_1
                childlist[i + 1] = child_2

        return parentlist, childlist

    def job_order_crossover(self, populationlist, j, crossover_rate):
        parentlist = copy.deepcopy(populationlist)
        childlist = copy.deepcopy(populationlist)
        for i in range(len(parentlist), 2):
            sample_prob = np.random.rand()
            if sample_prob <= crossover_rate:
                parent_id = np.random.choice(len(populationlist), 2, replace=False)
                select_job = np.random.choice(j, 1, replace=False)[0]
                child_1 = self.job_order_implementation(parentlist[parent_id[0]], parentlist[parent_id[1]], select_job)
                child_2 = self.job_order_implementation(parentlist[parent_id[1]], parentlist[parent_id[0]], select_job)
                childlist[i] = child_1
                childlist[i + 1] = child_2

        return parentlist, childlist

    def job_order_implementation(parent1, parent2, select_job):
        other_job_order = []
        child = np.zeros(len(parent1))
        for j in parent2:
            if j != select_job:
                other_job_order.append(j)
        k = 0
        for i, j in enumerate(parent1):
            if j == select_job:
                child[i] = j
            else:
                child[i] = other_job_order[k]
                k += 1

        return child

    def repair(chromosome, j, m):
        job_count = np.zeros(j)
        for j in chromosome:
            job_count[j] += 1

        job_count = job_count - m

        much_less = [[], []]
        is_legall = True
        for j, count in enumerate(job_count):
            if count > 0:
                is_legall = False
                much_less[0].append(j)
            elif count < 0:
                is_legall = False
                much_less[1].append(j)

        if is_legall == False:
            for m in much_less[0]:
                for j in range(len(chromosome)):
                    if chromosome[j] == m:
                        less_id = np.random.choice(len(much_less[1]), 1)[0]
                        chromosome[j] = much_less[1][less_id]
                        job_count[m] -= 1
                        job_count[much_less[1][less_id]] += 1

                        if job_count[much_less[1][less_id]] == 0:
                            much_less[1].remove(much_less[1][less_id])

                        if job_count[m] == 0:
                            break

    #For childlist mutations
    def mutation(self, childlist, num_mutation_jobs, mutation_rate, dataset, config):
        current_childlist = copy.deepcopy(childlist)
        p_t, m_seq = dataset[0], dataset[1]
        for chromosome in childlist:
            sample_prob = np.random.rand()
            if sample_prob <= mutation_rate:
                mutationpoints = np.random.choice(len(chromosome), num_mutation_jobs, replace=False)
                chrom_copy = copy.deepcopy(chromosome)
                for i in range(len(mutationpoints) - 1):
                    chromosome[mutationpoints[i + 1]] = chrom_copy[mutationpoints[i]]

                chromosome[mutationpoints[0]] = chrom_copy[mutationpoints[-1]]
        makespan_list = np.zeros(len(childlist))
        # Configuring problem_name selection issues
        problem = eval(config.problem_name)(copy.deepcopy(config.Pn_j), copy.deepcopy(config.Pn_m))
        for i, chromosome in enumerate(childlist):
            makespan_list[i] = problem.cal_objective(chromosome, dataset)

        num_all_mut = int(0.1 * len(childlist))
        zipped = list(zip(makespan_list, np.arange(len(makespan_list))))
        sorted_zipped = sorted(zipped, key=lambda x: x[0])
        zipped = zip(*sorted_zipped)
        partial_mut_id = np.asarray(list(zipped)[1])[:-num_all_mut]
        all_mut = self.generate_init_pop(num_all_mut, p_t.shape[0], p_t.shape[1])
        childlist = np.concatenate((all_mut, copy.deepcopy(childlist)[partial_mut_id]), axis=0)


    def selection(self,populationlist, makespan_list):
        num_self_select = int(0.2 * len(populationlist) / 2)
        num_roulette_wheel = int(len(populationlist) / 2) - num_self_select
        zipped = list(zip(makespan_list, np.arange(len(makespan_list))))
        sorted_zipped = sorted(zipped, key=lambda x: x[0])
        zipped = zip(*sorted_zipped)
        self_select_id = np.asarray(list(zipped)[1])[:num_self_select]

        makespan_list = 1 / makespan_list
        selection_prob = makespan_list / sum(makespan_list)
        roulette_wheel_id = np.random.choice(len(populationlist), size=num_roulette_wheel, p=selection_prob)
        new_population = np.concatenate(
            (copy.deepcopy(populationlist)[self_select_id], copy.deepcopy(populationlist)[roulette_wheel_id]), axis=0)

        return new_population

    def binary_selection(self,populationlist, makespan_list):
        new_population = np.zeros((int(len(populationlist) / 2), populationlist.shape[1]), dtype=np.int32)

        num_self_select = int(0.1 * len(populationlist) / 2)
        num_binary = int(len(populationlist) / 2) - num_self_select
        zipped = list(zip(makespan_list, np.arange(len(makespan_list))))
        sorted_zipped = sorted(zipped, key=lambda x: x[0])
        zipped = zip(*sorted_zipped)
        self_select_id = np.asarray(list(zipped)[1])[:num_self_select]

        for i in range(num_binary):
            select_id = np.random.choice(len(makespan_list), 2, replace=False)
            if makespan_list[select_id[0]] < makespan_list[select_id[1]]:
                new_population[i] = populationlist[select_id[0]]
            else:
                new_population[i] = populationlist[select_id[1]]

        new_population[-num_self_select:] = copy.deepcopy(populationlist)[self_select_id]

        return new_population

    # def run_episode(self, problem,dataset):
    #     config = self.__config
    #     population_size = self.population_size
    #     population_list = self.generate_init_pop(population_size,config.Pn_j, config.Pn_m)
    #     # Configure problem_name
    #     # problem = eval(config.problem_name)(copy.deepcopy(config.Pn_j),copy.deepcopy(config.Pn_m))
    #     min_makespan=999999
    #     for i in tqdm(range(self.num_iteration)):
    #         parentlist, childlist = self.job_order_crossover(population_list, config.Pn_j, self.crossover_rate)
    #         self.mutation(childlist, self.num_mutation_jobs, self.mutation_rate, dataset,config)
    #         population_list = np.concatenate((parentlist, childlist), axis=0)
    #         makespan_list = np.zeros(len(population_list))
    #
    #         for k in range(len(population_list)):
    #             makespan_list[k] = problem.cal_objective(population_list[k], dataset)
    #             if makespan_list[k] < self.min_makespan:
    #                 makespan = makespan_list[k]
    #                 best_job_order = population_list[k]
    #         population_list = self.binary_selection(population_list, makespan_list)
    #         self.min_makespan_record.append(makespan)
    #         i += 1
    #         if(min_makespan>makespan):
    #             min_makespan=makespan
    #         print(min_makespan)
    #
    #     return min_makespan
    def run_episode(self, problem, dataset):
        config = self.config
        population_size = self.population_size
        max_time = config.Pn_j * config.Pn_m
        with open('./Result/JSP/'+self.config.optimizer + '-times runing solution.txt', 'a') as f:
            f.write(self.config.test_datas_type + str(self.config.Pn_j) + 'x' + str(self.config.Pn_m) + '\n')
            best_minfiness=9999999
            for repeat in range(10):
                min_makespan = 999999
                repeat += 1
                start_time = time.time()
                population_list = self.generate_init_pop(population_size, config.Pn_j, config.Pn_m)
                for run in range(100000):
                    # Configure problem_name
                    # problem = eval(config.problem_name)(copy.deepcopy(config.Pn_j),copy.deepcopy(config.Pn_m))
                    parentlist, childlist = self.job_order_crossover(population_list, config.Pn_j, self.crossover_rate)
                    self.mutation(childlist, self.num_mutation_jobs, self.mutation_rate, dataset, config)
                    population_list = np.concatenate((parentlist, childlist), axis=0)
                    makespan_list = np.zeros(len(population_list))
                    for k in range(len(population_list)):
                        makespan_list[k] = problem.cal_objective(population_list[k], dataset)
                        if makespan_list[k] < min_makespan:
                            min_makespan = makespan_list[k]
                    population_list = self.binary_selection(population_list, makespan_list)
                    for k in range(len(population_list)):
                        makespan = problem.cal_objective(population_list[k], dataset)
                        if (makespan < min_makespan):
                            min_makespan = makespan
                        if best_minfiness>min_makespan:
                            best_minfiness = min_makespan
                    end_time = time.time()
                    if end_time - start_time > max_time:
                        break

                f.write(str(min_makespan) + '\n')
                print('min_makespan', min_makespan)
        return best_minfiness
