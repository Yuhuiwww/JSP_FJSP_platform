import copy
from random import random
import random
import numpy as np
import time
from Problem.JSP import JSP
from Test.optimizer.JSP_optimizer.JSP_Traditional_algorithm.Bassic_algorithm import Basic_Algorithm
class ABC(Basic_Algorithm):
    def __init__(self, config):
        self.config = config
        self.pt = config.Pn_j  # processing time
        self.ms = config.Pn_m  # machine sequence
        self.J_num = config.Pn_j  # Job num
        self.M_num = config.Pn_m  # Machine num
        self.population_size = 100
        self.empoyed_beesrate = 0.5
        self.limit=10

    def initJobSequence(self):
        population_list = np.zeros((self.population_size, int(self.J_num * self.M_num)), dtype=np.int32)
        Jobsequence = np.zeros(self.J_num * self.M_num)
        start = 0
        for i in range(self.J_num):
            Jobsequence[start:start + self.M_num] = i
            start += self.M_num

        for i in range(self.population_size):
            np.random.shuffle(Jobsequence)
            population_list[i] = Jobsequence

        return population_list
    def empoyed_bees(self, config, individual):
        self.n_dim = config.Pn_j * config.Pn_m  # 搜索空间维度
        # init_solution = self.initJobSequence(config)
        for j in range(self.n_dim):
            individual[j] = individual[j] % config.Pn_j  # convert to job number format, every job appears m times
        return individual
    def swap(self,arr, idx1, idx2):
        arr[idx1], arr[idx2] = arr[idx2], arr[idx1]
    def onlooker_bees(self, individual):
        positions = random.sample(range(len(individual)), 2)
        self.swap(individual,positions[0], positions[1])
        return individual
    def scout_bees(self,config,population_list,data):
        problem = eval(config.problem_name)(copy.deepcopy(config.Pn_j), copy.deepcopy(config.Pn_m))
        population_list1=copy.deepcopy(population_list)
        for i in range(self.population_size):
            positions = random.sample(range(len(population_list[i])), 2)
            iter=0
            while iter <self.limit:
                num = random.randint(0, 1)
                if num==0:
                    self.swap(population_list[i],positions[0],positions[1])
                else:
                    new_population_list = np.delete(population_list[i], positions[1])
                    a=population_list[i][positions[1]]
                    population_list[i] = np.insert(new_population_list, positions[0], a)

                iter+=1
                currentmakespan = problem.cal_objective(population_list[i],data)
            oldmakespan = problem.cal_objective(population_list1[i], data)
            if currentmakespan > oldmakespan:
                population_list[i] = copy.deepcopy(population_list1[i])
        return population_list
    def run_episode(self, problem,dataset):
        best_minfiness=9999999
        max_run_time =self.config.Pn_j*self.config.Pn_m*0.1
        with open('./Result/JSP/'+self.config.optimizer + '-20 times runing solution.txt', 'a') as f:
            f.write(self.config.test_datas_type + str(self.config.Pn_j) + 'x' + str(self.config.Pn_m) + '\n')
            for repeat in range(10):
                min_makespan = 999999
                start_time = time.time()
                population_list = self.initJobSequence()
                for run in range(100000000):
                    for i in range(self.population_size):
                        current = copy.deepcopy(population_list[i])
                        currentmakespan = problem.cal_objective(current, dataset)
                        if( i < int(self.empoyed_beesrate * self.population_size ) ):
                            population_list[i] = self.empoyed_bees(self.config, population_list[i])
                        else:
                            population_list[i] = self.onlooker_bees(population_list[i])
                        popmakespan = problem.cal_objective(population_list[i], dataset)
                        if (currentmakespan<popmakespan):
                            population_list[i]=current
                    population_list=self.scout_bees(self.config,population_list,dataset)
                    for i in range(self.population_size):
                        makespan = problem.cal_objective(population_list[i], dataset)
                        if (min_makespan > makespan):
                            min_makespan = makespan
                        if best_minfiness>min_makespan:
                            best_minfiness = min_makespan
                    if time.time() - start_time > max_run_time:
                        break
                f.write(str(min_makespan) + '\n')
                print("min_min_makespan", min_makespan)
        return best_minfiness
