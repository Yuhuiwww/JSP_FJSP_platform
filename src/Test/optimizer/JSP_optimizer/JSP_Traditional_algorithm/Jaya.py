import time
import random
import numpy as np
from Problem.JSP import JSP
from Test.optimizer.JSP_optimizer.JSP_Traditional_algorithm.Bassic_algorithm import Basic_Algorithm
class Jaya(Basic_Algorithm):
    def __init__(self, config):
        self.config = config
        self.population_size = 100
        self.global_best=[]#global optimum solution (GOS)
        self.global_best_fit = 100000  # global optimum solution (GOS)
        self.worst_fit = 0
        self.global_worst=[]#the worst
        self.best_number=0
        self.worst_number=0


    def generate_init_pop(self,population_size, j, m):
        population_list = np.zeros((population_size, int(j * m)), dtype=np.int32)
        indivudial=np.zeros(j*m)#Create an array of zeros of size j * m
        strat=0
        for i in range(j):
            indivudial[strat:strat+m]=i
            strat+=m
        for i in range(population_size):
            np.random.shuffle(indivudial)
            population_list[i]=indivudial
        return population_list
    def far_close_process(self,population_list,m_job):
        #Pick an individual who, with a certain probability, inherits the BEST and, with a certain probability, stays away from the WORST
        best_Individual=self.global_best
        worst_Individual=self.global_worst
        new_pop = []
        for num1,individual in enumerate(population_list):
            copy=individual.copy()
            if 0.8 >= 0.5:  # Perform a stay away operation with a probability of stay away of 0.5.
                all_list = []
                out_list = []  # Popup subscripts for the same task location
                for num, i in enumerate(worst_Individual):  # Find similarities and differences
                    all_list.append(num)
                    if individual[num] == i:
                        out_list.append(num)  # Same tasking as the worst DNA in the same location, away from it to be different from it
                back_up_list = [i for i in all_list if i not in out_list]  # Alternative sets, which refer to task subscripts that are different from the sort position of the worst DNA.
                if len(out_list) <= len(back_up_list):  # When there are more different positions than the same position
                    while len(out_list) > 0:
                        exchange1 = out_list.pop()
                        exchange2 = random.choice(back_up_list)
                        back_up_list.remove(exchange2)
                        temp = copy[exchange1]  # Get specific exchange points based on subscripts
                        copy[exchange1] = individual[exchange2]  #
                        copy[exchange2] = temp
                else:  #
                    if len(out_list) == len(all_list):  # Iterate to the worst value and just skip over it:
                        continue
                    else:
                        while len(out_list) > 1:
                            exchange1 = out_list.pop()
                            exchange2 = out_list.pop()
                            temp = copy[exchange1]  # exchange point
                            copy[exchange1] = individual[exchange2]
                            copy[exchange2] = temp
                        if len(out_list) == 1:
                            exchange1 = out_list.pop()
                            exchange2 = random.choice(back_up_list)
                            back_up_list.remove(exchange2)
                            temp = copy[exchange1]  # exchange point
                            copy[exchange1] = individual[exchange2]
                            copy[exchange2] = temp
                        else:
                            pass
                            # print('end of exchange')
            indivial = copy
                # Proximity operation
            if 0.9 >= 0.5:  # print('come close')
                select_list = []
                for selet in range(m_job):  # Selection of proximity points with 0.4 probability
                    if random.random() > 0.4:
                        select_list.append(selet)  # Mission points selected for assignment in the same location as the best DNA
                current_list=[]
                for i in select_list:  # Close to the best point.
                    index_select = np.where(i== best_Individual)[0] # Where the replacement value was originally located when it was close
                    #Remove the value of i from the indivial
                    individual = individual[individual != i]
                    for j, index in enumerate(index_select):
                        individual = np.insert(individual, index, i)

            individual=indivial  # To prepend 0 to indivial
            population_list[num1]=individual
            return population_list

    def run_episode(self, problem,dataset):
        max_time=self.config.Pn_j*self.config.Pn_m

        with open('./Result/JSP/'+self.config.optimizer + '-20 times runing solution.txt', 'a') as f:
            f.write(self.config.test_datas_type + str(self.config.Pn_j) + 'x' + str(self.config.Pn_m) + '\n')
            best_minfiness = 9999999
            for repeat in range(10):
                min_makespan=999999
                start_time=time.time()
                population_list=self.generate_init_pop(self.population_size, self.config.Pn_j, self.config.Pn_m)
                for run in range(1000000):
                    run+=1
                    for i in range(len(population_list)):
                       fitness=problem.cal_objective(population_list[i],dataset)
                       if fitness<self.global_best_fit:
                           self.global_best_fit=fitness
                           self.global_best=population_list[i]
                           self.best_number=i
                       if fitness>self.worst_fit:
                           self.worst_fit=fitness
                           self.global_worst=population_list[i]
                           self.worst_number=i

                    population_list=self.far_close_process(population_list,self.config.Pn_j)
                    makespan_list = np.zeros(len(population_list))
                    for k in range(len(population_list)):
                        makespan_list[k] = problem.cal_objective(population_list[k], dataset)
                        if makespan_list[k] < min_makespan:
                            min_makespan = makespan_list[k]
                        if best_minfiness>min_makespan:
                            best_minfiness = min_makespan
                    end_time = time.time()
                    if end_time - start_time > max_time:
                        break
                f.write(str(min_makespan) + '\n')
                print('min_makespan',min_makespan)
        return best_minfiness

