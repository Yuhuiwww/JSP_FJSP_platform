import time
import random
import numpy as np
from Problem.JSP import JSP
from Test.optimizer.JSP_optimizer.JSP_Traditional_algorithm.Bassic_algorithm import Basic_Algorithm
class Jaya(Basic_Algorithm):
    def __init__(self, config):
        self.config = config
        self.population_size = 100
        self.global_best=[]#全局最优解
        self.global_best_fit = 100000  # 全局最优解
        self.worst_fit = 0
        self.global_worst=[]#最差
        self.best_number=0
        self.worst_number=0


    def generate_init_pop(self,population_size, j, m):
        population_list = np.zeros((population_size, int(j * m)), dtype=np.int32)
        indivudial=np.zeros(j*m)#创建一个大小为 j * m 的零数组
        strat=0
        for i in range(j):
            indivudial[strat:strat+m]=i
            strat+=m
        for i in range(population_size):
            np.random.shuffle(indivudial)
            population_list[i]=indivudial
        return population_list
    def far_close_process(self,population_list,m_job):
        #挑选一个个体，按一定的概率继承best，按一定的概率，远离worst
        best_Individual=self.global_best
        worst_Individual=self.global_worst
        new_pop = []
        for num1,individual in enumerate(population_list):
            copy=individual.copy()
            if 0.8 >= 0.5:  # 进行远离操作,远离概率0.5
                all_list = []
                out_list = []  # 弹出相同任务位置的下标
                for num, i in enumerate(worst_Individual):  # 找出相同与不同
                    all_list.append(num)
                    if individual[num] == i:
                        out_list.append(num)  # 与最差的DNA在同一个位置的任务分配相同，远离它要和它不一样
                back_up_list = [i for i in all_list if i not in out_list]  # 备选集，指的是与最差DNA的排序位置不同的任务下标。
                if len(out_list) <= len(back_up_list):  # 不同位置多于相同位置时
                    while len(out_list) > 0:
                        exchange1 = out_list.pop()
                        exchange2 = random.choice(back_up_list)
                        back_up_list.remove(exchange2)
                        temp = copy[exchange1]  # 根据下标，获取具体的交换点
                        copy[exchange1] = individual[exchange2]  #
                        copy[exchange2] = temp
                else:  #
                    if len(out_list) == len(all_list):  # 遍历到最差值，直接跳过就好:
                        continue
                    else:
                        while len(out_list) > 1:
                            exchange1 = out_list.pop()
                            exchange2 = out_list.pop()
                            temp = copy[exchange1]  # 交换点
                            copy[exchange1] = individual[exchange2]
                            copy[exchange2] = temp
                        if len(out_list) == 1:
                            exchange1 = out_list.pop()
                            exchange2 = random.choice(back_up_list)
                            back_up_list.remove(exchange2)
                            temp = copy[exchange1]  # 交换点
                            copy[exchange1] = individual[exchange2]
                            copy[exchange2] = temp
                        else:
                            pass
                            # print('交换完毕')
            indivial = copy
                # 靠近操作
            if 0.9 >= 0.5:  # print('发生靠近')
                select_list = []
                for selet in range(m_job):  # 按0.4概率选择接近点
                    if random.random() > 0.4:
                        select_list.append(selet)  # 被选中分配位置和最佳DNA一样的任务点
                current_list=[]
                for i in select_list:  # 靠近best点
                    index_select = np.where(i== best_Individual)[0] # 靠近时，替换值原本在哪个位置
                    #将值为i的从indivial中删除
                    individual = individual[individual != i]
                    for j, index in enumerate(index_select):
                        individual = np.insert(individual, index, i)

            individual=indivial  # 要为indivial前面加上0
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

