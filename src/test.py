import copy
import os
import time

from tqdm import tqdm

from FJSP_config import get_FJSPconfig
from Test.optimizer.FJSP_optimizer.FJSP_Gurobi import FJSPModel
from Test.optimizer.JSP_optimizer.JSP_Gurobi import JSPModel

from Problem.FJSP import FJSP
from Test.optimizer.FJSP_optimizer.FJSP_Traditional_algorithm.FJSP_GA import FJSP_GA
from Test.optimizer.FJSP_optimizer.FJSP_Traditional_algorithm.FJSP_ABC import FJSP_ABC
from Test.optimizer.FJSP_optimizer.FJSP_Traditional_algorithm.FJSP_PSO import FJSP_PSO
from Test.optimizer.FJSP_optimizer.FJSP_heuristic.FJSP_heuristic_framework import Heuristic_Framework
from Test.optimizer.FJSP_optimizer.FJSP__RL_algorithm import FJSP_GNN_optimizer
from Test.optimizer.FJSP_optimizer.FJSP__RL_algorithm.FJSP_DAN_optimizer import FJSP_DAN_optimizer
from Test.optimizer.FJSP_optimizer.FJSP__RL_algorithm.FJSP_GNN_optimizer import FJSP_GNN_optimizer
from Problem.JSP import JSP
from Test.optimizer.JSP_optimizer.JSP_RL_algorithm.End2End_optimizer import End2End_optimizer
from Test.optimizer.JSP_optimizer.JSP_RL_algorithm.L2S_optimizer import L2S_optimizer
from Test.optimizer.JSP_optimizer.JSP_RL_algorithm.L2D_optimizer import L2D_optimizer
from Test.optimizer.JSP_optimizer.JSP_RL_algorithm.ScheduleNet_optimizer import ScheduleNet_optimizer
from Test.optimizer.JSP_optimizer.JSP_RL_algorithm.GNN_optimizer import GNN_optimizer
from Test.optimizer.JSP_optimizer.JSP_Simple_heuristic.LPT import LPT
from Test.optimizer.JSP_optimizer.JSP_Traditional_algorithm.ABC import ABC
from Test.optimizer.JSP_optimizer.JSP_Traditional_algorithm.GA import GA
from Test.optimizer.JSP_optimizer.JSP_Traditional_algorithm.Jaya import Jaya
from Test.optimizer.JSP_optimizer.JSP_Traditional_algorithm.PSO import PSO

from Test.optimizer.JSP_optimizer.JSP_Simple_heuristic import *
from Test.optimizer.JSP_optimizer.JSP_RL_algorithm.JSS_Env_optimizer import JSS_Env_optimizer
from Test.optimizer.JSP_optimizer.JSP_Simple_heuristic.LPT import LPT
from Test.optimizer.JSP_optimizer.JSP_Simple_heuristic.SPT import SPT
from Test.optimizer.JSP_optimizer.JSP_Simple_heuristic.LRPT import LRPT
from Test.optimizer.JSP_optimizer.JSP_Simple_heuristic.SRPT import SRPT


from JSP_config import get_config
from LoadUtils import load_data, FJSP_load_gurobi, load_dataDAN
from Test.environment.Basic_environment import PBO_Env


class Tester(object):
    def __init__(self, config):
        self.config = config
        self.test_results = {'cost': {},
                             'fes': {},
                             'T0': 0.,
                             'T1': {},
                             'T2': {}}

    def test_JSP(self):
        print(f'start testing: {self.config.run_time}')
        optimizer = eval(self.config.optimizer)(copy.deepcopy(self.config))
        # optimizer = JSS_Env_optimizer(self.config)
        dataset = load_data(self.config)
        problem = eval(self.config.problem_name)(copy.deepcopy(self.config.Pn_j), copy.deepcopy(self.config.Pn_m))
        env = PBO_Env(problem, optimizer)
        with open('./Result/JSP/'+self.config.optimizer + 'solution.txt', 'a') as f:
            f.write(self.config.test_datas_type + str(self.config.Pn_j) + 'x' + str(self.config.Pn_m) + '\n')
            pbar_len = len(dataset)
            with tqdm(range(pbar_len), desc='Testing') as pbar:
                for i in range(len(dataset)):
                    env.reset(dataset[i], self.config)
                    makespan, runtime = env.step(dataset[i], self.config)
                    f.write(str(makespan) + ' ' + str(runtime) + '\n')
                    self.config.itration += 1
                    pbar.update(1)

    def test_FJSP(self):
        print(f'start testing: {self.config.run_time}')
        optimizer = eval(self.config.optimizer)(copy.deepcopy(self.config))
        # optimizer = JSS_Env_optimizer(self.config)
        dataset, self.config = load_dataDAN(self.config)
        # test_data = pack_data_from_config(config.data_source, config.test_data)
        problem = eval(self.config.problem_name)(copy.deepcopy(self.config))
        env = PBO_Env(problem, optimizer)
        env.step(dataset, self.config)

    def JSP_Gurobi(self):
        dataset = load_data(self.config)
        with open('./Result/JSP/'+self.config.optimizer + 'solution.txt', 'a') as f:
            f.write(self.config.test_datas_type + str(self.config.Pn_j) + 'x' + str(self.config.Pn_m) + '\n')
            for i in range(len(dataset)):
                machies_array = []
                new_array = dataset[i][1]
                for element in new_array:
                    new_element = element - 1
                    machies_array.append(new_element)
                model = JSPModel(self.config.Pn_m, self.config.Pn_j, machies_array, dataset[i][0],self.config.JSP_gurobi_time_limit)
                makespan,runtime = model.solveGurobi(save_to_file=True, save_file_name="results")
                f.write(str(makespan) + ' ' + str(runtime) + '\n')

    def FJSP_Gurobi(self):
        # path problem/FJSP_test_datas/SD1/10x5
        # filesPath =  "./problem/FJSP_Gurobi_test/{}".format(str(self.config.Pn_j)+'x'+str(self.config.Pn_m))
        filesPath = self.config.test_datas+"/{}".format(
            self.config.test_datas_type + '/' + str(self.config.Pn_j) + 'x' + str(self.config.Pn_m))
        with open('./Result/FJSP/'+self.config.optimizer + 'solution.txt', 'a') as f:
            f.write(self.config.test_datas_type+ str(self.config.Pn_j) + 'x' + str(self.config.Pn_m) + '\n')
            for file_name in os.listdir(filesPath):
                if file_name.endswith('.fjs'):
                    file_path = os.path.join(filesPath, file_name)
                    data=FJSP_load_gurobi(file_path)
                    print("Perform a Gurobi validation......")
                    model=FJSPModel(self.config)
                    makespan,runtime=model.FJSP_solveGurobi(data)
                    f.write(str(makespan) + ' ' + str(runtime) + '\n')
        f.close()

    def test_FJSP_traditionalAlgorithm(self):
        optimizer = eval(self.config.optimizer)(copy.deepcopy(self.config))
        problem = eval(self.config.problem_name)(copy.deepcopy(self.config))
        filesPath = self.config.test_datas+"{}".format(
            self.config.test_datas_type + '/' + str(self.config.Pn_j) + 'x' + str(self.config.Pn_m))
        with open(self.config.optimizer + '-best-solution.txt', 'a') as f:
            f.write(self.config.test_datas_type + str(self.config.Pn_j) + 'x' + str(self.config.Pn_m) + '\n')
            for file_name in os.listdir(filesPath):
                if file_name.endswith('.fjs'):
                    file_path = os.path.join(filesPath, file_name)
                    min_makespan = optimizer.FJSP_run_episode(problem, file_path)
                    print("makespan", min_makespan)
                    f.write(str(min_makespan) + '\n')

    # def test_JSP_algorithm(self):
    #     optimizer = eval(self.config.optimizer)(copy.deepcopy(self.config))
    #     dataset = load_data(self.config)
    #     problem = eval(self.config.problem_name)(copy.deepcopy(self.config.Pn_j), copy.deepcopy(self.config.Pn_m))
    #     with open(self.config.optimizer + '-solution.txt', 'a') as f:
    #         f.write(self.config.test_datas_type + str(self.config.Pn_j) + 'x' + str(self.config.Pn_m) + '\n')
    #         for i in range(len(dataset)):
    #             print("The ",i," example")
    #             # optimizer.run_episode(problem, dataset[i])
    #             min_makespan = optimizer.run_episode(problem, dataset[i])
    #             f.write(str(min_makespan) + '\n')
    def test_JSP_algorithm(self):
        optimizer = eval(self.config.optimizer)(copy.deepcopy(self.config))
        dataset = load_data(self.config)
        problem = eval(self.config.problem_name)(copy.deepcopy(self.config.Pn_j), copy.deepcopy(self.config.Pn_m))
        with open('./Result/JSP/'+self.config.optimizer + '-best-solution.txt', 'a') as f:
            f.write(self.config.test_datas_type + str(self.config.Pn_j) + 'x' + str(self.config.Pn_m) + '\n')
            for i in range(len(dataset)):
                print("The ",i," example")
                # optimizer.run_episode(problem, dataset[i])
                min_makespan = optimizer.run_episode(problem, dataset[i])
                f.write(str(min_makespan) + '\n')
    def test_JSP_heuristic(self):
        optimizer = eval(self.config.optimizer)(copy.deepcopy(self.config))
        dataset = load_data(self.config)
        problem = eval(self.config.problem_name)(copy.deepcopy(self.config.Pn_j), copy.deepcopy(self.config.Pn_m))
        with open('./Result/JSP/'+self.config.optimizer + '00solution.txt', 'a') as f:
            f.write(self.config.test_datas_type + str(self.config.Pn_j) + 'x' + str(self.config.Pn_m) + '\n')
            for i in range(len(dataset)):
                start_time = time.time()
                print("The ", i, " example")
                min_makespan = optimizer.run_rule(problem, dataset[i])
                runtime = time.time() - start_time
                f.write(str(min_makespan) + ' ' + str(runtime) + '\n')

    def test_FJSP_Heuristic(self):
        filesPath=self.config.test_datas+'/'+self.config.test_datas_type+"/{}".format(str(self.config.Pn_j)+'x'+str(self.config.Pn_m))
        optimizer = eval(self.config.optimizer)(copy.deepcopy(self.config))
        with open('./Result/FJSP/' + self.config.dispatching_rule+self.config.machine_assignment_rule + '00solution.txt', 'a') as f:
            f.write(self.config.test_datas_type + str(self.config.Pn_j) + 'x' + str(self.config.Pn_m) + '\n')
            for file_name in os.listdir(filesPath):
                if file_name.endswith('.fjs'):
                    file_path = os.path.join(filesPath, file_name)
                    min_makespan,runtime=optimizer.run_rule(file_path)
                    # print("Minimum Completion Time ", min_makespan)
                    f.write(str(min_makespan) + ' ' + str(runtime) + '\n')
