from Problem.Basic_problem import Basic_Problem

class FJSP_Basic_algorithm:
    def __init__(self, config):
        self.__config = config

    def FJSP_run_episode(self, problem: Basic_Problem,parseFile):
        raise NotImplementedError

    # class FJSP_Basic_algorithm:
    #
    #     def __init__(self,
    #                  problem,
    #                  algorithm,
    #                  config
    #                  ):
    #         self.problem = problem
    #         self.algorithm = algorithm
    #         self.config = config
    #
    #     def FJSP_run_episode(self, problem: Basic_Problem, parseFile):
    #         raise NotImplementedError