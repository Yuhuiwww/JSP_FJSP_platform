from Problem.Basic_problem import Basic_Problem


class Basic_Algorithm:
    def __init__(self, config):
        self.__config = config

    def run_episode(self, problem: Basic_Problem,dataset):
        raise NotImplementedError
