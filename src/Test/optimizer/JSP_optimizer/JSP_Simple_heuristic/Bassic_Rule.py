
from Problem.Basic_problem import Basic_Problem
class Basic_Rule:
    def __init__(self, config):
        self.config = config
    def run_rule(self, problem: Basic_Problem,dataset):
        raise NotImplementedError
