from typing import Any



class PBO_Env:
    """
    An environment with a problem and an optimizer.
    """
    def __init__(self,
                 problem,
                 optimizer,
                 ):
        self.problem = problem
        self.optimizer = optimizer

    def reset(self,data,config):
        self.problem.reset(data,config)
        return self.optimizer.init_population(self.problem,data,config)

    def step(self,data, config):
        return self.optimizer.update(data, config)
