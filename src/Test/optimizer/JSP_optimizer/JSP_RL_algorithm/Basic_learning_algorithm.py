from Problem.Basic_problem import Basic_Problem
from typing import Any, Tuple

class Basic_learning_algorithm:
    """
    Abstract super class for learnable backbone optimizers.
    """
    def __init__(self,config):
        super().__init__()
        self.config = config

    def init_population(self,
                        problem: Basic_Problem,data,config) -> Any:
        raise NotImplementedError

    def update(self,
               data: Any,
               config) -> Tuple[Any]:
        raise NotImplementedError
