import time
import numpy as np
from Test.optimizer.JSP_optimizer.JSP_Simple_heuristic.Bassic_Rule import Basic_Rule
class SPT(Basic_Rule):
    def __init__(self, config):
        self.config=config
    def run_rule(self,problem,dataset):#
        Sequence=[]
        n_Job = len(dataset[0])
        m_Machine = len(dataset[0][0])
        start_time = time.time()
        machine_op = np.zeros(n_Job, dtype=np.int32)  # operand
        for i in range(m_Machine):
            # Get all elements in the current column
            current_col = [(row_idx, dataset[0][row_idx][i]) for row_idx in range(n_Job)]
            # Sort by element size from smallest to largest
            current_col_sorted = sorted(current_col, key=lambda x: x[1])
            # Add the sorted row numbers to the Sequence array.
            Sequence.extend([x[0] for x in current_col_sorted])
        return problem.cal_objective(Sequence, dataset)
