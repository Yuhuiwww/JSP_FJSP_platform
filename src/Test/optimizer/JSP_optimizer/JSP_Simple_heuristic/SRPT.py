import time
import numpy as np
from Test.optimizer.JSP_optimizer.JSP_Simple_heuristic.Bassic_Rule import Basic_Rule
class SRPT(Basic_Rule):
    def __init__(self, config):
        self.config=config
    def run_rule(self,problem,dataset):
        Sequence=[]
        n_Job = len(dataset[0])
        m_Machine = len(dataset[0][0])
        job_op = np.zeros(n_Job, dtype=np.int32)  # operand
        remaining_time = [0] * n_Job  # Used to store the remaining processing time for each job
        start_time = time.time()
        for i in range(n_Job):
            for j in range(m_Machine):
                remaining_time[i] += dataset[0][i][j]  # Starting, the remaining time is the total processing time
            # Get all elements in the current column
        for i in range(m_Machine * n_Job):
            current_col = [(row_idx, remaining_time[row_idx]) for row_idx in range(n_Job) if
                           remaining_time[row_idx] != 0]
            current_col_sorted = sorted(current_col, key=lambda x: x[1])
            if len(current_col_sorted)!=0:
                Job = current_col_sorted[0][0]
                Sequence.append(current_col_sorted[0][0])
                Op = job_op[Job]
                # Update remaining processing time
                remaining_time[Job] -= dataset[0][Job][Op]
                job_op[Job] += 1

        return problem.cal_objective(Sequence,dataset)
