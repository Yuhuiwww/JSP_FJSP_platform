import time
from Test.optimizer.JSP_optimizer.JSP_Simple_heuristic.Bassic_Rule import Basic_Rule


class LPT(Basic_Rule):
    def __init__(self, config):
        self.config=config

    def run_rule(self,problem, dataset):
        time_start=time.time()
        Sequence = []
        n_Job = len(dataset[0])
        m_Machine = len(dataset[0][0])
        for i in range(m_Machine):
            # Get all elements in the current column
            current_col = [(row_idx, dataset[0][row_idx][i]) for row_idx in range(n_Job)]
            current_col_sorted = sorted(current_col, key=lambda x: x[1], reverse=True)
            Sequence.extend([x[0] for x in current_col_sorted])
            makespan=problem.cal_objective(Sequence,dataset)

        return makespan