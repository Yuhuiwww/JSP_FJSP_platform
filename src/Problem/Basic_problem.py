

class Basic_Problem:

    # Problem reset
    def reset(self,data,config):
        self.global_time = 0  # -1 matters a lot

    def eval(self, x):
        pass

    # Solve makespan
    def cal_objective(self, sequence,dataset):
        raise NotImplementedError
