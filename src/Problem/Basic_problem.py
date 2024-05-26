

class Basic_Problem:

    # 问题重置
    def reset(self,data,config):
        self.global_time = 0  # -1 matters a lot

    def eval(self, x):
        pass

    # 求解makespan
    def cal_objective(self, sequence,dataset):
        raise NotImplementedError
