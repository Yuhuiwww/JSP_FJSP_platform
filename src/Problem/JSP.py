import numpy as np

from Problem.Basic_problem import Basic_Problem
class JSP(Basic_Problem):
    def __init__(self, job_num,machine_num):
        super().__init__()
        self.job_num = job_num
        self.machine_num = machine_num

    def cal_objective(self,sequence,dataset): #调度序列和处理时间，加工机床数据集
        job_op = np.zeros(self.job_num,dtype = np.int32)  # 操作数
        cmp_job = np.zeros(self.job_num)  # 工件的完工时间
        Idel_machine = np.zeros(self.machine_num)  # 机床的空闲时间
        for i in (sequence):
            n_job=i
            m_machine = dataset[1][n_job][job_op[n_job]]-1  # 获得工件所在的机床
            process_time = dataset[0][n_job][job_op[n_job]]  # 获取处理时间
            completion_t = max(cmp_job[n_job], Idel_machine[m_machine]) + process_time #获得工件的完工时间和机床的空闲时间
            cmp_job[n_job]=completion_t
            Idel_machine[m_machine]=completion_t
            job_op[n_job]+=1
        makespan = max(cmp_job)
        return makespan








