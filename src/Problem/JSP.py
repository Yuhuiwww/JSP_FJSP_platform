import numpy as np

from Problem.Basic_problem import Basic_Problem
class JSP(Basic_Problem):
    def __init__(self, job_num,machine_num):
        super().__init__()
        self.job_num = job_num
        self.machine_num = machine_num

    def cal_objective(self,sequence,dataset): #Scheduling Sequence and Processing Time, Machining Machine Data Set
        job_op = np.zeros(self.job_num,dtype = np.int32)  # operand
        cmp_job = np.zeros(self.job_num)  # Completion time of workpieces
        Idel_machine = np.zeros(self.machine_num)  # Idle time of the machine
        for i in (sequence):
            n_job=i
            m_machine = dataset[1][n_job][job_op[n_job]]-1  # Get the machine where the workpiece is located
            process_time = dataset[0][n_job][job_op[n_job]]  # Get processing time
            completion_t = max(cmp_job[n_job], Idel_machine[m_machine]) + process_time #Obtaining the completion time of the workpiece and the free time of the machine.
            cmp_job[n_job]=completion_t
            Idel_machine[m_machine]=completion_t
            job_op[n_job]+=1
        makespan = max(cmp_job)
        return makespan








