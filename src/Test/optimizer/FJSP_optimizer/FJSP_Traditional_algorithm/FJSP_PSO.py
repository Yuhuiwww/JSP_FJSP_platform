import copy
import time

from Test.optimizer.FJSP_optimizer.FJSP_Traditional_algorithm.FJSP_Bassic_algorithm import FJSP_Basic_algorithm
import numpy as np

class Input:
    def __init__(self, inputFile: str,config):
        self.__MAC_INFO = []
        self.__PRO_INTO = []
        self.__proNum = []
        self.__lines = None
        self.__input = inputFile
        self.Mac_Num=0
        self.Job_Num=0
        self.job_op_num=[]
        self.config=config

    def getMatrix(self):
        self.__readExample()
        self.__initMatrix()
        for i in range(len(self.__lines)-1):
            lo = 0
            hi = 0
            for j in range(self.__proNum[i]):
                head = int(self.__lines[i][lo])
                hi = lo + 2 * head + 1

                lo += 1
                while lo < hi:
                    self.__MAC_INFO[i][j].append(int(self.__lines[i][lo]))
                    self.__PRO_INTO[i][j].append(int(self.__lines[i][lo + 1]))
                    lo += 2


        p_table=self.DataConversion()
        return p_table,self.job_op_num




    def __initMatrix(self):
        for i in range(len(self.__proNum)):
            self.__MAC_INFO.append([])
            self.__PRO_INTO.append([])
            for j in range(self.__proNum[i]):
                self.__MAC_INFO[i].append([])
                self.__PRO_INTO[i].append([])

    def __readExample(self):
        with open(self.__input) as fileObject:
            self.__lines = fileObject.readlines()

        self.__lines[0] = self.__lines[0].lstrip().rstrip().split("\t")

        self.Job_Num=self.config.Pn_j
        self.Mac_Num=self.config.Pn_m

        # 数据调整
        del self.__lines[0]
        #这里要少一个

        for i in range(len(self.__lines)-1):

            self.__lines[i] = self.__lines[i].lstrip().rstrip().split(" ")
            operation=int(self.__lines[i].pop(0))
            self.job_op_num.append(operation)
            self.__proNum.append(operation)
            while "" in self.__lines[i]:
                self.__lines[i].remove("")

    def DataConversion(self):

        total_op = np.sum(self.job_op_num)
        #加工时间矩阵p_table：总的工序数*m；其中不能进行加工用-1表示
        p_table = np.ones((total_op,self.Mac_Num),dtype=int)*(-1)
        index =0
        for (i1,i2) in zip(self.__MAC_INFO,self.__PRO_INTO):
            for (j1,j2) in zip(i1,i2):
                for (k1,k2) in zip(j1,j2):
                    p_table[index][k1-1]=k2
                index += 1

        return p_table

class Encode:
    def __init__(self,Pop_size,p_table,job_op_num):
        #Pop_size为种群个数
        self.GS_num = int(0.6 * Pop_size)  # 全局选择的个数
        self.LS_num = int(0.2 * Pop_size)  # 局部选择的个数
        self.RS_num = int(0.2 * Pop_size)  # 随机选择的个数
        self.p_table = p_table
        self.half_chr = p_table.shape[0]
        self.job_op_num = job_op_num
        self.m = p_table.shape[1]
        self.n = len(job_op_num)
    #得到初始有序的一维os
    def order_os(self):
        order_os=[(index+1) for index,op in enumerate(self.job_op_num) for i in range(op)]
        # for index,op in enumerate(self.job_op_num):
        #     for i in range(op):
        #         order_OS.append(index+1)
        #
        return order_os
    def random_selection(self):
        MS=np.empty((self.RS_num,self.half_chr),dtype=int)
        #随机选择OS
        OS = np.empty((self.RS_num, self.half_chr),dtype=int)
        order_os = self.order_os()[:]
        # 随机选择MS
        for episode in range(self.RS_num):
            #打乱os的顺序
            np.random.shuffle(order_os)
            #随机选择os
            OS[episode]=order_os[:]
            for op_index,p in enumerate(self.p_table):
                #找出该工件能够加工的机器的序号
                ava_m = [(index+1) for index in range(len(p)) if p[index]!=-1]
                #随机选择,先这样写着
                MS[episode][op_index]=np.random.choice(np.arange(len(ava_m)))+1

        chr = np.hstack((MS, OS))
        return chr
    def global_selection(self):
        MS = np.empty((self.GS_num, self.half_chr), dtype=int)
        # 随机选择OS
        OS = np.empty((self.GS_num, self.half_chr), dtype=int)
        order_os = self.order_os()[:]
        # 随机选择MS
        for episode in range(self.GS_num):
            # 打乱os的顺序
            np.random.shuffle(order_os)
            # 随机选择os
            OS[episode] = order_os[:]
            # 用于随机选择的工件集合
            job_list = [i for i in range(self.n)]
            # 初始化值为0的长度为m的负荷数组
            M_load = np.zeros(self.m, dtype=int)
            for i in range(self.n):
                #随机选择一个工件
                job_num=np.random.choice(job_list)
                #在这个工件的所有工序上进行遍历
                for op in range(sum(self.job_op_num[:job_num]), sum(self.job_op_num[:job_num])+self.job_op_num[job_num]):
                    #得到临时的机器负荷数组
                    temp_load = np.array([pro + load for (pro, load) in zip(self.p_table[op], M_load) if pro != -1])
                    #得到临时的机器负荷索引
                    temp_index = [index for (index, pro) in enumerate(self.p_table[op]) if pro != -1]
                    #选取临时的机器符合最小的索引
                    ava_min_index = np.argmin(temp_load)
                    #将最小的索引+1放入MS中，即最好的可用的机器号,注意这里的下标是op,因为是随机找的工件
                    MS[episode][op]=ava_min_index+1
                    #更新机器负荷列表
                    M_load[temp_index[ava_min_index]] = temp_load[ava_min_index]
                #删除刚刚随机的工件号，继续随机进行
                job_list.remove(job_num)

        chr = np.hstack((MS, OS))
        return chr
    def local_selection(self):
        MS = np.empty((self.LS_num, self.half_chr), dtype=int)
        # 随机选择OS
        OS = np.empty((self.LS_num, self.half_chr), dtype=int)
        order_os = self.order_os()[:]
        # 随机选择MS
        for episode in range(self.LS_num):
            # 打乱os的顺序
            np.random.shuffle(order_os)
            # 随机选择os
            OS[episode] = order_os[:]
            # 因为不能直接得到二维矩阵的列索引，这里手工设置一个
            chr_index = 0


            #依次遍历整个工件
            for i in range(self.n):
                # 初始化值/重新置为0的长度为m的负荷数组
                M_load = np.zeros(self.m, dtype=int)

                # 在这个工件的所有工序上进行遍历
                for op in range(sum(self.job_op_num[:i]),
                                sum(self.job_op_num[:i]) + self.job_op_num[i]):
                    # 得到临时的机器负荷数组
                    temp_load = np.array([pro + load for (pro, load) in zip(self.p_table[op], M_load) if pro != -1])
                    # 得到临时的机器负荷索引
                    temp_index = [index for (index, pro) in enumerate(self.p_table[op]) if pro != -1]
                    # 选取临时的机器符合最小的索引
                    ava_min_index = np.argmin(temp_load)
                    # 将最小的索引+1放入MS中，即最好的可用的机器号
                    MS[episode][chr_index] = ava_min_index + 1
                    # 列索引加1
                    chr_index += 1
                    # 更新机器负荷列表
                    M_load[temp_index[ava_min_index]] = temp_load[ava_min_index]

        chr = np.hstack((MS, OS))
        return chr

class FJSP_PSO(FJSP_Basic_algorithm):
    def __init__(self, config):
        self.config = config
        self.Popsize = 200
        self.o_mega = 0.15
        self.c1 = 0.5
        self.c2 = 0.7
        self.pf_max = 0.8
        self.pf_min = 0.2
        self.Iter = 1
    def parse(self,path):
        file = open(path, 'r')

        firstLine = file.readline()
        firstLineValues = list(map(int, firstLine.split()[0:2]))

        jobsNb = firstLineValues[0]
        machinesNb = firstLineValues[1]

        jobs = []

        for i in range(jobsNb):
            currentLine = file.readline()
            currentLineValues = list(map(int, currentLine.split()))

            operations = []

            j = 1
            while j < len(currentLineValues):
                k = currentLineValues[j]
                j = j + 1

                operation = []

                for ik in range(k):
                    machine = currentLineValues[j]
                    j = j + 1
                    processingTime = currentLineValues[j]
                    j = j + 1

                    operation.append({'machine': machine, 'processingTime': processingTime})

                operations.append(operation)

            jobs.append(operations)

        file.close()

        return {'machinesNb': machinesNb, 'jobs': jobs}
    def init_Individual(self,file):
        input=Input(file,self.config)
        p_table, job_op_num = input.getMatrix()
        encode = Encode(self.Popsize, p_table, job_op_num)
        # 全局选择的染色体
        global_chrs = encode.global_selection()
        # #局部选择的染色体
        local_chrs = encode.local_selection()
        # #随机选择的染色体
        random_chrs = encode.random_selection()
        # 合并三者,得到初始的种群
        chrs = np.vstack((global_chrs, local_chrs, random_chrs))

        # 得到初始的全局最优位置
        # Decode.decode(chr,job_op_num,p_table,'decode')，其中的‘decode’表示不画图，只是计算适应度
        return chrs,p_table,job_op_num
    def split_ms(self,pb_instance, ms):
        jobs = []
        current = 0
        for index, job in enumerate(pb_instance['jobs']):
            jobs.append(ms[current:current + len(job)])
            current += len(job)
        return jobs

    def is_free(self, tab, start, duration):
        for k in range(start, start + duration):
            if not tab[k]:
                return False
        return True
    def find_first_available_place(self,start_ctr, duration, machine_jobs):
        max_duration_list = []
        max_duration = start_ctr + duration

        # max_duration is either the start_ctr + duration or the max(possible starts) + duration
        if machine_jobs:
            for job in machine_jobs:
                max_duration_list.append(job[3] + job[1])  # start + process time

            max_duration = max(max(max_duration_list), start_ctr) + duration

        machine_used = [True] * max_duration

        # Updating array with used places
        for job in machine_jobs:
            start = job[3]
            long = job[1]
            for k in range(start, start + long):
                machine_used[k] = False

        # Find the first available place that meets constraint
        for k in range(start_ctr, len(machine_used)):
            if self.is_free(machine_used, k, duration):
                return k


    def decode(self, pb_instance, os, ms):
        o = pb_instance['jobs']
        machine_operations = [[] for i in range(pb_instance['machinesNb'])]

        ms_s = self.split_ms(pb_instance, ms)  # machine for each operations

        indexes = [0] * len(ms_s)
        start_task_cstr = [0] * len(ms_s)

        # Iterating over OS to get task execution order and then checking in
        # MS to get the machine
        for job in os:
            # print('indexes[job]',indexes)
            # print('job',job)
            # print(indexes[job])
            #
            # print(type(indexes[job]))


            # 确保 indexes[job] 是整数类型
            if isinstance(indexes[job], int):
                index_machine = ms_s[job][indexes[job]]
            else:
                # 处理 indexes[job] 不是整数的情况
                print("indexes[job] 不是整数类型")
            index_machine = ms_s[job][indexes[job]]
            machine = o[job][indexes[job]][index_machine]['machine']
            prcTime = o[job][indexes[job]][index_machine]['processingTime']
            start_cstr = start_task_cstr[job]

            # Getting the first available place for the operation
            start = self.find_first_available_place(start_cstr, prcTime, machine_operations[machine - 1])
            name_task = "{}-{}".format(job, indexes[job] + 1)

            machine_operations[machine - 1].append((name_task, prcTime, start_cstr, start))

            # Updating indexes (one for the current task for each job, one for the start constraint
            # for each job)
            indexes[job] += 1
            start_task_cstr[job] = (start + prcTime)

        return machine_operations

    def f1_operator(self,chr, half_chr, p_table):
        ms = chr[:half_chr]
        os = chr[half_chr:]
        np.random.shuffle(os)
        chr = np.hstack((ms, os))

        return chr

    def op_in_m(self,i, j, job_op_num):
        # 求出这道工序在相应个机器上的位置，用job_op_num来求,
        if i == 1:
            op_index = j - 1
        else:
            # 切片是左闭右开
            op_index = sum(job_op_num[:i - 1]) + j - 1
        return op_index
    def f2_operator(self,n, half_chr, chr_Ek, single_best_chr, job_op_num):
        # #由操作1得到的染色体
        # chr_Ek = f1_operator()
        # 初始化工件编号列表
        job_num_list = [i + 1 for i in range(n)]
        # 打乱列表
        np.random.shuffle(job_num_list)
        # 随机选择一个从1——n的数,保证任何一个集合不为空
        index = np.random.randint(1, n)
        # 得到工件集1和工件集2
        job_set1 = job_num_list[:index]
        job_set2 = job_num_list[index:]
        # 分成ms和os两部分
        ms_Ek = chr_Ek[:half_chr]
        os_Ek = chr_Ek[half_chr:]
        ms_P = single_best_chr[:half_chr]
        os_P = single_best_chr[half_chr:]
        # 子代的os和ms
        os_F = []
        ms_F = [0 for i in range(half_chr)]
        # 遍历ms和os,
        # 论文上画的图有大问题!!!!!不能简单的将os与ms对应起来，然后给ms赋值，
        # 那样会导致比如某个工序本来最多可加工的机器个数为2，但是给它的染色体上的基因却为3了。
        # 正确的理解是：将选中的比如工序O（1,2）在父代对应的ms的基因值赋值到子代的ms上O（1，2）的位置

        # 用来存工件出现过几次的字典，形式{1：2}表示工件1出现了2次
        os_Ek_dict = {}
        os_P_dict = {}
        for os1, os2 in zip(os_Ek, os_P):
            # 现在默认Ek的在前面
            if os1 in job_set1:
                os_F.append(os1)
                if os1 in os_Ek_dict:
                    os_Ek_dict[os1] += 1
                else:
                    os_Ek_dict[os1] = 1
                op_index = self.op_in_m(os1, os_Ek_dict[os1], job_op_num)
                ms_F[op_index] = ms_Ek[op_index]
                # ms_F.append(ms_Ek[os_index])
            if os2 in job_set2:
                os_F.append(os2)
                if os2 in os_P_dict:
                    os_P_dict[os2] += 1
                else:
                    os_P_dict[os2] = 1
                op_index = self.op_in_m(os2, os_P_dict[os2], job_op_num)
                ms_F[op_index] = ms_P[op_index]
                # ms_F.append(ms_P[os_index])

            # # 如果都不满足，继续看下后面的基因点
            # else:
            #     continue
        # 合并子代的ms和os
        chr = np.hstack((ms_F, os_F))
        return chr

    def f3_operator(self,half_chr, chr_Fk, global_best_chr, pf, job_op_num):
        # 分解成ms和os,其中os_Xk是不变化的
        # if chr_Fk is None:
        #     # 如果 chr_Fk 是 NoneType
        #     print("chr_Fk is None")
        # if global_best_chr is None:
        #     print("global_best_chr is None")
        #
        # else:
        ms_Xk = chr_Fk[:half_chr]
        os_Xk = chr_Fk[half_chr:]
        ms_Pg = global_best_chr[:half_chr]

        # 产生随机向量R,值为0-1
        R = np.random.random_sample(half_chr)
        # 找出R中小于pf的位置
        R_bool = R < pf

        # 论文上画的图同样有大问题!!!!!不能简单的将os与ms对应起来，然后给ms赋值，
        # 那样会导致比如某个工序本来最多可加工的机器个数为2，但是给它的染色体上的基因却为3了。
        # 正确的理解是：将选中的比如工序O（1,2）在父代对应的ms的基因值赋值到子代的ms上O（1，2）的位置
        # 用来存工件出现过几次的字典，形式{1：2}表示工件1出现了2次
        # 跟算子2太像了没意思哎
        os_F_dict = {}

        for os_index, os in enumerate(os_Xk):
            if os in os_F_dict:
                os_F_dict[os] += 1
            else:
                os_F_dict[os] = 1
                # 只关心R中小于pf的部分,那部分才用全局最优去替换子代中的部分
                if R_bool[os_index]:
                    op_index = self.op_in_m(os, os_F_dict[os], job_op_num)
                    ms_Xk[op_index] = ms_Pg[op_index]
        # 合并子代的ms和os
        chr = np.hstack((ms_Xk, os_Xk))
        return chr

    def f_operator(self,job_op_num, p_table, chr, single_best_chr, global_best_chr, pf, o_mega, c1, c2):
        # 获得额外的参数
        half_chr = p_table.shape[0]
        n = len(job_op_num)
        chr_Ek = chr
        chr_Fk = chr_Ek
        # 产生0-1的随机数r1
        r1 = np.random.random()
        if r1 < o_mega:
            # 执行f1操作
            chr_Ek = self.f1_operator(chr, half_chr, p_table)


        r2 = np.random.random()
        if r2 < c1:
            # 执行f2操作
            if chr_Ek is None:
                print('null')
            chr_Fk = self.f2_operator(n, half_chr, chr_Ek, single_best_chr, job_op_num)



        r3 = np.random.random()
        if r3 < c2:
            # 执行f3操作
            chr_Xk = self.f3_operator(half_chr, chr_Fk, global_best_chr, pf, job_op_num)
        else:
            chr_Xk = chr_Fk

        return chr_Xk

    def FJSP_run_episode(self, problem, file):
        parameters = self.parse(file)
        best_minfiness = 9999999
        max_run_time = self.config.Pn_j * self.config.Pn_m
        with open(self.config.optimizer + '-10 times runing solution.txt', 'a') as f:
            f.write(self.config.test_datas_type + str(self.config.Pn_j) + 'x' + str(self.config.Pn_m) + '\n')
            for repeat in range(10):
                min_makespan = 999999
                start_time = time.time()
                chrs, p_table, job_op_num = self.init_Individual(file)
                P = copy.deepcopy(chrs)
                n = len(job_op_num)
                # 机器数
                m = p_table.shape[1]
                # 染色体长度的一半
                half_chr = p_table.shape[0]
                # 最大工序数
                max_op = np.max(job_op_num)
                fitness_list = []
                for chr in chrs:
                    # 染色体长度的一半 len(p_table)，得到MS和OS
                    MS = chr[:half_chr]
                    OS = chr[half_chr:]

                    fitness_list.append(
                        problem.cal_FJSP_objective(self.decode(parameters, [x - 1 for x in OS], [x - 1 for x in MS])))
                Pg = P[np.argmin(fitness_list)]
                fitness_listall = []
                for run in range(100000000):
                    pf = self.pf_max - (self.pf_max - self.pf_min) / self.Iter * run
                    # 更新种群中所有的染色体
                    copy_chrs = copy.deepcopy(chrs)
                    chrs = [self.f_operator(job_op_num, p_table, chr, P[index], Pg, pf, self.o_mega, self.c1, self.c2)
                            for index, chr in
                            enumerate(copy_chrs)]
                    # 更新个体最优位置
                    for i in range(len(P)):
                        MSp = P[i][:half_chr]
                        OSp = P[i][half_chr:]
                        MSchar = chrs[i][:half_chr]
                        OSchar = chrs[i][half_chr:]
                        fitness_p = problem.cal_FJSP_objective(
                            self.decode(parameters, [x - 1 for x in OSp], [x - 1 for x in MSp]))

                        fitness_char = problem.cal_FJSP_objective(
                            self.decode(parameters, [x - 1 for x in OSchar], [x - 1 for x in MSchar]))
                        if fitness_p <= fitness_char:
                            P[i] = P[i]
                        else:
                            P[i] = chrs[i]
                        # 更新全局最优位置

                        fitness_listall.append(problem.cal_FJSP_objective(
                            self.decode(parameters, [x - 1 for x in P[i][half_chr:]],
                                        [x - 1 for x in P[i][:half_chr]])))
                    Pg = P[np.argmin(fitness_list)]
                    if time.time() - start_time > max_run_time:
                        break
                    fitness = problem.cal_FJSP_objective(
                        self.decode(parameters, [x - 1 for x in Pg[half_chr:]], [x - 1 for x in Pg[:half_chr]]))
                    if min_makespan > fitness:
                        min_makespan = fitness
                print('fitness', min_makespan)
                f.write(str(min_makespan) + '\n')
                if best_minfiness > min_makespan:
                    best_minfiness = min_makespan
        return best_minfiness





