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

        # Data adjustment
        del self.__lines[0]
        #There's one less here.

        for i in range(len(self.__lines)-1):

            self.__lines[i] = self.__lines[i].lstrip().rstrip().split(" ")
            operation=int(self.__lines[i].pop(0))
            self.job_op_num.append(operation)
            self.__proNum.append(operation)
            while "" in self.__lines[i]:
                self.__lines[i].remove("")

    def DataConversion(self):

        total_op = np.sum(self.job_op_num)
        #Processing time matrix p_table: total number of processes * m; where unavailability is denoted by -1
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
        #Pop_size is the number of populations
        self.GS_num = int(0.6 * Pop_size)  # Number of global Selections
        self.LS_num = int(0.2 * Pop_size)  # Number of Local Selections
        self.RS_num = int(0.2 * Pop_size)  # Number of Random Selection
        self.p_table = p_table
        self.half_chr = p_table.shape[0]
        self.job_op_num = job_op_num
        self.m = p_table.shape[1]
        self.n = len(job_op_num)
    #Get the initial ordered one-dimensional os
    def order_os(self):
        order_os=[(index+1) for index,op in enumerate(self.job_op_num) for i in range(op)]
        # for index,op in enumerate(self.job_op_num):
        #     for i in range(op):
        #         order_OS.append(index+1)
        #
        return order_os
    def random_selection(self):
        MS=np.empty((self.RS_num,self.half_chr),dtype=int)
        #Random selection of OS
        OS = np.empty((self.RS_num, self.half_chr),dtype=int)
        order_os = self.order_os()[:]
        # Random selection of MS
        for episode in range(self.RS_num):
            #Disrupting the order of os
            np.random.shuffle(order_os)
            #Random selection of os
            OS[episode]=order_os[:]
            for op_index,p in enumerate(self.p_table):
                #Determine the machine numbers that can process the workpiece
                ava_m = [(index+1) for index in range(len(p)) if p[index]!=-1]
                #Random selection. Let's start with this.
                MS[episode][op_index]=np.random.choice(np.arange(len(ava_m)))+1

        chr = np.hstack((MS, OS))
        return chr
    def global_selection(self):
        MS = np.empty((self.GS_num, self.half_chr), dtype=int)
        # Random selection of OS
        OS = np.empty((self.GS_num, self.half_chr), dtype=int)
        order_os = self.order_os()[:]
        # Random selection of MS
        for episode in range(self.GS_num):
            # Disrupting the order of os
            np.random.shuffle(order_os)
            # Random selection of os
            OS[episode] = order_os[:]
            # Collection of artifacts for random selection
            job_list = [i for i in range(self.n)]
            # Initialize an array of loads of length m with value 0
            M_load = np.zeros(self.m, dtype=int)
            for i in range(self.n):
                #Randomly Select a Workpiece
                job_num=np.random.choice(job_list)
                #Iterate over all processes of this workpiece
                for op in range(sum(self.job_op_num[:job_num]), sum(self.job_op_num[:job_num])+self.job_op_num[job_num]):
                    #Get a temporary array of machine loads
                    temp_load = np.array([pro + load for (pro, load) in zip(self.p_table[op], M_load) if pro != -1])
                    #Get a temporary machine load index
                    temp_index = [index for (index, pro) in enumerate(self.p_table[op]) if pro != -1]
                    #Pick the temporary machine that matches the smallest index
                    ava_min_index = np.argmin(temp_load)
                    #Put the smallest index +1 into the MS, that is, the best available machine number, note that the subscript here is op, because it is a random artifact find
                    MS[episode][op]=ava_min_index+1
                    #Update machine load list
                    M_load[temp_index[ava_min_index]] = temp_load[ava_min_index]
                #Delete the artifact number just randomized and continue to randomize it
                job_list.remove(job_num)

        chr = np.hstack((MS, OS))
        return chr
    def local_selection(self):
        MS = np.empty((self.LS_num, self.half_chr), dtype=int)
        # Random selection of OS
        OS = np.empty((self.LS_num, self.half_chr), dtype=int)
        order_os = self.order_os()[:]
        # Random selection of MS
        for episode in range(self.LS_num):
            # Disrupting the order of os
            np.random.shuffle(order_os)
            # Random selection of os
            OS[episode] = order_os[:]
            # Because it is not possible to get the column indices of the two-dimensional matrix directly, here a manual setup of a
            chr_index = 0


            #Iterating over the workpiece
            for i in range(self.n):
                # Array of loads of length m with initialization value/reset to 0
                M_load = np.zeros(self.m, dtype=int)

                # Iterate over all processes in this workpiece
                for op in range(sum(self.job_op_num[:i]),
                                sum(self.job_op_num[:i]) + self.job_op_num[i]):
                    # Get a temporary array of machine loads
                    temp_load = np.array([pro + load for (pro, load) in zip(self.p_table[op], M_load) if pro != -1])
                    # Get a temporary machine load index
                    temp_index = [index for (index, pro) in enumerate(self.p_table[op]) if pro != -1]
                    # Pick the temporary machine that matches the smallest index
                    ava_min_index = np.argmin(temp_load)
                    # Put the smallest index +1 into MS, the best available machine number
                    MS[episode][chr_index] = ava_min_index + 1
                    # Column index plus 1
                    chr_index += 1
                    # Update machine load list
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
        # Globally Selected Chromosome
        global_chrs = encode.global_selection()
        # Locally Selected Chromosome
        local_chrs = encode.local_selection()
        # Randomly Selected Chromosome
        random_chrs = encode.random_selection()
        # Combine the three to get the initial population.
        chrs = np.vstack((global_chrs, local_chrs, random_chrs))

        # Get the initial global optimal position
        # Decode.decode(chr,job_op_num,p_table,'decode')，where 'decode' means no drawing, just calculating the fitness
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


            #Ensure that indexes[job] is an integer type
            if isinstance(indexes[job], int):
                index_machine = ms_s[job][indexes[job]]
            else:
                # Handling cases where indexes[job] is not an integer
                print("indexes[job] is not an integer type")
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
        # Find the position of the process on the corresponding machine, using job_op_num.
        if i == 1:
            op_index = j - 1
        else:
            # The slices are left closed and right open
            op_index = sum(job_op_num[:i - 1]) + j - 1
        return op_index
    def f2_operator(self,n, half_chr, chr_Ek, single_best_chr, job_op_num):
        # Chromosomes obtained from operation 1
        # chr_Ek = f1_operator()
        # Initialize the workpiece number list
        job_num_list = [i + 1 for i in range(n)]
        # disrupt the list
        np.random.shuffle(job_num_list)
        # Randomly select a number from 1 to n, ensuring that no set is empty.
        index = np.random.randint(1, n)
        # Get workpiece set 1 and workpiece set 2
        job_set1 = job_num_list[:index]
        job_set2 = job_num_list[index:]
        # Both ms and os
        ms_Ek = chr_Ek[:half_chr]
        os_Ek = chr_Ek[half_chr:]
        ms_P = single_best_chr[:half_chr]
        os_P = single_best_chr[half_chr:]
        # The os and ms of the child
        os_F = []
        ms_F = [0 for i in range(half_chr)]
        # Iterate over ms and os.
        #There is a big problem with the diagram drawn on the paper !!!!! Can't simply map os to ms and assign a value to ms.
        # That would result in, for example, a process that had a maximum number of machines that could be processed of 2, but was given a chromosome with 3 genes on it.
        # The correct understanding is that the gene value of the selected, say, process O(1,2) on the corresponding ms of the parent is assigned to the position O(1, 2) on the ms of the child

        # is used to store a dictionary of how many times an artifact has appeared, of the form {1:2} which means that artifact 1 has appeared 2 times
        os_Ek_dict = {}
        os_P_dict = {}
        for os1, os2 in zip(os_Ek, os_P):
            # Now the default Ek's come first
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

            # # If none of these satisfy you, keep looking at the gene points that follow
            # else:
            #     continue
        # Merging the ms and os of children
        chr = np.hstack((ms_F, os_F))
        return chr

    def f3_operator(self,half_chr, chr_Fk, global_best_chr, pf, job_op_num):
        # Decomposition into ms and os, where os_Xk is unchanged
        # if chr_Fk is None:
        #     # if chr_Fk is NoneType
        #     print("chr_Fk is None")
        # if global_best_chr is None:
        #     print("global_best_chr is None")
        #
        # else:
        ms_Xk = chr_Fk[:half_chr]
        os_Xk = chr_Fk[half_chr:]
        ms_Pg = global_best_chr[:half_chr]

        # Generate a random vector R with values 0-1.
        R = np.random.random_sample(half_chr)
        # Find the position in R that is less than pf
        R_bool = R < pf

        # The diagram drawn on the paper has the same big problem !!!!! Can't simply map os to ms and assign a value to ms.
        # That would result in, for example, a process that had a maximum number of machines that could be processed of 2, but was given a chromosome with 3 genes on it.
        # The correct understanding is that the gene value of the selected, say, process O(1,2) on the corresponding ms of the parent is assigned to the position O(1, 2) on the ms of the child
        # is used to store a dictionary of how many times an artifact has appeared, of the form {1:2} which means that artifact 1 has appeared 2 times
        # It's too similar to Calculator 2. It's no fun.
        os_F_dict = {}

        for os_index, os in enumerate(os_Xk):
            if os in os_F_dict:
                os_F_dict[os] += 1
            else:
                os_F_dict[os] = 1
                # It only cares about the part of R that is smaller than pf , and that part is replaced by the global optimum in the offspring.
                if R_bool[os_index]:
                    op_index = self.op_in_m(os, os_F_dict[os], job_op_num)
                    ms_Xk[op_index] = ms_Pg[op_index]
        # Merging the ms and os of children
        chr = np.hstack((ms_Xk, os_Xk))
        return chr

    def f_operator(self,job_op_num, p_table, chr, single_best_chr, global_best_chr, pf, o_mega, c1, c2):
        # Get additional parameters
        half_chr = p_table.shape[0]
        n = len(job_op_num)
        chr_Ek = chr
        chr_Fk = chr_Ek
        # Generate 0-1 random number r1
        r1 = np.random.random()
        if r1 < o_mega:
            # Execute the f1 operation
            chr_Ek = self.f1_operator(chr, half_chr, p_table)


        r2 = np.random.random()
        if r2 < c1:
            # Perform the f2 operation
            if chr_Ek is None:
                print('null')
            chr_Fk = self.f2_operator(n, half_chr, chr_Ek, single_best_chr, job_op_num)



        r3 = np.random.random()
        if r3 < c2:
            # Execute the f3 operation
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
                # Number of machines
                m = p_table.shape[1]
                # Half the length of a chromosome
                half_chr = p_table.shape[0]
                # Maximum number of processes
                max_op = np.max(job_op_num)
                fitness_list = []
                for chr in chrs:
                    # half of the chromosome length len(p_table) to get MS and OS
                    MS = chr[:half_chr]
                    OS = chr[half_chr:]

                    fitness_list.append(
                        problem.cal_FJSP_objective(self.decode(parameters, [x - 1 for x in OS], [x - 1 for x in MS])))
                Pg = P[np.argmin(fitness_list)]
                fitness_listall = []
                for run in range(100000000):
                    pf = self.pf_max - (self.pf_max - self.pf_min) / self.Iter * run
                    # Update all chromosomes in the population
                    copy_chrs = copy.deepcopy(chrs)
                    chrs = [self.f_operator(job_op_num, p_table, chr, P[index], Pg, pf, self.o_mega, self.c1, self.c2)
                            for index, chr in
                            enumerate(copy_chrs)]
                    # Updating the individual optimal position
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
                        # Update global optimal position

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





