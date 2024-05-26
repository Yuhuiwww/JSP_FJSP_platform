import time
import random
import itertools
import numpy as np

from Test.optimizer.FJSP_optimizer.FJSP_Traditional_algorithm.FJSP_Bassic_algorithm import FJSP_Basic_algorithm


class FJSP_ABC(FJSP_Basic_algorithm):
    def __init__(self, config):
        self.config = config
        self.pt = config.Pn_j  # processing time
        self.ms = config.Pn_m  # machine sequence
        self.J_num = config.Pn_j  # Job num
        self.M_num = config.Pn_m  # Machine num
        self.population_size = 100
        self.empoyed_beesrate = 0.5
        self.num_onlooker_bees=0.3
        self.limit = 10

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
    def generateOS(self,parameters):
        jobs = parameters['jobs']

        OS = []
        i = 0
        for job in jobs:
            for op in job:
                OS.append(i)
            i = i + 1

        random.shuffle(OS)

        return OS

    def generateMS(self,parameters):
        jobs = parameters['jobs']

        MS = []
        for job in jobs:
            for op in job:
                randomMachine = random.randint(0, len(op) - 1)
                MS.append(randomMachine)

        return MS
    def initializePopulation(self, parameters):
        gen1 = []
        for i in range(self.population_size):
            tem = []
            OS = self.generateOS(parameters)
            MS = self.generateMS(parameters)
            tem.append(OS)
            tem.append(MS)
            gen1.append(tem)
            # gen1.append((OS, MS))
        return gen1

    def halfMutation(self, p, parameters):
        o = p
        jobs = parameters['jobs']

        size = len(p)
        r = int(size / 2)

        positions = random.sample(range(size), r)

        i = 0
        for job in jobs:
            for op in job:
                if i in positions:
                    o[i] = random.randint(0, len(op) - 1)
                i = i + 1

        return o

    def swap(self,p):
        pos1 = random.randint(0, len(p) - 1)
        pos2 = random.randint(0, len(p) - 1)

        if pos1 == pos2:
            return p

        if pos1 > pos2:
            pos1, pos2 = pos2, pos1

        os1 = p[:pos1] + [p[pos2]] + \
                    p[pos1 + 1:pos2] + [p[pos1]] + \
                    p[pos2 + 1:]

        return os1
    def insert(self,p):
        pos1 = random.randint(0, len(p) - 1)
        pos2 = random.randint(0, len(p) - 1)

        if pos1 == pos2:
            return p

        if pos1 > pos2:
            pos1, pos2 = pos2, pos1
        #将Pos位置的元素放到pos1位置，pos1 及其后面的元素后移
        p.insert(pos1, p.pop(pos2))
        return p

    def timeTaken(self,os_ms, pb_instance):
        (os1, ms) = os_ms
        decoded = self.decode(pb_instance, os1, ms)

        # Getting the max for each machine
        max_per_machine = []
        for machine in decoded:
            max_d = 0
            for job in machine:
                end = job[3] + job[1]
                if end > max_d:
                    max_d = end
            max_per_machine.append(max_d)

        return max(max_per_machine)

    def empoyed_bees(self, Individual, parameter):
        MS = Individual[1]
        OS = Individual[0]
        MS2=self.halfMutation(MS, parameter)
        # MS2=MS

        if random.choice([True, False]):
            OS2=self.swap(OS)
        else:
            OS2= self.insert(OS)
        Individual[1] = MS2
        Individual[0] = OS2
        return Individual
    def split_ms(self,pb_instance, ms):
        jobs = []
        current = 0
        for index, job in enumerate(pb_instance['jobs']):
            jobs.append(ms[current:current + len(job)])
            current += len(job)
        return jobs
    def is_free(self,tab, start, duration):
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
    def decode(self,pb_instance, os, ms):
        # print('os',os)
        # if(ms=='None'):
        #     print('ms is None')
        o = pb_instance['jobs']
        machine_operations = [[] for i in range(pb_instance['machinesNb'])]

        ms_s = self.split_ms(pb_instance, ms)  # machine for each operations

        indexes = [0] * len(ms_s)
        start_task_cstr = [0] * len(ms_s)

        # Iterating over OS to get task execution order and then checking in
        # MS to get the machine
        for job in os:
            index_machine = ms_s[job][indexes[job]]
            if indexes[job] >= len(o[job]):
                print("Error: indexes[job] >= len(o[job])")
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
    def calculate_probabilities(self,problem,population,parameters):
        fitness_values = [problem.cal_FJSP_objective(self.decode(parameters, population[sol][0], population[sol][1])) for sol in range(len(population))]
        total_fitness = sum(fitness_values)
        probabilities = [fitness / total_fitness for fitness in fitness_values]
        return probabilities
    def neighborhoodSearch(self,p):
        pos3 = pos2 = pos1 = random.randint(0, len(p) - 1)

        while p[pos2] == p[pos1]:
            pos2 = random.randint(0, len(p) - 1)

        while p[pos3] == p[pos2] or p[pos3] == p[pos1]:
            pos3 = random.randint(0, len(p) - 1)

        sortedPositions = sorted([pos1, pos2, pos3])
        pos1 = sortedPositions[0]
        pos2 = sortedPositions[1]
        pos3 = sortedPositions[2]

        e1 = p[sortedPositions[0]]
        e2 = p[sortedPositions[1]]
        e3 = p[sortedPositions[2]]

        permutations = list(itertools.permutations([e1, e2, e3]))
        permutation = random.choice(permutations)

        offspring = p[:pos1] + [permutation[0]] + \
                    p[pos1 + 1:pos2] + [permutation[1]] + \
                    p[pos2 + 1:pos3] + [permutation[2]] + \
                    p[pos3 + 1:]

        return offspring
    def onlooker_bees(self,problem, population,Individual, parameters):
        probabilities = self.calculate_probabilities(problem, population, parameters)
        num_selected_solutions = 0
        i = 0
        while num_selected_solutions < self.num_onlooker_bees * self.population_size:
            if np.random.rand() < probabilities[i]:
                OS1 = self.neighborhoodSearch(Individual[0])
                Individual[0] =OS1
                num_selected_solutions += 1
        return Individual

    def compare(self,problem,parameters, a, b):
        # print('a',a)
        fitness_a = problem.cal_FJSP_objective(self.decode(parameters, a[0], a[1]))
        fitness_b = problem.cal_FJSP_objective(self.decode(parameters, b[0], b[1]))
        if(fitness_a < fitness_b):
            return a
        else:
            return b
    def FJSP_run_episode(self, problem, file):
        Empoyed_bees = []
        parameters = self.parse(file)
        t0 = time.time()
        best_minfiness = 9999999
        max_run_time = self.config.Pn_j * self.config.Pn_m
        with open(self.config.optimizer + '-10 times runing solution.txt', 'a') as f:
            f.write(self.config.test_datas_type + str(self.config.Pn_j) + 'x' + str(self.config.Pn_m) + '\n')
            for repeat in range(10):
                min_makespan = 999999
                start_time = time.time()
                population = self.initializePopulation(parameters)
                population =sorted(population, key=lambda cpl: self.timeTaken(cpl, parameters))
                for run in range(100000000):
                    for i in range(self.population_size):
                        if(i<self.population_size/2):
                            Empoyed_bees.append(self.empoyed_bees(population[i],parameters))
                            # print('Empoyed_bees',Empoyed_bees[i])
                            population[i]=self.compare(problem,parameters,Empoyed_bees[i],population[i])
                        else:
                            Onlook_bee=self.onlooker_bees(problem,population,population[i],parameters)
                            population[i]=self.compare(problem,parameters,Onlook_bee,population[i])
                    for i in range(self.population_size):
                        if(problem.cal_FJSP_objective(self.decode(parameters, population[i][0], population[i][1]))<min_makespan):
                            fitness=problem.cal_FJSP_objective(self.decode(parameters, population[i][0], population[i][1]))
                    if min_makespan > fitness:
                        min_makespan = fitness
                    if time.time() - start_time > max_run_time:
                        break
                if best_minfiness > min_makespan:
                    best_minfiness = min_makespan
                print('最小完工时间',min_makespan)
                f.write(str(min_makespan) + '\n')
        return best_minfiness

