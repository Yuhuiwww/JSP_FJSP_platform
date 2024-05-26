# -*- coding: utf-8 -*-

import itertools
import random
import time
from Test.optimizer.FJSP_optimizer.FJSP_Traditional_algorithm.FJSP_Bassic_algorithm import FJSP_Basic_algorithm



class FJSP_GA(FJSP_Basic_algorithm):
    def __init__(self, config):
        self.config = config
        self.popSize = 100
        self.maxGen = 200
        self.pr = 0.005
        self.pc = 0.8
        self.pm = 0.1
        self.latex_export = False

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
    def initializePopulation(self,parameters):
        gen1 = []
        for i in range(self.popSize):
            OS = self.generateOS(parameters)
            MS = self.generateMS(parameters)
            gen1.append((OS, MS))

        return gen1

    def shouldTerminate(self,population, gen):
        return gen > self.maxGen

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
        o = pb_instance['jobs']
        machine_operations = [[] for i in range(pb_instance['machinesNb'])]

        ms_s = self.split_ms(pb_instance, ms)  # machine for each operations

        indexes = [0] * len(ms_s)
        start_task_cstr = [0] * len(ms_s)

        # Iterating over OS to get task execution order and then checking in
        # MS to get the machine
        for job in os:
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

    def timeTaken(self,os_ms, pb_instance):
        (os, ms) = os_ms
        decoded = self.decode(pb_instance, os, ms)

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
    def elitistSelection(self,population, parameters):
        keptPopSize = int(self.pr * len(population))
        sortedPop = sorted(population, key=lambda cpl: self.timeTaken(cpl, parameters))
        return sortedPop[:keptPopSize]

    def tournamentSelection(self,population, parameters):
        b = 2

        selectedIndividuals = []
        for i in range(b):
            randomIndividual = random.randint(0, len(population) - 1)
            selectedIndividuals.append(population[randomIndividual])

        return min(selectedIndividuals, key=lambda cpl: self.timeTaken(cpl, parameters))
    def selection(self,population, parameters):
        newPop = self.elitistSelection(population, parameters)
        while len(newPop) < len(population):
            newPop.append(self.tournamentSelection(population, parameters))

        return newPop

    def precedenceOperationCrossover(self,p1, p2, parameters):
        J = parameters['jobs']
        jobNumber = len(J)
        jobsRange = range(1, jobNumber + 1)
        sizeJobset1 = random.randint(0, jobNumber)

        jobset1 = random.sample(jobsRange, sizeJobset1)

        o1 = []
        p1kept = []
        for i in range(len(p1)):
            e = p1[i]
            if e in jobset1:
                o1.append(e)
            else:
                o1.append(-1)
                p1kept.append(e)

        o2 = []
        p2kept = []
        for i in range(len(p2)):
            e = p2[i]
            if e in jobset1:
                o2.append(e)
            else:
                o2.append(-1)
                p2kept.append(e)

        for i in range(len(o1)):
            if o1[i] == -1:
                o1[i] = p2kept.pop(0)

        for i in range(len(o2)):
            if o2[i] == -1:
                o2[i] = p1kept.pop(0)

        return (o1, o2)

    def jobBasedCrossover(self,p1, p2, parameters):
        J = parameters['jobs']
        jobNumber = len(J)
        jobsRange = range(0, jobNumber)
        sizeJobset1 = random.randint(0, jobNumber)

        jobset1 = random.sample(jobsRange, sizeJobset1)
        jobset2 = [item for item in jobsRange if item not in jobset1]

        o1 = []
        p1kept = []
        for i in range(len(p1)):
            e = p1[i]
            if e in jobset1:
                o1.append(e)
                p1kept.append(e)
            else:
                o1.append(-1)

        o2 = []
        p2kept = []
        for i in range(len(p2)):
            e = p2[i]
            if e in jobset2:
                o2.append(e)
                p2kept.append(e)
            else:
                o2.append(-1)

        for i in range(len(o1)):
            if o1[i] == -1:
                o1[i] = p2kept.pop(0)

        for i in range(len(o2)):
            if o2[i] == -1:
                o2[i] = p1kept.pop(0)

        return (o1, o2)

    def crossoverOS(self,p1, p2, parameters):
        if random.choice([True, False]):
            return self.precedenceOperationCrossover(p1, p2, parameters)
        else:
            return self.jobBasedCrossover(p1, p2, parameters)

    def twoPointCrossover(self,p1, p2):
        pos1 = random.randint(0, len(p1) - 1)
        pos2 = random.randint(0, len(p1) - 1)

        if pos1 > pos2:
            pos2, pos1 = pos1, pos2

        offspring1 = p1
        if pos1 != pos2:
            offspring1 = p1[:pos1] + p2[pos1:pos2] + p1[pos2:]

        offspring2 = p2
        if pos1 != pos2:
            offspring2 = p2[:pos1] + p1[pos1:pos2] + p2[pos2:]

        return (offspring1, offspring2)

    def crossoverMS(self,p1, p2):
        return self.twoPointCrossover(p1, p2)
    def crossover(self,population, parameters):
        newPop = []
        i = 0
        while i < len(population):
            (OS1, MS1) = population[i]
            (OS2, MS2) = population[i + 1]

            if random.random() < self.pc:
                (oOS1, oOS2) = self.crossoverOS(OS1, OS2, parameters)
                (oMS1, oMS2) = self.crossoverMS(MS1, MS2)
                newPop.append((oOS1, oMS1))
                newPop.append((oOS2, oMS2))
            else:
                newPop.append((OS1, MS1))
                newPop.append((OS2, MS2))

            i = i + 2

        return newPop

    def swappingMutation(self,p):
        pos1 = random.randint(0, len(p) - 1)
        pos2 = random.randint(0, len(p) - 1)

        if pos1 == pos2:
            return p

        if pos1 > pos2:
            pos1, pos2 = pos2, pos1

        offspring = p[:pos1] + [p[pos2]] + \
                    p[pos1 + 1:pos2] + [p[pos1]] + \
                    p[pos2 + 1:]

        return offspring

    def neighborhoodMutation(self,p):
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

    def mutationOS(self,p):
        if random.choice([True, False]):
            return self.swappingMutation(p)
        else:
            return self.neighborhoodMutation(p)

    def halfMutation(self,p, parameters):
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
    def mutationMS(self,p, parameters):
        return self.halfMutation(p, parameters)
    def mutation(self,population, parameters):
        newPop = []

        for (OS, MS) in population:
            if random.random() < self.pm:
                oOS = self.mutationOS(OS)
                oMS = self.mutationMS(MS, parameters)
                newPop.append((oOS, oMS))
            else:
                newPop.append((OS, MS))

        return newPop
    def FJSP_run_episode(self,prblem,file):
        parameters = self.parse(file)
        t0 = time.time()
        best_minfiness = 9999999
        max_run_time = self.config.Pn_j * self.config.Pn_m
        gen = 1
        with open('./Result/' + self.config.optimizer + '-10 times runing solution.txt', 'a') as f:
            f.write(self.config.test_datas_type + str(self.config.Pn_j) + 'x' + str(self.config.Pn_m) + '\n')
            for repeat in range(10):
                min_makespan = 999999
                start_time = time.time()
                population = self.initializePopulation(parameters)
                for run in range(100000000):
                    population = self.selection(population, parameters)
                    population = self.crossover(population, parameters)
                    population = self.mutation(population, parameters)
                    sortedPop = sorted(population, key=lambda cpl: self.timeTaken(cpl, parameters))
                    fitness = prblem.cal_FJSP_objective(self.decode(parameters, sortedPop[0][0], sortedPop[0][1]))
                    if min_makespan > fitness:
                        min_makespan = fitness
                    if time.time() - start_time > max_run_time:
                        break

                if best_minfiness > min_makespan:
                    best_minfiness = min_makespan
                print('------------min_makespan:', min_makespan)
                f.write(str(min_makespan) + '\n')
        return best_minfiness



