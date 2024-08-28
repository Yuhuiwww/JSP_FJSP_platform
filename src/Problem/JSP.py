import numpy as np

from Problem.Basic_problem import Basic_Problem


class JSP(Basic_Problem):
    def __init__(self, job_num,machine_num):
        super().__init__()
        self.job_num = job_num
        self.machine_num = machine_num
    def permissibleLeftShift(self,job_op, a, durMat, mchMat, mchsStartTimes, opIDsOnMchs,mchsEndTimes,SpacestartTimesForMchOfaandendTimesForMchOfa):
        jobRdyTime_a, mchRdyTime_a = self.calJobAndMchRdyTimeOfa(job_op,a, mchMat, durMat, mchsStartTimes, opIDsOnMchs,mchsEndTimes)
        dur_a = durMat[a[0]][a[1]]
        mch_a = mchMat[a[0]][a[1]] - 1
        startTimesForMchOfa = mchsStartTimes[mch_a]
        endTimesForMchOfa = mchsEndTimes[mch_a]
        #SpacestartTimesForMchOfaandendTimesForMchOfa[mch_a] =endTimesForMchOfa-startTimesForMchOfa
        opsIDsForMchOfa = opIDsOnMchs[mch_a]
        flag = False

        possiblePos = np.where(jobRdyTime_a + dur_a <= startTimesForMchOfa)[0]

        if len(possiblePos) == 0:
            startTime_a = self.putInTheEnd(a,durMat, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa, endTimesForMchOfa)
        else:
            idxLegalPos, legalPos, endTimesForPossiblePos = self.calLegalPos(dur_a, jobRdyTime_a, durMat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa)
            if len(legalPos) == 0:
                startTime_a = self.putInTheEnd(a,durMat, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa, endTimesForMchOfa)
            else:
                flag = True
                startTime_a = self.putInBetween(a, durMat,idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa, opsIDsForMchOfa, endTimesForMchOfa)
        return startTime_a, flag



    def putInTheEnd(self, a,dur_a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa, endTimesForMchOfa):
        index = np.where(startTimesForMchOfa == -99999)[0][0]
        startTime_a = max(jobRdyTime_a, mchRdyTime_a)
        startTimesForMchOfa[index] = startTime_a
        endTimesForMchOfa[index] = startTime_a + dur_a[a[0]][a[1]]
        opsIDsForMchOfa[index] = a[0]
        return startTime_a

    def putInBetween(self, a,durMat, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa, opsIDsForMchOfa,
                     endTimesForMchOfa):
        earlstIdx = idxLegalPos[0]
        earlstPos = legalPos[0]
        startTime_a = endTimesForPossiblePos[earlstIdx]
        startTimesForMchOfa[:] = np.insert(startTimesForMchOfa, earlstPos, startTime_a)[:-1]
        endTimesForMchOfa[:] = np.insert(endTimesForMchOfa, earlstPos, startTime_a + durMat[a[0]][a[1]])[:-1]
        opsIDsForMchOfa[:] = np.insert(opsIDsForMchOfa, earlstPos, a[0])[:-1]
        return startTime_a


    def calLegalPos(self, dur_a, jobRdyTime_a, durMat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa):
        startTimesOfPossiblePos = startTimesForMchOfa[possiblePos]
        durOfPossiblePos = np.take(durMat, opsIDsForMchOfa[possiblePos])
        startTimeEarlst = max(jobRdyTime_a, startTimesForMchOfa[possiblePos[0]-1] + np.take(durMat, [opsIDsForMchOfa[possiblePos[0]-1]]))
        endTimesForPossiblePos = np.append(startTimeEarlst, (startTimesOfPossiblePos + durOfPossiblePos))[:-1]# end time for last ops don't care
        possibleGaps = startTimesOfPossiblePos - endTimesForPossiblePos
        idxLegalPos = np.where(dur_a <= possibleGaps)[0]
        legalPos = np.take(possiblePos, idxLegalPos)
        legalPos = []
        for idx in idxLegalPos:
            pos = possiblePos[idx]
            if startTimesForMchOfa[pos] == -99999 or startTimesForMchOfa[pos] > startTimesOfPossiblePos[idx]:
                legalPos.append(pos)

        legalPos = np.array(legalPos)
        return idxLegalPos, legalPos, endTimesForPossiblePos




    def calJobAndMchRdyTimeOfa(self, job_op,a, mchMat, durMat, mchsStartTimes, opIDsOnMchs, mchsEndTimes):
        mch_a = mchMat[a[0]][a[1]] - 1
        # cal jobRdyTime_a
        jobPredecessor = a[0] if a[1] != 0 else None
        if jobPredecessor is not None:
            durJobPredecessor = durMat[a[0]][a[1]-1]
            mchJobPredecessor = mchMat[a[0]][a[1]-1] - 1

            jobRdyTime_a = (mchsStartTimes[mchJobPredecessor][np.where(opIDsOnMchs[mchJobPredecessor] == jobPredecessor)] + durJobPredecessor).item()

        else:
            jobRdyTime_a = 0
        # cal mchRdyTime_a
        mchPredecessor = opIDsOnMchs[mch_a][np.where(opIDsOnMchs[mch_a] >= 0)][-1] if len(np.where(opIDsOnMchs[mch_a] >= 0)[0]) != 0 else None
        if mchPredecessor is not None:
            durMchPredecessor = durMat[mchPredecessor][mch_a]
           # mchRdyTime_a = (mchsStartTimes[mch_a][np.where(opIDsOnMchs[mch_a] >= 0)][-1] + durMchPredecessor).item()
            mchRdyTime_a =(mchsEndTimes[mch_a][np.where(opIDsOnMchs[mch_a] >= 0)][-1]).item()
            # print("------------------------")
            # print('opIDsOnMchs', opIDsOnMchs)
            # print('Job', a[0])
            # print('machine', mch_a)
            # print('mchPredecessor', mchPredecessor)
            #print('mchRdyTime_a', mchRdyTime_a)
            # print('mchRdyTime_a', mchRdyTime_a)
            # print('St', mchsStartTimes[mch_a][np.where(opIDsOnMchs[mch_a] >= 0)][-1])
            # print('process', durMchPredecessor)
            # print("start", mchsStartTimes)
            # print('end', mchsEndTimes)

        else:
            mchRdyTime_a = 0



        return jobRdyTime_a, mchRdyTime_a


    def cal_objective(self, sequence, dataset):
        cmp_job = np.zeros(self.job_num)  # Completion time of workpieces
        idle_machine = np.zeros(self.machine_num)  # Idle time of the machine
        job_op = np.zeros(self.job_num, dtype=np.int32)  # Operand
        # Initialize machine schedules
        machine_schedules = {i: [] for i in range(self.machine_num)}
        machine_start_end_times = {i: [] for i in range(self.machine_num)}  # Save start and end times

        # Initialize the machine start times and operation IDs
        mchsStartTimes = -99999 * np.ones((self.machine_num, self.job_num), dtype=np.int32)
        mchsEndTimes = -99999 * np.ones((self.machine_num, self.job_num), dtype=np.int32)
        opIDsOnMchs = -self.job_num * np.ones((self.machine_num, self.job_num), dtype=np.int32)
        SpacestartTimesForMchOfaandendTimesForMchOfa = -99999 * np.ones((self.machine_num, self.job_num), dtype=np.int32)
        flags = []
        number = 0;
        # sequence=[4 ,5, 5, 5, 1, 0, 4, 1, 3, 0, 2, 1, 1, 2, 2, 1, 0, 0, 0, 3, 0, 4, 2, 3, 3, 3 ,1, 3, 4, 5, 2 ,2, 5 ,4 ,4, 5];
        for operation in sequence:
            n_job = operation
            startTime_a, flag = self.permissibleLeftShift(job_op,
                (n_job, job_op[n_job]),
                durMat=dataset[0].astype(np.single),
                mchMat=dataset[-1],
                mchsStartTimes=mchsStartTimes,
                opIDsOnMchs=opIDsOnMchs,
                mchsEndTimes=mchsEndTimes,SpacestartTimesForMchOfaandendTimesForMchOfa=SpacestartTimesForMchOfaandendTimesForMchOfa)
            if(job_op[n_job]<self.machine_num):
                job_op[n_job] += 1
            number+=1
            flags.append(flag)
        makespan = np.max(mchsEndTimes)
        # print('Squence',sequence)
        #   print('--------------------------------operationMachine')
        #     print('------------opIDsOnMchs', opIDsOnMchs)
        #     print('------------mchsStartTimes', mchsStartTimes)
        #     print('---------------mchsEndTimes', mchsEndTimes)
        #     print('makespan', makespan)
        print('makespan', makespan)
        return makespan