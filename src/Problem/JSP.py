import numpy as np

from Problem.Basic_problem import Basic_Problem

class JSP(Basic_Problem):
    def __init__(self, job_num, machine_num,DecodingScheme):
        super().__init__()
        self.job_num = job_num
        self.machine_num = machine_num
        self.DecodingScheme =DecodingScheme

    def permissibleLeftShift(self, job_op, a, durMat, mchMat, mchsStartTimes, opIDsOnMchs, mchsEndTimes,
                             SpacestartTimesForMchOfaandendTimesForMchOfa):
        # print("----------a[0]",a[0])
        jobRdyTime_a, mchRdyTime_a = self.calJobAndMchRdyTimeOfa(job_op, a, mchMat, durMat, mchsStartTimes, opIDsOnMchs,
                                                                 mchsEndTimes)
        dur_a = durMat[a[0]][a[1]]
        mch_a = mchMat[a[0]][a[1]] - 1
        # print('dua',dur_a)
        # print('mch_a', mch_a)
        startTimesForMchOfa = mchsStartTimes[mch_a]
        endTimesForMchOfa = mchsEndTimes[mch_a]
        SpaceTime = SpacestartTimesForMchOfaandendTimesForMchOfa[mch_a]
        opsIDsForMchOfa = opIDsOnMchs[mch_a]

        flag = False

        possiblePos = np.where(dur_a < SpaceTime)[0]

        if len(possiblePos) == 0:
            startTime_a = self.putInTheEnd(a, durMat, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa,
                                           endTimesForMchOfa, SpaceTime)
        else:
            idxLegalPos, legalPos, startTimeEarlst = self.calLegalPos(dur_a, jobRdyTime_a, possiblePos,
                                                                      startTimesForMchOfa, endTimesForMchOfa,
                                                                      opsIDsForMchOfa, SpaceTime)

            if len(legalPos) == 0:
                startTime_a = self.putInTheEnd(a, durMat, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa,
                                               opsIDsForMchOfa, endTimesForMchOfa, SpaceTime)
            else:
                flag = True
                startTime_a = self.putInBetween(a, durMat, idxLegalPos, legalPos, startTimeEarlst, startTimesForMchOfa,
                                                endTimesForMchOfa, opsIDsForMchOfa, SpaceTime)
        return startTime_a, flag

    def putInTheEnd(self, a, dur_a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa, endTimesForMchOfa,
                    SpaceTime):
        index = np.where(startTimesForMchOfa == -99999)[0][0]
        startTime_a = max(jobRdyTime_a, mchRdyTime_a)
        startTimesForMchOfa[index] = startTime_a
        endTimesForMchOfa[index] = startTime_a + dur_a[a[0]][a[1]]
        opsIDsForMchOfa[index] = a[0]
        if index == 0:
            SpaceTime[index] = startTimesForMchOfa[index]
        else:
            SpaceTime[index] = startTimesForMchOfa[index] - endTimesForMchOfa[index - 1]
        return startTime_a

    def putInBetween(self, a, durMat, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa,
                     endTimesForMchOfa, opsIDsForMchOfa, SpaceTime):
        earlstIdx = idxLegalPos[0]
        earlstPos = legalPos[0]
        startTime_a = endTimesForPossiblePos
        ST = startTimesForMchOfa[earlstPos]
        ET = endTimesForMchOfa[earlstPos]
        startTimesForMchOfa[:] = np.insert(startTimesForMchOfa, earlstPos, startTime_a)[:-1]
        endTimesForMchOfa[:] = np.insert(endTimesForMchOfa, earlstPos, startTime_a + durMat[a[0]][a[1]])[:-1]
        opsIDsForMchOfa[:] = np.insert(opsIDsForMchOfa, earlstPos, a[0])[:-1]
        if (earlstPos == 0):
            SpaceTime[earlstPos] = startTime_a
        else:
            SpaceTime[earlstPos] = startTime_a - endTimesForMchOfa[earlstPos - 1]
        SpaceTime[:] = np.insert(SpaceTime, earlstPos + 1, ST - (startTime_a + durMat[a[0]][a[1]]))[:-1]

        return startTime_a

    def calLegalPos(self, dur_a, jobRdyTime_a, possiblePos, startTimesForMchOfa, endTimesForMchOfa, opsIDsForMchOfa,
                    SpaceTime):

        startTimeEarlst = max(jobRdyTime_a,
                              startTimesForMchOfa[possiblePos[0]] - SpaceTime[possiblePos[0]])  # possiblePos
        endTimesForPossiblePos = startTimeEarlst + dur_a
        idxLegalPos = np.where(endTimesForPossiblePos <= startTimesForMchOfa[possiblePos[0]])[0]
        legalPos = np.take(possiblePos, idxLegalPos)

        return idxLegalPos, legalPos, startTimeEarlst

    def calJobAndMchRdyTimeOfa(self, job_op, a, mchMat, durMat, mchsStartTimes, opIDsOnMchs, mchsEndTimes):
        mch_a = mchMat[a[0]][a[1]] - 1
        # cal jobRdyTime_a
        jobPredecessor = a[0] if a[1] != 0 else None
        if jobPredecessor is not None:
            durJobPredecessor = durMat[a[0]][a[1] - 1]
            mchJobPredecessor = mchMat[a[0]][a[1] - 1] - 1

            idx = np.where(opIDsOnMchs[mchJobPredecessor] == jobPredecessor)
            if len(idx[0]) == 1:
                jobRdyTime_a = (mchsStartTimes[mchJobPredecessor][idx] + durJobPredecessor).item()
            elif len(idx[0]) == 0:
                jobRdyTime_a = 0
            else:
                raise ValueError("Multiple indices found for job predecessor.")
        else:
            jobRdyTime_a = 0
        # cal mchRdyTime_a
        mchPredecessor = opIDsOnMchs[mch_a][np.where(opIDsOnMchs[mch_a] >= 0)][-1] if len(
            np.where(opIDsOnMchs[mch_a] >= 0)[0]) != 0 else None
        if mchPredecessor is not None:
            durMchPredecessor = durMat[mchPredecessor][mch_a]
            mchRdyTime_a = (mchsEndTimes[mch_a][np.where(opIDsOnMchs[mch_a] >= 0)][-1]).item()
        else:
            mchRdyTime_a = 0

        return jobRdyTime_a, mchRdyTime_a

    def cal_objective(self, sequence, dataset):

        if(self.DecodingScheme=='full'):
            cmp_job = np.zeros(self.job_num)  # Completion time of workpieces
            idle_machine = np.zeros(self.machine_num)  # Idle time of the machine
            job_op = np.zeros(self.job_num, dtype=np.int32)  # Operand

            # Initialize the machine start times and operation IDs
            mchsStartTimes = -99999 * np.ones((self.machine_num, self.job_num), dtype=np.int32)
            mchsEndTimes = -99999 * np.ones((self.machine_num, self.job_num), dtype=np.int32)
            opIDsOnMchs = -self.job_num * np.ones((self.machine_num, self.job_num), dtype=np.int32)
            OperationNumber = -self.job_num * np.ones((self.machine_num, self.job_num), dtype=np.int32)
            SpacestartTimesForMchOfaandendTimesForMchOfa = -99999 * np.ones((self.machine_num, self.job_num),
                                                                            dtype=np.int32)
            flags = []
            number = 0
            for operation in sequence:
                n_job = operation
                startTime_a, flag = self.permissibleLeftShift(job_op,
                                                              (n_job, job_op[n_job]),
                                                              durMat=dataset[0].astype(np.single),
                                                              mchMat=dataset[-1],
                                                              mchsStartTimes=mchsStartTimes,
                                                              opIDsOnMchs=opIDsOnMchs,
                                                              mchsEndTimes=mchsEndTimes,
                                                              SpacestartTimesForMchOfaandendTimesForMchOfa=SpacestartTimesForMchOfaandendTimesForMchOfa)
                job_op[n_job] += 1
                number += 1
                flags.append(flag)
            makespan = np.max(mchsEndTimes)
            return makespan
        else:
            job_op = np.zeros(self.job_num, dtype=np.int32)  # operand
            cmp_job = np.zeros(self.job_num)  # Completion time of workpieces
            Idel_machine = np.zeros(self.machine_num)  # Idle time of the machine
            for i in (sequence):
                n_job = i
                m_machine = dataset[1][n_job][job_op[n_job]] - 1  # Get the machine where the workpiece is located
                process_time = dataset[0][n_job][job_op[n_job]]  # Get processing time
                completion_t = max(cmp_job[n_job], Idel_machine[
                    m_machine]) + process_time  # Obtaining the completion time of the workpiece and the free time of the machine.
                cmp_job[n_job] = completion_t
                Idel_machine[m_machine] = completion_t
                job_op[n_job] += 1
            makespan = max(cmp_job)
            return makespan
