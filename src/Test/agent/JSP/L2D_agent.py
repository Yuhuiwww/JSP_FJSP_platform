import numpy as np
from gym.utils import EzPickle
from torch.distributions.categorical import Categorical

from Test.agent.JSP.Basic_agent import Basic_Agent


def lastNonZero(arr, axis, invalid_val=-1):
    mask = arr != 0
    val = arr.shape[axis] - np.flip(mask, axis=axis).argmax(axis=axis) - 1
    yAxis = np.where(mask.any(axis=axis), val, invalid_val)
    xAxis = np.arange(arr.shape[0], dtype=np.int64)
    xRet = xAxis[yAxis >= 0]
    yRet = yAxis[yAxis >= 0]
    return xRet, yRet

def permute_rows(x):
    '''
    x is a np array
    '''
    ix_i = np.tile(np.arange(x.shape[0]), (x.shape[1], 1)).T
    ix_j = np.random.sample(x.shape).argsort(axis=1)
    return x[ix_i, ix_j]


def uni_instance_gen(n_j, n_m, low, high):
    times = np.random.randint(low=low, high=high, size=(n_j, n_m))
    machines = np.expand_dims(np.arange(1, n_m+1), axis=0).repeat(repeats=n_j, axis=0)
    machines = permute_rows(machines)
    return times, machines

def calEndTimeLB(temp1, dur_cp):
    x, y = lastNonZero(temp1, 1, invalid_val=-1)
    dur_cp[np.where(temp1 != 0)] = 0
    dur_cp[x, y] = temp1[x, y]
    temp2 = np.cumsum(dur_cp, axis=1)
    temp2[np.where(temp1 != 0)] = 0
    ret = temp1+temp2
    return ret
def getActionNbghs(action, opIDsOnMchs):
    coordAction = np.where(opIDsOnMchs == action)
    precd = opIDsOnMchs[coordAction[0], coordAction[1] - 1 if coordAction[1].item() > 0 else coordAction[1]].item()
    succdTemp = opIDsOnMchs[coordAction[0], coordAction[1] + 1 if coordAction[1].item() + 1 < opIDsOnMchs.shape[-1] else coordAction[1]].item()
    succd = action if succdTemp < 0 else succdTemp
    # precedX = coordAction[0]
    # precedY = coordAction[1] - 1 if coordAction[1].item() > 0 else coordAction[1]
    # succdX = coordAction[0]
    # succdY = coordAction[1] + 1 if coordAction[1].item()+1 < opIDsOnMchs.shape[-1] else coordAction[1]
    return precd, succd

# evaluate the actions
def eval_actions(p: object, actions: object) -> object:
    softmax_dist = Categorical(p)
    ret = softmax_dist.log_prob(actions).reshape(-1)
    entropy = softmax_dist.entropy().mean()
    return ret, entropy

def select_action(p, cadidate, memory):
    dist = Categorical(p.squeeze())
    s = dist.sample()
    if memory is not None: memory.logprobs.append(dist.log_prob(s))
    return cadidate[s], s
 # select action method for test
def greedy_select_action(p, candidate):
    _, index = p.squeeze().max(0)
    action = candidate[index]
    return action

# select action method for test
def sample_select_action(p, candidate):
    dist = Categorical(p.squeeze())
    s = dist.sample()
    return candidate[s]
class L2D_agent(Basic_Agent):
    def __init__(self,config):
        EzPickle.__init__(self)

        self.step_count = 0
        self.number_of_jobs = config.Pn_j
        self.number_of_machines = config.Pn_m
        self.number_of_tasks = self.number_of_jobs * self.number_of_machines
        # the task id for first column
        self.first_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, 0]
        # the task id for last column
        self.last_col = np.arange(start=0, stop=self.number_of_tasks, step=1).reshape(self.number_of_jobs, -1)[:, -1]
        self.getEndTimeLB = calEndTimeLB
        self.getNghbs = getActionNbghs

    def done(self):
        if len(self.partial_sol_sequeence) == self.number_of_tasks:
            return True
        return False

    def transit(self, action):
        # action is a int 0 - 224 for 15x15 for example
        # redundant action makes no effect
        if action not in self.partial_sol_sequeence:

            # UPDATE BASIC INFO:
            row = action // self.number_of_machines
            col = action % self.number_of_machines
            self.step_count += 1
            self.finished_mark[row, col] = 1
            dur_a = self.dur[row, col]
            self.partial_sol_sequeence.append(action)

            # UPDATE STATE:
            # permissible left shift
            startTime_a, flag = self.permissibleLeftShift(a=action, durMat=self.dur, mchMat=self.m,
                                                          mchsStartTimes=self.mchsStartTimes, opIDsOnMchs=self.opIDsOnMchs)
            self.flags.append(flag)
            # update omega or mask
            if action not in self.last_col:
                self.omega[action // self.number_of_machines] += 1
            else:
                self.mask[action // self.number_of_machines] = 1

            self.temp1[row, col] = startTime_a + dur_a

            self.LBs = calEndTimeLB(self.temp1, self.dur_cp)

            # adj matrix
            precd, succd = self.getNghbs(action, self.opIDsOnMchs)
            self.adj[action] = 0
            self.adj[action, action] = 1
            if action not in self.first_col:
                self.adj[action, action - 1] = 1
            self.adj[action, precd] = 1
            self.adj[succd, action] = 1
            if flag and precd != action and succd != action:  # Remove the old arc when a new operation inserts between two operations
                self.adj[succd, precd] = 0

        # prepare for return
        fea = np.concatenate((self.LBs.reshape(-1, 1) / self.configs.et_normalize_coef,
                              self.finished_mark.reshape(-1, 1)), axis=1)
        reward = - (self.LBs.max() - self.max_endTime)
        if reward == 0:
            reward = self.configs.rewardscale
            self.posRewards += reward
        self.max_endTime = self.LBs.max()

        return self.adj, fea, reward, self.done(), self.omega, self.mask
    def reset(self, data,config):
        self.configs=config
        self.step_count = 0
        self.m = data[-1]
        self.dur = data[0].astype(np.single)
        self.dur_cp = np.copy(self.dur)
        # record action history
        self.partial_sol_sequeence = []
        self.flags = []
        self.posRewards = 0

        # initialize adj matrix
        conj_nei_up_stream = np.eye(self.number_of_tasks, k=-1, dtype=np.single)
        conj_nei_low_stream = np.eye(self.number_of_tasks, k=1, dtype=np.single)
        # first column does not have upper stream conj_nei
        conj_nei_up_stream[self.first_col] = 0
        # last column does not have lower stream conj_nei
        conj_nei_low_stream[self.last_col] = 0
        self_as_nei = np.eye(self.number_of_tasks, dtype=np.single)
        self.adj = self_as_nei + conj_nei_up_stream

        # initialize features
        self.LBs = np.cumsum(self.dur, axis=1, dtype=np.single)
        self.initQuality = self.LBs.max() if not self.configs.init_quality_flag else 0
        self.max_endTime = self.initQuality
        self.finished_mark = np.zeros_like(self.m, dtype=np.single)

        fea = np.concatenate((self.LBs.reshape(-1, 1) / self.configs.et_normalize_coef,
                              # self.dur.reshape(-1, 1)/configs.high,
                              # wkr.reshape(-1, 1)/configs.wkr_normalize_coef,
                              self.finished_mark.reshape(-1, 1)), axis=1)
        # initialize feasible omega
        self.omega = self.first_col.astype(np.int64)

        # initialize mask
        self.mask = np.full(shape=self.number_of_jobs, fill_value=0, dtype=bool)

        # start time of operations on machines
        self.mchsStartTimes = -self.configs.high * np.ones_like(self.dur.transpose(), dtype=np.int32)
        # Ops ID on machines
        self.opIDsOnMchs = -self.number_of_jobs * np.ones_like(self.dur.transpose(), dtype=np.int32)

        self.temp1 = np.zeros_like(self.dur, dtype=np.single)

        return self.adj, fea, self.omega, self.mask

    def permissibleLeftShift(self, a, durMat, mchMat, mchsStartTimes, opIDsOnMchs):
        jobRdyTime_a, mchRdyTime_a = self.calJobAndMchRdyTimeOfa(a, mchMat, durMat, mchsStartTimes, opIDsOnMchs)
        dur_a = np.take(durMat, a)
        mch_a = np.take(mchMat, a) - 1
        startTimesForMchOfa = mchsStartTimes[mch_a]
        opsIDsForMchOfa = opIDsOnMchs[mch_a]
        flag = False

        possiblePos = np.where(jobRdyTime_a < startTimesForMchOfa)[0]
        # print('possiblePos:', possiblePos)
        if len(possiblePos) == 0:
            startTime_a = self.putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa)
        else:
            idxLegalPos, legalPos, endTimesForPossiblePos = self.calLegalPos(dur_a, jobRdyTime_a, durMat, possiblePos,
                                                                        startTimesForMchOfa, opsIDsForMchOfa)
            # print('legalPos:', legalPos)
            if len(legalPos) == 0:
                startTime_a = self.putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa)
            else:
                flag = True
                startTime_a = self.putInBetween(a, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa,
                                           opsIDsForMchOfa)
        return startTime_a, flag

    def putInTheEnd(self,a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa):
        idx = np.where(startTimesForMchOfa == -self.configs.high)
        if len(idx[0]) == 0:
            print("没有满足条件的索引")
        index = np.where(startTimesForMchOfa == -self.configs.high)[0][0]
        startTime_a = max(jobRdyTime_a, mchRdyTime_a)
        startTimesForMchOfa[index] = startTime_a
        opsIDsForMchOfa[index] = a
        return startTime_a

    def calLegalPos(self, dur_a, jobRdyTime_a, durMat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa):
        startTimesOfPossiblePos = startTimesForMchOfa[possiblePos]
        durOfPossiblePos = np.take(durMat, opsIDsForMchOfa[possiblePos])
        startTimeEarlst = max(jobRdyTime_a, startTimesForMchOfa[possiblePos[0] - 1] + np.take(durMat, [
            opsIDsForMchOfa[possiblePos[0] - 1]]))
        endTimesForPossiblePos = np.append(startTimeEarlst, (startTimesOfPossiblePos + durOfPossiblePos))[
                                 :-1]  # end time for last ops don't care
        possibleGaps = startTimesOfPossiblePos - endTimesForPossiblePos
        idxLegalPos = np.where(dur_a <= possibleGaps)[0]
        legalPos = np.take(possiblePos, idxLegalPos)
        return idxLegalPos, legalPos, endTimesForPossiblePos

    def putInBetween(self,a, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa, opsIDsForMchOfa):
        earlstIdx = idxLegalPos[0]
        # print('idxLegalPos:', idxLegalPos)
        earlstPos = legalPos[0]
        startTime_a = endTimesForPossiblePos[earlstIdx]
        # print('endTimesForPossiblePos:', endTimesForPossiblePos)
        startTimesForMchOfa[:] = np.insert(startTimesForMchOfa, earlstPos, startTime_a)[:-1]
        opsIDsForMchOfa[:] = np.insert(opsIDsForMchOfa, earlstPos, a)[:-1]
        return startTime_a

    def calJobAndMchRdyTimeOfa(self,a, mchMat, durMat, mchsStartTimes, opIDsOnMchs):
        mch_a = np.take(mchMat, a) - 1
        # cal jobRdyTime_a
        jobPredecessor = a - 1 if a % mchMat.shape[1] != 0 else None
        if jobPredecessor is not None:
            durJobPredecessor = np.take(durMat, jobPredecessor)
            mchJobPredecessor = np.take(mchMat, jobPredecessor) - 1
            jobRdyTime_a = (mchsStartTimes[mchJobPredecessor][
                                np.where(opIDsOnMchs[mchJobPredecessor] == jobPredecessor)] + durJobPredecessor).item()
        else:
            jobRdyTime_a = 0
        # cal mchRdyTime_a
        mchPredecessor = opIDsOnMchs[mch_a][np.where(opIDsOnMchs[mch_a] >= 0)][-1] if len(
            np.where(opIDsOnMchs[mch_a] >= 0)[0]) != 0 else None
        if mchPredecessor is not None:
            durMchPredecessor = np.take(durMat, mchPredecessor)
            mchRdyTime_a = (mchsStartTimes[mch_a][np.where(mchsStartTimes[mch_a] >= 0)][-1] + durMchPredecessor).item()
        else:
            mchRdyTime_a = 0

        return jobRdyTime_a, mchRdyTime_a


