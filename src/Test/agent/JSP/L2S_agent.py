import collections
import os
import sys
from torch import Tensor
from typing import Any, Dict, Optional, Tuple, Union
from ortools.sat.python import cp_model
from torch_geometric.typing import OptPairTensor, Adj, Size

import torch
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.nn.conv import MessagePassing

from Test.agent.JSP.Basic_agent import Basic_Agent

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = torch.device(device)
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
def _rebuild_from_type_v2(func, new_type, args, state):
    ret = func(*args)
    if type(ret) is not new_type:
        ret = ret.as_subclass(new_type)
    # Tensor does define __setstate__ even though it doesn't define
    # __getstate__. So only use __setstate__ if it is NOT the one defined
    # on Tensor
    if (
        getattr(ret.__class__, "__setstate__", Tensor.__setstate__)
        is not Tensor.__setstate__
    ):
        ret.__setstate__(state)
    else:
        ret = torch._utils._set_obj_state(ret, state)
    return ret

def processing_order_to_edge_index(order, instance):
    """
    order: [n_m, n_j] a numpy array specifying the processing order on each machine, each row is a machine
    instance: [1, n_j, n_m] an instance as numpy array
    RETURN: edge index: [2, n_j * n_m +2] tensor for the directed disjunctive graph
    """
    dur, mch = instance[0], instance[1]
    n_j, n_m = dur.shape[0], dur.shape[1]
    n_opr = n_j*n_m

    adj = np.eye(n_opr, k=-1, dtype=int)  # Create adjacent matrix for precedence constraints
    adj[np.arange(start=0, stop=n_opr, step=1).reshape(n_j, -1)[:, 0]] = 0  # first column does not have upper stream conj_nei
    adj = np.pad(adj, 1, 'constant', constant_values=0)  # pad dummy S and T nodes
    adj[[i for i in range(1, n_opr + 2 - 1, n_m)], 0] = 1  # connect S with 1st operation of each job
    adj[-1, [i for i in range(n_m, n_opr + 2 - 1, n_m)]] = 1  # connect last operation of each job to T
    adj = np.transpose(adj)

    # rollout ortools solution
    steps_basedon_sol = []
    for i in range(n_m):
        get_col_position_unsorted = np.argwhere(mch == (i + 1))
        get_col_position_sorted = get_col_position_unsorted[order[i]]
        sol_i = order[i] * n_m + get_col_position_sorted[:, 1]
        steps_basedon_sol.append(sol_i.tolist())

    for operations in steps_basedon_sol:
        for i in range(len(operations) - 1):
            adj[operations[i]+1][operations[i+1]+1] += 1

    return torch.nonzero(torch.from_numpy(adj)).t().contiguous()


def forward_pass(graph, topological_order=None):  # graph is a nx.DiGraph;
    # assert (graph.in_degree(topological_order[0]) == 0)
    earliest_ST = dict.fromkeys(graph.nodes, -float('inf'))
    if topological_order is None:
        topo_order = list(nx.topological_sort(graph))
    else:
        topo_order = topological_order
    earliest_ST[topo_order[0]] = 0.
    for n in topo_order:
        for s in graph.successors(n):
            if earliest_ST[s] < earliest_ST[n] + graph.edges[n, s]['weight']:
                earliest_ST[s] = earliest_ST[n] + graph.edges[n, s]['weight']
    # return is a dict where key is each node's ID, value is the length from source node s
    return earliest_ST


def backward_pass(graph, makespan, topological_order=None):
    if topological_order is None:
        reverse_order = list(reversed(list(nx.topological_sort(graph))))
    else:
        reverse_order = list(reversed(topological_order))
    latest_ST = dict.fromkeys(graph.nodes, float('inf'))
    latest_ST[reverse_order[0]] = float(makespan)
    for n in reverse_order:
        for p in graph.predecessors(n):
            if latest_ST[p] > latest_ST[n] - graph.edges[p, n]['weight']:
                # assert latest_ST[n] - graph.edges[p, n]['weight'] >= 0, 'latest start times should is negative, BUG!'  # latest start times should be non-negative
                latest_ST[p] = latest_ST[n] - graph.edges[p, n]['weight']
    return latest_ST


def forward_and_backward_pass(G):
    # calculate topological order
    topological_order = list(nx.topological_sort(G))
    # forward and backward pass
    est = np.fromiter(forward_pass(graph=G, topological_order=topological_order).values(), dtype=np.float32)
    lst = np.fromiter(backward_pass(graph=G, topological_order=topological_order, makespan=est[-1]).values(), dtype=np.float32)
    # assert np.where(est > lst)[0].shape[0] == 0, 'latest starting time is smaller than earliest starting time, bug!'  # latest starting time should be larger or equal to earliest starting time
    return est, lst, est[-1]


def CPM_batch_G(Gs, dev):
    multi_est = []
    multi_lst = []
    multi_makespan = []
    for G in Gs:
        est, lst, makespan = forward_and_backward_pass(G)
        multi_est.append(est)
        multi_lst.append(lst)
        multi_makespan.append([makespan])
    multi_est = torch.from_numpy(np.concatenate(multi_est, axis=0)).view(-1, 1).to(dev)
    multi_lst = torch.from_numpy(np.concatenate(multi_lst, axis=0)).view(-1, 1).to(dev)
    multi_makespan = torch.tensor(multi_makespan, device=dev)
    return multi_est, multi_lst, multi_makespan
def permissibleLeftShift(a, durMat, mchMat, mchsStartTimes, opIDsOnMchs,config):
    jobRdyTime_a, mchRdyTime_a = calJobAndMchRdyTimeOfa(a, mchMat, durMat, mchsStartTimes, opIDsOnMchs)
    dur_a = np.take(durMat, a)
    mch_a = np.take(mchMat, a) - 1
    startTimesForMchOfa = mchsStartTimes[mch_a]
    opsIDsForMchOfa = opIDsOnMchs[mch_a]
    flag = False

    possiblePos = np.where(jobRdyTime_a < startTimesForMchOfa)[0]
    # print('possiblePos:', possiblePos)
    if len(possiblePos) == 0:
        startTime_a = putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa,config)
    else:
        idxLegalPos, legalPos, endTimesForPossiblePos = calLegalPos(dur_a, jobRdyTime_a, durMat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa)
        # print('legalPos:', legalPos)
        if len(legalPos) == 0:
            startTime_a = putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa,config)
        else:
            flag = True
            startTime_a = putInBetween(a, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa, opsIDsForMchOfa)
    return startTime_a, flag


def putInTheEnd(a, jobRdyTime_a, mchRdyTime_a, startTimesForMchOfa, opsIDsForMchOfa,config):
    # index = first position of -config.high in startTimesForMchOfa
    # print('Yes!OK!')
    index = np.where(startTimesForMchOfa == -config.high)[0][0]
    startTime_a = max(jobRdyTime_a, mchRdyTime_a)
    startTimesForMchOfa[index] = startTime_a
    opsIDsForMchOfa[index] = a
    return startTime_a


def calLegalPos(dur_a, jobRdyTime_a, durMat, possiblePos, startTimesForMchOfa, opsIDsForMchOfa):
    startTimesOfPossiblePos = startTimesForMchOfa[possiblePos]
    durOfPossiblePos = np.take(durMat, opsIDsForMchOfa[possiblePos])
    startTimeEarlst = max(jobRdyTime_a, startTimesForMchOfa[possiblePos[0]-1] + np.take(durMat, [opsIDsForMchOfa[possiblePos[0]-1]]))
    endTimesForPossiblePos = np.append(startTimeEarlst, (startTimesOfPossiblePos + durOfPossiblePos))[:-1]# end time for last ops don't care
    possibleGaps = startTimesOfPossiblePos - endTimesForPossiblePos
    idxLegalPos = np.where(dur_a <= possibleGaps)[0]
    legalPos = np.take(possiblePos, idxLegalPos)
    return idxLegalPos, legalPos, endTimesForPossiblePos


def putInBetween(a, idxLegalPos, legalPos, endTimesForPossiblePos, startTimesForMchOfa, opsIDsForMchOfa):
    earlstIdx = idxLegalPos[0]
    # print('idxLegalPos:', idxLegalPos)
    earlstPos = legalPos[0]
    startTime_a = endTimesForPossiblePos[earlstIdx]
    # print('endTimesForPossiblePos:', endTimesForPossiblePos)
    startTimesForMchOfa[:] = np.insert(startTimesForMchOfa, earlstPos, startTime_a)[:-1]
    opsIDsForMchOfa[:] = np.insert(opsIDsForMchOfa, earlstPos, a)[:-1]
    return startTime_a


def calJobAndMchRdyTimeOfa(a, mchMat, durMat, mchsStartTimes, opIDsOnMchs):
    mch_a = np.take(mchMat, a) - 1
    # cal jobRdyTime_a
    jobPredecessor = a - 1 if a % mchMat.shape[1] != 0 else None
    if jobPredecessor is not None:
        durJobPredecessor = np.take(durMat, jobPredecessor)
        mchJobPredecessor = np.take(mchMat, jobPredecessor) - 1
        jobRdyTime_a = (mchsStartTimes[mchJobPredecessor][np.where(opIDsOnMchs[mchJobPredecessor] == jobPredecessor)] + durJobPredecessor).item()
    else:
        jobRdyTime_a = 0
    # cal mchRdyTime_a
    mchPredecessor = opIDsOnMchs[mch_a][np.where(opIDsOnMchs[mch_a] >= 0)][-1] if len(np.where(opIDsOnMchs[mch_a] >= 0)[0]) != 0 else None
    if mchPredecessor is not None:
        durMchPredecessor = np.take(durMat, mchPredecessor)
        mchRdyTime_a = (mchsStartTimes[mch_a][np.where(mchsStartTimes[mch_a] >= 0)][-1] + durMchPredecessor).item()
    else:
        mchRdyTime_a = 0

    return jobRdyTime_a, mchRdyTime_a
def MinimalJobshopSat(data):
    """Minimal jobshop problem."""
    # Create the model.
    model = cp_model.CpModel()

    jobs_data = data
    n_j = len(jobs_data)
    n_m = len(jobs_data[0])

    machines_count = 1 + max(task[0] for job in jobs_data for task in job)
    all_machines = range(machines_count)

    # Computes horizon dynamically as the sum of all durations.
    horizon = sum(task[1] for job in jobs_data for task in job)

    # Named tuple to store information about created variables.
    task_type = collections.namedtuple('task_type', 'start end interval')
    # Named tuple to manipulate solution information.
    assigned_task_type = collections.namedtuple('assigned_task_type',
                                                'start job index duration')

    # Creates job intervals and add to the corresponding machine lists.
    all_tasks = {}
    machine_to_intervals = collections.defaultdict(list)

    for job_id, job in enumerate(jobs_data):
        for task_id, task in enumerate(job):
            machine = task[0]
            duration = task[1]
            suffix = '_%i_%i' % (job_id, task_id)
            start_var = model.NewIntVar(0, horizon, 'start' + suffix)
            end_var = model.NewIntVar(0, horizon, 'end' + suffix)
            interval_var = model.NewIntervalVar(start_var, duration, end_var,
                                                'interval' + suffix)
            all_tasks[job_id, task_id] = task_type(
                start=start_var, end=end_var, interval=interval_var)
            machine_to_intervals[machine].append(interval_var)

    # Create and add disjunctive constraints.
    for machine in all_machines:
        model.AddNoOverlap(machine_to_intervals[machine])

    # Precedences inside a job.
    for job_id, job in enumerate(jobs_data):
        for task_id in range(len(job) - 1):
            model.Add(all_tasks[job_id, task_id +
                                1].start >= all_tasks[job_id, task_id].end)

    # Makespan objective.
    obj_var = model.NewIntVar(0, horizon, 'makespan')
    model.AddMaxEquality(obj_var, [
        all_tasks[job_id, len(job) - 1].end
        for job_id, job in enumerate(jobs_data)
    ])
    model.Minimize(obj_var)

    # Solve model.
    solver = cp_model.CpSolver()
    # solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)

    if status == cp_model.OPTIMAL:
        # Create one list of assigned tasks per machine.
        assigned_jobs = collections.defaultdict(list)
        for job_id, job in enumerate(jobs_data):
            for task_id, task in enumerate(job):
                machine = task[0]
                assigned_jobs[machine].append(
                    assigned_task_type(
                        start=solver.Value(all_tasks[job_id, task_id].start),
                        job=job_id,
                        index=task_id,
                        duration=task[1]))

        # Create per machine output lines.
        output = ''
        machine_assign_mat = []
        for machine in all_machines:
            # Sort by starting time.
            assigned_jobs[machine].sort()
            sol_line_tasks = 'Machine ' + str(machine) + ': '
            sol_line = '           '

            for assigned_task in assigned_jobs[machine]:
                name = 'job_%i_%i' % (assigned_task.job, assigned_task.index)
                machine_assign_mat.append(assigned_task.job)
                # Add spaces to output to align columns.
                sol_line_tasks += '%-10s' % name

                start = assigned_task.start
                duration = assigned_task.duration
                sol_tmp = '[%i,%i]' % (start, start + duration)
                # Add spaces to output to align columns.
                sol_line += '%-10s' % sol_tmp

            sol_line += '\n'
            sol_line_tasks += '\n'
            output += sol_line_tasks
            output += sol_line

        return solver.ObjectiveValue(), np.array(machine_assign_mat).reshape((n_m, n_j))

class ForwardPass(MessagePassing):
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'max')
        super(ForwardPass, self).__init__(**kwargs)

    def forward(self,
                x: Union[Tensor, OptPairTensor],
                edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        # propagate_type: (x: OptPairTensor)
        out = self.propagate(edge_index, x=x, size=size)
        return out


class BackwardPass(MessagePassing):
    def __init__(self, **kwargs):
        kwargs.setdefault('aggr', 'max')
        super(BackwardPass, self).__init__(**kwargs)

    def forward(self,
                x: Union[Tensor, OptPairTensor],
                edge_index: Adj,
                size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)
        out = self.propagate(edge_index, x=x, size=size)
        return out


class Evaluator:
    def __init__(self):
        self.forward_pass = ForwardPass(aggr='max', flow="source_to_target")
        self.backward_pass = BackwardPass(aggr='max', flow="target_to_source")

    def forward(self, edge_index, duration, n_j, n_m):
        """
        support batch version
        edge_index: [2, n_edges] tensor
        duration: [n_nodes, 1] tensor
        """
        n_nodes = duration.shape[0]
        n_nodes_each_graph = n_j * n_m + 2
        device = edge_index.device

        # forward pass...
        index_S = np.arange(n_nodes // n_nodes_each_graph, dtype=int) * n_nodes_each_graph
        earliest_start_time = torch.zeros_like(duration, dtype=torch.float32, device=device)
        mask_earliest_start_time = torch.ones_like(duration, dtype=torch.int8, device=device)
        mask_earliest_start_time[index_S] = 0
        for _ in range(n_nodes):
            if mask_earliest_start_time.sum() == 0:
                break
            x_forward = duration + earliest_start_time.masked_fill(mask_earliest_start_time.bool(), 0)
            earliest_start_time = self.forward_pass(x=x_forward, edge_index=edge_index)
            mask_earliest_start_time = self.forward_pass(x=mask_earliest_start_time, edge_index=edge_index)

        # backward pass...
        index_T = np.cumsum(np.ones(shape=[n_nodes // n_nodes_each_graph], dtype=int) * n_nodes_each_graph) - 1
        make_span = earliest_start_time[index_T]
        # latest_start_time = torch.zeros_like(duration, dtype=torch.float32, device=device)
        latest_start_time = - torch.ones_like(duration, dtype=torch.float32, device=device)
        latest_start_time[index_T] = - make_span
        mask_latest_start_time = torch.ones_like(duration, dtype=torch.int8, device=device)
        mask_latest_start_time[index_T] = 0
        for _ in range(n_nodes):
            if mask_latest_start_time.sum() == 0:
                break
            x_backward = latest_start_time.masked_fill(mask_latest_start_time.bool(), 0)
            latest_start_time = self.backward_pass(x=x_backward, edge_index=edge_index) + duration
            latest_start_time[index_T] = - make_span
            mask_latest_start_time = self.backward_pass(x=mask_latest_start_time, edge_index=edge_index)

        return earliest_start_time, torch.abs(latest_start_time), make_span


class BatchGraph:
    def __init__(self):
        self.x = None
        self.edge_index_pc = None
        self.edge_index_mc = None
        self.batch = None

    def wrapper(self, x, edge_index_pc, edge_index_mc, batch):
        self.x = x
        self.edge_index_pc = edge_index_pc
        self.edge_index_mc = edge_index_mc
        self.batch = batch

    def clean(self):
        self.x = None
        self.edge_index_pc = None
        self.edge_index_mc = None
        self.batch = None



class L2S_agent(Basic_Agent):
    def __init__(self, config):

        self.n_job = config.Pn_j
        self.n_mch = config.Pn_m
        self.n_oprs = self.n_job * self.n_mch
        self.low = config.low
        self.high = config.high
        self.itr = 0
        self.instances = None
        self.sub_graphs_mc = None
        self.current_graphs = None
        self.current_objs = None
        self.tabu_size = 1
        self.tabu_lists = None
        self.incumbent_objs = None
        self.fea_norm_const = 1000
        self.evaluator_type = 'message-passing'
        self.eva = Evaluator() if self.evaluator_type == 'message-passing' else CPM_batch_G
        self.adj_mat_pc = self._adj_mat_pc()


    def _adj_mat_pc(self):
        adj_mat_pc = np.eye(self.n_oprs, k=-1, dtype=int)  # Create adjacent matrix for precedence constraints
        adj_mat_pc[np.arange(start=0, stop=self.n_oprs, step=1).reshape(self.n_job, -1)[:, 0]] = 0  # first column does not have upper stream conj_nei
        adj_mat_pc = np.pad(adj_mat_pc, 1, 'constant', constant_values=0)  # pad dummy S and T nodes
        adj_mat_pc[[i for i in range(1, self.n_job * self.n_mch + 2 - 1, self.n_mch)], 0] = 1  # connect S with 1st operation of each job
        adj_mat_pc[-1, [i for i in range(self.n_mch, self.n_job * self.n_mch + 2 - 1, self.n_mch)]] = 1  # connect last operation of each job to T
        adj_mat_pc = np.transpose(adj_mat_pc)  # convert input adj from column pointing to row, to, row pointing to column
        return adj_mat_pc


    def _gen_moves(self, solution, mch_mat, tabu_list=None):
        """
        solution: networkx DAG conjunctive graph
        mch_mat: the same mch from our NeurIPS 2020 paper of solution
        """
        critical_path = nx.dag_longest_path(solution)[1:-1]
        critical_blocks_opr = np.array(critical_path)
        critical_blocks = mch_mat.take(critical_blocks_opr - 1)  # -1: ops id starting from 0
        pairs = self._get_pairs(critical_blocks, critical_blocks_opr, tabu_list)
        return pairs

    @staticmethod
    def _get_pairs(cb, cb_op, tabu_list=None):
        pairs = []
        rg = cb[:-1].shape[0]  # sliding window of 2
        for i in range(rg):
            if cb[i] == cb[i + 1]:  # find potential pair
                if i == 0:
                    if cb[i + 1] != cb[i + 2]:
                        if [cb_op[i], cb_op[i + 1]] not in tabu_list:
                            pairs.append([cb_op[i], cb_op[i + 1]])
                elif cb[i] != cb[i - 1]:
                    if [cb_op[i], cb_op[i + 1]] not in tabu_list:
                        pairs.append([cb_op[i], cb_op[i + 1]])
                elif i + 1 == rg:
                    if cb[i + 1] != cb[i]:
                        if [cb_op[i], cb_op[i + 1]] not in tabu_list:
                            pairs.append([cb_op[i], cb_op[i + 1]])
                elif cb[i + 1] != cb[i + 2]:
                    if [cb_op[i], cb_op[i + 1]] not in tabu_list:
                        pairs.append([cb_op[i], cb_op[i + 1]])
                else:
                    pass
        return pairs

    @staticmethod
    def _get_pairs_has_tabu(cb, cb_op):
        pairs = []
        rg = cb[:-1].shape[0]  # sliding window of 2
        for i in range(rg):
            if cb[i] == cb[i + 1]:  # find potential pair
                if i == 0:
                    if cb[i + 1] != cb[i + 2]:
                        pairs.append([cb_op[i], cb_op[i + 1]])
                elif cb[i] != cb[i - 1]:
                    pairs.append([cb_op[i], cb_op[i + 1]])
                elif i + 1 == rg:
                    if cb[i + 1] != cb[i]:
                        pairs.append([cb_op[i], cb_op[i + 1]])
                elif cb[i + 1] != cb[i + 2]:
                    pairs.append([cb_op[i], cb_op[i + 1]])
                else:
                    pass
        return pairs

    def show_state(self, G):
        x_axis = np.pad(np.tile(np.arange(1, self.n_mch + 1, 1), self.n_job), (1, 1), 'constant', constant_values=[0, self.n_mch + 1])
        y_axis = np.pad(np.arange(self.n_job, 0, -1).repeat(self.n_mch), (1, 1), 'constant', constant_values=np.median(np.arange(self.n_job, 0, -1)))
        pos = dict((n, (x, y)) for n, x, y in zip(G.nodes(), x_axis, y_axis))
        plt.figure(figsize=(15, 10))
        plt.tight_layout()
        nx.draw_networkx_edge_labels(G, pos=pos)  # show edge weight
        nx.draw(
            G, pos=pos, with_labels=True, arrows=True, connectionstyle='arc3, rad = 0.1'
            # <-- tune curvature and style ref:https://matplotlib.org/3.1.1/api/_as_gen/matplotlib.patches.ConnectionStyle.html
        )
        plt.show()

    def _p_list_solver(self, args, plot=False):
        instances, priority_lists, device = args[0], args[1], args[2]

        edge_indices_pc = []
        edge_indices_mc = []
        durations = []
        current_graphs = []
        sub_graphs_mc = []
        for i, (instance, priority_list) in enumerate(zip(instances, priority_lists)):
            dur_mat, mch_mat = instance[0], instance[1]
            n_jobs = mch_mat.shape[0]
            n_machines = mch_mat.shape[1]
            n_operations = n_jobs * n_machines

            # prepare NIPS adj
            ops_mat = np.arange(0, n_operations).reshape(mch_mat.shape).tolist()  # Init operations mat
            list_for_latest_task_onMachine = [None] * n_machines  # Init list_for_latest_task_onMachine

            adj_mat_mc = np.zeros(shape=[n_operations, n_operations], dtype=int)  # Create adjacent matrix for machine clique
            # Construct NIPS adjacent matrix only for machine cliques
            for job_id in priority_list:
                op_id = ops_mat[job_id][0]
                m_id_for_action = mch_mat[op_id // n_machines, op_id % n_machines] - 1
                if list_for_latest_task_onMachine[m_id_for_action] is not None:
                    adj_mat_mc[op_id, list_for_latest_task_onMachine[m_id_for_action]] = 1
                list_for_latest_task_onMachine[m_id_for_action] = op_id
                ops_mat[job_id].pop(0)
            adj_mat_mc = np.pad(adj_mat_mc, ((1, 1), (1, 1)), 'constant', constant_values=0)  # add S and T to machine clique adj
            adj_mat_mc = np.transpose(adj_mat_mc)  # convert input adj from column pointing to row, to, row pointing to column

            adj_all = self.adj_mat_pc + adj_mat_mc
            dur_mat = np.pad(dur_mat.reshape(-1, 1), ((1, 1), (0, 0)), 'constant', constant_values=0).repeat(self.n_oprs + 2, axis=1)
            edge_weight = np.multiply(adj_all, dur_mat)
            G = nx.from_numpy_matrix(edge_weight, parallel_edges=False, create_using=nx.DiGraph)  # create nx.DiGraph
            G.add_weighted_edges_from([(0, i, 0) for i in range(1, self.n_oprs + 2 - 1, self.n_mch)])  # add release time, here all jobs are available at t=0. This is the only way to add release date. And if you do not add release date, startime computation will return wired value
            current_graphs.append(G)
            G_mc = nx.from_numpy_matrix(adj_mat_mc, parallel_edges=False, create_using=nx.DiGraph)  # create nx.DiGraph
            sub_graphs_mc.append(G_mc)

            if plot:
                self.show_state(G)

            edge_indices_pc.append((torch.nonzero(torch.from_numpy(self.adj_mat_pc)).t().contiguous()) + (n_operations + 2) * i)
            edge_indices_mc.append((torch.nonzero(torch.from_numpy(adj_mat_mc)).t().contiguous()) + (n_operations + 2) * i)
            durations.append(torch.from_numpy(dur_mat[:, 0]).to(device))

        edge_indices_pc = torch.cat(edge_indices_pc, dim=-1).to(device)
        edge_indices_mc = torch.cat(edge_indices_mc, dim=-1).to(device)

        durations = torch.cat(durations, dim=0).reshape(-1, 1)
        if self.evaluator_type == 'message-passing':
            est, lst, make_span = self.eva.forward(edge_index=torch.cat([edge_indices_pc, edge_indices_mc], dim=-1), duration=durations, n_j=self.n_job, n_m=self.n_mch)
        else:
            est, lst, make_span = self.eva(current_graphs, dev=device)

            # prepare x
        x = torch.cat([durations / self.high, est / self.fea_norm_const, lst / self.fea_norm_const], dim=-1)
        # prepare batch
        batch = torch.from_numpy(np.repeat(np.arange(instances.shape[0], dtype=np.int64), repeats=self.n_job * self.n_mch + 2)).to(device)

        return (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, sub_graphs_mc, make_span


    def _rules_solver(self, data, config):
        instances=data
        edge_indices_pc = []
        edge_indices_mc = []
        durations = []
        current_graphs = []
        sub_graphs_mc = []
        dur_mat, dur_cp, mch_mat = instances[0], np.copy(instances[0]), instances[1]
        n_jobs, n_machines = config.Pn_j, config.Pn_m
        n_operations = n_jobs * n_machines
        last_col = np.arange(start=0, stop=n_operations, step=1).reshape(n_jobs, -1)[:, -1]
        candidate_oprs = np.arange(start=0, stop=n_operations, step=1).reshape(n_jobs, -1)[:,
                         0]  # initialize action space: [n_jobs, 1], the first column
        mask = np.zeros(shape=n_jobs, dtype=bool)  # initialize the mask: [n_jobs, 1]
        adj_mat_mc = np.zeros(shape=[n_operations, n_operations],
                              dtype=int)  # Create adjacent matrix for machine clique
        if isinstance(dur_mat, tuple):
            dur_mat = np.array(dur_mat)
        if isinstance(mch_mat, tuple):
            mch_mat = np.array(mch_mat)

        gant_chart = -self.high * np.ones_like(dur_mat.transpose(), dtype=np.int32)
        opIDsOnMchs = -n_jobs * np.ones_like(dur_mat.transpose(), dtype=np.int32)
        finished_mark = np.zeros_like(mch_mat, dtype=np.int32)

        actions = []
        for _ in range(n_operations):
            candidate_masked = candidate_oprs[np.where(~mask)]
            fdd = np.take(np.cumsum(dur_mat, axis=1), candidate_masked)
            wkr = np.take(np.cumsum(np.multiply(dur_mat, 1 - finished_mark), axis=1), last_col[np.where(~mask)])
            priority = fdd / wkr
            idx = np.random.choice(np.where(priority == np.min(priority))[0])
            action = candidate_masked[idx]
            actions.append(action)
            permissibleLeftShift(a=action, durMat=dur_mat, mchMat=mch_mat, mchsStartTimes=gant_chart,
                                 opIDsOnMchs=opIDsOnMchs,config=config)

            # update action space or mask
            if action not in last_col:
                candidate_oprs[action // n_machines] += 1
            else:
                mask[action // n_machines] = 1
            # update finished_mark:
            finished_mark[action // n_machines, action % n_machines] = 1

        for _ in range(opIDsOnMchs.shape[1] - 1):
            adj_mat_mc[opIDsOnMchs[:, _ + 1], opIDsOnMchs[:, _]] = 1

        # prepare augmented adj, augmented dur, and G
        adj_mat_mc = np.pad(adj_mat_mc, ((1, 1), (1, 1)), 'constant',
                            constant_values=0)  # add S and T to machine clique adj
        adj_mat_mc = np.transpose(
            adj_mat_mc)  # convert input adj from column pointing to row, to, row pointing to column
        adj_all = self.adj_mat_pc + adj_mat_mc
        dur_mat = np.pad(dur_mat.reshape(-1, 1), ((1, 1), (0, 0)), 'constant', constant_values=0).repeat(
            self.n_oprs + 2, axis=1)
        edge_weight = np.multiply(adj_all, dur_mat)
        G = nx.from_numpy_matrix(edge_weight, parallel_edges=False, create_using=nx.DiGraph)  # create nx.DiGraph
        G.add_weighted_edges_from([(0, i, 0) for i in range(1, self.n_oprs + 2 - 1,
                                                            self.n_mch)])  # add release time, here all jobs are available at t=0. This is the only way to add release date. And if you do not add release date, startime computation will return wired value
        current_graphs.append(G)
        G_mc = nx.from_numpy_matrix(adj_mat_mc, parallel_edges=False, create_using=nx.DiGraph)  # create nx.DiGraph
        sub_graphs_mc.append(G_mc)

        edge_indices_pc.append(
            (torch.nonzero(torch.from_numpy(self.adj_mat_pc)).t().contiguous())  )
        edge_indices_mc.append((torch.nonzero(torch.from_numpy(adj_mat_mc)).t().contiguous()) )
        durations.append(torch.from_numpy(dur_mat[:, 0]).to(device))

        edge_indices_pc = torch.cat(edge_indices_pc, dim=-1).to(device)
        edge_indices_mc = torch.cat(edge_indices_mc, dim=-1).to(device)
        durations = torch.cat(durations, dim=0).reshape(-1, 1)
        if self.evaluator_type == 'message-passing':
            est, lst, make_span = self.eva.forward(edge_index=torch.cat([edge_indices_pc, edge_indices_mc], dim=-1), duration=durations, n_j=self.n_job, n_m=self.n_mch)
        else:
            est, lst, make_span = self.eva(current_graphs, dev=device)

        # prepare x
        x = torch.cat([durations / self.high, est / self.fea_norm_const, lst / self.fea_norm_const], dim=-1)
        # prepare batch
        batch = torch.from_numpy(np.repeat(np.arange(1, dtype=np.int64), repeats=self.n_job * self.n_mch + 2)).to(device)

        return (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, sub_graphs_mc, make_span

    def dag2pyg(self, instances, nx_graphs, device):
        n_jobs, n_machines = instances[0].shape
        n_operations = n_jobs * n_machines

        edge_indices_pc = []
        edge_indices_mc = []
        durations = []
        for i, (instance, G_mc) in enumerate(zip(instances, nx_graphs)):
            durations.append(np.pad(instances[0].reshape(-1), (1, 1), 'constant', constant_values=0))
            adj_mat_mc = nx.adjacency_matrix(G_mc, weight=None).todense()
            edge_indices_pc.append((torch.nonzero(torch.from_numpy(self.adj_mat_pc)).t().contiguous()) + (n_operations + 2) * i)
            edge_indices_mc.append((torch.nonzero(torch.from_numpy(adj_mat_mc)).t().contiguous()) + (n_operations + 2) * i)

        edge_indices_pc = torch.cat(edge_indices_pc, dim=-1).to(device)
        edge_indices_mc = torch.cat(edge_indices_mc, dim=-1).to(device)
        durations = torch.from_numpy(np.concatenate(durations)).reshape(-1, 1).to(device)
        if self.evaluator_type == 'message-passing':
            est, lst, make_span = self.eva.forward(edge_index=torch.cat([edge_indices_pc, edge_indices_mc], dim=-1), duration=durations, n_j=self.n_job, n_m=self.n_mch)
        else:
            est, lst, make_span = self.eva(self.current_graphs, dev=device)
        # prepare x
        x = torch.cat([durations / self.high, est / self.fea_norm_const, lst / self.fea_norm_const], dim=-1)
        # prepare batch
        batch = torch.from_numpy(np.repeat(np.arange(1, dtype=np.int64), repeats=n_jobs * n_machines + 2)).to(device)

        return x, edge_indices_pc, edge_indices_mc, batch, make_span

    def change_nxgraph_topology(self, actions, plot=False):
        n_jobs, n_machines = self.instances[0].shape
        n_operations = n_jobs * n_machines
        action = actions[0]
        G = self.current_graphs[0]
        G_mc = self.sub_graphs_mc[0]
        instance = self.instances
    # for i, (action, G, G_mc, instance) in enumerate(zip(actions, self.current_graphs, self.sub_graphs_mc, self.instances)):
        if action == [0, 0]:  # if dummy action then do not transit
            pass
        else:  # change nx graph topology
            S = [s for s in G.predecessors(action[0]) if
                 int((s - 1) // n_machines) != int((action[0] - 1) // n_machines) and s != 0]
            T = [t for t in G.successors(action[1]) if
                 int((t - 1) // n_machines) != int((action[1] - 1) // n_machines) and t != n_operations + 1]
            s = S[0] if len(S) != 0 else None
            t = T[0] if len(T) != 0 else None

            if s is not None:  # connect s with action[1]
                G.remove_edge(s, action[0])
                G.add_edge(s, action[1], weight=np.take(instance[0], s - 1))
                G_mc.remove_edge(s, action[0])
                G_mc.add_edge(s, action[1], weight=np.take(instance[0], s - 1))
            else:
                pass

            if t is not None:  # connect action[0] with t
                G.remove_edge(action[1], t)
                G.add_edge(action[0], t, weight=np.take(instance[0], action[0] - 1))
                G_mc.remove_edge(action[1], t)
                G_mc.add_edge(action[0], t, weight=np.take(instance[0], action[0] - 1))
            else:
                pass

            # reverse edge connecting selected pair
            G.remove_edge(action[0], action[1])
            G.add_edge(action[1], action[0], weight=np.take(instance[0], action[1] - 1))
            G_mc.remove_edge(action[0], action[1])
            G_mc.add_edge(action[1], action[0], weight=np.take(instance[0], action[1] - 1))

        if plot:
            self.show_state(G)



    def transit(self, actions,plot=False):
        self.change_nxgraph_topology(actions, plot)  # change graph topology
        x, edge_indices_pc, edge_indices_mc, batch, makespan = self.dag2pyg(self.instances, self.sub_graphs_mc, device)  # generate new state data
        reward = torch.where(self.incumbent_objs - makespan > 0, self.incumbent_objs - makespan, torch.tensor(0, dtype=torch.float32, device=device))

        self.incumbent_objs = torch.where(makespan - self.incumbent_objs < 0, makespan, self.incumbent_objs)
        self.current_objs = makespan

        # update tabu list
        if self.tabu_size != 0:
            action_reversed = [a[::-1] for a in actions]
            for i, action in enumerate(action_reversed):
                if action == [0, 0]:  # if dummy action, don't update tabu list
                    pass
                else:
                    if len(self.tabu_lists[i]) == self.tabu_size:
                        self.tabu_lists[i].pop(0)
                        self.tabu_lists[i].append(action)
                    else:
                        self.tabu_lists[i].append(action)

        self.itr = self.itr + 1


        feasible_actions, flag = self.feasible_actions(device)  # new feasible actions w.r.t updated tabu list

        return (x, edge_indices_pc, edge_indices_mc, batch), reward, feasible_actions, ~flag

    def reset(self, data, config):
        self.instances = data
        (x, edge_indices_pc, edge_indices_mc, batch), current_graphs, sub_graphs_mc, make_span = self._rules_solver(data, config)

        self.sub_graphs_mc = sub_graphs_mc
        self.current_graphs = current_graphs
        self.current_objs = make_span
        self.incumbent_objs = make_span
        self.itr = 0
        self.tabu_lists = [[] for _ in range(len(self.instances))]
        feasible_actions, flag = self.feasible_actions(device)

        return (x, edge_indices_pc, edge_indices_mc, batch), feasible_actions, ~flag

    def feasible_actions(self, device):
        actions = []
        feasible_actions_flag = []  # False for no feasible operation pairs
        for i, (G, instance, tabu_list) in enumerate(zip(self.current_graphs, self.instances, self.tabu_lists)):
            action = self._gen_moves(solution=G, mch_mat=self.instances[1], tabu_list=tabu_list)
            # print(action)
            if len(action) != 0:
                actions.append(action)
                feasible_actions_flag.append(True)
            else:  # if no feasible actions available append dummy actions [0, 0]
                actions.append([[0, 0]])
                feasible_actions_flag.append(False)
        return actions, torch.tensor(feasible_actions_flag, device=device).unsqueeze(1)