import random
from collections import OrderedDict
import matplotlib.pyplot as plt
import networkx as nx
import plotly.figure_factory as ff
import numpy as np

from Test.agent.JSP.Basic_agent import Basic_Agent


def jssp_sampling(m, n, low=5, high=100):
    machine_mat = np.ndarray(shape=(n, m))
    process_time_mat = np.random.randint(low, high, size=(n, m))
    for i in range(n):
        machine_mat[i] = np.random.permutation(m)
    return machine_mat, process_time_mat
class ScheduleNet_agent (Basic_Agent):
    def __init__(self,config,data):
        self.num_machines=config.Pn_m
        self.num_jobs=config.Pn_j
        self.detach_done = False
        self.name = None
        self.machine_matrix = None
        self.processing_time_matrix = None
        self.embedding_dim = 16
        self.use_surrogate_index = True
        self.delay = False
        self.verbose = False

        if self.machine_matrix is None or self.processing_time_matrix is None:
            ms, prts = data[1]-1,data[0]
            self.machine_matrix = ms.astype(int)
            self.processing_time_matrix = prts.astype(float)
        else:
            self.machine_matrix = self.machine_matrix.astype(int)
            self.processing_time_matrix = self.processing_time_matrix.astype(float)

        if self.name is None:
            self.name = '{} machine {} job'.format(self.num_machines, self.num_jobs)


        self._machine_set = list(set(self.machine_matrix.flatten().tolist()))
        self.num_machine = len(self._machine_set)
        self.detach_done = self.detach_done
        self.embedding_dim = self.embedding_dim
        self.num_jobs = self.processing_time_matrix.shape[0]
        self.num_steps = self.processing_time_matrix.shape[1]
        self.use_surrogate_index = self.use_surrogate_index
        self.delay = self.delay
        self.verbose = self.verbose
        self.config = config
        self.reset()
        # simulation procedure : global_time +=1 -> do_processing -> transit

    def reset(self):
        self.job_manager = JobManager(self.config,self.machine_matrix,
                                      self.processing_time_matrix,
                                      embedding_dim=self.embedding_dim,
                                      use_surrogate_index=self.use_surrogate_index)
        self.machine_manager = MachineManager(self.config,
                                              self.machine_matrix,
                                              self.job_manager,
                                              self.delay,
                                              self.verbose)
        self.global_time = 0  # -1 matters a lot

    def process_one_time(self):
        self.global_time += 1
        self.machine_manager.do_processing(self.global_time)

    def transit(self, action=None):
        if action is None:
            # Perform random action
            machine = random.choice(self.machine_manager.get_available_machines())
            op_id = random.choice(machine.doable_ops_id)
            job_id, step_id = self.job_manager.sur_index_dict[op_id]
            operation = self.job_manager[job_id][step_id]
            action = operation
            # print(machine)
            machine.transit(self.global_time, action)
            return op_id
        else:
            if self.use_surrogate_index:
                if action in self.job_manager.sur_index_dict.keys():
                    job_id, step_id = self.job_manager.sur_index_dict[action]
                else:
                    raise RuntimeError("Input action is not valid")
            else:
                job_id, step_id = action

            operation = self.job_manager[job_id][step_id]
            machine_id = operation.machine_id
            machine = self.machine_manager[machine_id]
            action = operation
            machine.transit(self.global_time, action)

    def flush_trivial_ops(self, reward='utilization', gamma=1.0):
        done = False
        cum_reward = 0
        t = 0
        # print(random.random())
        sub_list = []
        while True:
            # print('in the while')
            t += 1
            m_list = []
            do_op_dict = self.get_doable_ops_in_dict()
            # print(do_op_dict)
            all_machine_work = False if bool(do_op_dict) else True

            if all_machine_work:  # all machines are on processing. keep process!
                self.process_one_time()
            else:  # some of machine has possibly trivial action. the others not.
                # load trivial ops to the machines
                num_ops_counter = 1
                for m_id, op_ids in do_op_dict.items():
                    num_ops = len(op_ids)
                    if num_ops == 1:
                        # print(op_ids[0])
                        sub_list.append(op_ids[0])
                        self.transit(op_ids[0])  # load trivial action
                        g, r, _ = self.observe(reward)
                        cum_reward = r + gamma * cum_reward
                    else:
                        m_list.append(m_id)
                        num_ops_counter *= num_ops

                # not-all trivial break the loop
                if num_ops_counter != 1:
                    break

            # if simulation is done
            jobs_done = [job.job_done for _, job in self.job_manager.jobs.items()]
            # print(jobs_done)
            done = True if np.prod(jobs_done) == 1 else False

            if done:
                # print('done')
                break
        # print(t)
        return m_list, cum_reward, done, sub_list

    def get_available_machines(self, shuffle_machine=True):
        return self.machine_manager.get_available_machines(shuffle_machine)

    def get_doable_ops_in_dict(self, machine_id=None, shuffle_machine=True):
        if machine_id is None:
            doable_dict = {}
            if self.get_available_machines():
                for m in self.get_available_machines(shuffle_machine):
                    _id = m.machine_id
                    _ops = m.doable_ops_id
                    doable_dict[_id] = _ops
            ret = doable_dict
        else:
            available_machines = [m.machine_id for m in self.get_available_machines()]
            if machine_id in available_machines:
                ret = self.machine_manager[machine_id].doable_ops_id
            else:
                raise RuntimeWarning("Access to the not available machine {}. Return is None".format(machine_id))
        return ret

    def get_doable_ops_in_list(self, machine_id=None, shuffle_machine=True):
        doable_dict = self.get_doable_ops_in_dict(machine_id, shuffle_machine)
        do_ops = []
        for _, v in doable_dict.items():
            do_ops += v
        return do_ops

    def get_doable_ops(self, machine_id=None, return_list=False, shuffle_machine=True):
        if return_list:
            ret = self.get_doable_ops_in_list(machine_id, shuffle_machine)
        else:
            ret = self.get_doable_ops_in_dict(machine_id, shuffle_machine)
        return ret

    def observe(self, reward='utilization', return_doable=True):
        # A simple wrapper for JobManager's observe function
        # and return current time step reward r
        # check all jobs are done or not, then return done = True or False

        jobs_done = [job.job_done for _, job in self.job_manager.jobs.items()]
        # check jobs_done contains only True or False
        if np.prod(jobs_done) == 1:
            done = True
        else:
            done = False
        if reward == 'makespan':
            if done:
                r = -self.global_time
            else:
                r = 0
        # return reward as total sum of queues for all machines
        elif reward == 'utilization':
            t_cost = self.machine_manager.cal_total_cost()
            r = -t_cost

        elif reward == 'idle_time':
            r = -float(len(self.machine_manager.get_idle_machines())) / float(self.num_machine)
        else:
            raise RuntimeError("Not support reward type")

        g = self.machine_manager.observe(detach_done=self.detach_done)

        if return_doable:
            if self.use_surrogate_index:
                do_ops_list = [doable_op + self.num_machine for doable_op in self.get_doable_ops(return_list=True)]
                for n in g.nodes:
                    if n in do_ops_list:
                        job_id, op_id = self.job_manager.sur_index_dict[n - self.num_machine]
                        m_id = self.job_manager[job_id][op_id].machine_id
                        g.nodes[n]['doable'] = True
                        g.nodes[n]['machine'] = m_id
                    else:
                        g.nodes[n]['doable'] = False
                        g.nodes[n]['machine'] = 0

        return g, r, done

    def draw_gantt_chart(self, path, benchmark_name, max_x):
        # Draw a gantt chart
        self.job_manager.draw_gantt_chart(path, benchmark_name, max_x)

    @staticmethod
    def _sample_jssp_graph(config,m, n):
        if not m % config.N_SEP == 0:
            m = int(config.N_SEP * (m // config.N_SEP))
            if m < config.N_SEP:
                m = config.N_SEP
        if not n % config.N_SEP == 0:
            n = int(config.N_SEP * (n // config.N_SEP))
            if n < config.N_SEP:
                n = config.N_SEP
        if m > n:
            raise RuntimeError(" m should be smaller or equal to n ")

        return jssp_sampling(m, n, 5, 100)
        # return jssp_sampling(m, n, 1, 5)

    @classmethod
    def from_path(self,cls, jssp_path, **kwargs):
        with open(jssp_path) as f:
            ms = []  # machines
            prts = []  # processing times
            for l in f:
                l_split = " ".join(l.split()).split(' ')
                m = l_split[0::2]
                prt = l_split[1::2]
                ms.append(np.array(m, dtype=int))
                prts.append(np.array(prt, dtype=float))

        ms = np.stack(ms)
        prts = np.stack(prts)
        num_job, num_machine = ms.shape
        name = jssp_path.split('/')[-1].replace('.txt', '')

        return cls(num_machines=num_machine,
                   num_jobs=num_job,
                   name=name,
                   machine_matrix=ms,
                   processing_time_matrix=prts,
                   **kwargs)

    @classmethod
    def from_TA_path(self,cls, pt_path, m_path, **kwargs):
        config=self.config
        with open(pt_path) as f1:
            prts = []
            for l in f1:
                l_split = l.split(config.SEP)
                prt = [e for e in l_split if e != '']
                if config.NEW in prt[-1]:
                    prt[-1] = prt[-1].split(config.NEW)[0]
                prts.append(np.array(prt, dtype=float))

        with open(m_path) as f2:
            ms = []
            for l in f2:
                l_split = l.split(config.SEP)
                m = [e for e in l_split if e != '']
                if config.NEW in m[-1]:
                    m[-1] = m[-1].split(config.NEW)[0]
                ms.append(np.array(m, dtype=int))

        ms = np.stack(ms) - 1
        prts = np.stack(prts)
        num_job, num_machine = ms.shape
        name = pt_path.split('/')[-1].replace('_PT.txt', '')

        return cls(num_machines=num_machine,
                   num_jobs=num_job,
                   name=name,
                   machine_matrix=ms,
                   processing_time_matrix=prts,
                   **kwargs)

class JobManager:
    def __init__(self,
                 config,
                 machine_matrix,
                 processing_time_matrix,
                 embedding_dim=16,
                 use_surrogate_index=True):

        machine_matrix = machine_matrix.astype(int)
        processing_time_matrix = processing_time_matrix.astype(float)

        self.jobs = OrderedDict()

        # Constructing conjunctive edges
        for job_i, (m, pr_t) in enumerate(zip(machine_matrix, processing_time_matrix)):
            m = m + 1  # To make machine index starts from 1
            self.jobs[job_i] = Job(job_i, m, pr_t, embedding_dim,config)  # connection happens by class initializing

        # Constructing disjunctive edges
        machine_index = list(set(machine_matrix.flatten().tolist()))
        for m_id in machine_index:
            job_ids, step_ids = np.where(machine_matrix == m_id)
            for job_id1, step_id1 in zip(job_ids, step_ids):
                op1 = self.jobs[job_id1][step_id1]
                ops = []
                for job_id2, step_id2 in zip(job_ids, step_ids):
                    if (job_id1 == job_id2) and (step_id1 == step_id2):
                        continue  # skip itself
                    else:
                        ops.append(self.jobs[job_id2][step_id2])
                op1.disjunctive_ops = ops

        self.use_surrogate_index = use_surrogate_index

        if self.use_surrogate_index:
            # Constructing surrogate indices:
            num_ops = 0
            self.sur_index_dict = dict()
            for job_id, job in self.jobs.items():
                for op in job.ops:
                    op.sur_id = num_ops
                    self.sur_index_dict[num_ops] = op._id
                    num_ops += 1

    def __call__(self, index):
        return self.jobs[index]

    def __getitem__(self, index):
        return self.jobs[index]

    def draw_gantt_chart(self, path, benchmark_name, max_x):
        gantt_info = []
        for _, job in self.jobs.items():
            for op in job.ops:
                if not isinstance(op, DummyOperation):
                    temp = OrderedDict()
                    temp['Task'] = "Machine" + str(op.machine_id)
                    temp['Start'] = op.start_time
                    temp['Finish'] = op.end_time
                    temp['Resource'] = "Job" + str(op.job_id)
                    gantt_info.append(temp)
        gantt_info = sorted(gantt_info, key=lambda k: k['Task'])
        color = OrderedDict()
        for g in gantt_info:
            _r = random.randrange(0, 255, 1)
            _g = random.randrange(0, 255, 1)
            _b = random.randrange(0, 255, 1)
            rgb = 'rgb({}, {}, {})'.format(_r, _g, _b)
            color[g['Resource']] = rgb
        fig = ff.create_gantt(gantt_info, colors=color, show_colorbar=True, group_tasks=True, index_col='Resource',
                              title=benchmark_name + ' gantt chart', showgrid_x=True, showgrid_y=True)
        fig['layout']['xaxis'].update({'type': None})
        fig['layout']['xaxis'].update({'range': [0, max_x]})
        fig['layout']['xaxis'].update({'title': 'time'})

        plt.plot(fig, filename=path)


class Job:
    def __init__(self, job_id, machine_order, processing_time_order, embedding_dim,config):
        self.config = config
        self.job_id = job_id
        self.ops = list()
        self.processing_time = np.sum(processing_time_order)
        self.num_sequence = processing_time_order.size

        # initialize operations
        cum_pr_t = 0
        for step_id, (m_id, pr_t) in enumerate(zip(machine_order, processing_time_order)):
            cum_pr_t += pr_t
            op = Operation(config=config,job_id=job_id, step_id=step_id, machine_id=m_id,
                           prev_op=None,
                           processing_time=pr_t,
                           complete_ratio=cum_pr_t / self.processing_time,
                           job=self)
            self.ops.append(op)

        # constructing job clique
        for op in self.ops:
            neighbour_ops = []
            for neighbour_op in self.ops:
                if op.id != neighbour_op.id:
                    neighbour_ops.append(neighbour_op)
            op.conjunctive_ops = neighbour_ops

        # Connecting backward paths (add prev_op to operations)
        for i, op in enumerate(self.ops[1:]):
            op.prev_op = self.ops[i]

        # Connecting forward paths (add next_op to operations)
        for i, node in enumerate(self.ops[:-1]):
            node.next_op = self.ops[i + 1]

    def __getitem__(self, index):
        return self.ops[index]

    # To check job is done or not using last operation's node status
    @property
    def job_done(self):
        if self.ops[-1].node_status == self.config.DONE_NODE_SIG:
            return True
        else:
            return False

    # To check the number of remaining operations
    @property
    def remaining_ops(self):
        c = 0
        for op in self.ops:
            if op.node_status != self.config.DONE_NODE_SIG:
                c += 1
        return c

class Machine:
    def __init__(self, machine_id, possible_ops, delay, verbose,config):
        self.machine_id = machine_id
        self.possible_ops = possible_ops
        self.remain_ops = possible_ops
        self.current_op = None
        self.delayed_op = None
        self.prev_op = None
        self.remaining_time = 0
        self.done_ops = []
        self.num_done_ops = 0
        self.cost = 0
        self.delay = delay
        self.verbose = verbose
        self.config = config

    def __str__(self):
        return "Machine {}".format(self.machine_id)

    def status(self):
        currently_not_processing_cond = self.current_op is None
        not_wait_for_delayed_cond = not self.wait_for_delayed()
        status = currently_not_processing_cond and not_wait_for_delayed_cond
        return status

    def available(self):
        future_work_exist_cond = bool(self.doable_ops())
        currently_not_processing_cond = self.current_op is None
        not_wait_for_delayed_cond = not self.wait_for_delayed()
        ret = future_work_exist_cond and currently_not_processing_cond and not_wait_for_delayed_cond
        return ret

    def wait_for_delayed(self):
        wait_for_delayed_cond = self.delayed_op is not None
        ret = wait_for_delayed_cond
        if wait_for_delayed_cond:
            delayed_op_ready_cond = self.delayed_op.prev_op.node_status == self.config.DONE_NODE_SIG
            ret = ret and not delayed_op_ready_cond
        return ret

    def doable_ops(self):
        # doable_ops are subset of remain_ops.
        # some ops are doable when the prev_op is 'done' or 'processing' or 'start'
        doable_ops = []
        for op in self.remain_ops:
            prev_start = op.prev_op is None
            if prev_start:
                doable_ops.append(op)
            else:
                prev_done = op.prev_op.node_status == self.config.DONE_NODE_SIG
                prev_process = op.prev_op.node_status == self.config.PROCESSING_NODE_SIG
                first_op = not bool(self.done_ops)
                if self.delay:
                    # each machine's first processing operation should not be a reserved operation
                    if first_op:
                        cond = prev_done
                    else:
                        cond = (prev_done or prev_process)
                else:
                    cond = prev_done

                if cond:
                    doable_ops.append(op)
                else:
                    pass

        return doable_ops

    @property
    def doable_ops_id(self):
        doable_ops_id = []
        doable_ops = self.doable_ops()
        for op in doable_ops:
            doable_ops_id.append(op.id)

        return doable_ops_id

    @property
    def doable_ops_no_delay(self):
        doable_ops = []
        for op in self.remain_ops:
            prev_start = op.prev_op is None
            if prev_start:
                doable_ops.append(op)
            else:
                prev_done = op.prev_op.node_status == self.config.DONE_NODE_SIG
                if prev_done:
                    doable_ops.append(op)
        return doable_ops

    def work_done(self):
        return not self.remain_ops

    def load_op(self, t, op):

        # Procedures for double-checkings
        # If machine waits for the delayed job is done:
        if self.wait_for_delayed():
            raise RuntimeError("Machine {} waits for the delayed job {} but load {}".format(self.machine_id,
                                                                                  print(self.delayed_op), print(op)))

        # ignore input when the machine is not available
        if not self.available():
            raise RuntimeError("Machine {} is not available".format(self.machine_id))

        # ignore when input op's previous op is not done yet:
        if not op.processable():
            raise RuntimeError("Operation {} is not accessible yet".format(print(op)))

        if op not in self.possible_ops:
            raise RuntimeError("Machine {} can't perform ops {}{}".format(self.machine_id,
                                                                          op.job_id,
                                                                          op.step_id))

        # Essential condition for checking whether input is delayed
        # if delayed then, flush dealed_op attr
        if op == self.delayed_op:
            if self.verbose:
                print("[DELAYED OP LOADED] / MACHINE {} / {} / at {}".format(self.machine_id, op, t))
            self.delayed_op = None

        else:
            if self.verbose:
                print("[LOAD] / Machine {} / {} on at {}".format(self.machine_id, op, t))

        # Update operation's attributes
        op.node_status = self.config.PROCESSING_NODE_SIG
        op.remaining_time = op.processing_time
        op.start_time = t

        # Update machine's attributes
        self.current_op = op
        self.remaining_time = op.processing_time
        self.remain_ops.remove(self.current_op)

    def unload(self, t):
        if self.verbose:
            print("[UNLOAD] / Machine {} / Op {} / t = {}".format(self.machine_id, self.current_op, t))
        self.current_op.node_status = self.config.DONE_NODE_SIG
        self.current_op.end_time = t
        self.done_ops.append(self.current_op)
        self.num_done_ops += 1
        self.prev_op = self.current_op
        self.current_op = None
        self.remaining_time = 0

    def do_processing(self, t):
        if self.remaining_time > 0:  # When machine do some operation
            if self.current_op is not None:
                self.current_op.remaining_time -= 1
                if self.current_op.remaining_time <= 0:
                    if self.current_op.remaining_time < 0:
                        raise RuntimeWarning("Negative remaining time observed")
                    if self.verbose:
                        print("[OP DONE] : / Machine  {} / Op {}/ t = {} ".format(self.machine_id, self.current_op, t))
                    self.unload(t)
            # to compute idle_time reward, we need to count delayed_time
            elif self.delayed_op is not None:
                self.delayed_op.delayed_time += 1
                self.delayed_op.remaining_time -= 1

            doable_ops = self.doable_ops()
            if doable_ops:
                for op in doable_ops:
                    op.waiting_time += 1
            else:
                pass

            self.remaining_time -= 1

    def transit(self, t, a):
        if self.available():  # Machine is ready to process.
            if a.processable():  # selected action is ready to be loaded right now.
                self.load_op(t, a)
            else:  # When input operation turns out to be 'delayed'
                a.node_status = self.config.DELAYED_NODE_SIG
                self.delayed_op = a
                self.delayed_op.remaining_time = a.processing_time + a.prev_op.remaining_time
                self.remaining_time = a.processing_time + a.prev_op.remaining_time
                self.current_op = None  # MACHINE is now waiting for delayed ops
                if self.verbose:
                    print("[DELAYED OP CHOSEN] : / Machine  {} / Op {}/ t = {} ".format(self.machine_id, self.delayed_op, t))
        else:
            raise RuntimeError("Access to not available machine")


class MachineManager:
    def __init__(self,
                 config,
                 machine_matrix,
                 job_manager,
                 delay=True,  # True: prev op is processing, next op is processable
                 verbose=False):

        machine_matrix = machine_matrix.astype(int)
        self.job_manager = job_manager
        self.config=config

        # Parse machine indices
        machine_index = list(set(machine_matrix.flatten().tolist()))

        # Global machines dict
        self.machines = OrderedDict()
        for m_id in machine_index:
            job_ids, step_ids = np.where(machine_matrix == m_id)
            possible_ops = []
            for job_id, step_id in zip(job_ids, step_ids):
                possible_ops.append(job_manager[job_id][step_id])
            m_id += 1  # To make machine index starts from 1
            self.machines[m_id] = Machine(m_id, possible_ops, delay, verbose,config)

    def do_processing(self, t):
        for _, machine in self.machines.items():
            machine.do_processing(t)

    def load_op(self, machine_id, op, t):
        self.machines[machine_id].load_op(op, t)

    def __getitem__(self, index):
        return self.machines[index]

    # available: have remaining ops, idle, and not waiting for delayed op, i.e. those can be assigned
    def get_available_machines(self, shuffle_machine=True):
        m_list = []
        for _, m in self.machines.items():
            if m.available():
                m_list.append(m)

        if shuffle_machine:
            m_list = random.sample(m_list, len(m_list))

        return m_list

    # get idle machines' list
    def get_idle_machines(self):
        m_list = []
        for _, m in self.machines.items():
            if m.current_op is None and not m.work_done():
                m_list.append(m)
        return m_list

    # calculate the length of queues for all machines
    def cal_total_cost(self):
        c = 0
        for _, m in self.machines.items():
            c += len(m.doable_ops_no_delay)  # number of ready operations of m
        return c

    # update all cost functions of machines
    def update_cost_function(self, cost):
        for _, m in self.machines.items():
            m.cost += cost

    def get_machines(self):
        m_list = [m for _, m in self.machines.items()]
        return random.sample(m_list, len(m_list))

    def all_delayed(self):
        return np.product([m.delayed_op is not None for _, m in self.machines.items()])

    def fab_stuck(self):
        # All machines are not available and All machines are delayed.
        all_machines_not_available_cond = not self.get_available_machines()
        all_machines_delayed_cond = self.all_delayed()
        return all_machines_not_available_cond and all_machines_delayed_cond

    def observe(self, detach_done=True):
        """
        generate graph representation
        :return: nx.OrderedDiGraph
        """

        num_machine = len(self.machines)

        # create agents clique
        target_agents = self.get_available_machines(
            shuffle_machine=False)  # target agents are those idle and non-waiting
        g = nx.DiGraph()
        for m_id, m in self.machines.items():  # add node
            _x_machine = OrderedDict()
            _x_machine['agent'] = 1
            _x_machine['target_agent'] = 1 if m in target_agents else 0
            _x_machine['assigned'] = 1 - int(m.current_op is None)
            _x_machine['waiting'] = int(m.wait_for_delayed())
            _x_machine['processable'] = 0  # flag for operation node
            _x_machine['accessible'] = 0  # flag for operation node
            _x_machine['task_wait_time'] = m.delayed_op.wait_time if m.delayed_op is not None else -1
            _x_machine['task_processing_time'] = m.current_op.processing_time if m.current_op is not None else -1
            _x_machine['time_to_complete'] = m.remaining_time
            _x_machine['remain_ops'] = len(m.remain_ops)
            _x_machine['job_completion_ratio'] = m.current_op.complete_ratio if m.current_op is not None else -1
            # node type
            _x_machine['node_type'] = 'assigned_agent' if _x_machine['assigned'] == 1 else 'unassigned_agent'
            g.add_node(m_id - 1, **_x_machine)  # machine id from 0
            for neighbour_machine in self.machines.keys():  # fully connect to other machines
                if neighbour_machine != m_id:
                    g.add_edge(m_id - 1, neighbour_machine - 1,
                               edge_feature=[0])  # edge_feature = not processable by the source node

        # create task subgraph
        for job_id, job in self.job_manager.jobs.items():
            for op in job.ops:
                not_start_cond = (op.node_status == self.config.NOT_START_NODE_SIG)
                delayed_cond = (op.node_status == self.config.DELAYED_NODE_SIG)
                processing_cond = (op.node_status == self.config.PROCESSING_NODE_SIG)
                done_cond = (op.node_status == self.config.DONE_NODE_SIG)

                if not_start_cond:
                    _x_task = OrderedDict()
                    _x_task['id'] = op._id
                    _x_task["type"] = op.node_status
                    _x_task["job_completion_ratio"] = op.complete_ratio
                    _x_task['task_processing_time'] = op.processing_time
                    _x_task['remain_ops'] = op.remaining_ops
                    _x_task['task_wait_time'] = op.waiting_time
                    _x_task["time_to_complete"] = -1
                    # ScheduleNet feature
                    _x_task["agent"] = 0
                    _x_task["target_agent"] = 0
                    _x_task["assigned"] = 0  # not_start_cond = op not load, i.e. not assigned
                    _x_task["waiting"] = 0
                    processable = int(op in self.machines[op.machine_id].doable_ops() and self.machines[
                        op.machine_id] in target_agents)
                    _x_task["processable"] = processable
                    _x_task["accessible"] = processable * int(self.machines[op.machine_id].status())
                    _x_task['node_type'] = 'processable_task' if processable == 1 else 'unprocessable_task'
                elif processing_cond:
                    _x_task = OrderedDict()
                    _x_task['id'] = op._id
                    _x_task["type"] = op.node_status
                    _x_task["job_completion_ratio"] = op.complete_ratio
                    _x_task['task_processing_time'] = op.processing_time
                    _x_task['remain_ops'] = op.remaining_ops
                    _x_task['task_wait_time'] = 0
                    _x_task["time_to_complete"] = op.remaining_time
                    # ScheduleNet feature
                    _x_task["agent"] = 0
                    _x_task["target_agent"] = 0
                    _x_task["assigned"] = 1
                    _x_task["waiting"] = 0
                    _x_task["processable"] = 0
                    _x_task["accessible"] = 0
                    _x_task['node_type'] = 'assigned_task'
                elif done_cond:
                    _x_task = OrderedDict()
                    _x_task['id'] = op._id
                    _x_task["type"] = op.node_status
                    _x_task["job_completion_ratio"] = op.complete_ratio
                    _x_task['task_processing_time'] = op.processing_time
                    _x_task['remain_ops'] = op.remaining_ops
                    _x_task['task_wait_time'] = 0
                    _x_task["time_to_complete"] = -1
                    # ScheduleNet feature
                    _x_task["agent"] = 0
                    _x_task["target_agent"] = 0
                    _x_task["assigned"] = 1
                    _x_task["waiting"] = 0
                    _x_task["processable"] = 0
                    _x_task["accessible"] = 0
                    _x_task['node_type'] = 'completed_task'
                elif delayed_cond:
                    raise NotImplementedError("delayed operation")
                else:
                    raise RuntimeError("Not supporting node type")

                done_cond = _x_task["type"] == self.config.DONE_NODE_SIG

                node_id = op.id + num_machine  # task node iterate from num_machine + i
                g.add_node(node_id, **_x_task, task_done=done_cond)
                if detach_done:
                    if not done_cond:
                        g.add_edge(node_id, op.machine_id - 1, edge_feature=[0])  # task node -> agent node
                        machine_to_task_arc_feature = int(op in self.machines[op.machine_id].doable_ops())
                        g.add_edge(op.machine_id - 1, node_id,
                                   edge_feature=[machine_to_task_arc_feature])  # agent node -> task node
                        # out degrees for this op in job clique
                        for op_con in op.conjunctive_ops:
                            if op_con.node_status != self.config.DONE_NODE_SIG:
                                node_id_op_con = op_con.id + num_machine
                                g.add_edge(node_id, node_id_op_con, edge_feature=[0])
                else:
                    node_id = op.id + num_machine  # task node iterate from num_machine + i
                    g.add_node(node_id, **_x_task)
                    g.add_edge(node_id, op.machine_id - 1, edge_feature=[0])  # task node -> agent node
                    machine_to_task_arc_feature = int(op in self.machines[op.machine_id].doable_ops())
                    g.add_edge(op.machine_id - 1, node_id,
                               edge_feature=[machine_to_task_arc_feature])  # agent node -> task node
                    # out degrees for this op in job clique
                    for op_con in op.conjunctive_ops:
                        node_id_op_con = op_con.id + num_machine
                        g.add_edge(node_id, node_id_op_con, edge_feature=[0])

        return g

class DummyOperation:
    def __init__(self,config,
                 job_id,
                 step_id,
                 embedding_dim):
        self.job_id = job_id
        self.step_id = step_id
        self._id = (job_id, step_id)
        self.machine_id = 'NA'
        self.processing_time = 0
        self.embedding_dim = embedding_dim
        self.built = False
        self.type = config.DUMMY_NODE_SIG
        self._x = {'type': self.type}
        self.node_status = config.DUMMY_NODE_SIG
        self.remaining_time = 0

    @property
    def id(self):
        if hasattr(self, 'sur_id'):
            _id = self.sur_id
        else:
            _id = self._id
        return _id


class Operation:

    def __init__(self,
                 config,
                 job_id,
                 step_id,
                 machine_id,
                 complete_ratio,
                 prev_op,
                 processing_time,
                 job,
                 next_op=None,
                 disjunctive_ops=None,
                 conjunctive_ops=None):
        self.config=config

        self.job_id = job_id
        self.step_id = step_id
        self.job = job
        self._id = (job_id, step_id)
        self.machine_id = machine_id
        self.node_status = self.config.NOT_START_NODE_SIG
        self.complete_ratio = complete_ratio
        self.prev_op = prev_op
        self.delayed_time = 0
        self.processing_time = int(processing_time)
        self.remaining_time = - np.inf
        self.remaining_ops = self.job.num_sequence - (self.step_id + 1)
        self.waiting_time = 0
        self._next_op = next_op
        self._disjunctive_ops = disjunctive_ops
        self._conjunctive_ops = conjunctive_ops

        self.next_op_built = False
        self.disjunctive_built = False
        self.built = False

    def __str__(self):
        return "job {} step {}".format(self.job_id, self.step_id)

    def processable(self):
        prev_none = self.prev_op is None
        if self.prev_op is not None:
            prev_done = self.prev_op.node_status is self.config.DONE_NODE_SIG
        else:
            prev_done = False
        return prev_done or prev_none

    @property
    def id(self):
        if hasattr(self, 'sur_id'):
            _id = self.sur_id
        else:
            _id = self._id
        return _id

    @property
    def disjunctive_ops(self):
        return self._disjunctive_ops

    @disjunctive_ops.setter
    def disjunctive_ops(self, disj_ops):
        for ops in disj_ops:
            if not isinstance(ops, Operation):
                raise RuntimeError("Given {} is not Operation instance".format(ops))
        self._disjunctive_ops = disj_ops
        self.disjunctive_built = True
        if self.disjunctive_built and self.next_op_built:
            self.built = True

    @property
    def conjunctive_ops(self):
        return self._conjunctive_ops

    @conjunctive_ops.setter
    def conjunctive_ops(self, con_ops):
        for ops in con_ops:
            if not isinstance(ops, Operation):
                raise RuntimeError("Given {} is not Operation instance".format(ops))
        self._conjunctive_ops = con_ops

    @property
    def next_op(self):
        return self._next_op

    @next_op.setter
    def next_op(self, next_op):
        self._next_op = next_op
        self.next_op_built = True
        if self.disjunctive_built and self.next_op_built:
            self.built = True
