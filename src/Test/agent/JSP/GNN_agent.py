from collections import OrderedDict
import numpy as np
import networkx as nx
import random

from Test.agent.JSP.Basic_agent import Basic_Agent


class GNN_agent(Basic_Agent):
    """
       模拟器初始化函数

       参数：
       num_machines：int，作业车间的机器数量
       num_jobs：int，作业的数量
       detach_done：bool，指示作业是否已完成
       name：str，模拟器的名称
       machine_matrix：ndarray，机器矩阵
       processing_time_matrix：ndarray，处理时间矩阵
       embedding_dim：int，嵌入维度
       use_surrogate_index：bool，是否使用代理索引
       delay：bool，是否延迟
       verbose：bool，是否显示详细信息
       """
    def __init__(self,config,data):
        self.machine_manager = None
        self.job_manager = None
        self.config = config
        self.detach_done = False
        self.name=None
        self.machine_matrix = None
        self.processing_time_matrix = None
        self.embedding_dim = 16
        self.use_surrogate_index = True
        self.delay = False
        self.verbose = False

        if self.machine_matrix is None or self.processing_time_matrix is None:
            ms, prts = data[1],data[0]
            self.machine_matrix = ms.astype(int)
            self.processing_time_matrix = prts.astype(float)
        else:
            self.machine_matrix = self.machine_matrix.astype(int)
            self.processing_time_matrix = self.processing_time_matrix.astype(float)

        if self.name is None:
            self.name = '{} machine {} job'.format(self.config.Pn_m, self.config.Pn_j)

        self._machine_set = list(set(self.machine_matrix.flatten().tolist()))
        self.num_machine = len(self._machine_set)
        self.num_jobs = self.processing_time_matrix.shape[0]
        self.num_steps = self.processing_time_matrix.shape[1]

        self.reset(data,config)
        # simulation procedure : global_time +=1 -> do_processing -> transit

    def reset(self,data,config):
        """重置模拟器状态的方法"""
        self.global_time = 0  # -1 matters a lot
        self.job_manager = JobManager(config, self.machine_matrix,
                                      self.processing_time_matrix,
                                      embedding_dim=self.embedding_dim,
                                      use_surrogate_index=self.use_surrogate_index)
        self.machine_manager = MachineManager(config,self.machine_matrix,
                                              self.job_manager,
                                              self.delay,
                                              self.verbose)

    def process_one_time(self):
        self.global_time += 1
        self.machine_manager.do_processing(self.global_time)

    def transit(self, action=None,):
        if action is None:
            # 执行随机动作
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
                # 使用代理索引时进行状态转移
                if action in self.job_manager.sur_index_dict.keys():
                    job_id, step_id = self.job_manager.sur_index_dict[action]
                else:
                    raise RuntimeError("输入的动作无效")
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
        sub_list = []

        while True:
            t += 1
            m_list = []
            do_op_dict = self.get_doable_ops_in_dict()
            all_machine_work = False if bool(do_op_dict) else True

            if all_machine_work:
                # 所有机器在处理中，继续进行模拟
                self.process_one_time()
            else:
                num_ops_counter = 1
                for m_id, op_ids in do_op_dict.items():
                    num_ops = len(op_ids)
                    if num_ops == 1:
                        sub_list.append(op_ids[0])

                        self.transit(op_ids[0])
                        g, r, _ = self.observe(reward)
                        cum_reward = r + gamma * cum_reward
                    else:
                        m_list.append(m_id)
                        num_ops_counter *= num_ops

                if num_ops_counter != 1:
                    # 不是全部微不足道的动作，跳出循环
                    break

            jobs_done = [job.job_done for _, job in self.job_manager.jobs.items()]
            done = True if np.prod(jobs_done) == 1 else False

            if done:
                # 模拟结束
                break

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
                raise RuntimeWarning("访问不可用的机器 {}. 返回值为 None".format(machine_id))
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

        g = self.job_manager.observe(self.detach_done)

        if g.number_of_nodes()<225:
            print("g.number_of_nodes()",g.number_of_nodes())

        if return_doable:
            if self.use_surrogate_index:
                do_ops_list = self.get_doable_ops(return_list=True)
                for n in g.nodes:
                    if n in do_ops_list:
                        job_id, op_id = self.job_manager.sur_index_dict[n]
                        m_id = self.job_manager[job_id][op_id].machine_id
                        g.nodes[n]['doable'] = True
                        g.nodes[n]['machine'] = m_id
                    else:
                        g.nodes[n]['doable'] = False
                        g.nodes[n]['machine'] = 0
        if g.number_of_nodes() < 225:
            print("g.number_of_nodes()", g.number_of_nodes())

        return g, r, done

    def draw_gantt_chart(self, path, benchmark_name, max_x):
        # Draw a gantt chart
        self.job_manager.draw_gantt_chart(path, benchmark_name, max_x)


    @classmethod
    def from_path(self,cls, jssp_path, **kwargs):
        """
        从文件路径创建NodeProcessingTimeSimulator对象的方法

        参数：
        jssp_path: str，JSSP数据文件的路径
        kwargs: dict，其他参数

        返回：
        NodeProcessingTimeSimulator对象
        """
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
    """
    作业管理器
    """

    def __init__(self, config,
                 machine_matrix,
                 processing_time_matrix,
                 embedding_dim=16,
                 use_surrogate_index=True):
        self.config = config
        machine_matrix = machine_matrix.astype(int)
        processing_time_matrix = processing_time_matrix.astype(float)

        self.jobs = OrderedDict()

        # Constructing conjunctive edges
        for job_i, (m, pr_t) in enumerate(zip(machine_matrix, processing_time_matrix)):
            m = m + 1  # To make machine index starts from 1
            self.jobs[job_i] = Job(job_i, m, pr_t, embedding_dim,config)

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
            # Constructing surrogate indices
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

    def observe(self, detach_done=True):
        """
        :return: Current time stamp job-shop graph
        """

        g = nx.OrderedDiGraph()
        for job_id, job in self.jobs.items():
            for op in job.ops:
                not_start_cond = not (op == job.ops[0])
                not_end_cond = not (op == job.ops[-1])

                done_cond = op.x['type'] == self.config.DONE_NODE_SIG

                if detach_done:
                    if not done_cond:
                        g.add_node(op.id, **op.x)
                        if not_end_cond:  # Construct forward flow conjunctive edges only
                            g.add_edge(op.id, op.next_op.id,
                                       processing_time=op.processing_time,
                                       type=self.config.CONJUNCTIVE_TYPE,
                                       direction=self.config.FORWARD)

                        if not_start_cond:  # Construct backward flow conjunctive edges only
                            if op.prev_op.x['type'] != self.config.DONE_NODE_SIG:
                                g.add_edge(op.id, op.prev_op.id,
                                           processing_time=-1 * op.prev_op.processing_time,
                                           type=self.config.CONJUNCTIVE_TYPE,
                                           direction=self.config.BACKWARD)

                        for disj_op in op.disjunctive_ops:  # Construct disjunctive edges
                            if disj_op.x['type'] != self.config.DONE_NODE_SIG:
                                g.add_edge(op.id, disj_op.id, type=self.config.DISJUNCTIVE_TYPE)

                else:
                    g.add_node(op.id, **op.x)
                    if not_end_cond:  # Construct forward flow conjunctive edges only
                        g.add_edge(op.id, op.next_op.id,
                                   processing_time=op.processing_time,
                                   type=self.config.CONJUNCTIVE_TYPE,
                                   direction=self.config.FORWARD)

                    if not_start_cond:  # Construct backward flow conjunctive edges only
                        g.add_edge(op.id, op.prev_op.id,
                                   processing_time=-1 * op.prev_op.processing_time,
                                   type=self.config.CONJUNCTIVE_TYPE,
                                   direction=self.config.BACKWARD)

                    for disj_op in op.disjunctive_ops:  # Construct disjunctive edges
                        g.add_edge(op.id, disj_op.id, type=self.config.DISJUNCTIVE_TYPE)

        return g

class Job:
    """
    作业
    """

    def __init__(self, job_id, machine_order, processing_time_order, embedding_dim,config):
        self.config = config
        self.job_id = job_id
        self.ops = list()
        self.processing_time = np.sum(processing_time_order)
        self.num_sequence = processing_time_order.size
        # Connecting backward paths (add prev_op to operations)
        cum_pr_t = 0
        for step_id, (m_id, pr_t) in enumerate(zip(machine_order, processing_time_order)):
            cum_pr_t += pr_t
            op = Operation(self.config,job_id=job_id, step_id=step_id, machine_id=m_id,
                           prev_op=None,
                           processing_time=pr_t,
                           complete_ratio=cum_pr_t / self.processing_time,
                           job=self)
            self.ops.append(op)
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

class Operation:

    def __init__(self,config,
                 job_id,
                 step_id,
                 machine_id,
                 complete_ratio,
                 prev_op,
                 processing_time,
                 job,
                 next_op=None,
                 disjunctive_ops=None):
        self.config=config

        self.job_id = job_id
        self.step_id = step_id
        self.job = job
        self._id = (job_id, step_id)
        self.machine_id = machine_id
        self.node_status = config.NOT_START_NODE_SIG
        self.complete_ratio = complete_ratio
        self.prev_op = prev_op
        self.delayed_time = 0
        self.processing_time = int(processing_time)
        self.remaining_time = - np.inf
        self.remaining_ops = self.job.num_sequence - (self.step_id + 1)
        self.waiting_time = 0
        self._next_op = next_op
        self._disjunctive_ops = disjunctive_ops

        self.next_op_built = False
        self.disjunctive_built = False
        self.built = False

    def __str__(self):
        return "job {} step {}".format(self.job_id, self.step_id)

    def processible(self):
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
    def next_op(self):
        return self._next_op

    @next_op.setter
    def next_op(self, next_op):
        self._next_op = next_op
        self.next_op_built = True
        if self.disjunctive_built and self.next_op_built:
            self.built = True

    @property
    def x(self):  # return node attribute
        not_start_cond = (self.node_status == self.config.NOT_START_NODE_SIG)
        delayed_cond = (self.node_status == self.config.DELAYED_NODE_SIG)
        processing_cond = (self.node_status == self.config.PROCESSING_NODE_SIG)
        done_cond = (self.node_status == self.config.DONE_NODE_SIG)

        if not_start_cond:
            _x = OrderedDict()
            _x['id'] = self._id
            _x["type"] = self.node_status
            _x["complete_ratio"] = self.complete_ratio
            _x['processing_time'] = self.processing_time
            _x['remaining_ops'] = self.remaining_ops
            _x['waiting_time'] = self.waiting_time
            _x["remain_time"] = -1
        elif processing_cond:
            _x = OrderedDict()
            _x['id'] = self._id
            _x["type"] = self.node_status
            _x["complete_ratio"] = self.complete_ratio
            _x['processing_time'] = self.processing_time
            _x['remaining_ops'] = self.remaining_ops
            _x['waiting_time'] = 0
            _x["remain_time"] = self.remaining_time
        elif done_cond:
            _x = OrderedDict()
            _x['id'] = self._id
            _x["type"] = self.node_status
            _x["complete_ratio"] = self.complete_ratio
            _x['processing_time'] = self.processing_time
            _x['remaining_ops'] = self.remaining_ops
            _x['waiting_time'] = 0
            _x["remain_time"] = -1
        elif delayed_cond:
            raise NotImplementedError("delayed operation")
        else:
            raise RuntimeError("Not supporting node type")
        return _x

class NodeProcessingTimeJobManager(JobManager):
    """
    以节点处理时间为准的作业管理器
    """
    def __init__(self, config,machine_matrix, processing_time_matrix, embedding_dim=16, use_surrogate_index=True):
        super().__init__(config,machine_matrix, processing_time_matrix, embedding_dim, use_surrogate_index)
        machine_matrix = machine_matrix.astype(int)
        processing_time_matrix = processing_time_matrix.astype(float)

        self.jobs = OrderedDict()
        self.config = config

        # Constructing conjunctive edges
        for job_i, (m, pr_t) in enumerate(zip(machine_matrix, processing_time_matrix)):
            m = m + 1  # To make machine index starts from 1
            self.jobs[job_i] = NodeProcessingTimeJob(job_i, m, pr_t, embedding_dim,config)

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

    def observe(self, detach_done=True):
        """
        :return: Current time stamp job-shop graph
        """

        g = nx.OrderedDiGraph()
        for job_id, job in self.jobs.items():
            for op in job.ops:
                not_start_cond = not (op == job.ops[0])
                not_end_cond = not isinstance(op, EndOperation)

                done_cond = op.x['type'] == self.config.DONE_NODE_SIG

                if detach_done:
                    if not done_cond:
                        g.add_node(op.id, **op.x)
                        if not_end_cond:  # Construct forward flow conjunctive edges only
                            g.add_edge(op.id, op.next_op.id,
                                       distance=(op.next_op.complete_ratio - op.complete_ratio),
                                       type=self.config.CONJUNCTIVE_TYPE,
                                       direction=self.config.FORWARD)
                        if not_start_cond:  # Construct backward flow conjunctive edges only
                            g.add_edge(op.id, op.prev_op.id,
                                       distance=-(op.complete_ratio - op.prev_op.complete_ratio),
                                       type=self.config.CONJUNCTIVE_TYPE,
                                       direction=self.config.BACKWARD)

                        for disj_op in op.disjunctive_ops:  # Construct disjunctive edges
                            g.add_edge(op.id, disj_op.id, type=self.config.DISJUNCTIVE_TYPE)

                else:
                    g.add_node(op.id, **op.x)
                    if not_end_cond:  # Construct forward flow conjunctive edges only
                        g.add_edge(op.id, op.next_op.id,
                                   distance=(op.next_op.complete_ratio - op.complete_ratio),
                                   type=self.config.CONJUNCTIVE_TYPE,
                                   direction=self.config.FORWARD)

                    if not_start_cond:  # Construct backward flow conjunctive edges only
                        g.add_edge(op.id, op.prev_op.id,
                                   distance=-(op.complete_ratio - op.prev_op.complete_ratio),
                                   type=self.config.CONJUNCTIVE_TYPE,
                                   direction=self.config.BACKWARD)

                    for disj_op in op.disjunctive_ops:  # Construct disjunctive edges
                        g.add_edge(op.id, disj_op.id, type=self.config.DISJUNCTIVE_TYPE)
        return g

class NodeProcessingTimeJob(Job):

    def __init__(self, job_id, machine_order, processing_time_order, embedding_dim,config):
        super().__init__(job_id, machine_order, processing_time_order, embedding_dim,config)
        self.job_id = job_id
        self.ops = list()
        self.processing_time = np.sum(processing_time_order)
        # Connecting backward paths (add prev_op to operations)
        cum_pr_t = 0
        for step_id, (m_id, pr_t) in enumerate(zip(machine_order, processing_time_order)):
            op = NodeProcessingTimeOperation(config=config,job_id=job_id,
                                             step_id=step_id,
                                             machine_id=m_id,
                                             prev_op=None,
                                             processing_time=pr_t,
                                             complete_ratio=cum_pr_t / self.processing_time,
                                             job=self)
            cum_pr_t += pr_t
            self.ops.append(op)
        for i, op in enumerate(self.ops[1:]):
            op.prev_op = self.ops[i]

        # instantiate DUMMY END node
        _prev_op = self.ops[-1]
        self.ops.append(NodeProcessingTimeEndOperation(job_id=job_id,
                                                       step_id=_prev_op.step_id + 1,
                                                       embedding_dim=embedding_dim,
                                                       config=config))
        self.ops[-1].prev_op = _prev_op
        self.num_sequence = len(self.ops) - 1

        # Connecting forward paths (add next_op to operations)
        for i, node in enumerate(self.ops[:-1]):
            node.next_op = self.ops[i + 1]

class DummyOperation:
    """
    哑元运算
    """

    def __init__(self,
                 job_id,
                 step_id,
                 embedding_dim,config):
        self.job_id = job_id
        self.step_id = step_id
        self._id = (job_id, step_id)
        self.machine_id = 'NA'
        self.processing_time = 0
        self.embedding_dim = embedding_dim
        self.built = False
        self.remaining_time = 0
        self.config = config
        self.type = self.config.DUMMY_NODE_SIG
        self._x = {'type': self.type}
        self.node_status = self.config.DUMMY_NODE_SIG

    @property
    def id(self):
        if hasattr(self, 'sur_id'):
            _id = self.sur_id
        else:
            _id = self._id
        return _id

class StartOperation(DummyOperation):

    def __init__(self, job_id, embedding_dim,config):
        super().__init__(job_id=job_id, step_id=-1, embedding_dim=embedding_dim,config=config)
        self.complete_ratio = 0.0
        self._next_op = None

    @property
    def next_op(self):
        return self._next_op

    @next_op.setter
    def next_op(self, op):
        self._next_op = op
        self.built = True

    @property
    def x(self):
        ret = self._x
        ret['complete_ratio'] = self.complete_ratio
        return ret

class EndOperation(DummyOperation):

    def __init__(self, job_id, step_id, embedding_dim,config):
        super().__init__(job_id=job_id, step_id=step_id, embedding_dim=embedding_dim,config=config)
        self.remaining_time = -1.0
        self.complete_ratio = 1.0
        self._prev_op = None

    @property
    def prev_op(self):
        return self._prev_op

    @prev_op.setter
    def prev_op(self, op):
        self._prev_op = op
        self.built = True

    @property
    def x(self):
        ret = self._x
        ret['complete_ratio'] = self.complete_ratio
        ret['remain_time'] = self.remaining_time
        return ret

class NodeProcessingTimeEndOperation(EndOperation):

    @property
    def x(self):
        ret = self._x
        ret['processing_time'] = self.processing_time
        ret['remain_time'] = self.remaining_time
        return ret

class NodeProcessingTimeOperation(Operation):

    def __init__(self, config,job_id, step_id, machine_id, complete_ratio, prev_op, processing_time, job, next_op=None,
                 disjunctive_ops=None):

        super().__init__(config,job_id, step_id, machine_id, complete_ratio, prev_op, processing_time, job, next_op,
                         disjunctive_ops)

        self.config=config
        self.job_id = job_id
        self.step_id = step_id
        self.job = job
        self._id = (job_id, step_id)
        self.machine_id = machine_id
        self.node_status = None
        self.complete_ratio = complete_ratio
        self.prev_op = prev_op
        self.processing_time = int(processing_time)
        self.remaining_time = - np.inf
        self._next_op = next_op
        self._disjunctive_ops = disjunctive_ops

        self.start_time = None
        self.end_time = None

        self.next_op_built = False
        self.disjunctive_built = False
        self.built = False

    @property
    def x(self):  # return node attribute
        not_start_cond = (self.node_status == self.config.NOT_START_NODE_SIG)
        delayed_cond = (self.node_status == self.config.DELAYED_NODE_SIG)
        processing_cond = (self.node_status == self.config.PROCESSING_NODE_SIG)
        done_cond = (self.node_status == self.config.DONE_NODE_SIG)

        if not_start_cond:
            _x = OrderedDict()
            _x["processing_time"] = self.processing_time
            _x["type"] = self.node_status
            _x["remain_time"] = -1
        elif processing_cond or done_cond or delayed_cond:
            _x = OrderedDict()
            _x["processing_time"] = self.processing_time
            _x["type"] = self.node_status
            _x["remain_time"] = self.remaining_time
        else:
            raise RuntimeError("Not supporting node type")
        return _x

class MachineManager:
    def __init__(self,config,machine_matrix,job_manager,delay=True,verbose=False):
        """
        Initialize the MachineManager.

        Parameters:
        - machine_matrix: numpy array, the matrix representing the machines and their corresponding operations
        - job_manager: dictionary, mapping job id to corresponding operations
        - delay: bool, whether to allow delay or not
        - verbose: bool, whether to print detailed information during processing
        """

        machine_matrix = machine_matrix.astype(int)

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
        """
        Perform processing for all machines at time t.

        Parameters:
        - t: int, current time step
        """
        for _, machine in self.machines.items():
            machine.do_processing(t)

    def load_op(self, machine_id, op, t):
        """
        Load an operation to a specified machine at time t.

        Parameters:
        - machine_id: int, the id of the machine
        - op: Operation, the operation to be loaded
        - t: int, current time step
        """
        self.machines[machine_id].load_op(op, t)

    def __getitem__(self, index):
        return self.machines[index]

    def get_available_machines(self, shuffle_machine=True):
        """
        Get a list of available machines.

        Parameters:
        - shuffle_machine: bool, whether to shuffle the machine list

        Returns:
        - list, list of available machines
        """
        m_list = []
        for _, m in self.machines.items():
            if m.available():
                m_list.append(m)

        if shuffle_machine:
            m_list = random.sample(m_list, len(m_list))

        return m_list

    def get_idle_machines(self):
        """
        Get a list of idle machines.

        Returns:
        - list, list of idle machines
        """
        m_list = []
        for _, m in self.machines.items():
            if m.current_op is None and not m.work_done():
                m_list.append(m)
        return m_list

    def cal_total_cost(self):
        """
        Calculate the total cost (length of queues) for all machines.

        Returns:
        - int, total cost
        """
        c = 0
        for _, m in self.machines.items():
            c += len(m.doable_ops_no_delay)
        return c

    def update_cost_function(self, cost):
        """
        Update the cost function for all machines.

        Parameters:
        - cost: int, the cost to be added
        """
        for _, m in self.machines.items():
            m.cost += cost

    def get_machines(self):
        """
        Get a list of all machines.

        Returns:
        - list, list of all machines
        """
        m_list = [m for _, m in self.machines.items()]
        return random.sample(m_list, len(m_list))

    def all_delayed(self):
        """
        Check if all machines are delayed.

        Returns:
        - bool, indicating whether all machines are delayed
        """
        return np.product([m.delayed_op is not None for _, m in self.machines.items()])

    def fab_stuck(self):
        """
        Check if the fabrication is stuck (all machines are not available and all machines are delayed).

        Returns:
        - bool, indicating whether the fabrication is stuck
        """
        all_machines_not_available_cond = not self.get_available_machines()
        all_machines_delayed_cond = self.all_delayed()
        return all_machines_not_available_cond and all_machines_delayed_cond

class Machine:
    def __init__(self, machine_id, possible_ops, delay, verbose,config):
        """
        Initialize the Machine.

        Parameters:
        - machine_id: int, the id of the machine
        - possible_ops: list, the list of possible operations for the machine
        - delay: bool, whether to allow delay or not
        - verbose: bool, whether to print detailed information during processing
        """
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

    def available(self):
        """
        Check if the machine is available for processing.

        Returns:
        - bool, indicating whether the machine is available
        """
        future_work_exist_cond = bool(self.doable_ops())
        currently_not_processing_cond = self.current_op is None
        not_wait_for_delayed_cond = not self.wait_for_delayed()
        ret = future_work_exist_cond and currently_not_processing_cond and not_wait_for_delayed_cond
        return ret

    def wait_for_delayed(self):
        """
        Check if the machine is waiting for a delayed operation.

        Returns:
        - bool, indicating whether the machine is waiting for a delayed operation
        """
        wait_for_delayed_cond = self.delayed_op is not None
        ret = wait_for_delayed_cond
        if wait_for_delayed_cond:
            delayed_op_ready_cond = self.delayed_op.prev_op.node_status == self.config.DONE_NODE_SIG
            ret = ret and not delayed_op_ready_cond
        return ret

    def doable_ops(self):
        """
        Get the list of operations that are currently doable.

        Returns:
        - list, list of doable operations
        """
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
        """
        Get the ids of doable operations.

        Returns:
        - list, list of ids of doable operations
        """
        doable_ops_id = []
        doable_ops = self.doable_ops()
        for op in doable_ops:
            doable_ops_id.append(op.id)

        return doable_ops_id

    @property
    def doable_ops_no_delay(self):
        """
        Get the list of doable operations without delay.

        Returns:
        - list, list of doable operations without delay
        """
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
        """
        Check if all work is done for the machine.

        Returns:
        - bool, indicating whether all work is done
        """
        return not self.remain_ops

    def load_op(self, t, op):
        """
        Load an operation to the machine at time t.

        Parameters:
        - t: int, current time step
        - op: Operation, the operation to be loaded
        """

        # Procedures for double-checkings
        if self.wait_for_delayed():
            raise RuntimeError("Machine {} waits for the delayed job {} but load {}".format(self.machine_id,
                                                                                            print(self.delayed_op),
                                                                                            print(op)))

        if not self.available():
            raise RuntimeError("Machine {} is not available".format(self.machine_id))

        if not op.processible():
            raise RuntimeError("Operation {} is not processible yet".format(print(op)))

        if op not in self.possible_ops:
            raise RuntimeError("Machine {} can't perform ops {}{}".format(self.machine_id,
                                                                          op.job_id,
                                                                          op.step_id))

        if op == self.delayed_op:
            if self.verbose:
                print("[DELAYED OP LOADED] / MACHINE {} / {} / at {}".format(self.machine_id, op, t))
            self.delayed_op = None

        else:
            if self.verbose:
                print("[LOAD] / Machine {} / {} on at {}".format(self.machine_id, op, t))

        op.node_status = self.config.PROCESSING_NODE_SIG
        op.remaining_time = op.processing_time
        op.start_time = t

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
        if self.remaining_time > 0:
            if self.current_op is not None:
                self.current_op.remaining_time -= 1
                if self.current_op.remaining_time <= 0:
                    if self.current_op.remaining_time < 0:
                        raise RuntimeWarning("Negative remaining time observed")
                    if self.verbose:
                        print("[OP DONE] : / Machine  {} / Op {}/ t = {} ".format(self.machine_id, self.current_op, t))
                    self.unload(t)

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
        if self.available():
            if a.processible():
                self.load_op(t, a)
            else:
                a.node_status = self.config.DELAYED_NODE_SIG
                self.delayed_op = a
                self.delayed_op.remaining_time = a.processing_time + a.prev_op.remaining_time
                self.remaining_time = a.processing_time + a.prev_op.remaining_time
                self.current_op = None
                if self.verbose:
                    print(
                        "[DELAYED OP CHOSEN] : / Machine  {} / Op {}/ t = {} ".format(self.machine_id, self.delayed_op,
                                                                                      t))
        else:
            raise RuntimeError("Access to not available machine")



