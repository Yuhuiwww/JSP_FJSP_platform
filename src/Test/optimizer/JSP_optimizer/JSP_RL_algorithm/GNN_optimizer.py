import torch
import numpy as np
import networkx as nx
import time
from torch.nn.functional import relu, softmax
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch.distributions.categorical import Categorical
from Test.agent.JSP.GNN_agent import GNN_agent
from Test.optimizer.JSP_optimizer.JSP_RL_algorithm.Basic_learning_algorithm import Basic_learning_algorithm

dev = 'cuda' if torch.cuda.is_available() else 'cpu'

class MLP(torch.nn.Module):
    """
    MLP Model
    """

    def __init__(self,
                 num_layers=2,
                 in_chnl=8,
                 hidden_chnl=256,
                 out_chnl=8):
        super(MLP, self).__init__()

        self.layers = torch.nn.ModuleList()

        for l in range(num_layers):
            if l == 0:  # first layer
                self.layers.append(torch.nn.Linear(in_chnl, hidden_chnl))
                self.layers.append(torch.nn.ReLU())
                if num_layers == 1:
                    self.layers.append(torch.nn.Linear(hidden_chnl, out_chnl))
            elif l <= num_layers - 2:  # hidden layers
                self.layers.append(torch.nn.Linear(hidden_chnl, hidden_chnl))
                self.layers.append(torch.nn.ReLU())
            else:  # last layer
                self.layers.append(torch.nn.Linear(hidden_chnl, hidden_chnl))
                self.layers.append(torch.nn.ReLU())
                self.layers.append(torch.nn.Linear(hidden_chnl, out_chnl))

    def forward(self, h):
        for lyr in self.layers:
            h = lyr(h)
        return h

class RLGNNLayer(MessagePassing):
    """
    Graph Neural Network Layers
    """

    def __init__(self,
                 num_mlp_layer=2,
                 in_chnl=8,
                 hidden_chnl=256,
                 out_chnl=8):
        super(RLGNNLayer, self).__init__()

        self.module_pre = MLP(num_layers=num_mlp_layer, in_chnl=in_chnl, hidden_chnl=hidden_chnl, out_chnl=out_chnl)
        self.module_suc = MLP(num_layers=num_mlp_layer, in_chnl=in_chnl, hidden_chnl=hidden_chnl, out_chnl=out_chnl)
        self.module_dis = MLP(num_layers=num_mlp_layer, in_chnl=in_chnl, hidden_chnl=hidden_chnl, out_chnl=out_chnl)
        self.module_merge = MLP(num_layers=num_mlp_layer, in_chnl=6 * out_chnl, hidden_chnl=hidden_chnl,
                                out_chnl=out_chnl)
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.module_pre)
        reset(self.module_suc)
        reset(self.module_dis)

    def message(self, x_j: Tensor) -> Tensor:
        return x_j

    def forward(self, raw_feature, **graphs):
        graph_pre = graphs['pre']
        graph_suc = graphs['suc']
        graph_dis = graphs['dis']

        num_nodes = graph_pre.num_nodes  # either pre, suc, or dis will work
        h_before_process = graph_pre.x  # either pre, suc, or dis will work

        # message passing
        out_pre = self.propagate(graph_pre.edge_index, x=graph_pre.x, size=None)
        out_suc = self.propagate(graph_suc.edge_index, x=graph_suc.x, size=None)
        out_dis = self.propagate(graph_dis.edge_index, x=graph_dis.x, size=None)

        # process aggregated messages
        out_pre = self.module_pre(out_pre)
        out_suc = self.module_suc(out_suc)
        out_dis = self.module_dis(out_dis)

        # merge different h
        h = torch.cat([relu(out_pre),
                       relu(out_suc),
                       relu(out_dis),
                       relu(h_before_process.sum(dim=0).tile(num_nodes, 1)),
                       h_before_process,
                       raw_feature], dim=1)
        h = self.module_merge(h)

        # new graphs after processed by this layer
        graph_pre = Data(x=h, edge_index=graph_pre.edge_index)
        graph_suc = Data(x=h, edge_index=graph_suc.edge_index)
        graph_dis = Data(x=h, edge_index=graph_dis.edge_index)

        return {'pre': graph_pre, 'suc': graph_suc, 'dis': graph_dis}

class RLGNN(torch.nn.Module):
    """
    GNN Model
    """

    def __init__(self,
                 num_mlp_layer=2,
                 num_layer=3,
                 in_chnl=8,
                 hidden_chnl=256,
                 out_chnl=8):
        super(RLGNN, self).__init__()

        self.layers = torch.nn.ModuleList()

        for l in range(num_layer):
            if l == 0:  # initial layer
                self.layers.append(RLGNNLayer(num_mlp_layer=num_mlp_layer,
                                              in_chnl=in_chnl,
                                              hidden_chnl=hidden_chnl,
                                              out_chnl=out_chnl))
            else:  # the rest layers
                self.layers.append(RLGNNLayer(num_mlp_layer=num_mlp_layer,
                                              in_chnl=out_chnl,
                                              hidden_chnl=hidden_chnl,
                                              out_chnl=out_chnl))

    def forward(self, raw_feature, **graphs):
        for layer in self.layers:
            graphs = layer(raw_feature, **graphs)
        return graphs

class PolicyNet(torch.nn.Module):
    """
    Policy Network Model (PNM)
    """

    def __init__(self,
                 num_mlp_layer=2,
                 in_chnl=8,
                 hidden_chnl=256,
                 out_chnl=1):
        super(PolicyNet, self).__init__()

        self.policy = MLP(num_layers=num_mlp_layer, in_chnl=in_chnl, hidden_chnl=hidden_chnl, out_chnl=out_chnl)

    def forward(self, node_h, feasible_op_id):
        logit = self.policy(node_h).view(-1)
        pi = softmax(logit[feasible_op_id], dim=0)
        dist = Categorical(probs=pi)
        sampled_op_id = dist.sample()
        sampled_op = feasible_op_id[sampled_op_id.item()]
        log_prob = dist.log_prob(sampled_op_id)
        return sampled_op, log_prob

class CriticNet(torch.nn.Module):
    """
    Critic Network Model
    """

    def __init__(self,
                 num_mlp_layer=2,
                 in_chnl=8,
                 hidden_chnl=256,
                 out_chnl=1):
        super(CriticNet, self).__init__()

        self.critic = MLP(num_layers=num_mlp_layer, in_chnl=in_chnl, hidden_chnl=hidden_chnl, out_chnl=out_chnl)

    def forward(self, node_h):
        v = self.critic(node_h.sum(dim=0))
        return v

class GNN_optimizer(Basic_learning_algorithm):
    def __init__(self, config):
        super().__init__(config)
        self.config = config


    def init_population(self,problem,data,config):
        ms, prts = data[0], data[1]
        self.machine_matrix = ms.astype(int)
        self.processing_time_matrix = prts.astype(float)
        self.embedding_dim = 16
        self.use_surrogate_index = True,
        self.delay = False,
        self.verbose = False


    def to_pyg(self, g, dev):
        """
        Convert the original diagram to PyG format
        Args:
            g: networkx.Graph，original drawing
            dev: str，device name
        Returns:
            torch_geometric.Data，Processed Chart
        """
        x = []
        one_hot = np.eye(3, dtype=np.float32)[np.fromiter(nx.get_node_attributes(g, 'type').values(), dtype=np.int32)]
        x.append(one_hot)
        x.append(
            np.fromiter(nx.get_node_attributes(g, 'processing_time').values(), dtype=np.float32).reshape(-1, 1)
        )
        x.append(
            np.fromiter(nx.get_node_attributes(g, 'complete_ratio').values(), dtype=np.float32).reshape(-1, 1)
        )
        x.append(
            np.fromiter(nx.get_node_attributes(g, 'remaining_ops').values(), dtype=np.float32).reshape(-1, 1)
        )
        x.append(
            np.fromiter(nx.get_node_attributes(g, 'waiting_time').values(), dtype=np.float32).reshape(-1, 1)
        )
        x.append(
            np.fromiter(nx.get_node_attributes(g, 'remain_time').values(), dtype=np.float32).reshape(-1, 1)
        )
        x = np.concatenate(x, axis=1)
        x = torch.from_numpy(x)

        for n in g.nodes:
            if g.nodes[n]['type'] == 1:
                x[n] = 0  # Completed operations characterized as 0

        adj_pre = np.zeros([g.number_of_nodes(), g.number_of_nodes()], dtype=np.float32)
        adj_suc = np.zeros([g.number_of_nodes(), g.number_of_nodes()], dtype=np.float32)
        adj_dis = np.zeros([g.number_of_nodes(), g.number_of_nodes()], dtype=np.float32)
        xx,yy = adj_dis.shape
        for e in g.edges:
            s, t = e
            if g.nodes[s]['id'][0] == g.nodes[t]['id'][0]:  # conjunctive edge
                if g.nodes[s]['id'][1] < g.nodes[t]['id'][1]:  # forward
                    adj_pre[s, t] = 1
                else:  # backward
                    adj_suc[s, t] = 1
            else:  # disjunctive edge
                if s>xx:
                    print(s)
                if t>xx:
                    print(t)
                adj_dis[s, t] = 1
        edge_index_pre = torch.nonzero(torch.from_numpy(adj_pre)).t().contiguous()
        edge_index_suc = torch.nonzero(torch.from_numpy(adj_suc)).t().contiguous()
        edge_index_dis = torch.nonzero(torch.from_numpy(adj_dis)).t().contiguous()

        g_pre = Data(x=x, edge_index=edge_index_pre).to(dev)
        g_suc = Data(x=x, edge_index=edge_index_suc).to(dev)
        g_dis = Data(x=x, edge_index=edge_index_dis).to(dev)

        return g_pre, g_suc, g_dis

    def rollout(self, data, config, dev, embedding_net=None, policy_net=None, critic_net=None, verbose=True):
        if embedding_net is not None and policy_net is not None and critic_net is not None:
            embedding_net.to(dev)
            policy_net.to(dev)
            critic_net.to(dev)
        s=GNN_agent(config,data)
        s.reset(data,config)
        done = False
        # max_run_time=self.config.Pn_j*self.config.Pn_m*0.08
        p_list = []
        t1 = time.time()
        while True:
            do_op_dict = s.get_doable_ops_in_dict()
            all_machine_work = False if bool(do_op_dict) else True

            if all_machine_work:  # All machines are working. Keep going.
                s.process_one_time()
            else:  # Some machines may have trivial behavior, others do not
                _, _, done, sub_list = s.flush_trivial_ops(reward='makespan')  # Emptying trivial behavior
                p_list += sub_list
                if done:
                    break  # End of environment rollout
                g, r, done = s.observe(return_doable=True)
                if embedding_net is not None and policy_net is not None and critic_net is not None:  # Web Forward Propagation is here to stay
                    g_pre, g_suc, g_dis = self.to_pyg(g, dev)
                    raw_feature = g_pre.x  # Pre, suc or dis are all fine.
                    pyg_graphs = {'pre': g_pre, 'suc': g_suc, 'dis': g_dis}
                    pyg_graphs = embedding_net(raw_feature, **pyg_graphs)
                    feasible_op_id = s.get_doable_ops_in_list()
                    sampled_action, _ = policy_net(pyg_graphs['pre'].x, feasible_op_id)  # Pre, suc or dis are all fine.
                    s.transit(sampled_action)
                    p_list.append(sampled_action)
                    v = critic_net(pyg_graphs['pre'].x)  # Pre, suc or dis are all fine.
                else:
                    op_id = s.transit()
                    p_list.append(op_id)

            # if t2 - t1 > max_run_time:
            #     done=True
            if done:
                break  # End of environment rollout
        t2 = time.time()
        # if verbose:
        #     print('All tasks completed, makespan={}. Rollout spends time {} seconds'.format(s.global_time, t2 - t1))
        return p_list, t2 - t1, s.global_time

    def update(self, data, config):
        t1 = time.time()
        embed = RLGNN()  # GNN
        policy = PolicyNet()  # Policy Network Model
        critic = CriticNet()  # value
        _, t, min_makespan = GNN_optimizer.rollout(self, data, config, dev, embed, policy, critic)
        t2 = time.time()
        print('makespan', min_makespan, 'time', t2 - t1)
        return min_makespan, t2 - t1
