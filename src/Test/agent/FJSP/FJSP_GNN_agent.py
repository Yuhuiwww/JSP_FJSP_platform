import copy
import math

import numpy as np
from torch.distributions import Categorical
import torch
from torch import nn
from torch.nn import Identity
import torch.nn.functional as F
import gym
from dataclasses import dataclass

from Test.agent.FJSP.Basic_FJSP_agent import Basic_FJSP_agent


class Memory:
    def __init__(self):
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.action_indexes = []

        self.ope_ma_adj = []
        self.ope_pre_adj = []
        self.ope_sub_adj = []
        self.batch_idxes = []
        self.raw_opes = []
        self.raw_mas = []
        self.proc_time = []
        self.jobs_gather = []
        self.eligible = []
        self.nums_opes = []

    def clear_memory(self):
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.action_indexes[:]

        del self.ope_ma_adj[:]
        del self.ope_pre_adj[:]
        del self.ope_sub_adj[:]
        del self.batch_idxes[:]
        del self.raw_opes[:]
        del self.raw_mas[:]
        del self.proc_time[:]
        del self.jobs_gather[:]
        del self.eligible[:]
        del self.nums_opes[:]


class MLPs(nn.Module):
    '''
    MLPs in operation node embedding
    '''

    def __init__(self, W_sizes_ope, hidden_size_ope, out_size_ope, num_head, dropout):
        '''
        The multi-head and dropout mechanisms are not actually used in the final experiment.
        :param W_sizes_ope: A list of the dimension of input vector for each type,
        including [machine, operation (pre), operation (sub), operation (self)]
        :param hidden_size_ope: hidden dimensions of the MLPs
        :param out_size_ope: dimension of the embedding of operation nodes
        '''
        super(MLPs, self).__init__()
        self.in_sizes_ope = W_sizes_ope
        self.hidden_size_ope = hidden_size_ope
        self.out_size_ope = out_size_ope
        self.num_head = num_head
        self.dropout = dropout
        self.gnn_layers = nn.ModuleList()

        # A total of five MLPs and MLP_0 (self.project) aggregates information from other MLPs
        for i in range(len(self.in_sizes_ope)):
            self.gnn_layers.append(MLPsim(self.in_sizes_ope[i], self.out_size_ope, self.hidden_size_ope, self.num_head,
                                          self.dropout, self.dropout))
        self.project = nn.Sequential(
            nn.ELU(),
            nn.Linear(self.out_size_ope * len(self.in_sizes_ope), self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.hidden_size_ope),
            nn.ELU(),
            nn.Linear(self.hidden_size_ope, self.out_size_ope),
        )

    def forward(self, ope_ma_adj_batch, ope_pre_adj_batch, ope_sub_adj_batch, batch_idxes, feats):
        '''
        :param ope_ma_adj_batch: Adjacency matrix of operation and machine nodes
        :param ope_pre_adj_batch: Adjacency matrix of operation and pre-operation nodes
        :param ope_sub_adj_batch: Adjacency matrix of operation and sub-operation nodes
        :param batch_idxes: Uncompleted instances
        :param feats: Contains operation, machine and edge features
        '''
        h = (feats[1], feats[0], feats[0], feats[0])
        # Identity matrix for self-loop of nodes
        self_adj = torch.eye(feats[0].size(-2),
                             dtype=torch.int64).unsqueeze(0).expand_as(ope_pre_adj_batch[batch_idxes])

        # Calculate an return operation embedding
        adj = (ope_ma_adj_batch[batch_idxes], ope_pre_adj_batch[batch_idxes],
               ope_sub_adj_batch[batch_idxes], self_adj)
        MLP_embeddings = []
        for i in range(len(adj)):
            MLP_embeddings.append(self.gnn_layers[i](h[i], adj[i]))
        MLP_embedding_in = torch.cat(MLP_embeddings, dim=-1)
        mu_ij_prime = self.project(MLP_embedding_in)
        return mu_ij_prime

class GATedge(nn.Module):
    '''
    Machine node embedding
    '''
    def __init__(self,
                 in_feats,
                 out_feats,
                 num_head,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        '''
        :param in_feats: tuple, input dimension of (operation node, machine node)
        :param out_feats: Dimension of the output (machine embedding)
        :param num_head: Number of heads
        '''
        super(GATedge, self).__init__()
        self._num_heads = num_head  # single head is used in the actual experiment
        self._in_src_feats = in_feats[0]
        self._in_dst_feats = in_feats[1]
        self._out_feats = out_feats

        if isinstance(in_feats, tuple):
            self.fc_src = nn.Linear(
                self._in_src_feats, out_feats * num_head, bias=False)
            self.fc_dst = nn.Linear(
                self._in_dst_feats, out_feats * num_head, bias=False)
            self.fc_edge = nn.Linear(
                1, out_feats * num_head, bias=False)
        else:
            self.fc = nn.Linear(
                self._in_src_feats, out_feats * num_head, bias=False)
        self.attn_l = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.attn_r = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.attn_e = nn.Parameter(torch.rand(size=(1, num_head, out_feats), dtype=torch.float))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)

        # Deprecated in final experiment
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, num_head * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        if hasattr(self, 'fc'):
            nn.init.xavier_normal_(self.fc.weight, gain=gain)
        else:
            nn.init.xavier_normal_(self.fc_src.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_dst.weight, gain=gain)
            nn.init.xavier_normal_(self.fc_edge.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        nn.init.xavier_normal_(self.attn_e, gain=gain)

    def forward(self, ope_ma_adj_batch, batch_idxes, feat):
        # Two linear transformations are used for the machine nodes and operation nodes, respective
        # In linear transformation, an W^O (\in R^{d \times 7}) for \mu_{ijk} is equivalent to
        #   W^{O'} (\in R^{d \times 6}) and W^E (\in R^{d \times 1}) for the nodes and edges respectively
        if isinstance(feat, tuple):
            h_src = self.feat_drop(feat[0])
            h_dst = self.feat_drop(feat[1])
            if not hasattr(self, 'fc_src'):
                self.fc_src, self.fc_dst = self.fc, self.fc
            feat_src = self.fc_src(h_src)
            feat_dst = self.fc_dst(h_dst)
        else:
            # Deprecated in final experiment
            h_src = h_dst = self.feat_drop(feat)
            feat_src = feat_dst = self.fc(h_src).view(
                -1, self._num_heads, self._out_feats)
        feat_edge = self.fc_edge(feat[2].unsqueeze(-1))

        # Calculate attention coefficients
        el = (feat_src * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat_dst * self.attn_r).sum(dim=-1).unsqueeze(-1)
        ee = (feat_edge * self.attn_l).sum(dim=-1).unsqueeze(-1)
        el_add_ee = ope_ma_adj_batch[batch_idxes].unsqueeze(-1) * el.unsqueeze(-2) + ee
        a = el_add_ee + ope_ma_adj_batch[batch_idxes].unsqueeze(-1) * er.unsqueeze(-3)
        eijk = self.leaky_relu(a)
        ekk = self.leaky_relu(er + er)

        # Normalize attention coefficients
        mask = torch.cat((ope_ma_adj_batch[batch_idxes].unsqueeze(-1)==1,
                          torch.full(size=(ope_ma_adj_batch[batch_idxes].size(0), 1,
                                           ope_ma_adj_batch[batch_idxes].size(2), 1),
                                     dtype=torch.bool, fill_value=True)), dim=-3)
        e = torch.cat((eijk, ekk.unsqueeze(-3)), dim=-3)
        e[~mask] = float('-inf')
        alpha = F.softmax(e.squeeze(-1), dim=-2)
        alpha_ijk = alpha[..., :-1, :]
        alpha_kk = alpha[..., -1, :].unsqueeze(-2)

        # Calculate an return machine embedding
        Wmu_ijk = feat_edge + feat_src.unsqueeze(-2)
        a = Wmu_ijk * alpha_ijk.unsqueeze(-1)
        b = torch.sum(a, dim=-3)
        c = feat_dst * alpha_kk.squeeze().unsqueeze(-1)
        nu_k_prime = torch.sigmoid(b+c)
        return nu_k_prime

class MLPsim(nn.Module):
    '''
    Part of operation node embedding
    '''
    def __init__(self,
                 in_feats,
                 out_feats,
                 hidden_dim,
                 num_head,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False):
        '''
        :param in_feats: Dimension of the input vectors of the MLPs
        :param out_feats: Dimension of the output (operation embedding) of the MLPs
        :param hidden_dim: Hidden dimensions of the MLPs
        :param num_head: Number of heads
        '''
        super(MLPsim, self).__init__()
        self._num_heads = num_head
        self._in_feats = in_feats
        self._out_feats = out_feats

        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        self.project = nn.Sequential(
            nn.Linear(self._in_feats, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ELU(),
            nn.Linear(hidden_dim, self._out_feats),
        )

        # Deprecated in final experiment
        if residual:
            if self._in_dst_feats != out_feats:
                self.res_fc = nn.Linear(
                    self._in_dst_feats, self._num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)

    def forward(self, feat, adj):
        # MLP_{\theta_x}, where x = 1, 2, 3, 4
        # Note that message-passing should along the edge (according to the adjacency matrix)
        a = adj.unsqueeze(-1) * feat.unsqueeze(-3)
        b = torch.sum(a, dim=-2)
        c = self.project(b)
        return c


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLP, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
            return self.linears[self.num_layers - 1](h)


class MLPActor(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLPActor, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                h = torch.tanh(self.linears[layer](h))
                # h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)


class MLPCritic(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        '''
            num_layers: number of layers in the neural networks (EXCLUDING the input layer). If num_layers=1, this reduces to linear model.
            input_dim: dimensionality of input features
            hidden_dim: dimensionality of hidden units at ALL layers
            output_dim: number of classes for prediction
            device: which device to use
        '''

        super(MLPCritic, self).__init__()

        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            '''
            self.batch_norms = torch.nn.ModuleList()
            '''

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))
            '''
            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))
            '''

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for layer in range(self.num_layers - 1):
                '''
                h = F.relu(self.batch_norms[layer](self.linears[layer](h)))
                '''
                h = torch.tanh(self.linears[layer](h))
                # h = F.relu(self.linears[layer](h))
            return self.linears[self.num_layers - 1](h)

class HGNNScheduler(nn.Module):
    def __init__(self, config):
        super(HGNNScheduler, self).__init__()
        self.device = config.device
        self.in_size_ma = config.in_size_ma  # Dimension of the raw feature vectors of machine nodes
        self.out_size_ma = config.out_size_ma # Dimension of the embedding of machine nodes
        self.in_size_ope = config.in_size_ope  # Dimension of the raw feature vectors of operation nodes
        self.out_size_ope = config.out_size_ope  # Dimension of the embedding of operation nodes
        self.hidden_size_ope = config.hidden_size_ope  # Hidden dimensions of the MLPs
        self.actor_dim = config.actor_in_dim  # Input dimension of actor
        self.critic_dim = config.critic_in_dim  # Input dimension of critic
        self.n_latent_actor = config.n_latent_actor  # Hidden dimensions of the actor
        self.n_latent_critic = config.n_latent_critic  # Hidden dimensions of the critic
        self.n_hidden_actor = config.n_hidden_actor  # Number of layers in actor
        self.n_hidden_critic = config.n_hidden_critic  # Number of layers in critic
        self.action_dim = config.action_dim  # Output dimension of actor

        # len() means of the number of HGNN iterations
        # and the element means the number of heads of each HGNN (=1 in final experiment)
        self.num_heads = config.num_heads
        self.dropout = config.dropout

        # Machine node embedding
        self.get_machines = nn.ModuleList()
        self.get_machines.append(GATedge((self.in_size_ope, self.in_size_ma), self.out_size_ma, self.num_heads[0],
                                         self.dropout, self.dropout, activation=F.elu))
        for i in range(1, len(self.num_heads)):
            self.get_machines.append(GATedge((self.out_size_ope, self.out_size_ma), self.out_size_ma, self.num_heads[i],
                                             self.dropout, self.dropout, activation=F.elu))

        # Operation node embedding
        self.get_operations = nn.ModuleList()
        self.get_operations.append(MLPs([self.out_size_ma, self.in_size_ope, self.in_size_ope, self.in_size_ope],
                                        self.hidden_size_ope, self.out_size_ope, self.num_heads[0], self.dropout))
        for i in range(len(self.num_heads) - 1):
            self.get_operations.append(MLPs([self.out_size_ma, self.out_size_ope, self.out_size_ope, self.out_size_ope],
                                            self.hidden_size_ope, self.out_size_ope, self.num_heads[i], self.dropout))

        self.actor = MLPActor(self.n_hidden_actor, self.actor_dim, self.n_latent_actor, self.action_dim).to(self.device)
        self.critic = MLPCritic(self.n_hidden_critic, self.critic_dim, self.n_latent_critic, 1).to(self.device)

    def forward(self):
        '''
        Replaced by separate act and evaluate functions
        '''
        raise NotImplementedError

    def feature_normalize(self, data):
        return (data - torch.mean(data)) / ((data.std() + 1e-5))

    '''
        raw_opes: shape: [len(batch_idxes), max(num_opes), in_size_ope]
        raw_mas: shape: [len(batch_idxes), num_mas, in_size_ma]
        proc_time: shape: [len(batch_idxes), max(num_opes), num_mas]
    '''

    def get_normalized(self, raw_opes, raw_mas, proc_time, batch_idxes, nums_opes, flag_sample=False, flag_train=False):
        '''
        :param raw_opes: Raw feature vectors of operation nodes
        :param raw_mas: Raw feature vectors of machines nodes
        :param proc_time: Processing time
        :param batch_idxes: Uncompleted instances
        :param nums_opes: The number of operations for each instance
        :param flag_sample: Flag for DRL-S
        :param flag_train: Flag for training
        :return: Normalized feats, including operations, machines and edges
        '''
        batch_size = batch_idxes.size(0)  # number of uncompleted instances

        # There may be different operations for each instance, which cannot be normalized directly by the matrix
        if not flag_sample and not flag_train:
            mean_opes = []
            std_opes = []
            for i in range(batch_size):
                mean_opes.append(torch.mean(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True))
                std_opes.append(torch.std(raw_opes[i, :nums_opes[i], :], dim=-2, keepdim=True))
                proc_idxes = torch.nonzero(proc_time[i])
                proc_values = proc_time[i, proc_idxes[:, 0], proc_idxes[:, 1]]
                proc_norm = self.feature_normalize(proc_values)
                proc_time[i, proc_idxes[:, 0], proc_idxes[:, 1]] = proc_norm
            mean_opes = torch.stack(mean_opes, dim=0)
            std_opes = torch.stack(std_opes, dim=0)
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)
            proc_time_norm = proc_time
        # DRL-S and scheduling during training have a consistent number of operations
        else:
            mean_opes = torch.mean(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
            mean_mas = torch.mean(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
            std_opes = torch.std(raw_opes, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ope]
            std_mas = torch.std(raw_mas, dim=-2, keepdim=True)  # shape: [len(batch_idxes), 1, in_size_ma]
            proc_time_norm = self.feature_normalize(proc_time)  # shape: [len(batch_idxes), num_opes, num_mas]
        return ((raw_opes - mean_opes) / (std_opes + 1e-5), (raw_mas - mean_mas) / (std_mas + 1e-5),
                proc_time_norm)

    def get_action_prob(self, state, memories, flag_sample=False, flag_train=False):
        '''
        Get the probability of selecting each action in decision-making
        '''
        # Uncompleted instances
        batch_idxes = state.batch_idxes
        # Raw feats
        raw_opes = state.feat_opes_batch.transpose(1, 2)[batch_idxes]
        raw_mas = state.feat_mas_batch.transpose(1, 2)[batch_idxes]
        proc_time = state.proc_times_batch[batch_idxes]
        # Normalize
        nums_opes = state.nums_opes_batch[batch_idxes]
        features = self.get_normalized(raw_opes, raw_mas, proc_time, batch_idxes, nums_opes, flag_sample, flag_train)
        norm_opes = (copy.deepcopy(features[0]))
        norm_mas = (copy.deepcopy(features[1]))
        norm_proc = (copy.deepcopy(features[2]))

        # L iterations of the HGNN
        for i in range(len(self.num_heads)):
            # First Stage, machine node embedding
            # shape: [len(batch_idxes), num_mas, out_size_ma]
            h_mas = self.get_machines[i](state.ope_ma_adj_batch, state.batch_idxes, features)
            features = (features[0], h_mas, features[2])
            # Second Stage, operation node embedding
            # shape: [len(batch_idxes), max(num_opes), out_size_ope]
            h_opes = self.get_operations[i](state.ope_ma_adj_batch, state.ope_pre_adj_batch, state.ope_sub_adj_batch,
                                            state.batch_idxes, features)
            features = (h_opes, features[1], features[2])

        # Stacking and pooling
        h_mas_pooled = h_mas.mean(dim=-2)  # shape: [len(batch_idxes), out_size_ma]
        # There may be different operations for each instance, which cannot be pooled directly by the matrix
        if not flag_sample and not flag_train:
            h_opes_pooled = []
            for i in range(len(batch_idxes)):
                h_opes_pooled.append(torch.mean(h_opes[i, :nums_opes[i], :], dim=-2))
            h_opes_pooled = torch.stack(h_opes_pooled)  # shape: [len(batch_idxes), d]
        else:
            h_opes_pooled = h_opes.mean(dim=-2)  # shape: [len(batch_idxes), out_size_ope]

        # Detect eligible O-M pairs (eligible actions) and generate tensors for actor calculation
        ope_step_batch = torch.where(state.ope_step_batch > state.end_ope_biases_batch,
                                     state.end_ope_biases_batch, state.ope_step_batch)
        jobs_gather = ope_step_batch[..., :, None].expand(-1, -1, h_opes.size(-1))[batch_idxes]
        h_jobs = h_opes.gather(1, jobs_gather)
        # Matrix indicating whether processing is possible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        eligible_proc = state.ope_ma_adj_batch[batch_idxes].gather(1,
                                                                   ope_step_batch[..., :, None].expand(-1, -1,
                                                                                                       state.ope_ma_adj_batch.size(
                                                                                                           -1))[
                                                                       batch_idxes])
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, state.proc_times_batch.size(-1), -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)
        # Matrix indicating whether machine is eligible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        ma_eligible = ~state.mask_ma_procing_batch[batch_idxes].unsqueeze(1).expand_as(h_jobs_padding[..., 0])
        # Matrix indicating whether job is eligible
        # shape: [len(batch_idxes), num_jobs, num_mas]
        job_eligible = ~(state.mask_job_procing_batch[batch_idxes] +
                         state.mask_job_finish_batch[batch_idxes])[:, :, None].expand_as(h_jobs_padding[..., 0])
        # shape: [len(batch_idxes), num_jobs, num_mas]
        eligible = job_eligible & ma_eligible & (eligible_proc == 1)
        if (~(eligible)).all():
            print("No eligible O-M pair!")
            return
        # Input of actor MLP
        # shape: [len(batch_idxes), num_mas, num_jobs, out_size_ma*2+out_size_ope*2]
        h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
                              dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)  # deprecated
        mask = eligible.transpose(1, 2).flatten(1)

        # Get priority index and probability of actions with masking the ineligible actions
        scores = self.actor(h_actions).flatten(1)
        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)

        # Store data in memory during training
        if flag_train == True:
            memories.ope_ma_adj.append(copy.deepcopy(state.ope_ma_adj_batch))
            memories.ope_pre_adj.append(copy.deepcopy(state.ope_pre_adj_batch))
            memories.ope_sub_adj.append(copy.deepcopy(state.ope_sub_adj_batch))
            memories.batch_idxes.append(copy.deepcopy(state.batch_idxes))
            memories.raw_opes.append(copy.deepcopy(norm_opes))
            memories.raw_mas.append(copy.deepcopy(norm_mas))
            memories.proc_time.append(copy.deepcopy(norm_proc))
            memories.nums_opes.append(copy.deepcopy(nums_opes))
            memories.jobs_gather.append(copy.deepcopy(jobs_gather))
            memories.eligible.append(copy.deepcopy(eligible))

        return action_probs, ope_step_batch, h_pooled

    def act(self, state, memories, dones, flag_sample=True, flag_train=True):
        # Get probability of actions and the id of the current operation (be waiting to be processed) of each job
        action_probs, ope_step_batch, _ = self.get_action_prob(state, memories, flag_sample, flag_train=flag_train)

        # DRL-S, sampling actions following \pi
        if flag_sample:
            dist = Categorical(action_probs)
            action_indexes = dist.sample()
        # DRL-G, greedily picking actions with the maximum probability
        else:
            action_indexes = action_probs.argmax(dim=1)

        # Calculate the machine, job and operation index based on the action index
        mas = (action_indexes / state.mask_job_finish_batch.size(1)).long()
        jobs = (action_indexes % state.mask_job_finish_batch.size(1)).long()
        opes = ope_step_batch[state.batch_idxes, jobs]

        # Store data in memory during training
        if flag_train == True:
            # memories.states.append(copy.deepcopy(state))
            memories.logprobs.append(dist.log_prob(action_indexes))
            memories.action_indexes.append(action_indexes)

        return torch.stack((opes, mas, jobs), dim=1).t()

    def evaluate(self, ope_ma_adj, ope_pre_adj, ope_sub_adj, raw_opes, raw_mas, proc_time,
                 jobs_gather, eligible, action_envs, flag_sample=False):
        batch_idxes = torch.arange(0, ope_ma_adj.size(-3)).long()
        features = (raw_opes, raw_mas, proc_time)

        # L iterations of the HGNN
        for i in range(len(self.num_heads)):
            h_mas = self.get_machines[i](ope_ma_adj, batch_idxes, features)
            features = (features[0], h_mas, features[2])
            h_opes = self.get_operations[i](ope_ma_adj, ope_pre_adj, ope_sub_adj, batch_idxes, features)
            features = (h_opes, features[1], features[2])

        # Stacking and pooling
        h_mas_pooled = h_mas.mean(dim=-2)
        h_opes_pooled = h_opes.mean(dim=-2)

        # Detect eligible O-M pairs (eligible actions) and generate tensors for critic calculation
        h_jobs = h_opes.gather(1, jobs_gather)
        h_jobs_padding = h_jobs.unsqueeze(-2).expand(-1, -1, proc_time.size(-1), -1)
        h_mas_padding = h_mas.unsqueeze(-3).expand_as(h_jobs_padding)
        h_mas_pooled_padding = h_mas_pooled[:, None, None, :].expand_as(h_jobs_padding)
        h_opes_pooled_padding = h_opes_pooled[:, None, None, :].expand_as(h_jobs_padding)

        h_actions = torch.cat((h_jobs_padding, h_mas_padding, h_opes_pooled_padding, h_mas_pooled_padding),
                              dim=-1).transpose(1, 2)
        h_pooled = torch.cat((h_opes_pooled, h_mas_pooled), dim=-1)
        scores = self.actor(h_actions).flatten(1)
        mask = eligible.transpose(1, 2).flatten(1)

        scores[~mask] = float('-inf')
        action_probs = F.softmax(scores, dim=1)
        state_values = self.critic(h_pooled)
        dist = Categorical(action_probs.squeeze())
        action_logprobs = dist.log_prob(action_envs)
        dist_entropys = dist.entropy()
        return action_logprobs, state_values.squeeze().double(), dist_entropys


class PPO:
    def __init__(self, config, num_envs=None):
        self.lr = config.lr  # learning rate
        self.betas = config.betas  # default value for Adam
        self.gamma = config.gamma  # discount factor
        self.eps_clip = config.eps_clip  # clip ratio for PPO
        self.K_epochs = config.K_epochs  # Update policy for K epochs
        self.A_coeff = config.A_coeff # coefficient for policy loss
        self.vf_coeff = config.vf_coeff  # coefficient for value loss
        self.entropy_coeff = config.entropy_coeff  # coefficient for entropy term
        self.num_envs = num_envs # Number of parallel instances
        self.device = config.device  # PyTorch device

        self.policy = HGNNScheduler(config).to(self.device)
        self.policy_old = copy.deepcopy(self.policy)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=self.lr, betas=self.betas)
        self.MseLoss = nn.MSELoss()

    def update(self, memory, config):
        device = config.device
        minibatch_size = config.minibatch_size  # batch size for updating

        # Flatten the data in memory (in the dim of parallel instances and decision points)
        old_ope_ma_adj = torch.stack(memory.ope_ma_adj, dim=0).transpose(0, 1).flatten(0, 1)
        old_ope_pre_adj = torch.stack(memory.ope_pre_adj, dim=0).transpose(0, 1).flatten(0, 1)
        old_ope_sub_adj = torch.stack(memory.ope_sub_adj, dim=0).transpose(0, 1).flatten(0, 1)
        old_raw_opes = torch.stack(memory.raw_opes, dim=0).transpose(0, 1).flatten(0, 1)
        old_raw_mas = torch.stack(memory.raw_mas, dim=0).transpose(0, 1).flatten(0, 1)
        old_proc_time = torch.stack(memory.proc_time, dim=0).transpose(0, 1).flatten(0, 1)
        old_jobs_gather = torch.stack(memory.jobs_gather, dim=0).transpose(0, 1).flatten(0, 1)
        old_eligible = torch.stack(memory.eligible, dim=0).transpose(0, 1).flatten(0, 1)
        memory_rewards = torch.stack(memory.rewards, dim=0).transpose(0, 1)
        memory_is_terminals = torch.stack(memory.is_terminals, dim=0).transpose(0, 1)
        old_logprobs = torch.stack(memory.logprobs, dim=0).transpose(0, 1).flatten(0, 1)
        old_action_envs = torch.stack(memory.action_indexes, dim=0).transpose(0, 1).flatten(0, 1)

        # Estimate and normalize the rewards
        rewards_envs = []
        discounted_rewards = 0
        for i in range(self.num_envs):
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memory_rewards[i]), reversed(memory_is_terminals[i])):
                if is_terminal:
                    discounted_rewards += discounted_reward
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)
            discounted_rewards += discounted_reward
            rewards = torch.tensor(rewards, dtype=torch.float64).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            rewards_envs.append(rewards)
        rewards_envs = torch.cat(rewards_envs)

        loss_epochs = 0
        full_batch_size = old_ope_ma_adj.size(0)
        num_complete_minibatches = math.floor(full_batch_size / minibatch_size)
        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            for i in range(num_complete_minibatches + 1):
                if i < num_complete_minibatches:
                    start_idx = i * minibatch_size
                    end_idx = (i + 1) * minibatch_size
                else:
                    start_idx = i * minibatch_size
                    end_idx = full_batch_size
                logprobs, state_values, dist_entropy = \
                    self.policy.evaluate(old_ope_ma_adj[start_idx: end_idx, :, :],
                                         old_ope_pre_adj[start_idx: end_idx, :, :],
                                         old_ope_sub_adj[start_idx: end_idx, :, :],
                                         old_raw_opes[start_idx: end_idx, :, :],
                                         old_raw_mas[start_idx: end_idx, :, :],
                                         old_proc_time[start_idx: end_idx, :, :],
                                         old_jobs_gather[start_idx: end_idx, :, :],
                                         old_eligible[start_idx: end_idx, :, :],
                                         old_action_envs[start_idx: end_idx])

                ratios = torch.exp(logprobs - old_logprobs[i * minibatch_size:(i + 1) * minibatch_size].detach())
                advantages = rewards_envs[i * minibatch_size:(i + 1) * minibatch_size] - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                loss = - self.A_coeff * torch.min(surr1, surr2) \
                       + self.vf_coeff * self.MseLoss(state_values,
                                                      rewards_envs[i * minibatch_size:(i + 1) * minibatch_size]) \
                       - self.entropy_coeff * dist_entropy
                loss_epochs += loss.mean().detach()

                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        return loss_epochs.item() / self.K_epochs, \
               discounted_rewards.item() / (self.num_envs * config.update_timestep)

@dataclass
class EnvState:
    '''
    Class for the state of the environment
    '''
    # static
    opes_appertain_batch: torch.Tensor = None
    ope_pre_adj_batch: torch.Tensor = None
    ope_sub_adj_batch: torch.Tensor = None
    end_ope_biases_batch: torch.Tensor = None
    nums_opes_batch: torch.Tensor = None

    # dynamic
    batch_idxes: torch.Tensor = None
    feat_opes_batch: torch.Tensor = None
    feat_mas_batch: torch.Tensor = None
    proc_times_batch: torch.Tensor = None
    ope_ma_adj_batch: torch.Tensor = None
    time_batch: torch.Tensor = None

    mask_job_procing_batch: torch.Tensor = None
    mask_job_finish_batch: torch.Tensor = None
    mask_ma_procing_batch: torch.Tensor = None
    ope_step_batch: torch.Tensor = None

    def update(self, batch_idxes, feat_opes_batch, feat_mas_batch, proc_times_batch, ope_ma_adj_batch,
               mask_job_procing_batch, mask_job_finish_batch, mask_ma_procing_batch, ope_step_batch, time):
        self.batch_idxes = batch_idxes
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch
        self.proc_times_batch = proc_times_batch
        self.ope_ma_adj_batch = ope_ma_adj_batch

        self.mask_job_procing_batch = mask_job_procing_batch
        self.mask_job_finish_batch = mask_job_finish_batch
        self.mask_ma_procing_batch = mask_ma_procing_batch
        self.ope_step_batch = ope_step_batch
        self.time_batch = time
class FJSPEnv(gym.Env,Basic_FJSP_agent):
    '''
    FJSP environment
    '''
    def __init__(self, config, data):
        self.config = config
        self.data = data
        # load instance
        num_data = 8  # The amount of data extracted from instance
        tensors = [[] for _ in range(num_data)]
        self.num_opes = 0
        lines = []

        self.batch_size = self.config.batch_size  # Number of parallel instances during training
        self.num_jobs = self.config.Pn_j  # Number of jobs
        self.num_mas = self.config.Pn_m  # Number of machines
        self.device = self.config.device  # Computing device for PyTorch
        if config.data_source == 'case':  # Generate instances through generators
            for i in range(self.batch_size):
                lines.append(data.get_case(i)[0])  # Generate an instance and save it
                num_jobs, num_mas, num_opes = self.nums_detec(lines[i])
                # Records the maximum number of operations in the parallel instances
                self.num_opes = max(self.num_opes, num_opes)
        else:  # Load instances from files
            for i in range(self.batch_size):
                with open(data[i]) as file_object:
                    line = file_object.readlines()
                    lines.append(line)
                num_jobs, num_mas, num_opes = self.nums_detec(lines[i])
                self.num_opes = max(self.num_opes, num_opes)
        # load feats
        for i in range(self.batch_size):
            load_data = self.load_fjs(lines[i], num_mas, self.num_opes)
            for j in range(num_data):
                tensors[j].append(load_data[j])

        # shape: (batch_size, num_opes, num_mas)
        self.proc_times_batch = torch.stack(tensors[0], dim=0)
        # shape: (batch_size, num_opes, num_mas)
        self.ope_ma_adj_batch = torch.stack(tensors[1], dim=0).long()
        # shape: (batch_size, num_opes, num_opes), for calculating the cumulative amount along the path of each job
        self.cal_cumul_adj_batch = torch.stack(tensors[7], dim=0).float()

        # static feats
        # shape: (batch_size, num_opes, num_opes)
        self.ope_pre_adj_batch = torch.stack(tensors[2], dim=0)
        # shape: (batch_size, num_opes, num_opes)
        self.ope_sub_adj_batch = torch.stack(tensors[3], dim=0)
        # shape: (batch_size, num_opes), represents the mapping between operations and jobs
        self.opes_appertain_batch = torch.stack(tensors[4], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the first operation of each job
        self.num_ope_biases_batch = torch.stack(tensors[5], dim=0).long()
        # shape: (batch_size, num_jobs), the number of operations for each job
        self.nums_ope_batch = torch.stack(tensors[6], dim=0).long()
        # shape: (batch_size, num_jobs), the id of the last operation of each job
        self.end_ope_biases_batch = self.num_ope_biases_batch + self.nums_ope_batch - 1
        # shape: (batch_size), the number of operations for each instance
        self.nums_opes = torch.sum(self.nums_ope_batch, dim=1)

        # dynamic variable
        self.batch_idxes = torch.arange(self.batch_size)  # Uncompleted instances
        self.time = torch.zeros(self.batch_size)  # Current time of the environment
        self.N = torch.zeros(self.batch_size).int()  # Count scheduled operations
        # shape: (batch_size, num_jobs), the id of the current operation (be waiting to be processed) of each job
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        # Generate raw feature vectors
        feat_opes_batch = torch.zeros(size=(self.batch_size, config.ope_feat_dim, self.num_opes))
        feat_mas_batch = torch.zeros(size=(self.batch_size, config.ma_feat_dim, num_mas))

        feat_opes_batch[:, 1, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=2)
        feat_opes_batch[:, 2, :] = torch.sum(self.proc_times_batch, dim=2).div(feat_opes_batch[:, 1, :] + 1e-9)
        feat_opes_batch[:, 3, :] = self.convert_feat_job_2_ope(self.nums_ope_batch, self.opes_appertain_batch)
        feat_opes_batch[:, 5, :] = torch.bmm(feat_opes_batch[:, 2, :].unsqueeze(1),
                                             self.cal_cumul_adj_batch).squeeze()
        end_time_batch = (feat_opes_batch[:, 5, :] +
                          feat_opes_batch[:, 2, :]).gather(1, self.end_ope_biases_batch)
        feat_opes_batch[:, 4, :] = self.convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch)
        feat_mas_batch[:, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch, dim=1)
        self.feat_opes_batch = feat_opes_batch
        self.feat_mas_batch = feat_mas_batch

        # Masks of current status, dynamic
        # shape: (batch_size, num_jobs), True for jobs in process
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False)
        # shape: (batch_size, num_jobs), True for completed jobs
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, num_jobs), dtype=torch.bool, fill_value=False)
        # shape: (batch_size, num_mas), True for machines in process
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, num_mas), dtype=torch.bool, fill_value=False)
        '''
        Partial Schedule (state) of jobs/operations, dynamic
            Status
            Allocated machines
            Start time
            End time
        '''
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))
        self.schedules_batch[:, :, 2] = feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = feat_opes_batch[:, 5, :] + feat_opes_batch[:, 2, :]
        '''
        Partial Schedule (state) of machines, dynamic
            idle
            available_time
            utilization_time
            id_ope
        '''
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]  # shape: (batch_size)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)  # shape: (batch_size)

        self.state = EnvState(batch_idxes=self.batch_idxes,
                              feat_opes_batch=self.feat_opes_batch, feat_mas_batch=self.feat_mas_batch,
                              proc_times_batch=self.proc_times_batch, ope_ma_adj_batch=self.ope_ma_adj_batch,
                              ope_pre_adj_batch=self.ope_pre_adj_batch, ope_sub_adj_batch=self.ope_sub_adj_batch,
                              mask_job_procing_batch=self.mask_job_procing_batch,
                              mask_job_finish_batch=self.mask_job_finish_batch,
                              mask_ma_procing_batch=self.mask_ma_procing_batch,
                              opes_appertain_batch=self.opes_appertain_batch,
                              ope_step_batch=self.ope_step_batch,
                              end_ope_biases_batch=self.end_ope_biases_batch,
                              time_batch=self.time, nums_opes_batch=self.nums_opes)

        # Save initial data for reset
        self.old_proc_times_batch = copy.deepcopy(self.proc_times_batch)
        self.old_ope_ma_adj_batch = copy.deepcopy(self.ope_ma_adj_batch)
        self.old_cal_cumul_adj_batch = copy.deepcopy(self.cal_cumul_adj_batch)
        self.old_feat_opes_batch = copy.deepcopy(self.feat_opes_batch)
        self.old_feat_mas_batch = copy.deepcopy(self.feat_mas_batch)
        self.old_state = copy.deepcopy(self.state)


    def nums_detec(self,lines):
        '''
        Count the number of jobs, machines and operations
        '''
        num_opes = 0
        for i in range(1, len(lines)):
            num_opes += int(lines[i].strip().split()[0]) if lines[i] != "\n" else 0
        line_split = lines[0].strip().split()
        num_jobs = int(line_split[0])
        num_mas = int(line_split[1])
        return num_jobs, num_mas, num_opes

    def load_fjs(self,lines, num_mas, num_opes):
        '''
        Load the local FJSP instance.
        '''
        flag = 0
        matrix_proc_time = torch.zeros(size=(num_opes, num_mas))
        matrix_pre_proc = torch.full(size=(num_opes, num_opes), dtype=torch.bool, fill_value=False)
        matrix_cal_cumul = torch.zeros(size=(num_opes, num_opes)).int()
        nums_ope = []  # A list of the number of operations for each job
        opes_appertain = np.array([])
        num_ope_biases = []  # The id of the first operation of each job
        # Parse data line by line
        for line in lines:
            # first line
            if flag == 0:
                flag += 1
            # last line
            elif line is "\n":
                break
            # other
            else:
                num_ope_bias = int(sum(nums_ope))  # The id of the first operation of this job
                num_ope_biases.append(num_ope_bias)
                # Detect information of this job and return the number of operations
                num_ope = self.edge_detec(line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul)
                nums_ope.append(num_ope)
                # nums_option = np.concatenate((nums_option, num_option))
                opes_appertain = np.concatenate((opes_appertain, np.ones(num_ope) * (flag - 1)))
                flag += 1
        matrix_ope_ma_adj = torch.where(matrix_proc_time > 0, 1, 0)
        # Fill zero if the operations are insufficient (for parallel computation)
        opes_appertain = np.concatenate((opes_appertain, np.zeros(num_opes - opes_appertain.size)))
        return matrix_proc_time, matrix_ope_ma_adj, matrix_pre_proc, matrix_pre_proc.t(), \
            torch.tensor(opes_appertain).int(), torch.tensor(num_ope_biases).int(), \
            torch.tensor(nums_ope).int(), matrix_cal_cumul

    def convert_feat_job_2_ope(self,feat_job_batch, opes_appertain_batch):
        '''
        Convert job features into operation features (such as dimension)
        '''
        return feat_job_batch.gather(1, opes_appertain_batch)

    def edge_detec(self,line, num_ope_bias, matrix_proc_time, matrix_pre_proc, matrix_cal_cumul):
        '''
        Detect information of a job
        '''
        line_split = line.split()
        flag = 0
        flag_time = 0
        flag_new_ope = 1
        idx_ope = -1
        num_ope = 0  # Store the number of operations of this job
        num_option = np.array([])  # Store the number of processable machines for each operation of this job
        mac = 0
        for i in line_split:
            x = int(i)
            # The first number indicates the number of operations of this job
            if flag == 0:
                num_ope = x
                flag += 1
            # new operation detected
            elif flag == flag_new_ope:
                idx_ope += 1
                flag_new_ope += x * 2 + 1
                num_option = np.append(num_option, x)
                if idx_ope != num_ope - 1:
                    matrix_pre_proc[idx_ope + num_ope_bias][idx_ope + num_ope_bias + 1] = True
                if idx_ope != 0:
                    vector = torch.zeros(matrix_cal_cumul.size(0))
                    vector[idx_ope + num_ope_bias - 1] = 1
                    matrix_cal_cumul[:, idx_ope + num_ope_bias] = matrix_cal_cumul[:,
                                                                  idx_ope + num_ope_bias - 1] + vector
                flag += 1
            # not proc_time (machine)
            elif flag_time == 0:
                mac = x - 1
                flag += 1
                flag_time = 1
            # proc_time
            else:
                matrix_proc_time[idx_ope + num_ope_bias][mac] = x
                flag += 1
                flag_time = 0
        return num_ope
    def step(self,actions=None,):
        '''
        Environment transition function
        '''
        opes = actions[0, :]
        mas = actions[1, :]
        jobs = actions[2, :]
        self.N += 1

        # Removed unselected O-M arcs of the scheduled operations
        remain_ope_ma_adj = torch.zeros(size=(self.batch_size, self.num_mas), dtype=torch.int64)
        remain_ope_ma_adj[self.batch_idxes, mas] = 1
        self.ope_ma_adj_batch[self.batch_idxes, opes] = remain_ope_ma_adj[self.batch_idxes, :]
        self.proc_times_batch *= self.ope_ma_adj_batch

        # Update for some O-M arcs are removed, such as 'Status', 'Number of neighboring machines' and 'Processing time'
        proc_times = self.proc_times_batch[self.batch_idxes, opes, mas]
        self.feat_opes_batch[self.batch_idxes, :3, opes] = torch.stack((torch.ones(self.batch_idxes.size(0), dtype=torch.float),
                                                                        torch.ones(self.batch_idxes.size(0), dtype=torch.float),
                                                                        proc_times), dim=1)
        last_opes = torch.where(opes - 1 < self.num_ope_biases_batch[self.batch_idxes, jobs], self.num_opes - 1, opes - 1)
        self.cal_cumul_adj_batch[self.batch_idxes, last_opes, :] = 0

        # Update 'Number of unscheduled operations in the job'
        start_ope = self.num_ope_biases_batch[self.batch_idxes, jobs]
        end_ope = self.end_ope_biases_batch[self.batch_idxes, jobs]
        for i in range(self.batch_idxes.size(0)):
            self.feat_opes_batch[self.batch_idxes[i], 3, start_ope[i]:end_ope[i]+1] -= 1

        # Update 'Start time' and 'Job completion time'
        self.feat_opes_batch[self.batch_idxes, 5, opes] = self.time[self.batch_idxes]
        is_scheduled = self.feat_opes_batch[self.batch_idxes, 0, :]
        mean_proc_time = self.feat_opes_batch[self.batch_idxes, 2, :]
        start_times = self.feat_opes_batch[self.batch_idxes, 5, :] * is_scheduled  # real start time of scheduled opes
        un_scheduled = 1 - is_scheduled  # unscheduled opes
        estimate_times = torch.bmm((start_times + mean_proc_time).unsqueeze(1),
                            self.cal_cumul_adj_batch[self.batch_idxes, :, :]).squeeze()\
                         * un_scheduled  # estimate start time of unscheduled opes
        self.feat_opes_batch[self.batch_idxes, 5, :] = start_times + estimate_times
        end_time_batch = (self.feat_opes_batch[self.batch_idxes, 5, :] +
                          self.feat_opes_batch[self.batch_idxes, 2, :]).gather(1, self.end_ope_biases_batch[self.batch_idxes, :])
        self.feat_opes_batch[self.batch_idxes, 4, :] = self.convert_feat_job_2_ope(end_time_batch, self.opes_appertain_batch[self.batch_idxes,:])

        # Update partial schedule (state)
        self.schedules_batch[self.batch_idxes, opes, :2] = torch.stack((torch.ones(self.batch_idxes.size(0)), mas), dim=1)
        self.schedules_batch[self.batch_idxes, :, 2] = self.feat_opes_batch[self.batch_idxes, 5, :]
        self.schedules_batch[self.batch_idxes, :, 3] = self.feat_opes_batch[self.batch_idxes, 5, :] + \
                                                       self.feat_opes_batch[self.batch_idxes, 2, :]
        self.machines_batch[self.batch_idxes, mas, 0] = torch.zeros(self.batch_idxes.size(0))
        self.machines_batch[self.batch_idxes, mas, 1] = self.time[self.batch_idxes] + proc_times
        self.machines_batch[self.batch_idxes, mas, 2] += proc_times
        self.machines_batch[self.batch_idxes, mas, 3] = jobs.float()

        # Update feature vectors of machines
        self.feat_mas_batch[self.batch_idxes, 0, :] = torch.count_nonzero(self.ope_ma_adj_batch[self.batch_idxes, :, :], dim=1).float()
        self.feat_mas_batch[self.batch_idxes, 1, mas] = self.time[self.batch_idxes] + proc_times
        utiliz = self.machines_batch[self.batch_idxes, :, 2]
        cur_time = self.time[self.batch_idxes, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[self.batch_idxes, None] + 1e-9)
        self.feat_mas_batch[self.batch_idxes, 2, :] = utiliz

        # Update other variable according to actions
        self.ope_step_batch[self.batch_idxes, jobs] += 1
        self.mask_job_procing_batch[self.batch_idxes, jobs] = True
        self.mask_ma_procing_batch[self.batch_idxes, mas] = True
        self.mask_job_finish_batch = torch.where(self.ope_step_batch==self.end_ope_biases_batch+1,
                                                 True, self.mask_job_finish_batch)
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        self.done = self.done_batch.all()

        max = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self.reward_batch = self.makespan_batch - max
        self.makespan_batch = max

        # Check if there are still O-M pairs to be processed, otherwise the environment transits to the next time
        flag_trans_2_next_time = self.if_no_eligible()
        while ~((~((flag_trans_2_next_time==0) & (~self.done_batch))).all()):
            self.next_time(flag_trans_2_next_time)
            flag_trans_2_next_time = self.if_no_eligible()

        # Update the vector for uncompleted instances
        mask_finish = (self.N+1) <= self.nums_opes
        if ~(mask_finish.all()):
            self.batch_idxes = torch.arange(self.batch_size)[mask_finish]

        # Update state of the environment
        self.state.update(self.batch_idxes, self.feat_opes_batch, self.feat_mas_batch, self.proc_times_batch,
            self.ope_ma_adj_batch, self.mask_job_procing_batch, self.mask_job_finish_batch, self.mask_ma_procing_batch,
                          self.ope_step_batch, self.time)
        return self.state, self.reward_batch, self.done_batch

    def if_no_eligible(self):
        '''
        Check if there are still O-M pairs to be processed
        '''
        ope_step_batch = torch.where(self.ope_step_batch > self.end_ope_biases_batch,
                                     self.end_ope_biases_batch, self.ope_step_batch)
        op_proc_time = self.proc_times_batch.gather(1, ope_step_batch.unsqueeze(-1).expand(-1, -1,
                                                                                        self.proc_times_batch.size(2)))
        ma_eligible = ~self.mask_ma_procing_batch.unsqueeze(1).expand_as(op_proc_time)
        job_eligible = ~(self.mask_job_procing_batch + self.mask_job_finish_batch)[:, :, None].expand_as(
            op_proc_time)
        flag_trans_2_next_time = torch.sum(torch.where(ma_eligible & job_eligible, op_proc_time.double(), 0.0).transpose(1, 2),
                                           dim=[1, 2])
        # shape: (batch_size)
        # An element value of 0 means that the corresponding instance has no eligible O-M pairs
        # in other words, the environment need to transit to the next time
        return flag_trans_2_next_time

    def next_time(self, flag_trans_2_next_time):
        '''
        Transit to the next time
        '''
        # need to transit
        flag_need_trans = (flag_trans_2_next_time==0) & (~self.done_batch)
        # available_time of machines
        a = self.machines_batch[:, :, 1]
        # remain available_time greater than current time
        b = torch.where(a > self.time[:, None], a, torch.max(self.feat_opes_batch[:, 4, :]) + 1.0)
        # Return the minimum value of available_time (the time to transit to)
        c = torch.min(b, dim=1)[0]
        # Detect the machines that completed (at above time)
        d = torch.where((a == c[:, None]) & (self.machines_batch[:, :, 0] == 0) & flag_need_trans[:, None], True, False)
        # The time for each batch to transit to or stay in
        e = torch.where(flag_need_trans, c, self.time)
        self.time = e

        # Update partial schedule (state), variables and feature vectors
        aa = self.machines_batch.transpose(1, 2)
        aa[d, 0] = 1
        self.machines_batch = aa.transpose(1, 2)

        utiliz = self.machines_batch[:, :, 2]
        cur_time = self.time[:, None].expand_as(utiliz)
        utiliz = torch.minimum(utiliz, cur_time)
        utiliz = utiliz.div(self.time[:, None] + 1e-5)
        self.feat_mas_batch[:, 2, :] = utiliz

        jobs = torch.where(d, self.machines_batch[:, :, 3].double(), -1.0).float()
        jobs_index = np.argwhere(jobs.cpu() >= 0).to(self.device)
        job_idxes = jobs[jobs_index[0], jobs_index[1]].long()
        batch_idxes = jobs_index[0]

        self.mask_job_procing_batch[batch_idxes, job_idxes] = False
        self.mask_ma_procing_batch[d] = False
        self.mask_job_finish_batch = torch.where(self.ope_step_batch == self.end_ope_biases_batch + 1,
                                                 True, self.mask_job_finish_batch)

    def reset(self):
        '''
        Reset the environment to its initial state
        '''
        self.proc_times_batch = copy.deepcopy(self.old_proc_times_batch)
        self.ope_ma_adj_batch = copy.deepcopy(self.old_ope_ma_adj_batch)
        self.cal_cumul_adj_batch = copy.deepcopy(self.old_cal_cumul_adj_batch)
        self.feat_opes_batch = copy.deepcopy(self.old_feat_opes_batch)
        self.feat_mas_batch = copy.deepcopy(self.old_feat_mas_batch)
        self.state = copy.deepcopy(self.old_state)

        self.batch_idxes = torch.arange(self.batch_size)
        self.time = torch.zeros(self.batch_size)
        self.N = torch.zeros(self.batch_size)
        self.ope_step_batch = copy.deepcopy(self.num_ope_biases_batch)
        self.mask_job_procing_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_job_finish_batch = torch.full(size=(self.batch_size, self.num_jobs), dtype=torch.bool, fill_value=False)
        self.mask_ma_procing_batch = torch.full(size=(self.batch_size, self.num_mas), dtype=torch.bool, fill_value=False)
        self.schedules_batch = torch.zeros(size=(self.batch_size, self.num_opes, 4))
        self.schedules_batch[:, :, 2] = self.feat_opes_batch[:, 5, :]
        self.schedules_batch[:, :, 3] = self.feat_opes_batch[:, 5, :] + self.feat_opes_batch[:, 2, :]
        self.machines_batch = torch.zeros(size=(self.batch_size, self.num_mas, 4))
        self.machines_batch[:, :, 0] = torch.ones(size=(self.batch_size, self.num_mas))

        self.makespan_batch = torch.max(self.feat_opes_batch[:, 4, :], dim=1)[0]
        self.done_batch = self.mask_job_finish_batch.all(dim=1)
        return self.state

    # def render(self, mode='human'):
    #     '''
    #     Deprecated in the final experiment
    #     '''
    #     if self.show_mode == 'draw':
    #         num_jobs = self.num_jobs
    #         num_mas = self.num_mas
    #         print(sys.argv[0])
    #         color = read_json("./futils/color_config")["gantt_color"]
    #         if len(color) < num_jobs:
    #             num_append_color = num_jobs - len(color)
    #             color += ['#' + ''.join([random.choice("0123456789ABCDEF") for _ in range(6)]) for c in
    #                       range(num_append_color)]
    #         write_json({"gantt_color": color}, "./futils/color_config")
    #         for batch_id in range(self.batch_size):
    #             schedules = self.schedules_batch[batch_id].to('cpu')
    #             fig = plt.figure(figsize=(10, 6))
    #             fig.canvas.set_window_title('Visual_gantt')
    #             axes = fig.add_axes([0.1, 0.1, 0.72, 0.8])
    #             y_ticks = []
    #             y_ticks_loc = []
    #             for i in range(num_mas):
    #                 y_ticks.append('Machine {0}'.format(i))
    #                 y_ticks_loc.insert(0, i + 1)
    #             labels = [''] * num_jobs
    #             for j in range(num_jobs):
    #                 labels[j] = "job {0}".format(j + 1)
    #             patches = [mpatches.Patch(color=color[k], label="{:s}".format(labels[k])) for k in range(self.num_jobs)]
    #             axes.cla()
    #             axes.set_title(u'FJSP Schedule')
    #             axes.grid(linestyle='-.', color='gray', alpha=0.2)
    #             axes.set_xlabel('Time')
    #             axes.set_ylabel('Machine')
    #             axes.set_yticks(y_ticks_loc, y_ticks)
    #             axes.legend(handles=patches, loc=2, bbox_to_anchor=(1.01, 1.0), fontsize=int(14 / pow(1, 0.3)))
    #             axes.set_ybound(1 - 1 / num_mas, num_mas + 1 / num_mas)
    #             for i in range(int(self.nums_opes[batch_id])):
    #                 id_ope = i
    #                 idx_job, idx_ope = self.get_idx(id_ope, batch_id)
    #                 id_machine = schedules[id_ope][1]
    #                 axes.barh(id_machine,
    #                          0.2,
    #                          left=schedules[id_ope][2],
    #                          color='#b2b2b2',
    #                          height=0.5)
    #                 axes.barh(id_machine,
    #                          schedules[id_ope][3] - schedules[id_ope][2] - 0.2,
    #                          left=schedules[id_ope][2]+0.2,
    #                          color=color[idx_job],
    #                          height=0.5)
    #             plt.show()
    #     return
    def get_idx(self, id_ope, batch_id):
        '''
        Get job and operation (relative) index based on instance index and operation (absolute) index
        '''
        idx_job = max([idx for (idx, val) in enumerate(self.num_ope_biases_batch[batch_id]) if id_ope >= val])
        idx_ope = id_ope - self.num_ope_biases_batch[batch_id][idx_job]
        return idx_job, idx_ope

    def validate_gantt(self):
        '''
        Verify whether the schedule is feasible
        '''
        ma_gantt_batch = [[[] for _ in range(self.num_mas)] for __ in range(self.batch_size)]
        for batch_id, schedules in enumerate(self.schedules_batch):
            for i in range(int(self.nums_opes[batch_id])):
                step = schedules[i]
                ma_gantt_batch[batch_id][int(step[1])].append([i, step[2].item(), step[3].item()])
        proc_time_batch = self.proc_times_batch

        # Check whether there are overlaps and correct processing times on the machine
        flag_proc_time = 0
        flag_ma_overlap = 0
        flag = 0
        for k in range(self.batch_size):
            ma_gantt = ma_gantt_batch[k]
            proc_time = proc_time_batch[k]
            for i in range(self.num_mas):
                ma_gantt[i].sort(key=lambda s: s[1])
                for j in range(len(ma_gantt[i])):
                    if (len(ma_gantt[i]) <= 1) or (j == len(ma_gantt[i])-1):
                        break
                    if ma_gantt[i][j][2]>ma_gantt[i][j+1][1]:
                        flag_ma_overlap += 1
                    if ma_gantt[i][j][2]-ma_gantt[i][j][1] != proc_time[ma_gantt[i][j][0]][i]:
                        flag_proc_time += 1
                    flag += 1

        # Check job order and overlap
        flag_ope_overlap = 0
        for k in range(self.batch_size):
            schedule = self.schedules_batch[k]
            nums_ope = self.nums_ope_batch[k]
            num_ope_biases = self.num_ope_biases_batch[k]
            for i in range(self.num_jobs):
                if int(nums_ope[i]) <= 1:
                    continue
                for j in range(int(nums_ope[i]) - 1):
                    step = schedule[num_ope_biases[i]+j]
                    step_next = schedule[num_ope_biases[i]+j+1]
                    if step[3] > step_next[2]:
                        flag_ope_overlap += 1

        # Check whether there are unscheduled operations
        flag_unscheduled = 0
        for batch_id, schedules in enumerate(self.schedules_batch):
            count = 0
            for i in range(schedules.size(0)):
                if schedules[i][0]==1:
                    count += 1
            add = 0 if (count == self.nums_opes[batch_id]) else 1
            flag_unscheduled += add

        if flag_ma_overlap + flag_ope_overlap + flag_proc_time + flag_unscheduled != 0:
            return False, self.schedules_batch
        else:
            return True, self.schedules_batch

    def close(self):
        pass
