import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import copy
import os
from collections import deque
import gym
import numpy as np
from gym.envs.registration import register
import random
import time
from FJSP_config import get_FJSPconfig
from Test.agent.FJSP.FJSP_GNN_agent import Memory
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from Test.agent.FJSP.FJSP_GNN_agent import GATedge, MLPsim
from Test.agent.FJSP.FJSP_GNN_agent import MLPCritic, MLPActor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class CaseGenerator:
    '''
    FJSP instance generator
    '''

    def __init__(self, job_init, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=None, path='../model_/FJSP/FJSP_GNN/',
                 flag_same_opes=True, flag_doc=False):
        if nums_ope is None:
            nums_ope = []
        self.flag_doc = flag_doc  # Whether save the instance to a file
        self.flag_same_opes = flag_same_opes
        self.nums_ope = nums_ope
        self.path = path  # Instance save path (relative path)
        self.job_init = job_init
        self.num_mas = num_mas

        self.mas_per_ope_min = 1  # The minimum number of machines that can process an operation
        self.mas_per_ope_max = num_mas
        self.opes_per_job_min = opes_per_job_min  # The minimum number of operations for a job
        self.opes_per_job_max = opes_per_job_max
        self.proctime_per_ope_min = 1  # Minimum average processing time
        self.proctime_per_ope_max = 20
        self.proctime_dev = 0.2

    def get_case(self, idx=0):
        '''
        Generate FJSP instance
        :param idx: The instance number
        '''
        self.num_jobs = self.job_init
        if not self.flag_same_opes:
            self.nums_ope = [random.randint(self.opes_per_job_min, self.opes_per_job_max) for _ in range(self.num_jobs)]
        self.num_opes = sum(self.nums_ope)
        self.nums_option = [random.randint(self.mas_per_ope_min, self.mas_per_ope_max) for _ in range(self.num_opes)]
        self.num_options = sum(self.nums_option)
        self.ope_ma = []
        for val in self.nums_option:
            self.ope_ma = self.ope_ma + sorted(random.sample(range(1, self.num_mas + 1), val))
        self.proc_time = []
        self.proc_times_mean = [random.randint(self.proctime_per_ope_min, self.proctime_per_ope_max) for _ in
                                range(self.num_opes)]
        for i in range(len(self.nums_option)):
            low_bound = max(self.proctime_per_ope_min, round(self.proc_times_mean[i] * (1 - self.proctime_dev)))
            high_bound = min(self.proctime_per_ope_max, round(self.proc_times_mean[i] * (1 + self.proctime_dev)))
            proc_time_ope = [random.randint(low_bound, high_bound) for _ in range(self.nums_option[i])]
            self.proc_time = self.proc_time + proc_time_ope
        self.num_ope_biass = [sum(self.nums_ope[0:i]) for i in range(self.num_jobs)]
        self.num_ma_biass = [sum(self.nums_option[0:i]) for i in range(self.num_opes)]
        line0 = '{0}\t{1}\t{2}\n'.format(self.num_jobs, self.num_mas, self.num_options / self.num_opes)
        lines = []
        lines_doc = []
        lines.append(line0)
        lines_doc.append('{0}\t{1}\t{2}'.format(self.num_jobs, self.num_mas, self.num_options / self.num_opes))
        for i in range(self.num_jobs):
            flag = 0
            flag_time = 0
            flag_new_ope = 1
            idx_ope = -1
            idx_ma = 0
            line = []
            option_max = sum(self.nums_option[self.num_ope_biass[i]:(self.num_ope_biass[i] + self.nums_ope[i])])
            idx_option = 0
            while True:
                if flag == 0:
                    line.append(self.nums_ope[i])
                    flag += 1
                elif flag == flag_new_ope:
                    idx_ope += 1
                    idx_ma = 0
                    flag_new_ope += self.nums_option[self.num_ope_biass[i] + idx_ope] * 2 + 1
                    line.append(self.nums_option[self.num_ope_biass[i] + idx_ope])
                    flag += 1
                elif flag_time == 0:
                    line.append(self.ope_ma[self.num_ma_biass[self.num_ope_biass[i] + idx_ope] + idx_ma])
                    flag += 1
                    flag_time = 1
                else:
                    line.append(self.proc_time[self.num_ma_biass[self.num_ope_biass[i] + idx_ope] + idx_ma])
                    flag += 1
                    flag_time = 0
                    idx_option += 1
                    idx_ma += 1
                if idx_option == option_max:
                    str_line = " ".join([str(val) for val in line])
                    lines.append(str_line + '\n')
                    lines_doc.append(str_line)
                    break
        lines.append('\n')
        if self.flag_doc:
            doc = open(self.path + '{0}j_{1}m_{2}.fjs'.format(self.num_jobs, self.num_mas, str.zfill(str(idx + 1), 3)),
                       'a')
            for i in range(len(lines_doc)):
                print(lines_doc[i], file=doc)
            doc.close()
        return lines, self.num_jobs, self.num_jobs


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


class HGNNScheduler(nn.Module):
    def __init__(self, config):
        super(HGNNScheduler, self).__init__()
        self.device = config.device
        self.in_size_ma = config.in_size_ma  # Dimension of the raw feature vectors of machine nodes
        self.out_size_ma = config.out_size_ma  # Dimension of the embedding of machine nodes
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
        self.A_coeff = config.A_coeff  # coefficient for policy loss
        self.vf_coeff = config.vf_coeff  # coefficient for value loss
        self.entropy_coeff = config.entropy_coeff  # coefficient for entropy term
        self.num_envs = num_envs  # Number of parallel instances
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


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def get_validate_env(config):
    '''
    Generate and return the validation environment from the validation set ()
    '''
    file_path = "./Train/FJSP_GNN_train/{}/".format(str(config.Pn_j)+'x'+str(config.Pn_m))
    valid_data_files = os.listdir(file_path)
    for i in range(len(valid_data_files)):
        valid_data_files[i] = file_path + valid_data_files[i]
        config.data_source = 'file'
    env = gym.make('fjsp-v0', data=valid_data_files, config=config)
    return env


def validate(env_paras, env, model_policy):
    '''
    Validate the policy during training, and the process is similar to test
    '''
    start = time.time()
    batch_size = env_paras["batch_size"]
    memory = Memory()
    print('There are {0} dev instances.'.format(batch_size))  # validation set is also called development set
    state = env.state
    done = False
    dones = env.done_batch
    while ~done:
        with torch.no_grad():
            actions = model_policy.act(state, memory, dones, flag_sample=False, flag_train=False)
        state, rewards, dones = env.step(actions)
        done = dones.all()
    gantt_result = env.validate_gantt()[0]
    if not gantt_result:
        print("Scheduling Error！！！！！！")
    makespan = copy.deepcopy(env.makespan_batch.mean())
    makespan_batch = copy.deepcopy(env.makespan_batch)
    env.reset()
    print('validating time: ', time.time() - start, '\n')
    return makespan, makespan_batch


def main(config):
    register(
        id='fjsp-v0',  # Environment name (including version number)
        entry_point='Test.agent.FJSP.FJSP_GNN_agent:FJSPEnv',
        # The location of the environment class, like 'foldername.filename:classname'
    )

    if device.type == 'cuda':
        torch.cuda.set_device(device)
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    else:
        torch.set_default_tensor_type('torch.FloatTensor')
    print("PyTorch device: ", device.type)
    torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None,
                           sci_mode=False)

    config.device = device
    # model_paras["device"] = device
    env_paras = {
        "Pn_j": config.Pn_j,
        "Pn_m": config.Pn_m,
        "batch_size": config.batch_size,
        "ope_feat_dim": config.ope_feat_dim,
        "ma_feat_dim": config.ma_feat_dim,
        "show_mode": config.show_mode,
        "valid_batch_size": config.valid_batch_size
    }
    env_valid_paras = copy.deepcopy(env_paras)
    env_valid_paras["batch_size"] = config.valid_batch_size
    config.actor_in_dim = config.out_size_ma * 2 + config.out_size_ope * 2
    config.critic_in_dim = config.out_size_ma + config.out_size_ope

    num_jobs = config.Pn_j
    num_mas = config.Pn_m
    opes_per_job_min = int(num_mas * 0.8)
    opes_per_job_max = int(num_mas * 1.2)

    memories = Memory()
    model = PPO(config, num_envs=config.batch_size)
    env_valid = get_validate_env(config)  # Create an environment for validation

    maxlen = 1  # Save the best model
    best_models = deque()
    makespan_best = float('inf')

    # Generate data files and fill in the header
    str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
    save_path = './Result/save/train_{0}'.format(str_time)
    os.makedirs(save_path)
    # Training curve storage path (average of validation set)
    # writer_ave = pd.ExcelWriter('{0}/training_ave_{1}.xlsx'.format(save_path, str_time))
    # # Training curve storage path (value of each validating instance)
    # writer_100 = pd.ExcelWriter('{0}/training_100_{1}.xlsx'.format(save_path, str_time))
    valid_results = []
    valid_results_100 = []
    # data_file = pd.DataFrame(np.arange(10, 1010, 10), columns=["iterations"])
    # data_file.to_excel(writer_ave, sheet_name='Sheet1', index=False)
    # # writer_ave.save()
    # writer_ave.close()
    # data_file = pd.DataFrame(np.arange(10, 1010, 10), columns=["iterations"])
    # data_file.to_excel(writer_100, sheet_name='Sheet1', index=False)
    # # writer_100.save()
    # writer_100.close()

    # Start training iteration
    start_time = time.time()
    env = None
    for i in range(1, config.max_iterations + 1):
        # Replace training instances every x iteration (x = 20 in paper)
        if (i - 1) % config.parallel_iter == 0:
            # \mathcal{B} instances use consistent operations to speed up training
            nums_ope = [random.randint(opes_per_job_min, opes_per_job_max) for _ in range(num_jobs)]
            case = CaseGenerator(num_jobs, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=nums_ope)
            # env = gym.make('fjsp-v0', case=case, env_paras=env_paras)
            config.data_source = 'case'
            env = gym.make('fjsp-v0', data=case,config=config)
            print('num_job: ', num_jobs, '\tnum_mas: ', num_mas, '\tnum_opes: ', sum(nums_ope))

        # Get state and completion signal
        state = env.state
        done = False
        dones = env.done_batch
        last_time = time.time()

        # Schedule in parallel
        while ~done:
            with torch.no_grad():
                actions = model.policy_old.act(state, memories, dones)
            state, rewards, dones = env.step(actions)
            done = dones.all()
            memories.rewards.append(rewards)
            memories.is_terminals.append(dones)
            # gpu_tracker.track()  # Used to monitor memory (of gpu)
        print("spend_time: ", time.time() - last_time)

        # Verify the solution
        gantt_result = env.validate_gantt()[0]
        if not gantt_result:
            print("Scheduling Error！！！！！！")
        # print("Scheduling Finish")
        env.reset()

        # if iter mod x = 0 then update the policy (x = 1 in paper)
        if i % config.update_timestep == 0:
            loss, reward = model.update(memories, config)
            print("reward: ", '%.3f' % reward, "; loss: ", '%.3f' % loss)
            memories.clear_memory()
            # if is_viz:
            #     viz.line(X=np.array([i]), Y=np.array([reward]),
            #              win='window{}'.format(0), update='append', opts=dict(title='reward of envs'))
            #     viz.line(X=np.array([i]), Y=np.array([loss]),
            #              win='window{}'.format(1), update='append', opts=dict(title='loss of envs'))  # deprecated

        # if iter mod x = 0 then validate the policy (x = 10 in paper)
        if i % config.save_timestep == 0:
            print('\nStart validating')
            # Record the average results and the results on each instance
            vali_result, vali_result_100 = validate(env_valid_paras, env_valid, model.policy_old)
            # valid_results.append(vali_result.item())
            # valid_results_100.append(vali_result_100)

            # Save the best model
            if vali_result < makespan_best:
                makespan_best = vali_result
                if len(best_models) == maxlen:
                    delete_file = best_models.popleft()
                    os.remove(delete_file)
                save_file = '{0}/save_best_{1}_{2}.pt'.format(save_path, num_jobs, num_mas)
                best_models.append(save_file)
                torch.save(model.policy.state_dict(), save_file)

            # if is_viz:
            #     viz.line(
            #         X=np.array([i]), Y=np.array([vali_result.item()]),
            #         win='window{}'.format(2), update='append', opts=dict(title='makespan of valid'))

    # Save the data of training curve to files
    # data = pd.DataFrame(np.array(valid_results).transpose(), columns=["res"])
    # data.to_excel(writer_ave, sheet_name='Sheet1', index=False, startcol=1)
    # # writer_ave.save()
    # writer_ave.close()
    # # column = [i_col for i_col in range(100)]
    # column = [i_col for i_col in range(config.batch_size)]
    # data = pd.DataFrame(np.array(torch.stack(valid_results_100, dim=0).to('cpu')), columns=column)
    # data.to_excel(writer_100, sheet_name='Sheet1', index=False, startcol=1)
    # # writer_100.save()
    # writer_100.close()

    print("total_time: ", time.time() - start_time)


if __name__ == '__main__':
    N = [10, 15, 20, 20]
    M = [5, 10, 5, 10]
    bach=[100,100,100,100]
    # N = [30, 40]
    # M = [10, 10]
    # bach = [100, 100]
    count=0
    # 循环prefer
    for j in range(len(N)):
        print('start---',  N[j + count], 'X', M[j + count])
        config = get_FJSPconfig()
        config.Pn_j = N[j]
        config.Pn_m = M[j]
        config.batch_size = bach[j]
        config.valid_batch_size = bach[j]
        main(config)

