import os

import torch
import time
import torch.nn as nn
import numpy as np
from Test.agent.JSP.L2D_agent import L2D_agent, eval_actions, greedy_select_action
import torch.nn.functional as F
from Test.agent.JSP.L2D_agent import L2D_agent
from Test.optimizer.JSP_optimizer.JSP_RL_algorithm.Basic_learning_algorithm import Basic_learning_algorithm
from copy import deepcopy
from LoadUtils import find_nearest_file

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device2 = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# device = torch.device(config.device)
# env=L2D_agent(config)
# N_JOBS_P = config.Pn_j
# N_MACHINES_P = config.Pn_m
# LOW = config.low
# HIGH = config.high
# SEED = config.seed
# N_JOBS_N = config.Pn_j
# N_MACHINES_N = config.Pn_m
def g_pool_cal(graph_pool_type, batch_size, n_nodes, device):
    # batch_size is the shape of batch
    # for graph pool sparse matrix
    if graph_pool_type == 'average':
        elem = torch.full(size=(batch_size[0]*n_nodes, 1),
                          fill_value=1 / n_nodes,
                          dtype=torch.float32,
                          device=device).view(-1)
    else:
        elem = torch.full(size=(batch_size[0] * n_nodes, 1),
                          fill_value=1,
                          dtype=torch.float32,
                          device=device).view(-1)
    idx_0 = torch.arange(start=0, end=batch_size[0],
                         device=device,
                         dtype=torch.long)
    # print(idx_0)
    idx_0 = idx_0.repeat(n_nodes, 1).t().reshape((batch_size[0]*n_nodes, 1)).squeeze()

    idx_1 = torch.arange(start=0, end=n_nodes*batch_size[0],
                         device=device,
                         dtype=torch.long)
    idx = torch.stack((idx_0, idx_1))
    # graph_pool = torch.sparse.FloatTensor(idx, elem,
    #                                       torch.Size([batch_size[0],
    #                                                   n_nodes*batch_size[0]])
    #                                       ).to(device)
    graph_pool = torch.sparse_coo_tensor(idx, elem,
                                          torch.Size([batch_size[0],
                                                      n_nodes * batch_size[0]])
                                          ).to(device)


    return graph_pool

def aggr_obs(obs_mb, n_node):
    # obs_mb is [m, n_nodes_each_state, fea_dim], m is number of nodes in batch
    idxs = obs_mb.coalesce().indices()
    vals = obs_mb.coalesce().values()
    new_idx_row = idxs[1] + idxs[0] * n_node
    new_idx_col = idxs[2] + idxs[0] * n_node
    idx_mb = torch.stack((new_idx_row, new_idx_col))
    # print(idx_mb)
    # print(obs_mb.shape[0])
    adj_batch = torch.sparse.FloatTensor(indices=idx_mb,
                                         values=vals,
                                         size=torch.Size([obs_mb.shape[0] * n_node,
                                                          obs_mb.shape[0] * n_node]),
                                         ).to(obs_mb.device)
    return adj_batch

def validate(config,vali_set, model):
    env = L2D_agent(config)
    N_JOBS = vali_set[0][0].shape[0]
    N_MACHINES = vali_set[0][0].shape[1]
    device = torch.device(config.device)
    g_pool_step = g_pool_cal(graph_pool_type=config.graph_pool_type,
                             batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
                             n_nodes=env.number_of_tasks,
                             device=device)
    make_spans = []
    # rollout using model
    for data in vali_set:
        adj, fea, candidate, mask = env.reset(data,config)
        rewards = - env.initQuality
        while True:
            fea_tensor = torch.from_numpy(np.copy(fea)).to(device)
            adj_tensor = torch.from_numpy(np.copy(adj)).to(device).to_sparse()
            candidate_tensor = torch.from_numpy(np.copy(candidate)).to(device)
            mask_tensor = torch.from_numpy(np.copy(mask)).to(device)
            with torch.no_grad():
                pi, _ = model(x=fea_tensor,
                              graph_pool=g_pool_step,
                              padded_nei=None,
                              adj=adj_tensor,
                              candidate=candidate_tensor.unsqueeze(0),
                              mask=mask_tensor.unsqueeze(0))
            # action = sample_select_action(pi, candidate)
            action = greedy_select_action(pi, candidate)
            adj, fea, reward, done, candidate, mask = env.transit(action.item())
            rewards += reward
            if done:
                break
        make_spans.append(rewards - env.posRewards)
        # print(rewards - env.posRewards)
    return np.array(make_spans)

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

class GraphCNN(nn.Module):
    def __init__(self,
                 num_layers,
                 num_mlp_layers,
                 input_dim,
                 hidden_dim,
                 # final_dropout,
                 learn_eps,
                 neighbor_pooling_type,
                 device):
        '''
        num_layers: number of layers in the neural networks (INCLUDING the input layer)
        num_mlp_layers: number of layers in mlps (EXCLUDING the input layer)
        input_dim: dimensionality of input features
        hidden_dim: dimensionality of hidden units at ALL layers
        output_dim: number of classes for prediction
        final_dropout: dropout ratio on the final linear layer
        learn_eps: If True, learn epsilon to distinguish center nodes from neighboring nodes. If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: how to aggregate neighbors (mean, average, or max)
        device: which device to use
        '''

        super(GraphCNN, self).__init__()

        # self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.neighbor_pooling_type = neighbor_pooling_type
        self.learn_eps = learn_eps
        # common out the eps if you do not need to use it, otherwise the it will cause
        # error "not in the computational graph"
        # if self.learn_eps:
        #     self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))

        # List of MLPs
        self.mlps = torch.nn.ModuleList()

        # List of batchnorms applied to the output of MLP (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers-1):
            if layer == 0:
                self.mlps.append(MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def next_layer_eps(self, h, layer, padded_neighbor_list = None, Adj_block = None):
        # pooling neighboring nodes and center nodes separately by epsilon reweighting.

        if self.neighbor_pooling_type == "max":
            # If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            pooled = torch.mm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.mm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree

        # Reweights the center node representation when aggregating it with its neighbors
        pooled = pooled + (1 + self.eps[layer])*h
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def next_layer(self, h, layer, padded_neighbor_list = None, Adj_block = None):

        # pooling neighboring nodes and center nodes altogether
        if self.neighbor_pooling_type == "max":
            # If max pooling
            pooled = self.maxpool(h, padded_neighbor_list)
        else:
            # If sum or average pooling
            # print(Adj_block.dtype)
            # print(h.dtype)
            pooled = torch.mm(Adj_block, h)
            if self.neighbor_pooling_type == "average":
                # If average pooling
                degree = torch.mm(Adj_block, torch.ones((Adj_block.shape[0], 1)).to(self.device))
                pooled = pooled/degree
        # representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](pooled)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h

    def forward(self,
                x,
                graph_pool,
                padded_nei,
                adj):

        x_concat = x
        graph_pool = graph_pool

        if self.neighbor_pooling_type == "max":
            padded_neighbor_list = padded_nei
        else:
            Adj_block = adj

        # list of hidden representation at each layer (including input)
        h = x_concat

        for layer in range(self.num_layers-1):
            if self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and self.learn_eps:
                h = self.next_layer_eps(h, layer, Adj_block=Adj_block)
            elif self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, padded_neighbor_list=padded_neighbor_list)
            elif not self.neighbor_pooling_type == "max" and not self.learn_eps:
                h = self.next_layer(h, layer, Adj_block=Adj_block)

        h_nodes = h.clone()
        # print(graph_pool.shape, h.shape)
        pooled_h = torch.sparse.mm(graph_pool, h)
        # pooled_h = graph_pool.spmm(h)

        return pooled_h, h_nodes

class ActorCritic(nn.Module):
    def __init__(self,
                 n_j,
                 n_m,
                 # feature extraction net unique attributes:
                 num_layers,
                 learn_eps,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 # feature extraction net MLP attributes:
                 num_mlp_layers_feature_extract,
                 # actor net MLP attributes:
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 # actor net MLP attributes:
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 # actor/critic/feature_extraction shared attribute
                 device
                 ):
        super(ActorCritic, self).__init__()
        # job size for problems, no business with network
        self.n_j = n_j
        # machine size for problems, no business with network
        self.n_m = n_m
        self.n_ops_perjob = n_m
        self.device = device

        self.feature_extract = GraphCNN(num_layers=num_layers,
                                        num_mlp_layers=num_mlp_layers_feature_extract,
                                        input_dim=input_dim,
                                        hidden_dim=hidden_dim,
                                        learn_eps=learn_eps,
                                        neighbor_pooling_type=neighbor_pooling_type,
                                        device=device).to(device)
        self.actor = MLPActor(num_mlp_layers_actor, hidden_dim*2, hidden_dim_actor, 1).to(device)
        self.critic = MLPCritic(num_mlp_layers_critic, hidden_dim, hidden_dim_critic, 1).to(device)

    def forward(self,
                x,
                graph_pool,
                padded_nei,
                adj,
                candidate,
                mask,
                ):

        h_pooled, h_nodes = self.feature_extract(x=x,
                                                 graph_pool=graph_pool,
                                                 padded_nei=padded_nei,
                                                 adj=adj)
        # prepare policy feature: concat omega feature with global feature
        dummy = candidate.unsqueeze(-1).expand(-1, self.n_j, h_nodes.size(-1))
        candidate_feature = torch.gather(h_nodes.reshape(dummy.size(0), -1, dummy.size(-1)), 1, dummy)
        h_pooled_repeated = h_pooled.unsqueeze(1).expand_as(candidate_feature)

        '''# prepare policy feature: concat row work remaining feature
        durfea2mat = x[:, 1].reshape(shape=(-1, self.n_j, self.n_m))
        mask_right_half = torch.zeros_like(durfea2mat)
        mask_right_half.put_(omega, torch.ones_like(omega, dtype=torch.float))
        mask_right_half = torch.cumsum(mask_right_half, dim=-1)
        # calculate work remaining and normalize it with job size
        wkr = (mask_right_half * durfea2mat).sum(dim=-1, keepdim=True)/self.n_ops_perjob'''

        # concatenate feature
        # concateFea = torch.cat((wkr, candidate_feature, h_pooled_repeated), dim=-1)
        concateFea = torch.cat((candidate_feature, h_pooled_repeated), dim=-1)
        candidate_scores = self.actor(concateFea)

        # perform mask
        mask_reshape = mask.reshape(candidate_scores.size())
        candidate_scores[mask_reshape] = float('-inf')

        pi = F.softmax(candidate_scores, dim=1)
        v = self.critic(h_pooled)
        return pi, v

class PPO:
    def __init__(self,
                 lr,
                 gamma,
                 k_epochs,
                 eps_clip,
                 n_j,
                 n_m,
                 num_layers,
                 neighbor_pooling_type,
                 input_dim,
                 hidden_dim,
                 num_mlp_layers_feature_extract,
                 num_mlp_layers_actor,
                 hidden_dim_actor,
                 num_mlp_layers_critic,
                 hidden_dim_critic,
                 decay_step_size,
                 decay_ratio,
                 config
                 ):
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.k_epochs = k_epochs
        self.config = config
        self.device = device2
        self.policy = ActorCritic(n_j=n_j,
                                  n_m=n_m,
                                  num_layers=num_layers,
                                  learn_eps=False,
                                  neighbor_pooling_type=neighbor_pooling_type,
                                  input_dim=input_dim,
                                  hidden_dim=hidden_dim,
                                  num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
                                  num_mlp_layers_actor=num_mlp_layers_actor,
                                  hidden_dim_actor=hidden_dim_actor,
                                  num_mlp_layers_critic=num_mlp_layers_critic,
                                  hidden_dim_critic=hidden_dim_critic,
                                  device=device2)
        self.policy_old = deepcopy(self.policy)

        '''self.policy.load_state_dict(
            torch.load(path='./{}.pth'.format(str(n_j) + '_' + str(n_m) + '_' + str(1) + '_' + str(99))))'''

        self.policy_old.load_state_dict(self.policy.state_dict())
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer,
                                                         step_size=decay_step_size,
                                                         gamma=decay_ratio)
        self.V_loss_2 = nn.MSELoss()

    # Define a method, update, which is used to update the policy network parameters with input parameters memories (a list of data structures that store information about the environment),
    # n_tasks (the number of tasks), and g_pool (the global graph pool)
    def update(self, memories, n_tasks, g_pool):
        # Getting the coefficients of the loss function from the configuration
        vloss_coef = self.config.vloss_coef  # Coefficient of loss of value function
        ploss_coef = self.config.ploss_coef  # Coefficient of loss of strategy
        entloss_coef = self.config.entloss_coef  # Coefficient of entropy loss

        # Initialize a list storing all environmental data
        rewards_all_env = []
        adj_mb_t_all_env = []
        fea_mb_t_all_env = []
        candidate_mb_t_all_env = []
        mask_mb_t_all_env = []
        a_mb_t_all_env = []
        old_logprobs_mb_t_all_env = []

        # Iterate through all environments, process and store data in each environment
        for i in range(len(memories)):
            # Calculate the cumulative rewards of discounts for each environment
            rewards = []
            discounted_reward = 0
            for reward, is_terminal in zip(reversed(memories[i].r_mb), reversed(memories[i].done_mb)):
                if is_terminal:
                    discounted_reward = 0
                discounted_reward = reward + (self.gamma * discounted_reward)
                rewards.insert(0, discounted_reward)

            # Converting rewards to PyTorch tensor with normalization
            rewards = torch.tensor(rewards, dtype=torch.float).to(device)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
            rewards_all_env.append(rewards)

            # Processing of observations, features, candidate nodes, masks and actions for each environment
            adj_mb_t_all_env.append(aggr_obs(torch.stack(memories[i].adj_mb).to(device), n_tasks))
            fea_mb_t = torch.stack(memories[i].fea_mb).to(device)
            fea_mb_t = fea_mb_t.reshape(-1, fea_mb_t.size(-1))  # Adjustment of feature dimensions
            fea_mb_t_all_env.append(fea_mb_t)
            candidate_mb_t_all_env.append(torch.stack(memories[i].candidate_mb).to(device).squeeze())
            mask_mb_t_all_env.append(torch.stack(memories[i].mask_mb).to(device).squeeze())
            a_mb_t_all_env.append(torch.stack(memories[i].a_mb).to(device).squeeze())
            old_logprobs_mb_t_all_env.append(torch.stack(memories[i].logprobs).to(device).squeeze().detach())

        # Compute the global graph pool mb_g_pool, which is the same for all environments
        mb_g_pool = g_pool_cal(g_pool, torch.stack(memories[0].adj_mb).to(device).shape, n_tasks, device)

        # Perform K rounds of optimization iterations on the strategy
        for _ in range(self.k_epochs):
            loss_sum = 0
            vloss_sum = 0

            # Traverse all environments for training
            for i in range(len(memories)):
                # Use the current policy network to compute the output probability distribution pis and state values vals
                pis, vals = self.policy(x=fea_mb_t_all_env[i],
                                        graph_pool=mb_g_pool,
                                        adj=adj_mb_t_all_env[i],
                                        candidate=candidate_mb_t_all_env[i],
                                        mask=mask_mb_t_all_env[i],
                                        padded_nei=None)

                # Calculating log probability and entropy loss
                logprobs, ent_loss = eval_actions(pis.squeeze(), a_mb_t_all_env[i])

                # Computational Advantage Ratio
                ratios = torch.exp(logprobs - old_logprobs_mb_t_all_env[i].detach())

                # Calculation of dominance value
                advantages = rewards_all_env[i] - vals.view(-1).detach()

                # Compute two different strategy loss terms surr1 and surr2
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # Calculate loss of value function
                v_loss = self.V_loss_2(vals.squeeze(), rewards_all_env[i])

                # Calculating Strategy Losses
                p_loss = - torch.min(surr1, surr2).mean()

                # Determining entropy loss
                ent_loss = - ent_loss.clone()

                # Total portfolio loss
                loss = vloss_coef * v_loss + ploss_coef * p_loss + entloss_coef * ent_loss

                # Updating cumulative losses
                loss_sum += loss
                vloss_sum += v_loss

            # Backpropagation computes the gradient
            self.optimizer.zero_grad()  # Clearing the gradient
            loss_sum.mean().backward()  # Calculate the gradient of the average loss

            # Updating Model Parameters
            self.optimizer.step()

        # Copying new policy weights to the old policy network
        self.policy_old.load_state_dict(self.policy.state_dict())

        # If the configuration calls for decaying the learning rate, perform the STEP operation of the learning rate scheduler
        if self.config.decayflag:
            self.scheduler.step()

        # Returns the average total loss and average value loss during this renewal process
        return loss_sum.mean().item(), vloss_sum.mean().item()


class L2D_optimizer(Basic_learning_algorithm):
    def __init__(self,config):
        super(L2D_optimizer,self).__init__(config)
        self.config = config

    # def update(self, data, config):
    #
    #     result = []
    #     # torch.cuda.synchronize()
    #     start_time = time.time()
    #     min_makespan = 1000000
    #     max_run_time=config.Pn_j*config.Pn_m
    #     for run in range(10000000000):
    #         run+=1
    #         env = L2D_agent(config)
    #         adj, fea, candidate, mask = env.reset(data,config)
    #         ep_reward = - env.max_endTime
    #         # delta_t = []
    #         # t5 = time.time()
    #         N_JOBS_P = config.Pn_j
    #         N_MACHINES_P = config.Pn_m
    #         N_JOBS_N = config.Pn_j
    #         N_MACHINES_N = config.Pn_m
    #         ppo = PPO(config.lr, config.gamma, config.k_epochs, config.eps_clip,
    #                   N_JOBS_P,
    #                   N_MACHINES_P,
    #                   config.num_layers,
    #                   config.neighbor_pooling_type,
    #                   config.input_dim,
    #                   config.hidden_dim,
    #                   config.num_mlp_layers_feature_extract,
    #                   config.num_mlp_layers_actor,
    #                   config.hidden_dim_actor,
    #                   config.num_mlp_layers_critic,
    #                   config.hidden_dim_critic,
    #                   config.decay_step_size,
    #                   config.decay_ratio,
    #                   config)
    #         # Get the path to the same level folder
    #         current_dir = os.path.dirname(os.path.abspath(__file__))
    #         folder_path = os.path.join(current_dir, 'SavedNetwork')
    #         path = folder_path + '/' + config.test_datas_type + '{}.pth'.format(str(N_JOBS_N) + 'x' + str(N_MACHINES_N))
    #
    #         ppo.policy.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    #         # ppo.policy.eval()
    #         g_pool_step = g_pool_cal(graph_pool_type=config.graph_pool_type,
    #                                  batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
    #                                  n_nodes=env.number_of_tasks,
    #                                  device=config.device)
    #         while True:
    #             # t3 = time.time()
    #             fea_tensor = torch.from_numpy(fea).to(config.device)
    #             adj_tensor = torch.from_numpy(adj).to(config.device)
    #             candidate_tensor = torch.from_numpy(candidate).to(config.device)
    #             mask_tensor = torch.from_numpy(mask).to(config.device)
    #             # t4 = time.time()
    #             # delta_t.append(t4 - t3)
    #
    #             with torch.no_grad():
    #                 pi, _ = ppo.policy(x=fea_tensor,
    #                                    graph_pool=g_pool_step,
    #                                    padded_nei=None,
    #                                    adj=adj_tensor,
    #                                    candidate=candidate_tensor.unsqueeze(0),
    #                                    mask=mask_tensor.unsqueeze(0))
    #                 # action = sample_select_action(pi, omega)
    #
    #                 action = greedy_select_action(pi, candidate)
    #
    #             adj, fea, reward, done, candidate, mask = env.transit(action)
    #             ep_reward += reward
    #
    #             if done:
    #                 break
    #         # t6 = time.time()
    #         # print(t6 - t5)
    #         makespan=-ep_reward + env.posRewards
    #         if(min_makespan>makespan):
    #             min_makespan=makespan
    #         print('makespan:', min_makespan)
    #         result.append(min_makespan)
    #         if time.time() - start_time > max_run_time:
    #             break
    #
    #     return min_makespan
    def update(self, data, config):
        result = []
        # torch.cuda.synchronize()
        t1 = time.time()
        env = L2D_agent(config)
        adj, fea, candidate, mask = env.reset(data, config)
        ep_reward = - env.max_endTime
        # delta_t = []
        # t5 = time.time()
        N_JOBS_P = config.Pn_j
        N_MACHINES_P = config.Pn_m
        N_JOBS_N = config.Pn_j
        N_MACHINES_N = config.Pn_m
        ppo = PPO(config.lr, config.gamma, config.k_epochs, config.eps_clip,
                  N_JOBS_P,
                  N_MACHINES_P,
                  config.num_layers,
                  config.neighbor_pooling_type,
                  config.input_dim,
                  config.hidden_dim,
                  config.num_mlp_layers_feature_extract,
                  config.num_mlp_layers_actor,
                  config.hidden_dim_actor,
                  config.num_mlp_layers_critic,
                  config.hidden_dim_critic,
                  config.decay_step_size,
                  config.decay_ratio,
                  config)
        # Get the path to the same level folder

        folder_path = 'Train/model_/JSP/L2D'
        # folder_path = os.path.join(current_dir, 'L2D')
        min_filename = find_nearest_file(folder_path, str(N_JOBS_N) + 'x' + str(N_MACHINES_N))
        path = folder_path + '/' + min_filename


        ppo.policy.load_state_dict(torch.load(path, map_location=torch.device(device)))
        # ppo.policy.eval()
        g_pool_step = g_pool_cal(graph_pool_type=config.graph_pool_type,
                                 batch_size=torch.Size([1, env.number_of_tasks, env.number_of_tasks]),
                                 n_nodes=env.number_of_tasks,
                                 device=device)
        while True:
            # t3 = time.time()
            fea_tensor = torch.from_numpy(fea).to(device)
            adj_tensor = torch.from_numpy(adj).to(device)
            candidate_tensor = torch.from_numpy(candidate).to(device)
            mask_tensor = torch.from_numpy(mask).to(device)
            # t4 = time.time()
            # delta_t.append(t4 - t3)

            with torch.no_grad():
                pi, _ = ppo.policy(x=fea_tensor,
                                   graph_pool=g_pool_step,
                                   padded_nei=None,
                                   adj=adj_tensor,
                                   candidate=candidate_tensor.unsqueeze(0),
                                   mask=mask_tensor.unsqueeze(0))
                # action = sample_select_action(pi, omega)

                action = greedy_select_action(pi, candidate)

            adj, fea, reward, done, candidate, mask = env.transit(action)
            ep_reward += reward

            if done:
                break
        # t6 = time.time()
        # print(t6 - t5)
        makespan = -ep_reward + env.posRewards

        result.append(-ep_reward + env.posRewards)

        # torch.cuda.synchronize()
        t2 = time.time()
        print(' makespan:', makespan,'time',t2 - t1)
        # file_writing_obj = open(
        #     './' + 'drltime_' + str(N_JOBS_N) + 'x' + str(N_MACHINES_N) + '_' + str(N_JOBS_P) + 'x' + str(
        #         N_MACHINES_P) + '.txt', 'w')
        # file_writing_obj.write(str((t2 - t1) / len(data)))

        # print(result)
        # print(np.array(result, dtype=np.single).mean())
        # SEED = config.seed
        # np.save('drlResult_' + str(N_JOBS_N) + 'x' + str(N_MACHINES_N) + '_' + str(N_JOBS_P) + 'x' + str(
        #     N_MACHINES_P) + '_Seed' + str(SEED), np.array(result, dtype=np.single))
        # print(np.array(result, dtype=np.single).mean())
        return makespan,t2 - t1

    def init_population(self,problem,data,config):
        pass

class Memory:
    def __init__(self):
        self.adj_mb = []
        self.fea_mb = []
        self.candidate_mb = []
        self.mask_mb = []
        self.a_mb = []
        self.r_mb = []
        self.done_mb = []
        self.logprobs = []

    def clear_memory(self):
        del self.adj_mb[:]
        del self.fea_mb[:]
        del self.candidate_mb[:]
        del self.mask_mb[:]
        del self.a_mb[:]
        del self.r_mb[:]
        del self.done_mb[:]
        del self.logprobs[:]




