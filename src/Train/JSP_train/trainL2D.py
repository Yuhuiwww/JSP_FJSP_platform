import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
import torch
import time
import numpy as np
from JSP_config import get_config
from Test.agent.JSP.L2D_agent import uni_instance_gen, L2D_agent, select_action
from Test.optimizer.JSP_optimizer.JSP_RL_algorithm.L2D_optimizer import g_pool_cal, Memory, PPO
from Test.optimizer.JSP_optimizer.JSP_RL_algorithm.L2D_optimizer import validate


class RL2D4JSSP:
    def __init__(self,config):
        self.config = config

    def train(self, dataLoaded):
        device = torch.device(config.device)
        envs = [L2D_agent(config) for _ in range(config.num_envs)]
        data_generator = uni_instance_gen
        vali_data = []
        for i in range(dataLoaded.shape[0]):
            vali_data.append((dataLoaded[i][0], dataLoaded[i][1]))

        torch.manual_seed(config.torch_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.torch_seed)
        np.random.seed(config.np_seed_train)

        memories = [Memory() for _ in range(config.num_envs)]

        ppo = PPO(config.lr, config.gamma, config.k_epochs, config.eps_clip,
                  n_j=config.Pn_j,
                  n_m=config.Pn_m,
                  num_layers=config.num_layers,
                  neighbor_pooling_type=config.neighbor_pooling_type,
                  input_dim=config.input_dim,
                  hidden_dim=config.hidden_dim,
                  num_mlp_layers_feature_extract=config.num_mlp_layers_feature_extract,
                  num_mlp_layers_actor=config.num_mlp_layers_actor,
                  hidden_dim_actor=config.hidden_dim_actor,
                  num_mlp_layers_critic=config.num_mlp_layers_critic,
                  hidden_dim_critic=config.hidden_dim_critic,
                  decay_step_size=config.decay_step_size,
                  decay_ratio=config.decay_ratio,
                  config=config)
        g_pool_step = g_pool_cal(graph_pool_type=config.graph_pool_type,
                                 batch_size=torch.Size([1, config.Pn_j * config.Pn_m, config.Pn_j * config.Pn_m]),
                                 n_nodes=config.Pn_j * config.Pn_m,
                                 device=device)
        # training loop
        log = []
        validation_log = []
        optimal_gaps = []
        optimal_gap = 1
        record = 100000
        for i_update in range(config.max_updates):

            t3 = time.time()

            ep_rewards = [0 for _ in range(config.num_envs)]
            adj_envs = []
            fea_envs = []
            candidate_envs = []
            mask_envs = []

            for i, env in enumerate(envs):
                adj, fea, candidate, mask = env.reset(
                    data_generator(n_j=config.Pn_j, n_m=config.Pn_m, low=config.low, high=config.high), config)
                adj_envs.append(adj)
                fea_envs.append(fea)
                candidate_envs.append(candidate)
                mask_envs.append(mask)
                ep_rewards[i] = - env.initQuality
            # rollout the env
            while True:
                fea_tensor_envs = [torch.from_numpy(np.copy(fea)).to(device) for fea in fea_envs]
                adj_tensor_envs = [torch.from_numpy(np.copy(adj)).to(device).to_sparse() for adj in adj_envs]
                candidate_tensor_envs = [torch.from_numpy(np.copy(candidate)).to(device) for candidate in
                                         candidate_envs]
                mask_tensor_envs = [torch.from_numpy(np.copy(mask)).to(device) for mask in mask_envs]

                with torch.no_grad():
                    action_envs = []
                    a_idx_envs = []
                    for i in range(config.num_envs):
                        pi, _ = ppo.policy_old(x=fea_tensor_envs[i],
                                               graph_pool=g_pool_step,
                                               padded_nei=None,
                                               adj=adj_tensor_envs[i],
                                               candidate=candidate_tensor_envs[i].unsqueeze(0),
                                               mask=mask_tensor_envs[i].unsqueeze(0))

                        action, a_idx = select_action(pi, candidate_envs[i], memories[i])
                        action_envs.append(action)
                        a_idx_envs.append(a_idx)

                adj_envs = []
                fea_envs = []
                candidate_envs = []
                mask_envs = []
                # Saving episode data
                for i in range(config.num_envs):
                    memories[i].adj_mb.append(adj_tensor_envs[i])
                    memories[i].fea_mb.append(fea_tensor_envs[i])
                    memories[i].candidate_mb.append(candidate_tensor_envs[i])
                    memories[i].mask_mb.append(mask_tensor_envs[i])
                    memories[i].a_mb.append(a_idx_envs[i])

                    adj, fea, reward, done, candidate, mask = envs[i].transit(action_envs[i].item())
                    adj_envs.append(adj)
                    fea_envs.append(fea)
                    candidate_envs.append(candidate)
                    mask_envs.append(mask)
                    ep_rewards[i] += reward
                    memories[i].r_mb.append(reward)
                    memories[i].done_mb.append(done)
                if envs[0].done():
                    break
            for j in range(config.num_envs):
                ep_rewards[j] -= envs[j].posRewards

            loss, v_loss = ppo.update(memories, config.Pn_j * config.Pn_m, config.graph_pool_type)
            for memory in memories:
                memory.clear_memory()
            mean_rewards_all_env = sum(ep_rewards) / len(ep_rewards)
            log.append([i_update, mean_rewards_all_env])
            if (i_update + 1) % 100 == 0:
                file_writing_obj = open(
                    './Result/' + 'log_' + str(config.Pn_j) + '_' + str(config.Pn_m) + '_' + str(
                        config.low) + '_' + str(
                        config.high) + '.txt', 'w')
                file_writing_obj.write(str(log))

            # log results
            print('Episode {}\t Last reward: {:.2f}\t Mean_Vloss: {:.8f}'.format(
                i_update + 1, mean_rewards_all_env, v_loss))

            # validate and save use mean performance
            t4 = time.time()
            if (i_update + 1) % 100 == 0:
                vali_result = - validate(config, vali_data, ppo.policy).mean()
                validation_log.append(vali_result)
                if vali_result < record:
                    torch.save(ppo.policy.state_dict(), './{}.pth'.format(
                        config.test_datas_type + str(config.Pn_j) + 'x' + str(config.Pn_m)))
                    record = vali_result
                print('The validation quality is:', vali_result)
                file_writing_obj1 = open(
                    './Result/' + 'vali_' + config.test_datas_type + str(config.Pn_j) + '_' + str(
                        config.Pn_m) + '_' + str(config.low) + '_' + str(
                        config.high) + '.txt', 'w')
                file_writing_obj1.write(str(validation_log))
            t5 = time.time()


if __name__ == '__main__':
    # N = [20, 20, 30, 30, 40, 40, 50, 50, 15, 20, 20, 30, 30, 50, 50, 100]
    # M = [15, 20, 15, 20, 15, 20, 15, 20, 15, 15, 20, 15, 20, 15, 20, 20]
    N = [6, 10, 10, 15, 15]
    M = [6, 5, 10, 5, 10]
    prefer = [('SL', 5)]
    count = 0
    # Loop prefer
    for i in range(len(prefer)):
        name, times = prefer[i]
        for j in range(times):
            print('start---', name, N[j + count], 'X', M[j + count])
            config = get_config()
            config.Pn_j = N[j + count]
            config.Pn_m = M[j + count]
            config.test_datas_type = name

            TEST_DATA = config.test_datas
            dataLoaded = np.load(
                str(TEST_DATA) + str(config.test_datas_type) + str(config.Pn_j) + 'x' + str(config.Pn_m) + '.npy')
            agent = RL2D4JSSP(config)
            agent.train(dataLoaded)
        count += times