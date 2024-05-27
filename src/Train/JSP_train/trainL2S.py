import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../../')
import time
import random
import numpy as np
import numpy.random
from Test.agent.JSP.L2S_agent import L2S_agent
from JSP_config import get_config
import torch
import torch.optim as optim
from Test.optimizer.JSP_optimizer.JSP_RL_algorithm.L2S_optimizer import BatchGraph,Actor
from Test.agent.JSP.L2D_agent import uni_instance_gen
from pathlib import Path
from LoadUtils import load_data

dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class RL2S4JSSP:
    def __init__(self,config):
        self.env_training = L2S_agent(config)
        self.env_validation = L2S_agent(config)
        self.eps = np.finfo(np.float32).eps.item()
        self.incumbent_validation_result = np.inf
        self.current_validation_result = np.inf
        # validation_data_path = Path(
        #     './problem/JSP/JSP_test_datas/rand{}x{}.npy'.format(config.Pn_j, config.Pn_m))
        # if validation_data_path.is_file():
        #     self.validation_data = np.load(
        #         './problem/JSP/JSP_test_datas/rand{}x{}.npy'.format(config.Pn_j, config.Pn_m))
        # else:
        #     print('No validation data for {}x{}, generating new one.'.format(config.Pn_j, config.Pn_m))
        #     self.validation_data = np.array(
        #         [uni_instance_gen(n_j=config.Pn_j, n_m=config.Pn_m, low=config.low, high=config.high) for _ in range(100)])
        #     np.save('./problem/JSP/JSP_test_datas/rand{}x{}.npy'.format(config.Pn_j, config.Pn_m),
        #             self.validation_data)
    def learn (self, rewards, log_probs, dones, optimizer):
        R = torch.zeros_like(rewards[0], dtype=torch.float, device=rewards[0].device)
        returns = []
        for r in rewards[::-1]:
            R = r + config.gamma * R
            returns.insert(0, R)
        returns = torch.cat(returns, dim=-1)
        dones = torch.cat(dones, dim=-1)
        log_probs = torch.cat(log_probs, dim=-1)

        losses = []
        for b in range(returns.shape[0]):
            masked_R = torch.masked_select(returns[b], ~dones[b])
            masked_R = (masked_R - masked_R.mean()) / (torch.std(masked_R, unbiased=False) + self.eps)
            masked_log_prob = torch.masked_select(log_probs[b], ~dones[b])
            loss = (- masked_log_prob * masked_R).sum()
            losses.append(loss)

        optimizer.zero_grad()
        mean_loss = torch.stack(losses).mean()
        mean_loss.backward()
        optimizer.step()

    def validation(self, policy, dev,data):
        # validating...
        validation_start = time.time()
        validation_batch_data = BatchGraph()
        states_val, feasible_actions_val, _ = self.env_validation.reset(data,config)
        while self.env_validation.itr < config.transit:
            validation_batch_data.wrapper(*states_val)
            actions_val, _ = policy(validation_batch_data, feasible_actions_val)
            states_val, _, feasible_actions_val, _ = self.env_validation.transit(actions_val)
        states_val, feasible_actions_val, actions_val, _ = None, None, None, None
        validation_batch_data.clean()
        validation_result1 = self.env_validation.incumbent_objs.mean().cpu().item()
        validation_result2 = self.env_validation.current_objs.mean().cpu().item()
        # saving model based on validation results
        if validation_result1 < self.incumbent_validation_result:
            print('Find better model w.r.t incumbent objs, saving model...')
            torch.save(policy.state_dict(),
                       './Train/model_/L2S_'  # saved model type
                       '{}{}x{}'  # training parameters
                       '.pth'
                       .format(config.test_datas_type,config.Pn_j, config.Pn_m))
            self.incumbent_validation_result = validation_result1
        if validation_result2 < self.current_validation_result:
            print('Find better model w.r.t final step objs, saving model...')
            torch.save(policy.state_dict(),
                       './Train/model_/last-step_'  # saved model type
                       '{}x{}'  # training parameters
                       '.pth'
                       .format(config.Pn_j, config.Pn_m))
            self.current_validation_result = validation_result2

        validation_end = time.time()
        # saved_data_path = './model_/incumbent_{}x{}.pth'.format(config.Pn_j, config.Pn_m)
        #
        # policy.load_state_dict(torch.load(saved_data_path, map_location=torch.device(dev)), strict=False)
        print('Incumbent objs and final step objs for validation are: {:.2f}  {:.2f}'.format(validation_result1,
                                                                                             validation_result2),
              'validation takes:{:.2f}'.format(validation_end - validation_start))

        return validation_result1, validation_result2

    def train(self, data):

        torch.manual_seed(1)
        random.seed(1)
        np.random.seed(1)

        policy = Actor(in_dim=3,
                       hidden_dim=config.hidden_dim,
                       embedding_l=config.embedding_layer,
                       policy_l=config.policy_layer,
                       embedding_type=config.embedding_type,
                       heads=config.heads,
                       dropout=config.drop_out).to(dev)

        optimizer = optim.Adam(policy.parameters(), lr=config.lr)

        batch_data = BatchGraph()
        log = []
        validation_log = []
        print()
        for batch_i in range(1, config.episodes // config.batch_size + 1):
            t1 = time.time()
            states, feasible_actions, dones = self.env_training.reset(data,config)

            rewards_buffer = []
            log_probs_buffer = []
            dones_buffer = [dones]

            while self.env_training.itr < config.transit:
                batch_data.wrapper(*states)
                actions, log_ps = policy(batch_data, feasible_actions)
                states, rewards, feasible_actions, dones = self.env_training.transit(actions)

                # store training data
                rewards_buffer.append(rewards)
                log_probs_buffer.append(log_ps)
                dones_buffer.append(dones)

                # logging reward...
                # reward_log.append(rewards)

                if self.env_training.itr % config.steps_learn == 0:
                    # training...
                    self.learn(rewards_buffer, log_probs_buffer, dones_buffer[:-1], optimizer)
                    # clean training data
                    rewards_buffer = []
                    log_probs_buffer = []
                    dones_buffer = [dones]

            # learn(rewards_buffer, log_probs_buffer, dones_buffer[:-1])  # old-school training scheme

            t2 = time.time()
            print('Batch {} training takes: {:.2f}'.format(batch_i, t2 - t1),
                  'Mean Performance: {:.2f}'.format(self.env_training.current_objs.cpu().mean().item()))
            log.append(self.env_training.current_objs.mean().cpu().item())

            # start validation and saving model & logs...
            if batch_i % config.step_validation == 0:
                # validating...
                validation_result1, validation_result2 = self.validation(policy, dev,data)
                validation_log.append([validation_result1, validation_result2])

                # # saving log
                # np.save('./log/training_log_'
                #         '{}x{}.npy'  # training parameters
                #         .format(config.Pn_j, config.Pn_m),
                #         np.array(log))
                # np.save('./log/validation_log_'
                #         '{}x{}.npy'  # training parameters
                #         .format(config.Pn_j, config.Pn_m),
                #         np.array(validation_log))


if __name__ == '__main__':
    N = [10, 15, 15, 20, 20]
    M = [10,10, 15, 10, 15]
    prefer = [('vali', 5)]
    count = 0
    # 循环prefer
    for i in range(len(prefer)):
        name, times = prefer[i]
        for j in range(times):
            print('start---', name, N[j + count], 'X', M[j + count])
            config = get_config()
            config.Pn_j = N[j + count]
            config.Pn_m = M[j + count]
            config.test_datas_type = name
            agent = RL2S4JSSP(config)
            data = load_data(config)
            for i in range(len(data)):
                agent.train(data[i])
        count += times
    print('结束111111111111111111111111111111111111111')
