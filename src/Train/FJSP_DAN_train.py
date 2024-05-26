import os,sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
from copy import deepcopy
import os
import random
import time
import sys
from FJSP_config import get_FJSPconfig
from LoadUtils import strToSuffix, sample_action, SD2_instance_generator
import numpy as np
from tqdm import tqdm
from Test.agent.FJSP.FJSP_DAN_agent import FJSP_DAN_agent, PPO_initialize, Memory, greedy_select_action, FJSPEnvForVariousOpNums
from Test.optimizer.FJSP_optimizer.FJSP__RL_algorithm.FJSP_DAN_optimizer import setup_seed
from LoadUtils import load_data_from_files, text_to_matrix
# os.environ["CUDA_VISIBLE_DEVICES"] = configs.device_id
import torch
device = 'cuda' if torch.cuda.is_available() else 'cpu'
str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))

class CaseGenerator:
    """

        the generator of SD1 data (imported from "songwenas12/fjsp-drl"),
        used for generating training instances

        Remark: the validation and testing intances of SD1 data are
        imported from "songwenas12/fjsp-drl"
    """

    def __init__(self, job_init, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=None, path='./  ',
                 flag_same_opes=True, flag_doc=False):
        # n_i
        self.str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
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
        """
        Generate FJSP instance
        :param idx: The instance number
        """
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
        lines_doc = []
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
                    lines_doc.append(str_line)
                    break
        job_length, op_pt = text_to_matrix(lines_doc)
        if self.flag_doc:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
            # doc = open(
            #     self.path + '/' + '{0}x{1}_{2}.fjs'.format(self.num_jobs, self.num_mas, str.zfill(str(idx + 1), 3)),
            #     'w')
            doc = open(self.path + f'/{self.str_time}.txt', 'a')
            # doc = open(self.path + f'/ours.txt', 'a')
            for i in range(len(lines_doc)):
                print(lines_doc[i], file=doc)
            doc.close()

        return job_length, op_pt, self.num_options / self.num_opes




class Trainer:
    def __init__(self, config):
        self.n_j = config.Pn_j
        self.n_m = config.Pn_m
        self.low = config.low
        self.high = config.high
        self.op_per_job_min = int(0.8 * self.n_m)
        self.op_per_job_max = int(1.2 * self.n_m)
        self.data_source = config.test_datas_type
        self.config = config
        self.max_updates = config.max_updates
        self.reset_env_timestep = config.reset_env_timestep
        self.validate_timestep = config.validate_timestep
        self.num_envs = config.FJSP_num_envs

        if not os.path.exists(f'./Train/model_/FJSP/FJSP_DAN/trained_network/'):
            os.makedirs(f'./Train/model_/FJSP/FJSP_DAN/trained_network/')#f'./Train/model_/FJSP/FJSP_DAN/trained_network/'
        if not os.path.exists(f'./Train/model_/FJSP/FJSP_DAN/train_log/'):
            os.makedirs(f'./Train/model_/FJSP/FJSP_DAN/train_log/{self.data_source}')

        if device == 'cuda':
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')

        if self.data_source == 'SD1':
            self.data_name = f'{self.n_j}x{self.n_m}'
        elif self.data_source == 'SD2':
            self.data_name = f'{self.n_j}x{self.n_m}{strToSuffix(config.data_suffix)}'
        else:
            self.data_name = f'{self.n_j}x{self.n_m}'

        # self.vali_data_path = f'./problem/FJSP_test_datas/data_train_vali/{self.data_source}/{self.data_name}'
        self.vali_data_path = f'./Train/FJSP_DAN_train/{self.data_source}/{self.data_name}'
        self.test_data_path = f'./Train/FJSP_DAN_train/{self.data_source}/{self.data_name}'
        self.model_name = f'{self.data_name}{strToSuffix(config.model_suffix)}'

        # seed
        self.seed_train = config.seed_train
        self.seed_test = config.seed_test
        setup_seed(self.seed_train)

        self.env = FJSP_DAN_agent(config,[])
        self.test_data = load_data_from_files(self.test_data_path)
        # validation data set
        vali_data = load_data_from_files(self.vali_data_path)

        if self.data_source == 'SD1':
            self.vali_env = FJSPEnvForVariousOpNums(config,[])
        elif self.data_source == 'SD2':
            self.vali_env = FJSP_DAN_agent(config,[])
        # else:
        #     self.vali_env = FJSP_DAN_agent(config, [])

        self.vali_env.set_initial_data(vali_data[0], vali_data[1])

        self.ppo = PPO_initialize(config)
        self.memory = Memory(gamma=config.gamma, gae_lambda=config.gae_lambda)

    def train(self):
        """
            train the model following the config
        """
        setup_seed(self.seed_train)
        self.log = []
        self.validation_log = []
        self.record = float('inf')

        # print the setting
        print("-" * 25 + "Training Setting" + "-" * 25)
        print(f"source : {self.data_source}")
        print(f"model name :{self.model_name}")
        print(f"test_data_path data :{self.test_data_path}")
        print("\n")

        self.train_st = time.time()

        for i_update in tqdm(range(self.max_updates), file=sys.stdout, desc="progress", colour='blue'):
            ep_st = time.time()

            # resampling the training data
            if i_update % self.reset_env_timestep == 0:
                dataset_job_length, dataset_op_pt = self.sample_training_instances()
                state = self.env.set_initial_data(dataset_job_length, dataset_op_pt)
            else:
                state = self.env.reset()

            ep_rewards = - deepcopy(self.env.init_quality)

            while True:

                # state store
                self.memory.push(state)
                with torch.no_grad():

                    pi_envs, vals_envs = self.ppo.policy_old(fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
                                                             op_mask=state.op_mask_tensor,  # [sz_b, N, N]
                                                             candidate=state.candidate_tensor,  # [sz_b, J]
                                                             fea_m=state.fea_m_tensor,  # [sz_b, M, 6]
                                                             mch_mask=state.mch_mask_tensor,  # [sz_b, M, M]
                                                             comp_idx=state.comp_idx_tensor,  # [sz_b, M, M, J]
                                                             dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                                                             # [sz_b, J, M]
                                                             fea_pairs=state.fea_pairs_tensor)  # [sz_b, J, M]

                # sample the action
                action_envs, action_logprob_envs = sample_action(pi_envs)

                # state transition
                state, reward, done = self.env.transit(actions=action_envs.cpu().numpy())
                ep_rewards += reward
                reward = torch.from_numpy(reward).to(device)

                # collect the transition
                self.memory.done_seq.append(torch.from_numpy(done).to(device))
                self.memory.reward_seq.append(reward)
                self.memory.action_seq.append(action_envs)
                self.memory.log_probs.append(action_logprob_envs)
                self.memory.val_seq.append(vals_envs.squeeze(1))

                if done.all():
                    break

            loss, v_loss = self.ppo.update(self.memory)
            self.memory.clear_memory()

            mean_rewards_all_env = np.mean(ep_rewards)
            mean_makespan_all_env = np.mean(self.env.current_makespan)

            # save the mean rewards of all instances in current training data
            self.log.append([i_update, mean_rewards_all_env])

            # validate the trained model
            if (i_update + 1) % self.validate_timestep == 0:
                if self.data_source == "SD1":
                    vali_result = self.validate_envs_with_various_op_nums().mean()
                elif self.data_source == "SD2":
                    vali_result = self.validate_envs_with_same_op_nums().mean()
                # else:
                #     vali_result = self.validate_envs_with_various_op_nums().mean()

                if vali_result < self.record:
                    self.save_model()
                    self.record = vali_result

                self.validation_log.append(vali_result)
                self.save_validation_log()
                tqdm.write(f'The validation quality is: {vali_result} (best : {self.record})')

            ep_et = time.time()

            # print the reward, makespan, loss and training time of the current episode
            tqdm.write(
                'Episode {}\t reward: {:.2f}\t makespan: {:.2f}\t Mean_loss: {:.8f},  training time: {:.2f}'.format(
                    i_update + 1, mean_rewards_all_env, mean_makespan_all_env, loss, ep_et - ep_st))

        self.train_et = time.time()

        # log results
        self.save_training_log()

    def save_training_log(self):
        """
            save reward data & validation makespan data (during training) and the entire training time
        """
        file_writing_obj = open(f'./Train/model_/FJSP/FJSP_DAN/train_log/{self.data_source}/' + 'reward_' + self.model_name + '.txt', 'w')
        file_writing_obj.write(str(self.log))

        file_writing_obj1 = open(f'./Train/model_/FJSP/FJSP_DAN/train_log/{self.data_source}/' + 'valiquality_' + self.model_name + '.txt', 'w')
        file_writing_obj1.write(str(self.validation_log))

        file_writing_obj3 = open(f'./Train/model_/FJSP/FJSP_DAN/train_time.txt', 'a')
        file_writing_obj3.write(
            f'model path: ./Train/model_/FJSP/FJSP_DAN/trained_network/{self.data_source}/{self.model_name}\t\ttraining time: '
            f'{round((self.train_et - self.train_st), 2)}\t\t local time: {str_time}\n')

    def save_validation_log(self):
        """
            save the results of validation
        """
        file_writing_obj1 = open(f'./Train/model_/FJSP/FJSP_DAN/train_log/{self.data_source}/' + 'valiquality_' + self.model_name + '.txt', 'w')
        file_writing_obj1.write(str(self.validation_log))

    def sample_training_instances(self):
        """
            sample training instances following the config,
            the sampling process of SD1 data is imported from "songwenas12/fjsp-drl"
        :return: new training instances
        """
        prepare_JobLength = [random.randint(self.op_per_job_min, self.op_per_job_max) for _ in range(self.n_j)]
        dataset_JobLength = []
        dataset_OpPT = []
        for i in range(self.num_envs):
            if self.data_source == 'SD1':
                case = CaseGenerator(self.n_j, self.n_m, self.op_per_job_min, self.op_per_job_max,
                                     nums_ope=prepare_JobLength, path='./Train/model_/FJSP/FJSP_GNN/fjsp_test', flag_doc=False)
                JobLength, OpPT, _ = case.get_case(i)

            else:
                JobLength, OpPT, _ = SD2_instance_generator(config=self.config)
            # else:
            #     case = CaseGenerator(self.n_j, self.n_m, self.op_per_job_min, self.op_per_job_max,
            #                          nums_ope=prepare_JobLength, path='./fjsp_test', flag_doc=False)
            #     JobLength, OpPT, _ = case.get_case(i)
            dataset_JobLength.append(JobLength)
            dataset_OpPT.append(OpPT)

        return dataset_JobLength, dataset_OpPT

    def validate_envs_with_same_op_nums(self):
        """
            validate the policy using the greedy strategy
            where the validation instances have the same number of operations
        :return: the makespan of the validation set
        """
        self.ppo.policy.eval()
        state = self.vali_env.reset()

        while True:

            with torch.no_grad():
                pi, _ = self.ppo.policy(fea_j=state.fea_j_tensor,  # [sz_b, N, 8]
                                        op_mask=state.op_mask_tensor,
                                        candidate=state.candidate_tensor,  # [sz_b, J]
                                        fea_m=state.fea_m_tensor,  # [sz_b, M, 6]
                                        mch_mask=state.mch_mask_tensor,  # [sz_b, M, M]
                                        comp_idx=state.comp_idx_tensor,  # [sz_b, M, M, J]
                                        dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [sz_b, J, M]
                                        fea_pairs=state.fea_pairs_tensor)  # [sz_b, J, M]

            action = greedy_select_action(pi)
            state, _, done = self.vali_env.transit(action.cpu().numpy())

            if done.all():
                break

        self.ppo.policy.train()
        return self.vali_env.current_makespan

    def validate_envs_with_various_op_nums(self):
        """
            validate the policy using the greedy strategy
            where the validation instances have various number of operations
        :return: the makespan of the validation set
        """
        self.ppo.policy.eval()
        state = self.vali_env.reset()

        while True:

            with torch.no_grad():
                batch_idx = ~torch.from_numpy(self.vali_env.done_flag)
                pi, _ = self.ppo.policy(fea_j=state.fea_j_tensor[batch_idx],  # [sz_b, N, 8]
                                        op_mask=state.op_mask_tensor[batch_idx],
                                        candidate=state.candidate_tensor[batch_idx],  # [sz_b, J]
                                        fea_m=state.fea_m_tensor[batch_idx],  # [sz_b, M, 6]
                                        mch_mask=state.mch_mask_tensor[batch_idx],  # [sz_b, M, M]
                                        comp_idx=state.comp_idx_tensor[batch_idx],  # [sz_b, M, M, J]
                                        dynamic_pair_mask=state.dynamic_pair_mask_tensor[batch_idx],  # [sz_b, J, M]
                                        fea_pairs=state.fea_pairs_tensor[batch_idx])  # [sz_b, J, M]

            action = greedy_select_action(pi)
            state, _, done = self.vali_env.transit(action.cpu().numpy())

            if done.all():
                break

        self.ppo.policy.train()
        return self.vali_env.current_makespan

    def save_model(self):
        """
            save the model
        """
        torch.save(self.ppo.policy.state_dict(), f'./Result/trained_network_FJSPDAN/{self.model_name}.pth')

    def load_model(self):
        """
            load the trained model
        """
        model_path = f'./Train/model_/FJSP/FJSP_DAN/trained_network/{self.data_source}/{self.model_name}.pth'
        self.ppo.policy.load_state_dict(torch.load(model_path, map_location=device))


def main():
    N=[10,15,20,20]
    M=[5,10,5,10]
    # bach=[115,2,16,1,15,2,115,15,116,117,1,115,100]
    count = 0
    for j in range(len(N)):
        print('start---', N[j + count], 'X', M[j + count])
        config = get_FJSPconfig()
        config.Pn_j = N[j]
        config.Pn_m = M[j]
        trainer = Trainer(config)
        trainer.train()


if __name__ == '__main__':
    main()