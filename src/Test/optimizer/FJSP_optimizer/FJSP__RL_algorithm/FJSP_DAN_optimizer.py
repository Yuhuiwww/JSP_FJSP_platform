import os
import random
import re
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
from Test.agent.FJSP.FJSP_DAN_agent import PPO, FJSP_DAN_agent, PPO_initialize
from Test.optimizer.FJSP_optimizer.FJSP__RL_algorithm.Basic_FJSP_optimizer import Bassic_FJSP_optimizer
from LoadUtils import strToSuffix, sample_action, SD2_instance_generator
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def setup_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True




class FJSP_DAN_optimizer(Bassic_FJSP_optimizer):
    def __init__(self,config):
        super(FJSP_DAN_optimizer,self).__init__(config)
        self.config = config

    def text_to_matrix(text):
        """
                Convert text form of the data into matrix form
        :param text: the standard text form of the instance
        :return:  the matrix form of the instance
                job_length: the number of operations in each job (shape [J])
                op_pt: the processing time matrix with shape [N, M],
                    where op_pt[i,j] is the processing time of the ith operation
                    on the jth machine or 0 if $O_i$ can not process on $M_j$
        """
        n_j = int(re.findall(r'\d+\.?\d*', text[0])[0])
        n_m = int(re.findall(r'\d+\.?\d*', text[0])[1])

        job_length = np.zeros(n_j, dtype='int32')
        op_pt = []

        for i in range(n_j):
            content = np.array([int(s) for s in re.findall(r'\d+\.?\d*', text[i + 1])])
            job_length[i] = content[0]

            idx = 1
            for j in range(content[0]):
                op_pt_row = np.zeros(n_m, dtype='int32')
                mch_num = content[idx]
                next_idx = idx + 2 * mch_num + 1
                for k in range(mch_num):
                    mch_idx = content[idx + 2 * k + 1]
                    pt = content[idx + 2 * k + 2]
                    op_pt_row[mch_idx - 1] = pt

                idx = next_idx
                op_pt.append(op_pt_row)

        op_pt = np.array(op_pt)

        return job_length, op_pt
    def load_data_from_files(self,directory):
        """
            load all files within the specified directory
        :param directory: the directory of files
        :return: a list of data (matrix form) in the directory
        """
        if not os.path.exists(directory):
            return [], []

        dataset_job_length = []
        dataset_op_pt = []
        for root, dirs, files in os.walk(directory):
            # sort files by index
            files.sort(key=lambda s: int(re.findall("\d+", s)[0]))
            files.sort(key=lambda s: int(re.findall("\d+", s)[-1]))
            for f in files:
                # print(f)
                g = open(os.path.join(root, f), 'r').readlines()
                job_length, op_pt = self.text_to_matrix(g)
                dataset_job_length.append(job_length)
                dataset_op_pt.append(op_pt)
        return dataset_job_length, dataset_op_pt
    # def pack_data_from_config(self,data_source, test_data):
    #     """
    #         load multiple data (specified by the variable 'test_data')
    #         of the specified data source.
    #     :param data_source: the source of data (SD1/SD2/BenchData)
    #     :param test_data: the list of data's name
    #     :return: a list of data (matrix form) and its name
    #     """
    #     data_list = []
    #     for data_name in test_data:
    #         data_path = f'./data/{data_source}/{data_name}'
    #         data_list.append((self.load_data_from_files(data_path), data_name))
    #     return data_list



    def matrix_to_text(self,job_length, op_pt, op_per_mch):
        """
            Convert matrix form of the data into test form
        :param job_length: the number of operations in each job (shape [J])
        :param op_pt: the processing time matrix with shape [N, M],
                    where op_pt[i,j] is the processing time of the ith operation
                    on the jth machine or 0 if $O_i$ can not process on $M_j$
        :param op_per_mch: the average number of compatible machines of each operation
        :return: the standard text form of the instance
        """
        n_j = job_length.shape[0]
        n_op, n_m = op_pt.shape
        text = [f'{n_j}\t{n_m}\t{op_per_mch}']

        op_idx = 0
        for j in range(n_j):
            line = f'{job_length[j]}'
            for _ in range(job_length[j]):
                use_mch = np.where(op_pt[op_idx] != 0)[0]
                line = line + ' ' + str(use_mch.shape[0])
                for k in use_mch:
                    line = line + ' ' + str(k + 1) + ' ' + str(op_pt[op_idx][k])
                op_idx += 1

            text.append(line)

        return text

    def generate_data_to_files(self,seed, directory, config):
        """
            Generate data and save it to the specified directory
        :param seed: seed for data generation
        :param directory: the directory for saving files
        :param config: other parameters related to data generation
        """
        n_j = config.Pn_j
        n_m = config.Pn_m
        source = config.test_datas_type
        batch_size = config.data_size
        data_suffix = config.data_suffix

        suffix = strToSuffix(data_suffix)
        low = config.low
        high = config.high

        filename = '{}x{}{}'.format(n_j, n_m, suffix)
        np.random.seed(seed)
        random.seed(seed)

        print("-" * 25 + "Data Setting" + "-" * 25)
        print(f"seed : {seed}")
        print(f"data size : {batch_size}")
        print(f"data source: {source}")
        print(f"filename : {filename}")
        print(f"processing time : [{low},{high}]")
        print(f"mode : {data_suffix}")
        print("-" * 50)

        path = directory + filename

        if (not os.path.exists(path)) or config.cover_data_flag:
            if not os.path.exists(path):
                os.makedirs(path)

            for idx in range(batch_size):
                if source == 'SD2':
                    job_length, op_pt, op_per_mch = SD2_instance_generator(config=config)

                    lines_doc = self.matrix_to_text(job_length, op_pt, op_per_mch)

                    doc = open(
                        path + '/' + filename + '_{}.fjs'.format(str.zfill(str(idx + 1), 3)),
                        'w')
                    for i in range(len(lines_doc)):
                        print(lines_doc[i], file=doc)
                    doc.close()
        else:
            print("the data already exists...")

    def greedy_select_action(self,p):
        _, index = torch.max(p, dim=1)
        return index
    def sampling_strategy(self,data_set, model_path, sample_times, seed):
        """
        test the model on the given data using the sampling strategy
        :param data_set: test data
        :param model_path: the path of the model file
        :param seed: the seed for testing
        :return: the test results including the makespan and time
        """
        ppo = PPO_initialize(self.config)
        setup_seed(seed)
        test_result_list = []
        ppo.policy.load_state_dict(torch.load(model_path, map_location=device))
        ppo.policy.eval()

        n_j = data_set[0][0].shape[0]
        n_op, n_m = data_set[1][0].shape
        env = FJSP_DAN_agent(self.config, data_set)
        makespan_list = [0] * len(data_set[0])
        for i in tqdm(range(len(data_set[0])), file=sys.stdout, desc="progress", colour='blue'):
            # copy the testing environment
            JobLength_dataset = np.tile(np.expand_dims(data_set[0][i], axis=0), (sample_times, 1))
            OpPT_dataset = np.tile(np.expand_dims(data_set[1][i], axis=0), (sample_times, 1, 1))

            state = env.set_initial_data(JobLength_dataset, OpPT_dataset)
            t1 = time.time()
            while True:

                with torch.no_grad():
                    pi, _ = ppo.policy(fea_j=state.fea_j_tensor,  # [100, N, 8]
                                       op_mask=state.op_mask_tensor,  # [100, N, N]
                                       candidate=state.candidate_tensor,  # [100, J]
                                       fea_m=state.fea_m_tensor,  # [100, M, 6]
                                       mch_mask=state.mch_mask_tensor,  # [100, M, M]
                                       comp_idx=state.comp_idx_tensor,  # [100, M, M, J]
                                       dynamic_pair_mask=state.dynamic_pair_mask_tensor,  # [100, J, M]
                                       fea_pairs=state.fea_pairs_tensor)  # [100, J, M]

                action_envs, _ = sample_action(pi)
                state, _, done = env.transit(action_envs.cpu().numpy())
                if done.all():
                    break

            t2 = time.time()
            best_makespan = np.min(env.current_makespan)
            test_result_list.append([best_makespan, t2 - t1])
            makespan_list[i] =best_makespan

        return np.array(test_result_list) ,makespan_list

    def greedy_strategy(self,data_set, model_path, seed,File_GAN):
        """
            test the model on the given data using the greedy strategy
        :param data_set: test data
        :param model_path: the path of the model file
        :param seed: the seed for testing
        :return: the test results including the makespan and time
        """
        ppo = PPO_initialize(self.config)
        test_result_list = []

        setup_seed(seed)
        ppo.policy.load_state_dict(torch.load(model_path, map_location=device))
        ppo.policy.eval()

        # n_j = data_set[0][0].shape[0]
        # n_op, n_m = data_set[1][0].shape
        env = FJSP_DAN_agent(self.config, data_set)
        makespan_list=[0]*len(data_set[0])
        for i in tqdm(range(len(data_set[0])), file=sys.stdout, desc="progress", colour='blue'):

            state = env.set_initial_data([data_set[0][i]], [data_set[1][i]])
            t1 = time.time()
            while True:

                with torch.no_grad():
                    pi, _ = ppo.policy(fea_j=state.fea_j_tensor,  # [1, N, 8]
                                       op_mask=state.op_mask_tensor,  # [1, N, N]
                                       candidate=state.candidate_tensor,  # [1, J]
                                       fea_m=state.fea_m_tensor,  # [1, M, 6]
                                       mch_mask=state.mch_mask_tensor,  # [1, M, M]
                                       comp_idx=state.comp_idx_tensor,  # [1, M, M, J]
                                       dynamic_pair_mask=state.dynamic_pair_mask_tensor,
                                       fea_pairs=state.fea_pairs_tensor)  # [1, J, M]

                action = self.greedy_select_action(pi)
                state, reward, done = env.transit(actions=action.cpu().numpy())
                if done:
                    break
            t2 = time.time()

            test_result_list.append([env.current_makespan[0], t2 - t1])

            print('makespan:',File_GAN[i], env.current_makespan[0])
            makespan_list[i]=env.current_makespan[0]

        return np.array(test_result_list),makespan_list
    def update(self, test_data, config):
        setup_seed(config.seed_test)
        if not os.path.exists(f'./Result/FJSP'):
            os.makedirs('./Result/FJSP')

        # collect the path of test models

        model = [['10x5'], ['15x10'], ['20x5'], ['20x10']]
        with open('./Result/FJSP'+self.config.optimizer + 'Sample--solution-------.txt', 'a') as f:
            for model_num in model:
                f.write(self.config.test_datas_type + str(self.config.Pn_j) + 'x' + str(self.config.Pn_m) + '\n')
                start_time = time.time()
                config.test_model=model_num
                for model_name in config.test_model:
                    test_model = []
                    test_model.append((f'./Train/model_/FJSP/FJSP_DAN/{model_name}.pth', model_name))

                # collect the test data
                # test_data = self.pack_data_from_config(config.data_source, config.test_data)

                if config.flag_sample:
                    model_prefix = "DANIELS"
                else:
                    model_prefix = "DANIELG"

                for data in test_data:
                    print("-" * 25 + "Test Learned Model" + "-" * 25)
                    print(f"test data name: {data[1]}")
                    print(f"test mode: {model_prefix}")
                    save_direc = f'./Result/FJSP/{config.test_datas_type}/{data[1]}'
                    if not os.path.exists(save_direc):
                        os.makedirs(save_direc)
                    min_makespan=[999999]*len(config.File_GAN)
                    time_total=[0]*len(config.File_GAN)
                    for model in test_model:
                        print('model--------------------',model)
                        save_path = save_direc + f'/Result_{model_prefix}+{model[1]}_{data[1]}.npy'
                        if (not os.path.exists(save_path)) or config.cover_flag:
                            print(f"Model name : {model[1]}")
                            print(f"data name: ./{config.test_datas}/{data[1]}")

                            if not config.flag_sample:
                                print("Test mode: Greedy")
                                result_5_times = []
                                # Greedy mode, test 5 times, record average time.
                                for j in range(5):
                                    result,makespan_list = self.greedy_strategy(data[0], model[0], config.seed_test,config.File_GAN)
                                    result_5_times.append(result)
                                    for file_num in range(len(config.File_GAN)):
                                        time_total[file_num]+=result[file_num][1]
                                        if (min_makespan[file_num] > makespan_list[file_num]):
                                            min_makespan[file_num] = makespan_list[file_num]
                                result_5_times = np.array(result_5_times)

                                save_result = np.mean(result_5_times, axis=0)
                                print("testing results:")
                                print(f"makespan(greedy): ", save_result[:, 0])
                                print(f"average--makespan(greedy): ", save_result[:, 0].mean())
                                print(f"time: ", save_result[:, 1].mean())

                            else:
                                # Sample mode, test once.
                                print("Test mode: Sample")
                                save_result,makespan_list = self.sampling_strategy(data[0], model[0], config.sample_times, config.seed_test)
                                for file_num in range(len(config.File_GAN)):
                                    time_total[file_num] += save_result[file_num][1]
                                    if (min_makespan[file_num] > makespan_list[file_num]):
                                        min_makespan[file_num] = makespan_list[file_num]

                                print("testing results:")
                                print(f"makespan(sampling): ", save_result[:, 0])
                                print(f"average--makespan(greedy): ", save_result[:, 0].mean())
                                print(f"time: ", save_result[:, 1].mean())
                            runtime = time.time() - start_time
                            np.save(save_path, save_result)
                    for file_num1 in range(len(config.File_GAN)):
                        f.write(str(min_makespan[file_num1]) + ' ' + str(time_total[file_num1]) + '\n')
