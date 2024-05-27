import os
import time as time
import copy
import gym
import pandas as pd
import torch
import numpy as np
from Test.agent.FJSP.FJSP_GNN_agent import Memory, PPO
from Test.optimizer.FJSP_optimizer.FJSP__RL_algorithm.Basic_FJSP_optimizer import Bassic_FJSP_optimizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class FJSP_GNN_optimizer(Bassic_FJSP_optimizer):
    def __init__(self, config):
        super(FJSP_GNN_optimizer, self).__init__(config)
        self.config = config

    def nums_detec(self, lines):
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

    def schedule(self, env, model, memories, flag_sample=False):
        # Get state and completion signal
        state = env.state
        dones = env.done_batch
        done = False  # Unfinished at the beginning
        last_time = time.time()
        i = 0
        while ~done:
            i += 1
            with torch.no_grad():
                actions = model.policy_old.act(state, memories, dones, flag_sample=flag_sample, flag_train=False)
            state, rewards, dones = env.step(actions)  # environment transit
            done = dones.all()
        spend_time = time.time() - last_time  # The time taken to solve this environment (instance)
        # print("spend_time: ", spend_time)

        # Verify the solution
        gantt_result = env.validate_gantt()[0]
        if not gantt_result:
            print("Scheduling Error！！！！！！")
        return copy.deepcopy(env.makespan_batch), spend_time

    def update(self, data, config):

        if device.type == 'cuda':
            torch.cuda.set_device(device)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            torch.set_default_tensor_type('torch.FloatTensor')
        print("PyTorch device: ", device.type)
        torch.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, linewidth=None, profile=None,
                               sci_mode=False)

        # Load config and init objects
        # with open("./config.json", 'r') as load_f:
        #     load_dict = json.load(load_f)
        # env_paras = load_dict["env_paras"]
        # model_paras = load_dict["model_paras"]
        # train_paras = load_dict["train_paras"]
        # test_paras = load_dict["test_paras"]
        config.device = device
        # env_test_paras = copy.deepcopy(env_paras)
        num_ins = config.num_ins
        if config.sample:
            config.batch_size = config.num_sample
        else:
            config.batch_size = 1
        config.actor_in_dim = config.out_size_ma * 2 + config.out_size_ope * 2
        config.critic_in_dim = config.out_size_ma + config.out_size_ope
        # model_paras["actor_in_dim"] = model_paras["out_size_ma"] * 2 + model_paras["out_size_ope"] * 2
        # model_paras["critic_in_dim"] = model_paras["out_size_ma"] + model_paras["out_size_ope"]
        # data_path = "./problem/FJSP_test_datas/{}/".format(config.test_datas_type+'/'+str(config.Pn_j)+'x'+str(config.Pn_m))

        data_path = "./Test/data_test/FJSP_test_datas/{}/".format(config.test_datas_type+'/'+str(config.Pn_j)+'x'+str(config.Pn_m))
        test_files = os.listdir(data_path)
        test_files.sort(key=lambda x: x[:-4])
        test_files = test_files[:num_ins]


        memories = Memory()
        model = PPO(config)
        rules = config.rules
        envs = []  # Store multiple environments

        # Detect and add models to "rules"
        if "DRL" in rules:
            for root, ds, fs in os.walk('./Train/model_/FJSP/FJSP_GNN/{}/'.format(str(config.Pn_j)+'x'+str(config.Pn_m))):
                for f in fs:
                    if f.endswith('.pt'):
                        rules.append(f)
        if len(rules) != 1:
            if "DRL" in rules:
                rules.remove("DRL")

        # Generate data files and fill in the header
        str_time = time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))
        save_path = './Result/FJSP/save/test_{0}'.format(str_time)
        os.makedirs(save_path)
        writer = pd.ExcelWriter(
            '{0}/makespan_{1}.xlsx'.format(save_path, str_time))  # Makespan data storage path
        writer_time = pd.ExcelWriter('{0}/time_{1}.xlsx'.format(save_path, str_time))  # time data storage path
        file_name = [test_files[i] for i in range(num_ins)]
        data_file = pd.DataFrame(file_name, columns=["file_name"])
        data_file.to_excel(writer, sheet_name='Sheet1', index=False)
        # writer.save()
        writer.close()
        data_file.to_excel(writer_time, sheet_name='Sheet1', index=False)
        # writer_time.save()
        writer_time.close()

        # Rule-by-rule (model-by-model) testing
        start = time.time()
        for i_rules in range(len(rules)):
            rule = rules[i_rules]
            # Load trained model
            if rule.endswith('.pt'):
                if device.type == 'cuda':
                    model_CKPT = torch.load('./Train/model_/FJSP/FJSP_GNN/{}/'.format(str(config.Pn_j)+'x'+str(config.Pn_m))+ rules[i_rules])
                else:
                    model_CKPT = torch.load('./Train/model_/FJSP/FJSP_GNN/{}/'.format(str(config.Pn_j)+'x'+str(config.Pn_m)) +  rules[i_rules], map_location=device)
                print('\nloading checkpoint:', "save_best_{}.pt".format(str(config.Pn_j)+'_'+str(config.Pn_m)))

                model.policy.load_state_dict(model_CKPT)
                model.policy_old.load_state_dict(model_CKPT)
            print('rule:', rule)

            # Schedule instance by instance
            # step_time_last = time.time()

            times = []
            with open('./Result/FJSP/'+self.config.optimizer + 'Sample---solution-------.txt', 'a') as f:
                f.write(self.config.test_datas_type + str(self.config.Pn_j) + 'x' + str(self.config.Pn_m) + '\n')
                for i_ins in range(num_ins):

                    start = time.time()
                    test_file = data_path + test_files[i_ins]
                    with open(test_file) as file_object:
                        line = file_object.readlines()
                        ins_num_jobs, ins_num_mas, _ = self.nums_detec(line)
                    config.Pn_j = ins_num_jobs
                    config.Pn_m = ins_num_mas

                    # Environment object already exists
                    if len(envs) == num_ins:
                        env = envs[i_ins]
                    # Create environment object
                    else:
                        # Clear the existing environment
                        # meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                        # if meminfo.used / meminfo.total > 0.7:
                        #     envs.clear()
                        # DRL-S, each env contains multiple (=num_sample) copies of one instance
                        env_test_paras = {"Pn_j": config.Pn_j,
                                          "Pn_m": config.Pn_m,
                                          "batch_size": config.batch_size,
                                          "ope_feat_dim": config.ope_feat_dim,
                                          "ma_feat_dim": config.ma_feat_dim,
                                          "show_mode": config.show_mode,
                                          "valid_batch_size": config.valid_batch_size}
                        if config.sample:
                            config.data_source = 'file'
                            env = gym.make('fjsp-v0', data=[test_file] * config.num_sample, config=config)
                            # env = gym.make('fjsp-v0', case=[test_file] * config.num_sample,
                            #                env_paras=env_test_paras, data_source='file')
                        # DRL-G, each env contains one instance
                        else:
                            config.data_source = 'file'
                            env = gym.make('fjsp-v0', data=[test_file], config=config)
                            # env = gym.make('fjsp-v0', case=[test_file], env_paras=env_test_paras, data_source='file')
                        envs.append(copy.deepcopy(env))
                        # print("Create env[{0}]".format(i_ins))

                    # Schedule an instance/environment
                    # DRL-S

                    if config.sample:
                        print('------------------DRL-S')
                        Makespan = 9999999
                        makespans = []
                        makespan, time_re = self.schedule(env, model, memories, flag_sample=config.sample)
                        makespans.append(torch.min(makespan))
                        if (Makespan > makespans[0].item()):
                            Makespan = makespans[0].item()

                        times.append(time_re)
                        runtime = time.time() - start
                        f.write(str(Makespan) + ' ' + str(runtime) + '\n')
                        print('makespan', test_file, Makespan)
                    # DRL-G
                    else:
                        print('------------------DRL-G')
                        time_s = []
                        makespan_s = []  # In fact, the results obtained by DRL-G do not change
                        Makespan = 9999999
                        for j in range(config.num_average):
                            makespan, time_re = self.schedule(env, model, memories)
                            # print('makespan',test_file, makespan.item())
                            if (Makespan > makespan.item()):
                                Makespan = makespan.item()
                            makespan_s.append(makespan)
                            time_s.append(time_re)
                            env.reset()
                        runtime = time.time() - start
                        f.write(str(Makespan) + ' ' + str(runtime) + '\n')
                        print('makespan', test_file, Makespan)

                        # makespans.append(torch.mean(torch.tensor(makespan_s)))
                        # times.append(torch.mean(torch.tensor(time_s)))
                    print("finish env {0}".format(i_ins))
                # Save makespan and time data to files
                # data = pd.DataFrame(torch.tensor(makespans).t().tolist(), columns=[rule])
                # data.to_excel(writer, sheet_name='Sheet1', index=False, startcol=i_rules + 1)
                # # writer.save()
                # writer.close()
                # data = pd.DataFrame(torch.tensor(times).t().tolist(), columns=[rule])
                # data.to_excel(writer_time, sheet_name='Sheet1', index=False, startcol=i_rules + 1)
                # # writer_time.save()
                # writer_time.close()
                for env in envs:
                    env.reset()