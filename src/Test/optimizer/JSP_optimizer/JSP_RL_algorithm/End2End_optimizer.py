import json
import os
import random
import time
from typing import Tuple, Callable, Optional, List, Any, Type, Sequence, Union
import pandas as pd
import numpy as np
import torch
from compiled_jss.CPEnv import CompiledJssEnvCP
import ray
from ray.rllib.env.base_env import BaseEnv, ASYNC_RESET_RETURN
from ray.rllib.utils.annotations import PublicAPI
from ray.rllib.utils.typing import MultiEnvDict, EnvType, EnvID, MultiAgentDict
from stable_baselines3.common.vec_env.util import dict_to_obs
from stable_baselines3.common.vec_env import VecEnv
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvStepReturn, VecEnvObs
from Test.agent.JSP.End2End_agent import Agent, MyDummyVecEnv, End2End_agent
from collections import OrderedDict
from torch.distributions import Categorical
import gym
from Test.optimizer.JSP_optimizer.JSP_RL_algorithm.Basic_learning_algorithm import Basic_learning_algorithm

NPY = 'rand4x2'

def make_env(seed, instance):
    def thunk():
        _env = CompiledJssEnvCP(instance)
        return _env

    return thunk
# Changing styles by reading an example
def read_instance(data,config):
    row1 = data[0]
    row2 = data[1]
    h,l =np.shape(row1)# first row
    modified_lines = []
    for i in range(len(row1)):
        row_data = []
        for j in range(len(row1[i])):
            row_data.append([row2[i][j]-1, row1[i][j]])
        modified_lines.append(row_data)
    # Write new file, change npy to txt.
    new_filename = './Result/problem/JSP/JSP_test_datas/train_txt/'+config.test_datas_type+str(config.Pn_j)+'x'+str(config.Pn_m)+'_'+str(config.itration)+'.txt'
    with open(new_filename, 'w', encoding='utf-8') as file:
        # Adding a first line
        file.write(str(h)+'	'+str(l)+'\n')
        # Iterate through each sublist in the list Convert sublists to strings
        for sublist in modified_lines:
            sublist_str = ''.join(str(num).replace(" ", "") for num in sublist)
            # Remove the square brackets and commas to match the original.
            sublist_str = sublist_str.replace("[", "").replace("]", "	").replace(",", "	").rstrip()
            # Write sublists to file and add newlines
            file.write(sublist_str + '\n')
    return new_filename
# def read_instance(instance_filename):
#
#     new_instance = instance_filename.replace('instances_run/npy/', os.getcwd()+'/instances_run/instances_runNow/')
#     # Load data
#     dataLoaded = np.load(instance_filename)
#     dataset = []
#     for i in range(dataLoaded.shape[0]):
#         dataset.append((dataLoaded[i][0], dataLoaded[i][1]))
#     # Each time to generate a txt file, each cycle of data format conversion
#     for d in range(len(dataset)):
#         row1 = dataset[d][0]
#         row2 = dataset[d][1]
#         h,l =np.shape(row1)# first row
#         modified_lines = []
#         for i in range(len(row1)):
#             row_data = []
#             for j in range(len(row1[i])):
#                 row_data.append([row2[i][j]-1, row1[i][j]])
#             modified_lines.append(row_data)
#         # Write new file, change npy to txt.
#         new_filename = new_instance.replace('.npy', '_'+str(d)+'.txt')
#         with open(new_filename, 'w', encoding='utf-8') as file:
#             # Add first line
#             file.write(str(h)+'	'+str(l)+'\n')
#             # Iterate through each sublist in the list Convert sublists to strings
#             for sublist in modified_lines:
#                 sublist_str = ''.join(str(num).replace(" ", "") for num in sublist)
#                 # Remove the square brackets and commas to match the original.
#                 sublist_str = sublist_str.replace("[", "").replace("]", "	").replace(",", "	").rstrip()
#                 # Write sublists to file and add newlines
#                 file.write(sublist_str + '\n')

@ray.remote(num_cpus=1)
class _RemoteSingleAgentEnv:
    """Wrapper class for making a gym env a remote actor."""

    def __init__(self, make_env, i, env_per_worker):
        self.env = MyDummyVecEnv([lambda: make_env((i * env_per_worker) + k) for k in range(env_per_worker)])

    def reset(self):
        return self.env.reset(), 0, False, {}

    def step(self, actions):
        return self.env.step(actions)
@PublicAPI
class MyRemoteVectorEnv(BaseEnv):
    """Vector env that executes envs in remote workers.
    This provides dynamic batching of inference as observations are returned
    from the remote simulator actors. Both single and multi-agent child envs
    are supported, and envs can be stepped synchronously or async.
    You shouldn't need to instantiate this class directly. It's automatically
    inserted when you use the `remote_worker_envs` option for Trainers.
    """

    @property
    def observation_space(self):
        return self._observation_space

    def __init__(self, make_env: Callable[[int], EnvType], num_workers: int, env_per_worker: int, observation_space: Optional[gym.spaces.Space], device: torch.device):
        self.make_local_env = make_env
        self.num_workers = num_workers
        self.env_per_worker = env_per_worker
        self.num_envs = num_workers * env_per_worker
        self.poll_timeout = None

        self.actors = None  # lazy init
        self.pending = None  # lazy init

        self.observation_space = observation_space
        self.keys = []
        shapes = {}
        dtypes = {}
        for key, box in observation_space.items():
            self.keys.append(key)
            shapes[key] = box.shape
            dtypes[key] = box.dtype

        self.device = device

        self.buf_obs = OrderedDict(
            [(k, torch.zeros((self.num_envs,) + tuple(shapes[k]), dtype=torch.float, device=self.device)) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]

    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        for key in self.keys:
            self.buf_obs[key][env_idx * self.env_per_worker: (env_idx + 1) * self.env_per_worker] = torch.from_numpy(obs[key]).to(self.device,
                                                                                           non_blocking=True)

    def poll(self) -> Tuple[MultiEnvDict, MultiEnvDict, MultiEnvDict,
                            MultiEnvDict, MultiEnvDict]:
        if self.actors is None:

            def make_remote_env(i):
                return _RemoteSingleAgentEnv.remote(self.make_local_env, i, self.env_per_worker)

            self.actors = [make_remote_env(i) for i in range(self.num_workers)]

        if self.pending is None:
            self.pending = {a.reset.remote(): a for a in self.actors}

        # each keyed by env_id in [0, num_remote_envs)
        ready = []

        # Wait for at least 1 env to be ready here
        while not ready:
            ready, _ = ray.wait(
                list(self.pending),
                num_returns=len(self.pending),
                timeout=self.poll_timeout)

        for obj_ref in ready:
            actor = self.pending.pop(obj_ref)
            env_id = self.actors.index(actor)
            ob, rew, done, info = ray.get(obj_ref)

            self._save_obs(env_id, ob)
            self.buf_rews[env_id * self.env_per_worker: (env_id + 1) * self.env_per_worker] = rew
            self.buf_dones[env_id * self.env_per_worker: (env_id + 1) * self.env_per_worker] = done
            self.buf_infos[env_id * self.env_per_worker: (env_id + 1) * self.env_per_worker] = info
        return (self._obs_from_buf(), self.buf_rews, self.buf_dones, self.buf_infos)

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, self.buf_obs)

    @PublicAPI
    def send_actions(self, action_list) -> None:
        for worker_id in range(self.num_workers):
            actions = action_list[worker_id * self.env_per_worker: (worker_id + 1) * self.env_per_worker]
            actor = self.actors[worker_id]
            obj_ref = actor.step.remote(actions)
            self.pending[obj_ref] = actor

    @PublicAPI
    def try_reset(self,
                  env_id: Optional[EnvID] = None) -> Optional[MultiAgentDict]:
        actor = self.actors[env_id]
        obj_ref = actor.reset.remote()
        self.pending[obj_ref] = actor
        return ASYNC_RESET_RETURN

    @PublicAPI
    def stop(self) -> None:
        if self.actors is not None:
            for actor in self.actors:
                actor.__ray_terminate__.remote()

    @observation_space.setter
    def observation_space(self, value):
        self._observation_space = value
class WrapperRay(VecEnv):

    def __init__(self, make_env, num_workers, per_worker_env, device):
        self.one_env = make_env(0)
        self.remote: BaseEnv = MyRemoteVectorEnv(make_env, num_workers, per_worker_env, self.one_env.observation_space, device)
        super(WrapperRay, self).__init__(num_workers * per_worker_env, self.one_env.observation_space, self.one_env.action_space)

    def reset(self) -> VecEnvObs:
        return self.remote.poll()[0]

    def step_async(self, actions: np.ndarray) -> None:
        self.remote.send_actions(actions)

    def step_wait(self) -> VecEnvStepReturn:
        return self.remote.poll()

    def close(self) -> None:
        self.remote.stop()

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        pass

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        pass

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        pass

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        pass

    def get_images(self) -> Sequence[np.ndarray]:
        pass

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        pass
class End2End_optimizer(Basic_learning_algorithm):
    def __init__(self, config):
        self.config = config
        random.seed(0)
        np.random.seed(0)
        torch.manual_seed(0)


    def init_population(self,problem,data,config):
        ray.shutdown()
        ray.init(log_to_driver=False, include_dashboard=False)

    def update(self,data,config):

        with torch.inference_mode():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            agent = Agent()
            checkpoint = torch.load(os.getcwd()+"/Test/optimizer/JSP_optimizer/JSP_RL_algorithm/checkpoint/checkpoint.pt", map_location=device, encoding ='latin1')
            agent.load_state_dict(checkpoint["model"])
            actor = agent.actor
            actor = torch.jit.script(actor)
            actor = actor.to(device, non_blocking=True)
            actor.eval()
        print(f'Using device {device}')

        instances = []

        instances.append(read_instance(data,config))

        # wandb.login(key="3d6ab290f5a1128a46c4fe418c8e2187df67f1f0")
        experiment_name = f"benchmark__{0}__{int(time.time())}"
        # wandb.init(project=config.wandb_project_name + 'DEBUG', config=vars(config), name=experiment_name, save_code=True)

        all_datas = []
        with torch.inference_mode():
            # for each instance
            for instance in instances:
                print(f'Now solving instance {instance}')
                for iter_idx in range(1):
                    random.seed(iter_idx)
                    np.random.seed(iter_idx)
                    torch.manual_seed(iter_idx)
                    start_time = time.time()

                    fn_env = [make_env(0, instance)
                              for i in range(config.num_workers * 4)]
                    current_solution_cost = float('inf')
                    current_solution = []
                    ray_wrapper_env = WrapperRay(lambda n: fn_env[n](),
                                                 config.num_workers, 4, device)
                    envs = End2End_agent(ray_wrapper_env, device)
                    obs = envs.reset()
                    total_episode = 0
                    while total_episode < envs.num_envs:
                        logits = actor(obs['interval_rep'], obs['attention_interval_mask'], obs['job_resource_mask'],
                                       obs['action_mask'], obs['index_interval'], obs['start_end_tokens'])
                        # temperature vector
                        temperature = torch.arange(0.5, 2.0, step=(1.5 / (config.num_workers * 4)), device=device)
                        logits = logits / temperature[:, None]
                        probs = Categorical(logits=logits).probs
                        # random sample based on logits
                        actions = torch.multinomial(probs, probs.shape[1]).cpu().numpy()
                        obs, reward, done, infos = envs.step(actions)
                        total_episode += done.sum()
                        for env_idx, info in enumerate(infos):
                            if 'makespan' in info and int(info['makespan']) < current_solution_cost:
                                current_solution_cost = int(info['makespan'])
                                current_solution = json.loads(info['solution'])
                    total_time = time.time() - start_time
                    # write solution to file
                    with open(os.getcwd()+'/Train/model_/solutions_found/' + instance.split('/')[-1] + '_' + str(iter_idx) + '.json', 'w') as f:
                        json.dump(current_solution, f)
                    print(f'Instance {instance} - Iter {iter_idx} - Cost {current_solution_cost} - Time {total_time}')
                    all_datas.append({'instance': instance.split('/')[-1],
                                      'cost': current_solution_cost,
                                      'time': total_time})

                df = pd.DataFrame(all_datas)
                df.to_csv(os.getcwd()+'/Result/solutions_found/' + 'results.csv')
                # wandb.save('C:/Users/1/Desktop/results.csv')

        # log dataframe
        # wandb.log({"results": wandb.Table(dataframe=df)})
        # wandb.finish()
        return current_solution_cost,total_time
    # def update(self, data, config):
    #
    #     with torch.inference_mode():
    #         device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #         agent = Agent()
    #         checkpoint = torch.load(os.getcwd() + "/optimizer/RL_algorithm/checkpoint/checkpoint.pt",
    #                                 map_location=device, encoding='latin1')
    #         agent.load_state_dict(checkpoint["model"])
    #         actor = agent.actor
    #         actor = torch.jit.script(actor)
    #         actor = actor.to(device, non_blocking=True)
    #         actor.eval()
    #     print(f'Using device {device}')
    #
    #     # for each file in the 'instances_run' folder
    #     instances = []
    #     # Read npy file to generate txt file to instances_runNow folder
    #     # for file in sorted(os.listdir('instances_run/npy')):
    #     #     if file.startswith(NPY):
    #     #         read_instance('instances_run/npy/' + file)
    #     # for file2 in sorted(os.listdir('instances_run/instances_runNow/')):
    #     #     if file2.startswith(NPY):
    #     #         instances.append('instances_run/instances_runNow/' + file2)
    #     # Original logic, read txt file
    #     instances.append(read_instance(data, config))
    #
    #     # wandb.login(key="3d6ab290f5a1128a46c4fe418c8e2187df67f1f0")
    #     experiment_name = f"benchmark__{0}__{int(time.time())}"
    #     # wandb.init(project=config.wandb_project_name + 'DEBUG', config=vars(config), name=experiment_name, save_code=True)
    #
    #     all_datas = []
    #     with torch.inference_mode():
    #         # for each instance
    #         for instance in instances:
    #             print(f'Now solving instance {instance}')
    #             for iter_idx in range(10):
    #                 random.seed(iter_idx)
    #                 np.random.seed(iter_idx)
    #                 torch.manual_seed(iter_idx)
    #                 start_time = time.time()
    #
    #                 fn_env = [make_env(0, instance)
    #                           for i in range(config.num_workers * 4)]
    #                 current_solution_cost = float('inf')
    #                 current_solution = []
    #                 ray_wrapper_env = WrapperRay(lambda n: fn_env[n](),
    #                                              config.num_workers, 4, device)
    #                 envs = End2End_agent(ray_wrapper_env, device)
    #                 obs = envs.reset()
    #                 total_episode = 0
    #                 max_run_time = config.Pn_j * config.Pn_m * 0.05
    #                 start_time = time.time()
    #                 # while total_episode < envs.num_envs:
    #                 for run in range(10000000000):
    #                     run+=1
    #                     logits = actor(obs['interval_rep'], obs['attention_interval_mask'], obs['job_resource_mask'],
    #                                    obs['action_mask'], obs['index_interval'], obs['start_end_tokens'])
    #                     # temperature vector
    #                     temperature = torch.arange(0.5, 2.0, step=(1.5 / (config.num_workers * 4)), device=device)
    #                     logits = logits / temperature[:, None]
    #                     probs = Categorical(logits=logits).probs
    #                     # random sample based on logits
    #                     actions = torch.multinomial(probs, probs.shape[1]).cpu().numpy()
    #                     obs, reward, done, infos = envs.step(actions)
    #                     total_episode += done.sum()
    #                     for env_idx, info in enumerate(infos):
    #                         if 'makespan' in info and int(info['makespan']) < current_solution_cost:
    #                             current_solution_cost = int(info['makespan'])
    #                             current_solution = json.loads(info['solution'])
    #                     if time.time() - start_time > max_run_time:
    #                         break
    #                 end_time = time.time()
    #                 elapsed_time = end_time - start_time
    #                 with open(os.getcwd() + '/model_/solutions_found/' + instance.split('/')[-1] + '_' + str(
    #                         iter_idx) + '.json', 'w') as f:
    #                     json.dump(current_solution, f)
    #                 print(f'Instance {instance} - Iter {iter_idx} - Cost {current_solution_cost} - Time {elapsed_time}')
    #                 all_datas.append({'instance': instance.split('/')[-1],
    #                                   'cost': current_solution_cost,
    #                                   'time': elapsed_time})
    #
    #             df = pd.DataFrame(all_datas)
    #             df.to_csv('results.csv')
    #             # wandb.save('C:/Users/zhenzekun/Desktop/results.csv')
    #
    #     # log dataframe
    #     # wandb.log({"results": wandb.Table(dataframe=df)})
    #     # wandb.finish()
    #     return current_solution_cost
