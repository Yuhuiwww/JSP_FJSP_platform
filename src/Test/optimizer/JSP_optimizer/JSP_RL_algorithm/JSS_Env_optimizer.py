import time

import numpy as np
from gym.utils import EzPickle
from Test.agent.JSP.JSS_Env_agent import JssEnv, JSS_Env_agent
from Test.optimizer.JSP_optimizer.JSP_RL_algorithm.Basic_learning_algorithm import Basic_learning_algorithm
class JSS_Env_optimizer(Basic_learning_algorithm):
    def __init__(self,config):
        EzPickle.__init__(self)
        self.config=config
    def init_population(self,problem,data,config):
        pass

    def update(self, data, config):
        t1 = time.time()
        env = JSS_Env_agent(data, config)
        obs = env.reset(data, config)
        done = (False, 0)
        cum_reward = 0
        while not done[0]:
            legal_actions = obs["action_mask"]
            actions = np.random.choice(
                len(legal_actions), 1, p=(legal_actions / legal_actions.sum())
            )[0]
            obs, rewards, done, _ = env.step(actions)
            cum_reward += rewards
        t2 = time.time()
        print('makespan', done[1], 'time', t2 - t1)

        return done[1], t2 - t1

# def update(self,data,config):
#     max_run_time=config.Pn_m*config.Pn_j*0.08
#     start_time=time.time()
#     min_makespan=1000000
#     for run in range(10000000000):
#         run+=1
#         env = JSS_Env_agent(data,config)
#         obs = env.reset(data,config)
#         done = (False,0)
#         cum_reward = 0
#         while not done[0]:
#             legal_actions = obs["action_mask"]
#             actions = np.random.choice(
#                 len(legal_actions), 1, p=(legal_actions / legal_actions.sum())
#             )[0]
#             obs, rewards, done, _ = env.step(actions)
#             cum_reward += rewards
#         if (min_makespan > done[1]):
#             min_makespan = done[1]
#         print('makespan:', min_makespan)
#         if time.time() - start_time > max_run_time:
#             break
#
#     return min_makespan