import math
import gym
from typing import Any, Callable, List, Optional, Sequence, Type, Union
from stable_baselines3.common.vec_env import VecEnvWrapper
import numpy as np
from ray.rllib.examples.centralized_critic import nn
from stable_baselines3.common.vec_env.base_vec_env import VecEnvIndices, VecEnvObs, VecEnv, VecEnvStepReturn
from stable_baselines3.common.vec_env.util import dict_to_obs, obs_space_info
import torch
from torch import nn, Tensor
from torch.distributions import Categorical
from collections import OrderedDict

def layer_init_tanh(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 100):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, positions: Tensor) -> Tensor:
        return self.pe[positions]

class MyDummyVecEnv(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.

    :param env_fns: a list of functions
        that return environments to vectorize
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]]):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)
        self.buf_obs = OrderedDict(
            [(k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k])) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=bool)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        for env_idx in range(self.num_envs):
            obs, self.buf_rews[env_idx], self.buf_dones[env_idx], self.buf_infos[env_idx] = self.envs[env_idx].step(
                self.actions[env_idx]
            )
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                self.buf_infos[env_idx]["terminal_observation"] = obs
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), self.buf_rews, self.buf_dones, self.buf_infos)

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        seeds = list()
        for idx, env in enumerate(self.envs):
            seeds.append(env.seed(seed + idx))
        return seeds

    def reset(self) -> VecEnvObs:
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[np.ndarray]:
        return [env.render(mode="rgb_array") for env in self.envs]

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.
        Otherwise (if ``self.num_envs == 1``), we pass the render call directly to the
        underlying environment.

        Therefore, some arguments such as ``mode`` will have values that are valid
        only when ``num_envs == 1``.

        :param mode: The rendering type.
        """
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        for key in self.keys:
            if key is None:
                self.buf_obs[key][env_idx] = obs
            else:
                self.buf_obs[key][env_idx] = obs[key]

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, self.buf_obs)

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def env_is_wrapped(self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]

class Actor(nn.Module):

    def __init__(self, pos_encoder):
        super(Actor, self).__init__()
        self.activation = nn.Tanh()
        self.project = nn.Linear(4, 8)
        nn.init.xavier_uniform_(self.project.weight, gain=1.0)
        nn.init.constant_(self.project.bias, 0)
        self.pos_encoder = pos_encoder

        self.embedding_fixed = nn.Embedding(2, 1)
        self.embedding_legal_op = nn.Embedding(2, 1)

        self.tokens_start_end = nn.Embedding(3, 4)

        # self.conv_transform = nn.Conv1d(5, 1, 1)
        # nn.init.kaiming_normal_(self.conv_transform.weight, mode="fan_out", nonlinearity="relu")
        # nn.init.constant_(self.conv_transform.bias, 0)

        self.enc1 = nn.TransformerEncoderLayer(8, 1, dim_feedforward=8 * 4, dropout=0.0, batch_first=True,
                                               norm_first=True)
        self.enc2 = nn.TransformerEncoderLayer(8, 1, dim_feedforward=8 * 4, dropout=0.0, batch_first=True,
                                               norm_first=True)

        self.final_tmp = nn.Sequential(
            layer_init_tanh(nn.Linear(8, 32)),
            nn.Tanh(),
            layer_init_tanh(nn.Linear(32, 1), std=0.01)
        )
        self.no_op = nn.Sequential(
            layer_init_tanh(nn.Linear(8, 32)),
            nn.Tanh(),
            layer_init_tanh(nn.Linear(32, 1), std=0.01)
        )

    def forward(self, obs, attention_interval_mask, job_resource, mask, indexes_inter, tokens_start_end):
        embedded_obs = torch.cat((self.embedding_fixed(obs[:, :, :, 0].long()), obs[:, :, :, 1:3],
                                  self.embedding_legal_op(obs[:, :, :, 3].long())), dim=3)
        non_zero_tokens = tokens_start_end != 0
        t = tokens_start_end[non_zero_tokens].long()
        embedded_obs[non_zero_tokens] = self.tokens_start_end(t)
        pos_encoder = self.pos_encoder(indexes_inter.long())
        pos_encoder[non_zero_tokens] = 0
        obs = self.project(embedded_obs) + pos_encoder

        transformed_obs = obs.view(-1, obs.shape[2], obs.shape[3])
        attention_interval_mask = attention_interval_mask.view(-1, attention_interval_mask.shape[-1])
        transformed_obs = self.enc1(transformed_obs, src_key_padding_mask=attention_interval_mask == 1)
        transformed_obs = transformed_obs.view(obs.shape)
        obs = transformed_obs.mean(dim=2)

        job_resource = job_resource[:, :-1, :-1] == 0

        obs_action = self.enc2(obs, src_mask=job_resource) + obs

        logits = torch.cat((self.final_tmp(obs_action).squeeze(2), self.no_op(obs_action).mean(dim=1)), dim=1)
        return logits.masked_fill(mask == 0, -3.4028234663852886e+38)


class Agent(nn.Module):
    def __init__(self):
        super(Agent, self).__init__()
        self.pos_encoder = PositionalEncoding(8)
        self.actor = Actor(self.pos_encoder)

    def forward(self, data, attention_interval_mask, job_resource_masks, mask, indexes_inter, tokens_start_end,
                action=None):
        logits = self.actor(data, attention_interval_mask, job_resource_masks, mask, indexes_inter, tokens_start_end)
        probs = Categorical(logits=logits)
        if action is None:
            probabilities = probs.probs
            actions = torch.multinomial(probabilities, probabilities.shape[1])
            return actions, torch.log(probabilities), probs.entropy()
        else:
            return logits, probs.log_prob(action), probs.entropy()

    def get_action_only(self, data, attention_interval_mask, job_resource_masks, mask, indexes_inter, tokens_start_end):
        logits = self.actor(data, attention_interval_mask, job_resource_masks, mask, indexes_inter, tokens_start_end)
        probs = Categorical(logits=logits)
        return probs.sample()

    def get_logits_only(self,data, attention_interval_mask, job_resource_masks, mask, indexes_inter, tokens_start_end):
        logits = self.actor(data, attention_interval_mask, job_resource_masks, mask, indexes_inter, tokens_start_end)
        return logits


class End2End_agent(VecEnvWrapper):
    def __init__(self, venv, device):
        super(End2End_agent, self).__init__(venv)
        self.device = device

    def reset(self):
        return self.venv.reset()

    def step_async(self, actions):
        self.venv.step_async(actions)

    def step_wait(self):
        return self.venv.step_wait()
