import numpy as np
from environment import ARPOD_GYM
from dynamics import chaser_continous
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.policies import ActorCriticPolicy
from typing import Callable
import gymnasium as gym
import os
import torch as T
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecExtractDictObs, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback
from time import sleep
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter
from visualizer_all import write2text
from gymnasium.wrappers import TimeLimit

class TimeLimitWrapper(gym.Wrapper):
    """
    :param env: (gym.Env) Gym environment that will be wrapped
    :param max_steps: (int) Max number of steps per episode
    """

    def __init__(self, env, max_steps=2500):
        # Call the parent constructor, so we can access self.env later
        super(TimeLimitWrapper, self).__init__(env)
        self.max_steps = max_steps
        # Counter of steps per episode
        self.current_step = 0

    def reset(self, **kwargs):
        """
        Reset the environment
        """
        # Reset the counter
        self.current_step = 0
        obs, info = self.env.reset()
        return obs, info

    def step(self, action):
        """
        :param action: ([float] or int) Action taken by the agent
        :return: (np.ndarray, float, bool, bool, dict) observation, reward, is the episode      		over?, additional informations
        """
        self.current_step += 1
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Overwrite the truncation signal when when the number of steps reaches the maximum
        if self.current_step >= self.max_steps:
            truncated = True
        return obs, reward, terminated, truncated, info


env = make_vec_env(ARPOD_GYM, wrapper_class=TimeLimitWrapper, n_envs=1)

model_replay = 'chaser_vecnormalize_400000_steps.pkl'
stats_path = os.path.join('envstats', model_replay)
env = VecNormalize.load(stats_path, env)

#  do not update them at test time
env.training = False
# reward normalization is not needed at test time
env.norm_reward = False

modelname = 'vbar-finalv2.zip'
model_dir = f'models/{modelname}'
loaded_model = PPO.load(model_dir, env=env, print_system_info=True)
print(loaded_model.policy)
print(loaded_model.policy_kwargs)
evaluate_policy(model=loaded_model, env=env, n_eval_episodes=100, deterministic=False)
