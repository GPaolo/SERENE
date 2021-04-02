# Created by Giuseppe Paolo 
# Date: 15/09/2020

# Created by Giuseppe Paolo
# Date: 29/07/2020

from parameters import params
import environments
import os
from environments import registered_envs
import setuptools
import gym
import gym_dummy
import argparse
from stable_baselines3 import PPO, SAC, A2C, TD3
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.cmd_util import make_vec_env
import numpy as np


if __name__ == "__main__":
  parser = argparse.ArgumentParser('Run RL baseline script')
  parser.add_argument('-env', '--environment', help='Environment to use', choices=list(registered_envs.keys()))
  parser.add_argument('-exp', '--experiment', help='RL Baseline to use.', choices=['PPO', 'SAC', 'TD3', 'A2C'])
  parser.add_argument('-sp', '--save_path', help='Path where to save the experiment')
  parser.add_argument('-mp', '--multiprocesses', help='How many parallel workers need to use', type=int) # TODO find a way to use multiprocessing maybe running mulple runs in parallel
  parser.add_argument('-r', '--runs', help='Number of runs to perform', type=int, default=1)
  parser.add_argument('-ts', '--timesteps', help='Timesteps the algorithm is trained', type=int, default=25000)

  args = parser.parse_args(['-env', 'Walker2D', '-exp', 'PPO', '-r', '2'])

  for run in range(args.runs):

    if args.environment is not None: params.env_name = args.environment
    if args.experiment is not None: params.exp_type = args.experiment
    if args.save_path is not None: params.save_path = os.path.join(args.save_path, params.save_dir)
    if args.multiprocesses is not None: params.multiprocesses = args.multiprocesses

    print("SAVE PATH: {}".format(params.save_path))
    params.save()

    env = registered_envs[args.environment]
    gym_env = gym.make(env['gym_name'])

    # TODO tune the hyperparameters properly
    if args.experiment == 'PPO':
      model = PPO(MlpPolicy, gym_env, verbose=1)
    elif args.experiment == 'SAC':
      model = SAC(MlpPolicy, gym_env, verbose=1)
    elif args.experiment == 'TD3':
      n_actions = env.action_space.shape[-1]
      action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
      model = TD3(MlpPolicy, gym_env, action_noise=action_noise, verbose=1)
    elif args.experiment == 'A2C':
      model = A2C(MlpPolicy, gym_env, verbose=1)
    else:
      raise ValueError('Wrong experiment chosen: {}'.format(args.experiment))

    model.learn(total_timesteps=args.timesteps)
    model.save(os.path.join(params.save_path, "ppo_redarm"))
