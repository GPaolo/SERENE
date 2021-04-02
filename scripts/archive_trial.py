# Created by Giuseppe Paolo 
# Date: 29/09/2020

# This script is used to test the agents in the archive in the environment by rendering them


import os
import gym
import numpy as np
from environments import *
import gym_collectball
from core.population.archive import Archive
from parameters import params
import time

env_name = 'AntMaze'
exp_type = 'NS'

PATH = '/home/giuseppe/src/cmans/experiment_data/{}/{}_{}/2020_09_22_18:31_131792'.format(env_name, env_name, exp_type)

params.load(os.path.join(PATH, '_params.json'))
arch = Archive(parameters=params)
arch.load(os.path.join(PATH, 'archive_final.pkl'))

env = registered_envs[env_name]
controller = env['controller']['controller']
controller = controller(input_size=env['controller']['input_size'], output_size=env['controller']['output_size'])
gym_env = gym.make(env['gym_name'])

genome = arch['genome'][-1]

controller.load_genome(genome=genome)


gym_env.render()
obs = gym_env.reset()
for i in range(3000):
  action = controller(env['controller']['input_formatter'](i, obs))
  action = env['controller']['output_formatter'](action)
  obs, _, done, _ = gym_env.step(action)
  print(action)
  print(obs)
  print()
  gym_env.render()
  # time.sleep(.1)
  if done:
    print("AAAA: {}".format(i))
    time.sleep(20)
    break