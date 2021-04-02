# Created by Giuseppe Paolo 
# Date: 27/07/2020

#Here I make the class that creates everything. I pass the parameters as init arguments, this one creates the param class, and the pop, arch, opt alg

import os
from core.population import Population
from core.evolvers import NoveltySearch, CMAES, CMANS, NSGAII, SERENE, MAPElites, CMAME, RandomSearch
from core.behavior_descriptors.behavior_descriptors import BehaviorDescriptor
from core import Evaluator
import multiprocessing as mp
from timeit import default_timer as timer
import gc
from analysis.logger import Logger

evaluator = None
main_pool = None # Using pool as global prevents the creation of new environments at every generation


class Searcher(object):
  """
  This class creates the instance of the NS algorithm and everything related
  """
  def __init__(self, parameters):
    self.parameters = parameters
    self.bd_extractor = BehaviorDescriptor(self.parameters)

    self.generation = 1

    if self.parameters.multiprocesses:
      global main_pool
      main_pool = mp.Pool(initializer=self.init_process, processes=self.parameters.multiprocesses)

    self.evaluator = Evaluator(self.parameters)

    self.population = Population(self.parameters, init_size=self.parameters.pop_size)
    self.init_pop = True
    self.offsprings = None

    if self.parameters.exp_type == 'NS':
      self.evolver = NoveltySearch(self.parameters)

    elif self.parameters.exp_type == 'SIGN':
      self.evolver = NoveltySearch(self.parameters)

    elif self.parameters.exp_type == 'CMA-ES':
      self.evolver = CMAES(self.parameters)
      # Generate CMA-ES initial population
      del self.population
      self.population = Population(self.parameters, init_size=self.parameters.emitter_population)
      for agent in self.population:
        agent['genome'] = self.evolver.optimizer.ask()

    elif self.parameters.exp_type == 'CMA-NS':
      self.evolver = CMANS(self.parameters)
      self.reward_archive = self.evolver.rew_archive

    elif self.parameters.exp_type == 'NSGA-II':
      self.evolver = NSGAII(self.parameters)

    elif self.parameters.exp_type == 'SERENE':
      self.evolver = SERENE(self.parameters)

    elif self.parameters.exp_type == 'ME':
      self.evolver = MAPElites(self.parameters)

    elif self.parameters.exp_type == 'CMA-ME':
      self.evolver = CMAME(self.parameters)

    elif self.parameters.exp_type == 'RND':
      self.evolver = RandomSearch(self.parameters)

    else:
      print("Experiment type {} not implemented.".format(self.parameters.exp_type))
      raise ValueError

  def init_process(self):
    """
    This function is used to initialize the pool so each process has its own instance of the evaluator
    :return:
    """
    global evaluator
    evaluator = Evaluator(self.parameters)

  def _feed_eval(self, agent):
    """
    This function feeds the agent to the evaluator and returns the updated agent
    :param agent:
    :return:
    """
    global evaluator
    if agent['evaluated'] == None: # Agents are evaluated only once
      agent = evaluator(agent, self.bd_extractor.__call__)
    return agent

  def evaluate_in_env(self, pop, pool=None):
    """
    This function evaluates the population in the environment by passing it to the parallel evaluators.
    :return:
    """
    if self.parameters.verbose: print('Evaluating {} in environment.'.format(pop.name))
    if pool is not None:
      pop.pop = pool.map(self._feed_eval, pop.pop) # As long as the ID is fine, the order of the element in the list does not matter
    else:
      for i in range(pop.size):
        if self.parameters.verbose: print(".", end = '') # The end prevents the newline
        if pop[i]['evaluated'] is None: # Agents are evaluated only once
          pop[i] = self.evaluator(pop[i], self.bd_extractor)
      if self.parameters.verbose: print()

  def _main_search(self, budget_chunk):
    """
    This function performs the main search e.g. NS/NSGA/CMA-ES
    :return:
    """
    # Only log reward here if NS or NSGA. Emitter based log during the emitter evaluation
    if self.evolver.emitter_based:
      log_reward = False
    else:
      log_reward = True

    # Evaluate population in the environment only the first time
    if self.init_pop:
      self.evaluate_in_env(self.population, pool=main_pool)
      self.population['evaluated'] = list(range(self.evolver.evaluated_points, self.evolver.evaluated_points + self.population.size))
      self.evolver.evaluated_points += self.population.size
      self.evolver.evaluation_budget -= self.population.size
      budget_chunk -= self.population.size
      self.init_pop = False

      if not self.evolver.emitter_based:
        for area in self.population['rew_area']:
          if area is not None:
            name = 'rew_area_{}'.format(area)
            if name not in Logger.data:
              Logger.data[name] = 0
            Logger.data[name] += 1

    while budget_chunk > 0 and self.evolver.evaluation_budget > 0:
      self.offsprings = self.evolver.generate_offspring(self.population, pool=None, generation=self.generation)  # Generate offsprings

      # Evaluate offsprings in the env
      self.evaluate_in_env(self.offsprings, pool=main_pool)
      self.offsprings['evaluated'] = list(range(self.evolver.evaluated_points, self.evolver.evaluated_points + self.offsprings.size))
      self.evolver.evaluated_points += self.offsprings.size
      self.evolver.evaluation_budget -= self.offsprings.size
      budget_chunk -= self.offsprings.size
      self.evolver.init_emitters(self.population, self.offsprings)

      # Evaluate performances of pop and off and update archive
      self.evolver.evaluate_performances(self.population, self.offsprings, pool=main_pool)  # Calculate novelty/fitness/curiosity etc
      # Only update archive using NS stuff. No archive candidates from emitters
      self.evolver.update_archive(self.population, self.offsprings, generation=self.generation)

      # Save pop, archive and off
      if self.generation % 1 == 0:
        self.population.save(self.parameters.save_path, 'gen_{}'.format(self.generation))
        self.evolver.archive.save(self.parameters.save_path, 'gen_{}'.format(self.generation))
        self.offsprings.save(self.parameters.save_path, 'gen_{}'.format(self.generation))

      # Log reward only if not emitter based
      if not self.evolver.emitter_based:
        for area in self.offsprings['rew_area']:
          if area is not None:
            name = 'rew_area_{}'.format(area)
            if name not in Logger.data:
              Logger.data[name] = 0
            Logger.data[name] += 1

      # Last thing we do is to update the population
      self.generation += 1
      self.evolver.update_population(self.population, self.offsprings, generation=self.generation)

  def _emitter_search(self, budget_chunk):
    """
    This function performs the reward search through the emitters
    :return:
    """
    if self.evolver.emitter_based and (len(self.evolver.emitters) > 0 or len(self.evolver.emitter_candidate) > 0):
      self.evolver.emitter_step(self.evaluate_in_env,
                                self.generation,
                                ns_pop=self.population,
                                ns_off=self.offsprings,
                                budget_chunk=budget_chunk,
                                pool=None)
      self.evolver.rew_archive.save(self.parameters.save_path, 'gen_{}'.format(self.generation))

      # Update the performaces due to possible changes in the pop and archive given by the emitters
      self.evolver.evaluate_performances(self.population, self.offsprings, pool=None)
      # Update main archive with the archive candidates from the emitters
      self.evolver.elaborate_archive_candidates(self.generation)

  def chunk_step(self):
    """
    This function performs all the calculations needed for one generation.
    Generates offsprings, evaluates them and the parents in the environment, calculates the performance metrics,
    updates archive and population and finally saves offsprings, population and archive.
    :return: time taken for running the generation
    """
    global main_pool
    start_time = timer()

    print("\nRemaining budget: {}".format(self.evolver.evaluation_budget))

    # -------------------
    # Base part
    # -------------------
    budget_chunk = self.parameters.chunk_size
    if self.evolver.evaluation_budget > 0:
      print("MAIN")
      self._main_search(budget_chunk)

    # -------------------
    # Emitters part
    # -------------------
    budget_chunk = self.parameters.chunk_size
    # Starts only if a reward has been found.
    if self.evolver.evaluation_budget > 0:
      print("EMITTERS: {}".format(len(self.evolver.emitters)))
      self._emitter_search(budget_chunk)
    # -------------------

    return timer() - start_time, self.evolver.evaluated_points

  def load_generation(self, generation, path):
    """
    This function loads the population, the offsprings and the archive at a given generation, so it can restart the
    search from there.
    :param generation:
    :param path: experiment path
    :return:
    """
    self.generation = generation

    self.population.load(os.path.join(path, 'population_gen_{}.pkl'.format(self.generation)))
    self.offsprings.load(os.path.join(path, 'offsprings_gen_{}.pkl'.format(self.generation)))
    self.evolver.archive.load(os.path.join(path, 'archive_gen_{}.pkl'.format(self.generation)))

  def close(self):
    """
    This function closes the pool and deletes everything.
    :return:
    """
    if self.parameters.multiprocesses:
      global main_pool
      main_pool.close()
      main_pool.join()
    gc.collect()
