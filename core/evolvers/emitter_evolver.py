# Created by giuseppe
# Date: 19/10/20

from core.evolvers import BaseEvolver
from core.evolvers import utils
from core.population import Population, Archive
import numpy as np
from scipy import special
from environments.environments import registered_envs
from itertools import chain
import copy

class EmitterEvolver(BaseEvolver):
  """
  This class is the base for the evolver built around the emitter concept
  """
  def __init__(self, parameters, **kwargs):
    super().__init__(parameters)

    self.update_criteria = 'novelty'
    self.rew_archive = Archive(self.params, name='rew_archive')

    # Instantiated only to extract genome size
    controller = registered_envs[self.params.env_name]['controller']['controller'](**registered_envs[self.params.env_name]['controller'])

    self.genome_size = controller.genome_size
    self.bounds = self.params.genome_limit * np.ones((self.genome_size, len(self.params.genome_limit)))
    self.emitter_pop = Population(self.params, init_size=self.params.emitter_population, name='emitter')
    self.emitter_based = True
    self.archive_candidates = {}
    self.emitters = {}
    self.emitter_candidate = {}

  def evaluate_performances(self, population, offsprings, pool=None):
    """
    This function evaluates the novelty of population and offsprings wrt pop+off+archive reference set.
    The novelty is evaluated according to the given distance metric
    :param population:
    :param offsprings:
    :param pool: Multiprocessing pool
    :return:
    """
    # Get BSs
    population_bd = population['bd']
    offsprings_bd = offsprings['bd']

    reference_set = self.get_novelty_ref_set(population, offsprings)
    bd_set = population_bd + offsprings_bd

    novelties = utils.calculate_novelties(bd_set, reference_set, distance_metric=self.params.novelty_distance_metric,
                                          novelty_neighs=self.params.novelty_neighs, pool=pool)
    # Update population and offsprings
    population['novelty'] = novelties[:population.size]
    offsprings['novelty'] = novelties[population.size:]

  def calculate_init_sigma(self, ns_pop, ns_off, mean):
    """
    This function calculates the initial step size value for the emitter
    :param ns_pop:
    :param ns_off:
    :param mean
    :return: step size
    """
    # This is the factor multiplying the std deviation that is used to choose which percentage of the gaussian to select
    number_of_std = 3

    idxs = ns_pop['id'] + ns_off['id']
    idx = idxs.index(mean['id'])
    genomes = ns_pop['genome'] + ns_off['genome']
    genomes.pop(idx)  # Remove element taken into account
    distances = utils.calculate_distances([mean['genome']], genomes).flatten()
    sigma = min(distances) / number_of_std
    if sigma == 0.: sigma = self.params.mutation_parameters['sigma']
    return sigma

  def create_emitter(self, parent_id, ns_pop, ns_off):
    """
    This function creates the emitter
    :param parent:
    :param ns_pop:
    :param ns_off:
    :return:
    """
    raise NotImplementedError

  def init_emitters(self, ns_pop, ns_off):
    """
    This function initializes the emitters.
    :param ns_pop:
    :param ns_off:
    :return: True if any emitter has been init, False otherwise
    """
    if any(np.array(ns_pop['reward']) > 0) or any(np.array(ns_off['reward']) > 0):
      # Get rewarding genomes
      self.rewarding = {}
      for agent in chain(ns_pop, ns_off):
        if agent['reward'] > 0 and not agent['emitter']:
          agent['emitter'] = True
          self.rewarding[agent['id']] = agent
          self.rew_archive.store(agent) # They are also stored in the rew_archive

      # New emitters are added in the candidates list.
      # They will be added to the emitters list only if they can give some improvement
      for rew_agent in self.rewarding:
        self.emitter_candidate[rew_agent] = self.create_emitter(rew_agent, ns_pop, ns_off)

      return True
    return False

  def update_reward_archive(self, generation, emitters, emitter_idx):
    """
    This function updates the reward archive. That is the archive in which the rewarding functions found by CMA-ES are
    stored
    :param generation
    :param emitters: List of emitters. Can be either candidate emitters or full emitters
    :param emitter_idx: IDX of evaluated emitter
    :return:
    """
    emitters[emitter_idx].archived.append(0)
    if len(emitters[emitter_idx].archived_values) == 0: # If none added yet, add the one with highest reward
      agent_idx = np.argmax(emitters[emitter_idx].pop['reward'])
      if emitters[emitter_idx].pop[agent_idx]['reward'] > 0:
        emitters[emitter_idx].archived[-1] += 1
        emitters[emitter_idx].pop[agent_idx]['stored'] = generation
        emitters[emitter_idx].archived_values.append(emitters[emitter_idx].pop[agent_idx]['reward'])
        self.rew_archive.store(emitters[emitter_idx].pop[agent_idx])

    else: # Add only the ones whose reward is higher than the max of the emitter
      limit = np.max(emitters[emitter_idx].archived_values)
      for agent_idx in range(emitters[emitter_idx].pop.size):
        if emitters[emitter_idx].pop[agent_idx]['reward'] > limit:
          emitters[emitter_idx].archived[-1] += 1
          emitters[emitter_idx].pop[agent_idx]['stored'] = generation
          emitters[emitter_idx].archived_values.append(emitters[emitter_idx].pop[agent_idx]['reward'])
          self.rew_archive.store(emitters[emitter_idx].pop[agent_idx])

  def check_stopping_criteria(self, emitter_idx):
    if self.emitters[emitter_idx].steps == self.params.max_emitter_steps:
      return True
    elif self.emitters[emitter_idx].should_stop():
      return True
    elif self._stagnation(emitter_idx):
      return True
    else: return False

  def _stagnation(self, emitter_idx):
    """
    Calculates the stagnation criteria
    :param emitter_idx:
    :param ca_es_step:
    :return:
    """
    if self.params.stagnation == 'original':
      bottom = max(int(30 * self.genome_size / self.params.emitter_population + 120), int(self.emitters[emitter_idx].steps * .2))
      if self.emitters[emitter_idx].steps > bottom:
        limit = int(.3 * bottom)
        values = self.emitters[emitter_idx].values[-bottom:]
        maxes = np.max(values, 1)
        medians = np.median(values, 1)
        if np.median(maxes[:limit]) >= np.median(maxes[-limit:]) and \
                np.median(medians[:limit]) >= np.median(medians[-limit:]):
          return True
      return False
    elif self.params.stagnation == 'custom':
      bottom = int(20 * self.genome_size / self.params.emitter_population + 120)
      if self.emitters[emitter_idx].steps > bottom:
        values = self.emitters[emitter_idx].values[-bottom:]
        if np.median(values[:20]) >= np.median(values[-20:]) or np.max(values[:20]) >= np.max(values[-20:]):
          return True
      return False

  def choose_emitter(self):
    """
    This function is used to select the emitter with the biggest improvement.
    Emitters are chosen randomnly from the list by weighting the probability by their improvement
    :return:
    """
    # improvements = np.atleast_2d(np.array([[em, self.emitters[em].improvement] for em in self.emitters]))
    # idx = np.argmax(improvements[:, 1])
    # return improvements[idx, 0]
    # return np.random.choice(improvements[:, 0], p=special.softmax(improvements[:, 1]))

    emitters_data = np.atleast_2d(np.array([[em, self.emitters[em].ancestor['bd'], self.emitters[em].improvement] for em in self.emitters]))
    if self.rew_archive.size > 0: # Calculate pareto front between Novelty and Improvement
      reference_bd = self.rew_archive['bd']  # + [self.emitters[idx].ancestor['bd'] for idx in self.emitters]
      novelties = utils.calculate_novelties(np.stack(emitters_data[:, 1]),
                                            reference_bd,
                                            distance_metric=self.params.novelty_distance_metric,
                                            novelty_neighs=self.params.novelty_neighs, pool=None)
      fronts = utils.fast_non_dominated_sort(novelties, emitters_data[:, 2]) # Get pareto fronts
      idx = np.random.choice(fronts[0]) # Randomly sample from best front
      return emitters_data[idx, 0] # Return the one on the best front with the highest improv
    else: # Return the one with highest improvement
      idx = np.argmax(emitters_data[:, 2])
      return emitters_data[idx, 0]

  def get_novelty_ref_set(self, ns_pop, ns_off):
    """
    This function extracts the reference set for the novelty calculation
    :param ns_pop:
    :param ns_off:
    :return:
    """
    population_bd = ns_pop['bd']
    offsprings_bd = ns_off['bd']

    if self.archive.size > 0:
      archive_bd = self.archive['bd']
    else:
      archive_bd = []
    if self.rew_archive.size > 0:
      rew_archive_bd = self.rew_archive['bd']
    else:
      rew_archive_bd = []
    return population_bd + offsprings_bd + archive_bd + rew_archive_bd

  def update_emitter_novelties(self, ns_ref_set, ns_pop, emitter_idx, pool=None):
    """
    This function updates the most novel agent found by the emitter and the NOVELTY_CANDIDATES_BUFFER.
    It does this by calculating the novelty of the current pop of the emitter.
    :param ns_ref_set: Reference set to calculate Novelty
    :param ns_pop: Novelty Search population
    :param emitter_idx:
    :param pool:
    :return:
    """
    novelties = utils.calculate_novelties(self.emitters[emitter_idx].pop['bd'],
                                          ns_ref_set,
                                          distance_metric=self.params.novelty_distance_metric,
                                          novelty_neighs=self.params.novelty_neighs, pool=pool)

    self.emitters[emitter_idx].pop['novelty'] = novelties

    # Save in the NS archive candidates buffer the agents with a novelty higher than the previous most novel
    for emitter_agent in self.emitters[emitter_idx].pop:
      if emitter_agent['novelty'] > self.emitters[emitter_idx].most_novel['novelty']:
        self.emitters[emitter_idx].ns_arch_candidates.add(copy.deepcopy(emitter_agent))

    # Update emitter most novel
    most_novel = np.argmax(novelties)
    if novelties[most_novel] > self.emitters[emitter_idx].most_novel['novelty']:
      self.emitters[emitter_idx].pop[most_novel]['id'] = ns_pop.agent_id  # Recognize most novel agent by giving it a valid ID
      self.emitters[emitter_idx].pop[most_novel]['parent'] = emitter_idx  # The emitter idx is saved as the parent of the most novel
      ns_pop.agent_id += 1  # Update max ID reached

      self.emitters[emitter_idx].most_novel = self.emitters[emitter_idx].pop[most_novel].copy() # Update most novel

  def candidates_by_novelty(self, pool=None):
    """
    This function orders the candidates by their novelty wrt the Rew archive.
    This way most novel emitters are evaluated first helpin in better covering the space of reward
    :return: List of candidates idx ordered by novelty if rew_archive.size > 0, else just list of candidates emitters
    """
    # Get list of idx and of parent bd
    candidates_idx = list(self.emitter_candidate.keys())

    if self.rew_archive.size > 0 and len(candidates_idx) > 0:

      reference_bd = self.rew_archive['bd']# + [self.emitters[idx].ancestor['bd'] for idx in self.emitters]
      candidates_bd = [self.emitter_candidate[idx].ancestor['bd'] for idx in candidates_idx]
      novelties = utils.calculate_novelties(candidates_bd,
                                            reference_bd,
                                            distance_metric=self.params.novelty_distance_metric,
                                            novelty_neighs=self.params.novelty_neighs, pool=pool)

      # Order candidates idx based on their novelties
      sorted_zipped_lists = sorted(zip(novelties, candidates_idx), reverse=True)
      candidates_idx = [element for _, element in sorted_zipped_lists]
    return candidates_idx

  def update_archive(self, population, offsprings, generation):
    """
    Updates the archive according to the strategy and the criteria given.
    :param population:
    :param offsprings:
    :return:
    """
    # Get list of ordered indexes according to selection strategy
    if self.params.selection_operator == 'random':
      idx = list(range(offsprings.size))
      np.random.shuffle(idx)
    elif self.params.selection_operator == 'best':
      performances = offsprings[self.update_criteria]
      idx = np.argsort(performances)[::-1]  # Order idx according to performances. (From highest to lowest)
    else:
      raise ValueError(
        'Please specify a valid selection operator for the archive. Given {} - Valid: ["random", "best"]'.format(
          self.params.selection_operator))
    # Add to archive the first lambda offsprings in the idx list
    for i in idx[:self.params._lambda]:
      offsprings[i]['stored'] = generation
      self.archive.store(offsprings[i])

  def elaborate_archive_candidates(self, generation):
    """
    Chooses which archive candidates from the emitters to add in the ns archive
    :param generation:
    :return:
    """
    for em in self.archive_candidates:
      if self.params.selection_operator == 'random':
        idx = list(range(self.archive_candidates[em].size))
        np.random.shuffle(idx)
      elif self.params.selection_operator == 'best':
        performances = self.archive_candidates[em][self.update_criteria]
        idx = np.argsort(performances)[::-1]  # Order idx according to performances. (From highest to lowest)
      else:
        raise ValueError(
          'Please specify a valid selection operator for the archive. Given {} - Valid: ["random", "best"]'.format(
            self.params.selection_operator))

      for i in idx[:self.params._lambda]:
        self.archive_candidates[em][i]['stored'] = generation
        self.archive.store(self.archive_candidates[em][i])

    # In this are only the cands of the completed emitters, so it can be emptied after adding to the archive
    self.archive_candidates = {}

  def emitter_step(self, evaluate_in_env, generation, ns_pop, ns_off, budget_chunk, pool=None):
    """
    This function performs the steps for the FIT emitters
    :param evaluate_in_env: Function used to evaluate the agents in the environment
    :param generation: Generation at which the process is
    :param pool: Multiprocessing pool
    :return:
    """
    raise NotImplementedError
