# Created by giuseppe
# Date: 05/10/20

from core.evolvers import EmitterEvolver
from core.evolvers import utils
from core.population import Population, Archive
import numpy as np
import copy
from analysis.logger import Logger


class FitnessEmitter(object):
  """
  This class implements the fitness emitter
  """
  def __init__(self, ancestor, mutation_rate, parameters):
    self.ancestor = ancestor
    self._init_mean = self.ancestor['genome']
    self.id = ancestor['id']
    self._mutation_rate = mutation_rate
    self._params = parameters
    self._pop_size = self._params.emitter_population
    self.pop = self._init_pop()
    self.ns_arch_candidates = Population(self._params, init_size=0, name='ns_arch_cand')

    # List of lists. Each inner list corresponds to the values obtained during a step
    # We init with the ancestor reward so it's easier to calculate the improvement
    self.values = []
    self.archived_values = []
    self.improvement = 0
    self._init_values = None
    self.archived = []  # List containing the number of archived agents at each step
    self.most_novel = self.ancestor
    self.steps = 0

  def estimate_improvement(self):
    """
    This function calculates the improvement given by the last updates wrt the parent
    If negative improvement, set it to 0.
    If there have been no updates yet, return the ancestor parent as reward
    Called at the end of the emitter evaluation cycle
    :return:
    """
    if self._init_values is None: # Only needed at the fist time
      self._init_values = self.values[:3]
    self.improvement = np.max([np.mean(self.values[-3:]) - np.mean(self._init_values), 0])

    # If local improvement update init_values to have improvement calculated only on the last expl step
    if self._params.local_improvement:
      self._init_values = self.values[-3:]

  def mutate_genome(self, genome):
    """
    This function mutates the genome
    :param genome:
    :return:
    """
    genome = genome + np.random.normal(0, self._mutation_rate, size=np.shape(genome))
    return genome.clip(self._params.genome_limit[0], self._params.genome_limit[1])

  def _init_pop(self):
    """
    This function initializes the emitter pop around the parent
    :return:
    """
    pop = Population(self._params, self._pop_size)
    for agent in pop:
      agent['genome'] = self.mutate_genome(self._init_mean)
    return pop

  def generate_off(self, generation):
    """
    This function generates the offsprings of the emitter
    :return:
    """
    offsprings = Population(self._params, init_size=2*self.pop.size, name='offsprings')
    off_genomes = []
    off_ancestors = []
    off_parents = []
    for agent in self.pop:  # Generate 2 offsprings from each parent
      off_genomes.append(self.mutate_genome(agent['genome']))
      off_genomes.append(self.mutate_genome(agent['genome']))
      off_ancestors.append(agent['ancestor'] if agent['ancestor'] is not None else agent['id'])
      off_ancestors.append(agent['ancestor'] if agent['ancestor'] is not None else agent['id'])
      off_parents.append(agent['id'])
      off_parents.append(agent['id'])

    off_ids = self.pop.agent_id + np.array(range(len(offsprings)))

    offsprings['genome'] = off_genomes
    offsprings['id'] = off_ids
    offsprings['ancestor'] = off_ancestors
    offsprings['parent'] = off_parents
    offsprings['born'] = [generation] * offsprings.size
    self.pop.agent_id = max(off_ids) + 1 # This saves the maximum ID reached till now
    return offsprings

  def update_pop(self, offsprings):
    """
    This function chooses the agents between the pop and the off with highest reward to create the new pop
    :param offsprings:
    :return:
    """
    performances = self.pop['reward'] + offsprings['reward']
    idx = np.argsort(performances)[::-1]  # Order idx according to performances.
    parents_off = self.pop.pop + offsprings.pop
    # Update population list by going through it and putting an agent from parents+off at its place
    for new_pop_idx, old_pop_idx in zip(range(self.pop.size), idx[:self.pop.size]):
      self.pop.pop[new_pop_idx] = parents_off[old_pop_idx]

  def should_stop(self):
    """
    Checks internal stopping criteria
    :return:
    """
    return False


class SERENE(EmitterEvolver):
  """
  This class implements the SERENE evolver. It performs NS till a reward is found, then launches fitness based emitters
  to search the reward area.
  """
  def create_emitter(self, parent_id, ns_pop, ns_off):
    """
    This function creates the emitter
    :param parent_id:
    :param ns_pop:
    :param ns_off:
    :return:
    """
    return FitnessEmitter(ancestor=self.rewarding[parent_id].copy(),
                          mutation_rate=self.calculate_init_sigma(ns_pop, ns_off, self.rewarding[parent_id]),
                          parameters=self.params)

  def candidate_emitter_eval(self, evaluate_in_env, budget_chunk, generation, pool=None):
    """
    This function does a small evaluation for the cadidate emitters to calculate their initial improvement
    :return:
    """

    candidates = self.candidates_by_novelty(pool=pool)

    for candidate in candidates:
      # Bootstrap candidates improvements
      if budget_chunk <= self.params.chunk_size/3 or self.evaluation_budget <= 0:
        break

      # Initial population evaluation
      evaluate_in_env(self.emitter_candidate[candidate].pop, pool=pool)
      self.emitter_candidate[candidate].pop['evaluated'] = list(range(self.evaluated_points,
                                                                      self.evaluated_points + self.emitter_candidate[candidate].pop.size))
      self.emitter_candidate[candidate].values.append(self.emitter_candidate[candidate].pop['reward'])

      # Update counters
      self.evaluated_points += self.params.emitter_population
      self.evaluation_budget -= self.params.emitter_population
      budget_chunk -= self.params.emitter_population
      rew_area = 'rew_area_{}'.format(self.emitter_candidate[candidate].ancestor['rew_area'])
      if rew_area not in Logger.data:
        Logger.data[rew_area] = 0
      Logger.data[rew_area] += self.params.emitter_population

      for i in range(5): # Evaluate emitter on 6 generations
        offsprings = self.emitter_candidate[candidate].generate_off(generation)
        evaluate_in_env(offsprings, pool=pool)

        offsprings['evaluated'] = list(range(self.evaluated_points, self.evaluated_points + offsprings.size))

        self.emitter_candidate[candidate].update_pop(offsprings)
        self.emitter_candidate[candidate].values.append(self.emitter_candidate[candidate].pop['reward'])
        self.update_reward_archive(generation, self.emitter_candidate, candidate)

        # Update counters
        # step_count += 1
        self.emitter_candidate[candidate].steps += 1
        self.evaluated_points += offsprings.size
        self.evaluation_budget -= offsprings.size
        budget_chunk -= offsprings.size
        Logger.data[rew_area] += offsprings.size

      self.emitter_candidate[candidate].estimate_improvement()

      # Add to emitters list
      if self.emitter_candidate[candidate].improvement > 0:
        self.emitters[candidate] = copy.deepcopy(self.emitter_candidate[candidate])
      del self.emitter_candidate[candidate]
    return budget_chunk

  def emitter_step(self, evaluate_in_env, generation, ns_pop, ns_off, budget_chunk, pool=None):
    """
    This function performs the steps for the CMA-ES emitters
    :param evaluate_in_env: Function used to evaluate the agents in the environment
    :param generation: Generation at which the process is
    :param ns_pop: novelty search population
    :param ns_off: novelty search offsprings
    :param budget_chunk: budget chunk to allocate to search
    :param pool: Multiprocessing pool
    :return:
    """
    budget_chunk = self.candidate_emitter_eval(evaluate_in_env, budget_chunk, generation, pool)

    ns_reference_set = self.get_novelty_ref_set(ns_pop, ns_off)

    while self.emitters and budget_chunk > 0 and self.evaluation_budget > 0: # Till we have emitters or computation budget
      emitter_idx = self.choose_emitter()

      # Calculate parent novelty
      self.emitters[emitter_idx].ancestor['novelty'] = utils.calculate_novelties([self.emitters[emitter_idx].ancestor['bd']],
                                                                                 ns_reference_set,
                                                                                 distance_metric=self.params.novelty_distance_metric,
                                                                                 novelty_neighs=self.params.novelty_neighs,
                                                                                 pool=pool)[0]

      print("Emitter: {} - Improv: {}".format(emitter_idx, self.emitters[emitter_idx].improvement))
      rew_area = 'rew_area_{}'.format(self.emitters[emitter_idx].ancestor['rew_area'])

      # The emitter evaluation cycle breaks every X steps to choose a new emitter
      # ---------------------------------------------------
      while budget_chunk > 0 and self.evaluation_budget > 0:
        offsprings = self.emitters[emitter_idx].generate_off(generation)
        evaluate_in_env(offsprings, pool=pool)

        offsprings['evaluated'] = list(range(self.evaluated_points, self.evaluated_points + offsprings.size))

        self.emitters[emitter_idx].update_pop(offsprings)
        self.emitters[emitter_idx].values.append(self.emitters[emitter_idx].pop['reward'])
        self.update_reward_archive(generation, self.emitters, emitter_idx)

        # Now calculate novelties and update most novel
        self.update_emitter_novelties(ns_ref_set=ns_reference_set, ns_pop=ns_pop, emitter_idx=emitter_idx, pool=pool)

        # Update counters
        # step_count += 1
        self.emitters[emitter_idx].steps += 1
        self.evaluated_points += offsprings.size
        self.evaluation_budget -= offsprings.size
        budget_chunk -= offsprings.size
        Logger.data[rew_area] += offsprings.size

        if self.check_stopping_criteria(emitter_idx): # Only if emitter is finished
          self.emitters_data[int(emitter_idx)] = {'generation': generation,
                                                  'steps': self.emitters[emitter_idx].steps,
                                                  'rewards': self.emitters[emitter_idx].values,
                                                  'archived': self.emitters[emitter_idx].archived}

          self.archive_candidates[emitter_idx] = copy.deepcopy(self.emitters[emitter_idx].ns_arch_candidates)
          # Store parent once the emitter is finished
          self.rew_archive.store(self.emitters[emitter_idx].ancestor)
          print("Stopped after {} steps\n".format(self.emitters[emitter_idx].steps))
          del self.emitters[emitter_idx]
          break
        # ---------------------------------------------------
      # This is done only if the emitter still exists
      if emitter_idx in self.emitters:
        self.emitters[emitter_idx].estimate_improvement()
