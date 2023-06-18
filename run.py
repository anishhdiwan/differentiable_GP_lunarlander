import time

import gymnasium as gym

from genepro.node_impl import *
from genepro.evo import Evolution
from genepro.node_impl import Constant

import torch
import torch.optim as optim

import random
import os
import copy
from collections import namedtuple, deque

import matplotlib.pyplot as plt
from matplotlib import animation

from genepro.selection import tournament_selection
from genepro.variation import subtree_crossover, subtree_mutation, coeff_mutation

env = gym.make("LunarLander-v2", render_mode="rgb_array")

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def __iadd__(self, other):
        self.memory += other.memory
        return self

    def __add__(self, other):
        self.memory = self.memory + other.memory
        return self

def fitness_function_pt(multitree, num_episodes=5, episode_duration=300, render=False, ignore_done=False, frames=None):
    if frames is None:
        frames = []
    memory = ReplayMemory(10000)
    rewards = []

    for _ in range(num_episodes):
        # get initial state of the environment
        observation = env.reset()
        observation = observation[0]

        for _ in range(episode_duration):
            if render:
                frames.append(env.render())

            input_sample = torch.from_numpy(observation.reshape((1,-1))).float()

            # what goes here? TODO
            action = torch.argmax(multitree.get_output_pt(input_sample))
            observation, reward, terminated, truncated, info = env.step(action.item())
            rewards.append(reward)
            output_sample = torch.from_numpy(observation.reshape((1,-1))).float()
            memory.push(input_sample, torch.tensor([[action.item()]]), output_sample, torch.tensor([reward]))
            if (terminated or truncated) and not ignore_done:
                break

    fitness = np.sum(rewards)

    return fitness, memory


def get_test_score(tree):
    rewards = []

    for i in range(5):
        # get initial state
        observation = env.reset(seed=i)
        observation = observation[0]

        for _ in range(300):
            # build up the input sample for GP
            input_sample = torch.from_numpy(observation.reshape((1,-1))).float()
            # get output (squeezing because it is encapsulated in an array)
            output = tree.get_output_pt(input_sample)
            action = torch.argmax(tree.get_output_pt(input_sample))
            observation, reward, terminated, truncated, info = env.step(action.item())
            rewards.append(reward)


            output_sample = torch.from_numpy(observation.reshape((1,-1))).float()
            if (terminated or truncated):
                break

    fitness = np.sum(rewards)

    return fitness


def optimise_individual(individual):
    batch_size = 128
    GAMMA = 0.99

    constants = individual.get_subtrees_consts()

    if len(constants)>0:
        #                                     1e-3
        optimizer = optim.AdamW(constants, lr=1e-3, amsgrad=True)

    # 500
    for _ in range(500):

        if len(constants)>0 and len(evo.memory)>batch_size:
            target_tree = copy.deepcopy(individual)

            transitions = evo.memory.sample(batch_size)
            batch = Transition(*zip(*transitions))

            non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                    batch.next_state)), dtype=torch.bool)

            non_final_next_states = torch.cat([s for s in batch.next_state
                                               if s is not None])
            state_batch = torch.cat(batch.state)
            action_batch = torch.cat(batch.action)
            reward_batch = torch.cat(batch.reward)

            state_action_values = individual.get_output_pt(state_batch).gather(1, action_batch)
            next_state_values = torch.zeros(batch_size, dtype=torch.float)
            with torch.no_grad():
                next_state_values[non_final_mask] = target_tree.get_output_pt(non_final_next_states).max(1)[0].float()

            expected_state_action_values = (next_state_values * GAMMA) + reward_batch

            criterion = nn.SmoothL1Loss()
            loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_value_(constants, 100)
            optimizer.step()


num_features = env.observation_space.shape[0]
leaf_nodes = [Feature(i) for i in range(num_features)]
leaf_nodes = leaf_nodes + [Constant(),Constant(),Constant(),Constant(),Constant(),Constant(),Constant(),Constant()] # Think about the probability of sampling a coefficient
internal_nodes = [Plus(),Minus(),Times(),Div(), Sqrt(), Square()] #Add your own operators here

evo = Evolution(
    fitness_function_pt, internal_nodes, leaf_nodes,
    4,
    crossovers=[{"fun":subtree_crossover, "rate": 0.7}],
    mutations=[{"fun":subtree_mutation, "rate": 0.7}],
    coeff_opts=[{"fun":coeff_mutation, "rate": 0.1}],
    selection = {"fun":tournament_selection,"kwargs":{"tournament_size":16}},
    pop_size=256,
    max_gens=10,
    max_tree_size=31,
    n_jobs=16,
    verbose=True)

evo.start_time = time.time()

evo._initialize_population()

# generational loop
while not evo._must_terminate():
    # perform one generation
    print("Evolving generation")
    evo._perform_generation()

    # if evo.num_gens == 10:
    #     print("Optimizing individual")
    #     best = evo.best_of_gens[-1]
    #     best.get_readable_repr()
    #     optimise_individual(best)

    optimised_individuals = []
    for i in range(25):
        if random.random() < 0.1:
            individual = evo.population[np.argsort([t.fitness for t in evo.population])[-i]]
            copied = copy.deepcopy(individual)
            optimise_individual(copied)

            fitness = fitness_function_pt(copied)
            copied.fitness = fitness[0]
            optimised_individuals.append(copied)
            # evo.population[np.argsort([t.fitness for t in evo.population])[i]] = copied

    evo.population += optimised_individuals
    evo.population = list(sorted(evo.population, key=lambda p: p.fitness))[len(optimised_individuals):]

    if evo.verbose:
        print("gen: {},\tbest of gen fitness: {:.3f},\tbest of gen size: {}".format(
            evo.num_gens, evo.best_of_gens[-1].fitness, len(evo.best_of_gens[-1])
        ))
    print([p.fitness for p in evo.population])


besti = evo.best_of_gens[-1]

print(besti.get_readable_repr())
print(get_test_score(besti))

