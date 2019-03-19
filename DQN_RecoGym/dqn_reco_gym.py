import sys
sys.dont_write_bytecode = True

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


import numpy as np
import gym
import tensorflow as tf

from src.agents import DDPG_Agent, Random_Agent
from src.networks import ActorNetwork, CriticNetwork
from src.memory import Memory
from src.explorationnoise import UniformNoise
from src.environment import Environment


########################################################################
########################################################################

# Name of the OpenAI Gym Environment
PROBLEM = 'CartpoleDownwardsCont-v0'

############################# Hyperparameter ###########################
# Learning rate of the actor
ACTOR_LEARNING_RATE = 0.0001

# Learning rate of critic
CRITIC_LEARNING_RATE = 0.001

# Batch size
BATCH_SIZE = 64

# Maximum number of episodes
MAX_EPISODES = 100000

# warmup steps.
WARMUP_STEPS = 10000

# Exploration duration
EXPLORATION_EPISODES = 500

# Discount factor
GAMMA = 0.99

# Soft target update parameter
TAU = 0.001

# Size of replay buffer
BUFFER_SIZE = 1000000



########################################################################
########################################################################


def main(_):
    # Create a tensorflow session
    with tf.Session() as sess:

        # Create the environment
        environment = Environment(problem = PROBLEM, max_episodes = MAX_EPISODES, warm_up_steps = WARMUP_STEPS, rendering = True)

        # Transfer the dimensions of the state space and the action space into local variables
        state_dim = environment.env.observation_space.shape
        action_dim = 1

        # Set the action boundary
        action_boundary = 10.0

        # Model the network architecture for actor
        actor = ActorNetwork(sess, state_dim, action_dim, action_boundary, batch_size = BATCH_SIZE,
                             learning_rate = ACTOR_LEARNING_RATE, tau = TAU)

        # Model the network architecture for critic
        critic = CriticNetwork(sess, state_dim, action_dim, action_boundary,
                               actor.get_num_trainable_vars(), learning_rate = CRITIC_LEARNING_RATE,
                               tau = TAU)


        # Initialize the memory
        memory = Memory(capacity = BUFFER_SIZE)

        # Initialize the exploration noise
        noise = UniformNoise(bound = 20, n_steps_annealing = EXPLORATION_EPISODES)

        # Create the agents
        agent = DDPG_Agent(actor, critic, memory, action_boundary, GAMMA, BATCH_SIZE, noise = noise)
        random_agent = Random_Agent(memory, action_boundary)

        # Create the environment
        environment.solve_environment(sess, agent, random_agent)



if __name__ == '__main__':
    tf.app.run()
