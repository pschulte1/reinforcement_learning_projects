import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import gym
import My_Environments
from collections import deque
import numpy as np
import tensorflow as tf

from reco_gym import env_1_args


class Environment(object):
    def __init__(self, problem, max_episodes = 1e6, warm_up_steps = None, rendering = True):
        # Inputs:
        # problem = string of the name of the environment
        # max_episodes = maximum number of episodes
        # warm_up_steps = number of steps to fill the shared memory with the random agent
        # rendering = Whether rendering of the environment should be activated or not
        #
        # Description:
        # - Create the environment
        # - run the random agent as long as the number of warm-up-steps in the environment
        # - train the agent in the environment

        # Create the environment
        self.env = gym.make(problem)

        # Set the random seed for reproducability
        env_1_args['random_seed'] = 1234

        # Set the reco gym args
        self.env.init_gym(env_1_args)

        # Transfer the variables to the class
        self.max_episodes = max_episodes
        self.warm_up_steps = warm_up_steps
        self.rendering = rendering

    def solve_environment(self, sess, agent, random_agent = None):
        # Inputs:
        # agent = (learnable) agent which acts in the environment
        # random_agent = agent which randomly acts in the environment
        #
        # Outputs:
        # None
        #
        # Description:
        # - Initialize all variables of the neural networks
        # - Run first the random agent as long as the warm-up-steps
        # - Then train the (trainable) agent for the environment

        # Run tensorflow session to initialize all variables
        sess.run(tf.global_variables_initializer())

        # Check first that the warm-up steps are not None and the random-agent is not None
        if self.warm_up_steps is not None and random_agent is not None:

            # If there is a random-agent...
            # Printing...
            print("---------------------------------------------------")
            print("===============Start Warm-Up Phase=================")
            print("---------------------------------------------------")

            # Initialize the warm-up episode counter
            warm_up_episode = 0

            # As long as the steps that the random-agent did so far are lower than the warm-up steps
            while random_agent.steps < self.warm_up_steps:

                # Run one episode with the random agent and get the episode reward
                episode_reward = self.run_one_episode(random_agent)

                # Increment the counter
                warm_up_episode += 1

                # Printing the current warm-up episode and the corresponding episode reward
                print("[WARM-UP EPISODE {0}]\t\tEpisode Reward = {1:.2f}".format(warm_up_episode, episode_reward))


        # Printing that the training starts now...
        print("---------------------------------------------------")
        print("==================Start Training===================")
        print("---------------------------------------------------")

        # Initialize an empty array with fixed length of 100 for further calculations with the reward
        rewards = deque(maxlen = 100)

        # Start training of the agent in the environment as long as the maximum episodes variable
        for i in range(self.max_episodes):

            # Perform one episode of the trainable agent within the environment and get the episode reward
            episode_reward = self.run_one_episode(agent)

            # Add the episode reward to the array
            rewards.append(episode_reward)

            # Printing the current episode and the corresponding episode reward
            print("[EPISODE {0}]\t\tEpisode Reward = {1:.2f}".format(i, episode_reward))

            # Every 100 iterations...
            if i % 100 == 0 and not i == 0:

                # Printing the mean reward of the last 100 episodes
                print("---------------------------------------------------")
                print("[MEAN OVER LAST 100 EPISODES] = {0:.2f}".format(np.mean(rewards)))
                print("---------------------------------------------------")

            # Increase the step of the exploration noise every episode
            # Check first that the noise ist not None
            if agent.noise is not None:

                # Set the step to the current episode
                agent.noise.step = i




    # Define function to preprocess the state from the environemnt (OpenAI Gym)
    def preprocess_state(self, state):
        # Inputs:
        # state = current state of the agent in the environment
        #
        # Outputs:
        # The preprocessed state of the agent in the environment
        #
        # Description:
        # To

        return state

    def run_one_episode(self, agent):
        # Inputs:
        # agent = chosen agent to run one episode
        #
        # Outputs:
        # Accumulated reward over the episode
        #
        # Description:
        # - Resetting the environment
        # - Let the agent act in the environment
        # - Save the observations in the memory
        # - Replay experience

        # Get the initial state and preprocess it
        state = self.preprocess_state(self.env.reset())

        # Initialize accumulated reward
        accu_reward = 0

        # Run the episode
        while True:

            # If rendering is true...
            if self.rendering:

                # Render the current state
                self.env.render()

            # Get the action for the current state
            action = agent.act(state)

            # Perform the action in the environment
            next_state, reward, terminal, _ = self.env.step(action)

            # Transfer the observation to the agent
            agent.observe(state, action, reward, terminal, next_state)

            # Let the agent learn from its experience
            agent.replay()

            # The next state will be the current state in the next iteration
            state = next_state

            # Add the current reward to the accumulated reward
            accu_reward += reward

            # If the terminal state is reached...
            if terminal:

                # Ending the episode
                break

        # Return the accumulated reward
        return accu_reward












