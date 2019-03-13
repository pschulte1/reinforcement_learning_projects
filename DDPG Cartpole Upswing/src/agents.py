import sys
sys.dont_write_bytecode = True

import tensorflow as tf
import numpy as np
import random

from .explorationnoise import UniformNoise

class DDPG_Agent(object):
    def __init__(self, actor, critic, memory, action_boundary, gamma, batch_size, noise = None):
        # Inputs:
        # actor = neural network which predicts the action of the agent
        # critic = neural network which predicts the reward/future reward of the action
        # memory = shared memory of observations
        # action_boundary = boundary of the action space
        # gamma = discount factor for future rewards
        # batch_size = size of one minibatch
        # noise = class of an exploration noise
        #
        # Description:
        # - Creating an agent based on the DDPG algorithm
        # - Consisting of an actor neural network which predicts the action for the agent
        #   and of an critic neural network which predicts the quality (Q-value) of the action

        # Transfer all variables to the class
        self.actor = actor
        self.critic = critic
        self.gamma = gamma
        self.batch_size = batch_size
        self.noise = noise
        self.memory = memory
        self.action_boundary = action_boundary

        # Initialize number of steps of this agent with 0
        self.steps = 0

    # Define function to preprocess one batch
    def preprocess_batch(self, batch, size):
        # Input:
        # batch = batch to be reshaped
        # size = size of the batch ([batch_size, SIZE])
        #
        # Output:
        # Reshaped batch in the format: [batch_size, SIZE]
        #
        # Description:
        # Reshape one batch before feeding it into the neural network

        return np.reshape(batch, (self.batch_size, size))

    # Define function to force the agent to act
    def act(self, state):
        # Input:
        # state = current state of the agent
        #
        # Output:
        # Cliped output of the actor network
        #
        # Description:
        # Get prediction of the actor of the state and clip it by the action boundary

        # Get the prediction of the actor
        pred = self.actor.predict(np.expand_dims(state, 0))[0]

        # As long there is a noise function in the class...
        if self.noise is not None:

            # Generate noise
            noise = self.noise.generate()

            # Add the noise to the prediction, clip the sum by the action boundary and return it
            return np.clip(pred + noise, -self.action_boundary, self.action_boundary)[0]

        # If there is not a noise function...
        else:
            # Clip the prediction by the action boundary and return it
            return np.clip(pred, -self.action_boundary, self.action_boundary)[0]

    # Define function to let the agent learn
    def replay(self):
        # Input:
        # None
        #
        # Output:
        # Mean predicted Q-value of one minibatch
        #
        # Description:
        # - Sample one random batch from the memory
        # - Calculate estimated reward of the next state
        # - Train the critic network
        # - Get the actor gradients of the batch from the critic network
        # - Train the actor by the gradients

        # Sample one batch of the memory
        state_batch, action_batch, reward_batch, terminal_batch, next_state_batch = self.memory.sample_batch(self.batch_size)

        # Predict the action of the next state by the target-actor and predict afterwards the Q-value with this action and the next state by the target-critic network
        target_q = self.critic.predict_target(next_state_batch, self.actor.predict_target(next_state_batch))

        # If the next state is a terminal state, do NOT add the future reward, else add the future reward to the current reward by discounting it with gamma
        # Formular:
        # If NOT terminal state: y_i = reward_s + gamma*reward_(s+1)
        # If terminal state: y_i = reward_s
        y_i = self.preprocess_batch(reward_batch, 1) + (1 - self.preprocess_batch(terminal_batch, 1)).astype(float) * self.gamma * self.preprocess_batch(target_q, 1)

        # Train the critic network with the current state and the current action of the memory and return the predicted Q-values of the batch
        predicted_q_values, _ = self.critic.train(state_batch, self.preprocess_batch(action_batch, 1), y_i)

        # Predict the actor output (action)
        actor_outputs = self.actor.predict(state_batch)

        # Get the gradient for the actor from the critic network
        actor_grads = self.critic.action_gradients(state_batch, actor_outputs)

        # Train the actor by its gradients
        self.actor.train(state_batch, actor_grads[0])

        # Return the mean of the predicted Q-values
        return np.mean(predicted_q_values)

    # Define function to observe the current action
    def observe(self, state, action, reward, terminal, next_state):
        # Inputs:
        # state = current state of the agent
        # action = chosen action of the agent
        # reward = reward of the action
        # terminal = reached terminal state?
        # next_state = state after the action
        #
        # Outputs:
        # None
        #
        # Description:
        # - Add the current observation to the memory
        # - Update target networks

        # Add the current observation to the memory
        self.memory.add(state, action, reward, terminal, next_state)

        # Increase the number of the steps taken by this agent
        self.steps += 1

        # Update the target-actor network
        self.actor.update_target_network()

        # Update the target-critic network
        self.critic.update_target_network()

class Random_Agent(object):
    def __init__(self, memory, action_boundary):
        # Inputs:
        # memory = shared memory of observations
        # action_boundary = upper and lower bound of the action space
        #
        # Description:
        # Create an agent which will only perform uniformly selected actions

        # Transfer all variables to the class
        self.memory = memory
        self.boundary = action_boundary

        # Initialize the number of steps of this agent
        self.steps = 0

    # Define function to uniformly select an action
    def act(self, state):
        # Inputs:
        # state = negligible input
        #
        # Outputs:
        # Uniformly selected action with U(a,b); a = -boundary and b = boundary
        #
        # Description:
        # While doing warm-up-steps (fill the memory), select only random actions
        return random.uniform(-self.boundary, self.boundary)

    def observe(self, state, action, reward, terminal, next_state):
        self.memory.add(state, action, reward, terminal, next_state)
        self.steps += 1

    def replay(self):
        pass


