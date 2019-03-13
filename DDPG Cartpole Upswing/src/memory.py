import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

from collections import deque, namedtuple
import random
import numpy as np
import random

Transition = namedtuple('Transition', ['state', 'action', 'reward', 'terminal', 'next_state'])

class Memory(object):
    def __init__(self, capacity, random_seed = 1234):
        # Inputs:
        # capacity = size of the memory (How many experiences you want to store)
        # random_seed = random seed of the random library
        #
        # Description:
        # - Create the memory of the agent
        # - Sampling a batch for training

        # Transfer the variable to the class
        self.capacity = capacity

        # Set the random seed
        random.seed(random_seed)

        # Initialize counter for the number of experiences in the memory
        self.count = 0

        # Initialize a buffer with the maximum length of the capacity
        self.buffer = deque(maxlen = self.capacity)


    # Define function to add a new observation
    def add(self, state, action, reward, terminal, next_state):
        # Inputs:
        # state = current state of the environment
        # action = performed action at the current state
        # reward = reward received by performing the action
        # terminal = information if the terminal state is reached
        # next_state = new state after the action
        #
        # Outputs:
        # None
        #
        # Description:
        # - Add all observations to the buffer
        # - Increment the counter to track the all experiences

        # Create a named tuple with all observations
        experience = Transition(state, action, reward, terminal, next_state)

        # Append the experience to the buffer (right-side)
        # INFO: If the buffer is already full, then the left-side will be poped
        self.buffer.append(experience)

        # Increment the counter --> maximum is the capacity
        self.count = min(self.capacity, self.count+1)

    # Define function to sample one batch from the buffer
    def sample_batch(self, batch_size):
        # Inputs:
        # batch_size = size of one minibatch
        #
        # Outputs:
        # A random set of experiences from the buffer with the size of 'batch_size'
        #
        # Description:
        # - Checking the size of the memory
        # - Sample one random minibatch with experiences from the buffer
        # - Return the minibatch as numpy array

        # Initialize an empty batch
        batch = []

        # If the memory is smaller than the batch-size...
        if self.count < batch_size:

            # The minibatch will be as big as the memory
            batch = random.sample(self.buffer, self.count)

        # If the memory is bigger than the batch-size...
        else:

            # The minibatch will be as big as the batch-size
            batch = random.sample(self.buffer, batch_size)

        # Return the batch as numpy array
        return map(np.array, zip(*batch))

    # Define function to clear the memory/buffer
    def clear(self):
        # Inputs:
        # None
        #
        # Outputs:
        # None
        #
        # Description:
        # - Clear all experiences from the buffer/memory
        # - Set the counter to 0

        # Clear all experiences from the buffer
        self.buffer.clear()

        # Set the counter to 0
        self.count = 0

