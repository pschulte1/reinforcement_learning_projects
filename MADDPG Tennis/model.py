import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    """
    # Inputs:
    # layer = layer to be initialized
    #
    # Output:
    # Boundaries of the uniform distribution
    #
    # Description:
    # - Get the number of inputs
    # - Calculate the limits
    """

    # Get the number of inputs
    fan_in = layer.weight.data.size()[0]

    # Scale the limits by the square roots
    lim = 1. / np.sqrt(fan_in)

    # Return the limits
    return (-lim, lim)


class Actor(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        super(Actor, self).__init__()
        """
        # Inputs:
        # state_size = dimensions of the state
        # action_size = dimensions of the action
        # seed = random seed for reproduceability
        # fc1_units = number of neurons in the first hidden layer
        # fc2_units = number of neurons in the second hidden layer
        #
        # Description:
        # - Create all operations for the actor neural network
        # - Define function for the forward pass
        """

        # Set the random seed in pytorch
        self.seed = torch.manual_seed(seed)

        # Create the first hidden layer
        self.fc1 = nn.Linear(state_size, fc1_units)

        # Create the second hidden layer
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        # Create the output layer
        self.fc3 = nn.Linear(fc2_units, action_size)

        # Create one batch normalization layer
        self.bn1 = nn.BatchNorm1d(fc1_units)

        # Initialize the weights
        self.reset_parameters()

    def reset_parameters(self):
        """
        # Inputs:
        # None
        #
        # Outputs:
        # None
        #
        # Description:
        # - Initialize the hidden layer with a FAN-IN uniform method
        # - Initialize the output layer (tanh) between -0.003 and 0.003
        """

        # Initialize the first fully connected hidden layer
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))

        # Initialize the second fully connected hidden layer
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))

        # Initialize the output fully connected layer
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """
        # Inputs:
        # state = state to predict the action for
        #
        # Outputs:
        # Predicted output actions to the corresponding input state
        #
        # Description:
        # - Feeding the input state into the actor neural network
        # - Perform all operations
        # - Return the predicted actions
        """

        # Check if the state dimension is equal to 1...
        if state.dim() == 1:

            # If state dimension == 1 --> reshape state for batch normalization
            state = torch.unsqueeze(state,0)

        # Feed state into the neural network (first hidden layer + non-linearity)
        x = F.relu(self.fc1(state))

        # Feed output of the first hidden layer into the batch normalization
        x = self.bn1(x)

        # Feed output of the batch normalization into the second hidden layer + non-linearity
        x = F.relu(self.fc2(x))

        # Return the predicted action output (between -1 and 1)
        return F.tanh(self.fc3(x))


class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=400, fc2_units=300):
        super(Critic, self).__init__()
        """
        # Inputs:
        # state_size = dimension of the state
        # action_size = dimension of the actions
        # seed = random seed for reproduceability
        # fc1_units = number of neurons in the first hidden layer
        # fc2_units = number of neurons in the second hidden layer
        #
        # Description:
        # - Create all operations for the critic neural network
        # - Define function for the forward pass
        """

        # Set the random seed in pytorch
        self.seed = torch.manual_seed(seed)

        # Define the first hidden layer
        self.fc1 = nn.Linear(state_size+action_size, fc1_units)

        # Define the second hidden layer
        self.fc2 = nn.Linear(fc1_units, fc2_units)

        # Define the output layer
        self.fc3 = nn.Linear(fc2_units, 1)

        # Define batch normalization layer
        self.bn1 = nn.BatchNorm1d(fc1_units)

        # Initialize the weights
        self.reset_parameters()

    def reset_parameters(self):
        """
        # Inputs:
        # None
        #
        # Outputs:
        # None
        #
        # Description:
        # - Initialize the hidden layer with a FAN-IN uniform method
        # - Initialize the output layer (tanh) between -0.003 and 0.003
        """

        # Initialize the first fully connected hidden layer
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))

        # Initialize the second fully connected hidden layer
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))

        # Initialize the output fully connected layer
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, actions):
        """
        # Inputs:
        # state = state to feed into the critic neural network
        # actions = actions to feed into the critic neural network
        #
        # Outputs:
        # Predicted Q-Value of the state and the taken action
        #
        # Description:
        # - Feed state and actions into the critic neural network
        # - Perform all operations of the critic neural network
        """

        # If the state dimension is equal to 1...
        if state.dim() == 1:

            # Reshape the state for the batch normalization
            state = torch.unsqueeze(state,0)

        # Concatenate state and action for the input of the critic neural network
        xs = torch.cat((state, actions), dim=1)

        # Feed state + actions into the first hidden layer + non-linearity
        x = F.relu(self.fc1(xs))

        # Feed first hidden layer into the batch normalization
        x = self.bn1(x)

        # Feed the output of the batch normalization into the second hidden layer + non-linearity
        x = F.relu(self.fc2(x))

        # Feed the output of the second hidden layer into the output layer
        return self.fc3(x)
