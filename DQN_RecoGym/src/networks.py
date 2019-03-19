import sys
sys.dont_write_bytecode = True

import tensorflow as tf

from .layers import *
from .utils import lazy_property

class Double_Duel_DQN(object):
    def __init__(self, sess, state_dim, action_dim, batch_size, learning_rate, tau = 0.001):
        # Inputs:
        # sess = the tensorflow session
        # state_dim = dimension of the state
        # action_dim = dimension of the action(s)
        # learning_rate = factor to update the weights within the neural network
        # tau = rate to softly update the target network
        #
        # Description:
        # - Create the actor network
        # - Create the target-actor network
        # - Build the optimizer
        # - Function to train the network
        # - Function to update the target network
        # - Function to predict outputs

        # Transfer variables to the class
        self.sess = sessy
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Create the target actor and the online actor neural networks
        self._create_networks

        # Create the optimization operations for the online actor neural network and the soft update for the target network
        self._create_optimization_operations



    @lazy_property
    def _create_networks(self):
        # Inputs:
        # None
        #
        # Outputs:
        # None
        #
        # Description:
        # - Build the online DQN
        # - Get all trainable variables from the online DQN
        # - Build the target DQN
        # - Get all trainable variables from the target DQN, which also appear in the online DQN

        # Create the online DQN
        self.state, self.actions, self.training_phase, self.Q_pred, self.action_pred = self._build_network()

        # Get all trainable parameters from the online DQN
        self.net_params = tf.trainable_variables()

        # Create the target DQN
        self.target_state, self.target_actions, self.target_training_phase, self.target_Q_pred, self.target_action_pred = self._build_network()

        # Get all trainable parameters from the target DQN
        self.target_net_params = tf.trainable_variables()[len(self.net_params):]


    def _build_network(self):
        # Inputs:
        # None
        #
        # Outputs:
        # - Inputs of the neural network
        # - Training or Inference phase
        # - Outputs of the neural network
        # - Scaled outputs of the neural network
        #
        # Description:
        # - Define the inputs of the DQN
        # - Define the DQN architecture
        # - Create the outputs

        # Defining first the phase of the neural network
        # Inference or training?
        # If training, then the variable is true
        training_phase = tf.placeholder(tf.bool)

        # Starting with the inputs for the DQN
        # Dimensions: [batch_size, state_dim]
        state = tf.placeholder(tf.float32, shape=(None,) + self.state_dim)


        # Create the first hidden layer
        # Activation function = ReLU
        # Weight Initializer = Variance Scaling
        # Weight Regularizer = L2 with 0.001
        # Biases Initializer = Zeros
        # Biases Regularizer = L2 with 0.001
        net = fully_connected(state, 512)

        # Create the second hidden layer
        # Activation function = ReLU
        # Weight Initializer = Variance Scaling
        # Weight Regularizer = L2 with 0.001
        # Biases Initializer = Zeros
        # Biases Regularizer = L2 with 0.001
        net = fully_connected(net, 512)

        # Create two streams
        # First stream estimates the value of the state V(s)
        value_stream = fully_connected(net, 512)
        value_stream = fully_connected(value_stream, 1, activation_fn = None)

        # Second stream estimates the advantage of the actions A(s, a)
        advantage_stream = fully_connected(net, 512)
        advantage_stream = fully_connected(advantage_stream, self.action_dim, activation_fn = None)

        # Now aggregating the layers
        # Q(s, a) = V(s) - (A(s, a) - 1/|A| * sum A(s, a'))
        Q = value_stream + tf.substract(advantage_stream, tf.reduce_mean(advantage_stream, axis = 1, keepdims = True))

        # Get the action with the highest Q-value
        exploit_action = tf.argmax(Q, axis = 1)

        return state, training_phase, Q, exploit_action

    @lazy_property
    def _create_optimization_operations(self):
        # Inputs:
        # None
        #
        # Outputs:
        # None
        #
        # Description:
        # - Create soft-update operation for the target network
        # - Get the action gradients
        # - Normalize actor gradients
        # - Create optimization operation for the online actor neural network

        # Create the operation to soft-update the target network with the online network parameters
        self.update_target_net_params = [self.target_net_params[i].assign(
                                         tf.multiply(self.net_params[i], self.tau) +                # Online DQN Weights * tau
                                         tf.multiply(self.target_net_params[i], 1. - self.tau))     # Target DQN Weights * (1 - tau)
                                         for i in range(len(self.target_net_params))
                                         ]

        # Define placeholder for the true Q-Value
        self.true_Q = tf.placeholder(tf.float32, [None])

        # Calculate the loss: mean((Q_gt - Q_pred)^2)
        self.loss = tf.reduce_mean(tf.square(self.true_Q - self.Q_pred))

        # Create the optimization operation for the DQN
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Create class variable to count the number of trainable parameters
        self.num_trainable_vars = len(self.net_params) + len(self.target_net_params)

    def train(self, state):
        # Inputs:
        # state = state to feed into the DQN
        #
        # Outputs:
        # One optimization iteration
        #
        # Description:
        # - Run the tensorflow session for the DQN optimizer

        return self.sess.run(self.optimize,
                             feed_dict = {self.state: state,
                                          self.training_phase: True
                                        }
                            )

    def predict_action(self, state):
        # Inputs:
        # state = state to feed into the actor neural network
        #
        # Outputs:
        # Get the action prediction of the current state
        #
        # Description:
        # - Run the tensorflow session to predict the action prediction of the target DQN

        return self.sess.run(self.action_pred,
                             feed_dict = {self.state: state,
                                          self.training_phase: False
                                        }
                            )

    def predict_Q_target(self, state):
        # Inputs:
        # state = state to feed into the actor neural network
        #
        # Outputs:
        # Get the action prediction of the current state from the target DQN
        #
        # Description:
        # - Run the tensorflow session to predict the action prediction of the target DQN

        return self.sess.run(self.target_Q_pred,
                             feed_dict = {self.target_inputs: state,
                                          self.target_training_phase: False
                                        }
                            )

    def update_target_network(self):
        # Inputs:
        # None
        #
        # Outputs:
        # None
        #
        # Description:
        # - Perform one soft-update from the online DQN to the target DQN

        self.sess.run(self.update_target_net_params)

    def get_num_trainable_vars(self):
        # Inputs:
        # None
        #
        # Outputs:
        # Number of trainable variables of both networks
        #
        # Description:
        # - Return the number of trainable variables from online and target DQN

        return self.num_trainable_vars



