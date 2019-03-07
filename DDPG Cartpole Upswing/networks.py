import sys
sys.dont_write_bytecode = True

import tensorflow as tf

from layers import *
from utils import lazy_property

class ActorNetwork(object):
    def __init__(self, sess, state_dim, action_dim, action_boundary, batch_size, learning_rate, tau = 0.001):
        # Inputs:
        # sess = the tensorflow session
        # state_dim = dimension of the state
        # action_dim = dimension of the action(s)
        # action_boundary = boundary of the action space
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
        self.sess = sess
        self.action_boundary = action_boundary
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
        # - Build the online actor neural network
        # - Get all trainable variables from the online actor network
        # - Build the target actor neural network
        # - Get all trainable variables from the target actor network, which also appear in the online actor network

        # Create the actor network
        self.inputs, self.training_phase, self.outputs, self.scaled_outputs = self._build_network()

        # Get all trainable parameters from the actor network
        self.net_params = tf.trainable_variables()

        # Create the target actor network
        self.target_inputs, self.target_training_phase, self.target_outputs, self.target_scaled_outputs = self._build_network()

        # Get all trainable parameters from the target actor network
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
        # - Define the inputs of the actor network
        # - Define the actor network architecture
        # - Create the action outputs

        # Defining first the phase of the neural network
        # Inference or training?
        # If training, then the variable is true
        training_phase = tf.placeholder(tf.bool)

        # Starting with the input for the actor network
        # It is only the state
        # Dimensions: [batch_size, state_dim]
        state = tf.placeholder(tf.float32, shape=(None,) + self.state_dim)

        # Create the first hidden layer
        # Activation function = ReLU
        # Weight Initializer = Variance Scaling
        # Weight Regularizer = L2 with 0.001
        # Biases Initializer = Zeros
        # Biases Regularizer = L2 with 0.001
        net = fully_connected(state, 400)

        # Create the second hidden layer
        net = fully_connected(net, 300)

        # Create the output layer
        # Activation function = tanh
        # Weight Initializer = Uniform from -0.003 to 0.003 (does not get into the saturation area of tanh)
        outputs = fully_connected(net, self.action_dim, activation_fn = tf.tanh, weights_initializer = tf.random_uniform_initializer(-3e-3, 3e-3))

        # tanh is bounded between -1 and 1
        # Scaling the output between its boundaries
        scaled_outputs = tf.multiply(outputs, self.action_boundary)

        return state, training_phase, outputs, scaled_outputs

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
                                         tf.multiply(self.net_params[i], self.tau) +
                                         tf.multiply(self.target_net_params[i], 1. - self.tau))
                                         for i in range(len(self.target_net_params))
                                         ]

        # Combine d_netScaledOut/d_netParams with Critic2Action gradient to get actor gradient
        # Include a placeholder for the action gradient
        # Dimension = [batch_size, action_dim]
        self.action_gradients = tf.placeholder(tf.float32, [None, self.action_dim])

        # Get the actor gradient without normalization
        self.not_normalized_actor_gradient = tf.gradients(self.outputs, self.net_params, -self.action_gradients)

        # Normalizing the actor gradient by the batch_size
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.not_normalized_actor_gradient))

        # Create the optimization operation for the online actor
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).apply_gradients(zip(self.actor_gradients, self.net_params))

        # Create class variable to count the number of trainable parameters
        self.num_trainable_vars = len(self.net_params) + len(self.target_net_params)

    def train(self, state, action_gradients):
        # Inputs:
        # state = state to feed into the actor neural network
        # action_gradients = gradients to optimize the actor network
        #
        # Outputs:
        # One optimization iteration
        #
        # Description:
        # - Run the tensorflow session for the actor optimizer

        return self.sess.run(self.optimize,
                             feed_dict = {self.inputs: state,
                                          self.action_gradients: action_gradients,
                                          self.training_phase: True
                                        }
                            )

    def predict(self, state):
        # Inputs:
        # state = state to feed into the actor neural network
        #
        # Outputs:
        # Scaled action prediction from the online actor neural network
        #
        # Description:
        # - Run the tensorflow session to predict the scaled action to the input-state of the online actor neural network

        return self.sess.run(self.scaled_outputs,
                             feed_dict = {self.inputs: state,
                                          self.training_phase: False
                                        }
                            )

    def predict_target(self, state):
        # Inputs:
        # state = state to feed into the actor neural network
        #
        # Outputs:
        # Scaled action prediction from the target actor neural network
        #
        # Description:
        # - Run the tensorflow session to predict the scaled action to the input-state of the target actor neural network

        return self.sess.run(self.target_scaled_outputs,
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
        # - Perform one soft-update from the online actor network to the target actor network

        self.sess.run(self.update_target_net_params)

    def get_num_trainable_vars(self):
        # Inputs:
        # None
        #
        # Outputs:
        # Number of trainable variables of both networks
        #
        # Description:
        # - Return the number of trainable variables from online and target actor neural network

        return self.num_trainable_vars


class CriticNetwork(object):
    def __init__(self, sess, state_dim, action_dim, action_boundary, num_actor_vars, learning_rate, tau = 0.001):
        # Inputs:
        # sess = the tensorflow session
        # state_dim = dimension of the state
        # action_dim = dimension of the action(s)
        # action_boundary = boundary of the action space
        # num_actor_vars = number of variables from the actor
        # learning_rate = factor to update the weights within the neural network
        # tau = rate to softly update the target network
        #
        # Description:
        # - Create the critic network
        # - Create the target-critic network
        # - Build the optimizer
        # - Function to train the network
        # - Function to update the target network
        # - Function to predict outputs

        # Transfer variables to the class
        self.sess = sess
        self.action_boundary = action_boundary
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.tau = tau
        self.num_actor_vars = num_actor_vars
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
        # - Build the online actor neural network
        # - Get all trainable variables from the online actor network
        # - Build the target actor neural network
        # - Get all trainable variables from the target actor network, which also appear in the online actor network

        # Create the actor network
        self.inputs, self.training_phase, self.actions, self.output = self._build_network()

        # Get all trainable parameters from the actor network
        self.net_params = tf.trainable_variables()[self.num_actor_vars:]

        # Create the target actor network
        self.target_inputs, self.target_training_phase, self.target_actions, self.target_output = self._build_network()

        # Get all trainable parameters from the target actor network
        self.target_net_params = tf.trainable_variables()[len(self.net_params) + self.num_actor_vars:]


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
        # - Define the inputs of the actor network
        # - Define the critic network architecture
        # - Create the critic output (Q-value)

        # Defining first the phase of the neural network
        # Inference or training?
        # If training, then the variable is true
        training_phase = tf.placeholder(tf.bool)

        # Starting with the input for the critic network
        # First the state
        # Dimensions: [batch_size, state_dim]
        state = tf.placeholder(tf.float32, shape = (None,) + self.state_dim)

        # Then the action
        # Dimensions: [batch_size, action_dim]
        action = tf.placeholder(tf.float32, shape = [None, self.action_dim])


        # Create the first hidden layer
        # Activation function = ReLU
        # Weight Initializer = Variance Scaling
        # Weight Regularizer = L2 with 0.001
        # Biases Initializer = Zeros
        # Biases Regularizer = L2 with 0.001
        net = fully_connected(state, 400)

        # Create the second hidden layer
        net = fully_connected(tf.concat([net, action], 1), 300)

        # Create the output layer
        # Activation function = linear
        # Weight Initializer = Uniform from -0.003 to 0.003 (does not get into the saturation area of tanh)
        output = fully_connected(net, 1, activation_fn = None, weights_initializer = tf.random_uniform_initializer(-3e-3, 3e-3))


        return state, training_phase, action, output

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
        # - Create optimization operation for the online critic neural network
        # - Compute the gradients of the online critic neural network with respect to the actions

        # Create the operation to soft-update the target network with the online network parameters
        self.update_target_net_params = [self.target_net_params[i].assign(
                                         tf.multiply(self.net_params[i], self.tau) +
                                         tf.multiply(self.target_net_params[i], 1. - self.tau))
                                         for i in range(len(self.target_net_params))
                                         ]


        # Placeholder for the true Q-Values of the state
        self.q_value = tf.placeholder(tf.float32, [None, 1])

        # Define the loss
        # Here L2 of true_q_value and predicted_q_value
        self.loss = tf.reduce_mean(tf.squared_difference(self.q_value, self.output))

        # Create the optimization operation for the online critic neural network
        self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        # Get the gradient of the online critic neural network with respect to the actions
        self.action_grads = tf.gradients(self.output, self.actions)


    def train(self, state, actions, q_value):
        # Inputs:
        # state = state to feed into the critic neural network
        # actions = actions taken by the agent
        # q_value = true qualitiy value (Q-value) of the state
        #
        # Outputs:
        # One optimization iteration
        #
        # Description:
        # - Run the tensorflow session for the critic optimizer
        # - Return the prediction

        return self.sess.run([self.output, self.optimize],
                             feed_dict = {self.inputs: state,
                                          self.actions: actions,
                                          self.q_value: q_value,
                                          self.training_phase: True
                                        }
                            )

    def predict(self, state, actions):
        # Inputs:
        # state = state to feed into the critic neural network
        # actions = actions taken by the actor
        #
        # Outputs:
        # Predicted Q-value of the state and the action
        #
        # Description:
        # - Run the tensorflow session to predict the Q-value to the input-state and the taken action of the online critic neural network

        return self.sess.run(self.output,
                             feed_dict = {self.inputs: state,
                                          self.actions: actions,
                                          self.training_phase: False
                                        }
                            )

    def predict_target(self, state, actions):
        # Inputs:
        # state = state to feed into the critic neural network
        # actions = actions taken by the actor
        #
        # Outputs:
        # Predicted Q-value of the state and the action of the target critic neural network
        #
        # Description:
        # - Run the tensorflow session to predict the scaled action to the input-state of the target actor neural network

        return self.sess.run(self.target_output,
                             feed_dict = {self.target_inputs: state,
                                          self.target_actions: actions,
                                          self.target_training_phase: False
                                        }
                            )

    def action_gradients(self, state, actions):
        # Inputs:
        # state = state to feed into the critic neural network
        # actions = actions taken by the actor
        #
        # Outputs:
        # Gradients of the online critic neural network with respect to the taken action
        #
        # Description:
        # - Run the tensorflow session to calculate the gradient of the critic neural network with respect to the taken action

        return self.sess.run(self.action_grads,
                             feed_dict = {self.inputs: state,
                                          self.actions: actions,
                                          self.training_phase: False
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
        # - Perform one soft-update from the online actor network to the target critic network

        self.sess.run(self.update_target_net_params)


