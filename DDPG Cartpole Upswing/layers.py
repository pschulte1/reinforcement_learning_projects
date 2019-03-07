import sys
sys.dont_write_bytecode = True
import tensorflow as tf

def fully_connected(inputs,
                    output_size,
                    activation_fn = tf.nn.relu,
                    weights_initializer = tf.contrib.layers.variance_scaling_initializer(factor = 1.0),
                    weights_regularizer = tf.contrib.layers.l2_regularizer(0.001),
                    biases_initializer = tf.zeros_initializer(),
                    biases_regularizer = tf.contrib.layers.l2_regularizer(0.001)
                    ):
    return tf.contrib.layers.fully_connected(inputs,
                                            output_size,
                                            activation_fn = activation_fn,
                                            weights_initializer = weights_initializer,
                                            weights_regularizer = weights_regularizer,
                                            biases_initializer = biases_initializer,
                                            biases_regularizer = biases_regularizer
                                            )

