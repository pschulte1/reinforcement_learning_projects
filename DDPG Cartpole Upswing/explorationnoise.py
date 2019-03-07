import numpy as np
import random

class OrnsteinUhlenbeckProcess(object):
    def __init__(self, theta, mu=0, sigma=1, x0=0, dt=1e-2, n_steps_annealing=100, size=1):
        self.theta = theta
        self.sigma = sigma
        self.n_steps_annealing = n_steps_annealing
        self.sigma_step = - self.sigma / float(self.n_steps_annealing)
        self.x0 = x0
        self.mu = mu
        self.dt = dt
        self.size = size
        self.step = 0


    def generate(self):
        sigma = max(0, self.sigma_step * self.step + self.sigma)
        x = self.x0 + self.theta * (self.mu - self.x0) * self.dt + sigma * np.sqrt(self.dt) * np.random.normal(size=self.size)
        self.x0 = x
        return x

class UniformNoise(object):
    def __init__(self, bound, min_bound = 0.01, n_steps_annealing=2000, size=1):
        self.bound = bound
        self.max_bound = bound
        self.min_bound = min_bound
        self.n_steps_annealing = n_steps_annealing
        self.step = 0

    def generate(self):
        x = np.random.uniform(-self.bound, self.bound, size = 1)[0]
        self.bound = max(self.min_bound, ((self.max_bound - self.min_bound) / self.n_steps_annealing) * (self.n_steps_annealing-self.step))
        return x




class GreedyPolicy(object):
    def __init__(self, action_dim, n_steps_annealing, min_epsilon, max_epsilon):
        self.epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.action_dim = action_dim
        self.n_steps_annealing = n_steps_annealing
        self.epsilon_step = - (self.epsilon - self.min_epsilon) / float(self.n_steps_annealing)
        self.step = 0

    def generate(self, action):
        epsilon = max(self.min_epsilon, self.epsilon_step * self.step + self.epsilon)
        if random.random() < epsilon:
            return random.choice(range(self.action_dim))
        else:
            return action







