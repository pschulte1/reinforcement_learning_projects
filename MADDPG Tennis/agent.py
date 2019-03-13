import copy
from collections import namedtuple, deque

import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from model import Actor, Critic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent:
    def __init__(self, state_size, action_size, agent_no, params):
        """
        # Inputs:
        # state_size = dimensions of the state
        # actions_size = dimensions of the actions
        # agent_no = numerating of the current agent
        # params = parameter dictionary
        #
        # Description:
        # - Training the agent with experiences
        # - Performing actions in the environment
        """

        # Transfer variables to the class
        self.state_size = state_size
        self.action_size = action_size
        self.agent_no = agent_no
        self.seed = params['agent_seed']
        self.batch_size = params['batch_size']
        self.lr_actor = params['lr_actor']
        self.lr_critic = params['lr_critic']
        self.critic_weight_decay = params['critic_weight_decay']
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.update_step = params['update_step']
        self.num_agents = params['num_agents']
        self.consecutive_update_steps = params['consecutive_update_steps']

        # Set the random seed
        random.seed(self.seed)

        # Initialize step of the agent
        self.t_step = 0

        # ACTOR NETWORK ONLY FOR ONE AGENT
        # Create the online actor neural network
        self.actor_local = Actor(state_size, action_size, self.seed).to(device)

        # Create the target actor neural network
        self.actor_target = Actor(state_size, action_size, self.seed).to(device)

        # Create the optimizer for the actor
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=self.lr_actor)

        # CRITIC NETWORK RECEIVES INPUTS FROM ALL AGENTS
        # Create the online critic neural network
        self.critic_local = Critic(state_size * self.num_agents,
                                   action_size * self.num_agents,
                                   self.seed).to(device)

        # Create the target critic neural network
        self.critic_target = Critic(state_size * self.num_agents,
                                    action_size * self.num_agents,
                                    self.seed).to(device)

        # Create the optimizer for the critic network
        self.critic_optimizer = optim.Adam(self.critic_local.parameters(),
                                           lr=self.lr_critic,
                                           weight_decay=self.critic_weight_decay)

        # Create the exploration noise process for the actions
        self.noise = OUNoise(action_size, self.seed, sigma=params['noise_sigma'])

    def step(self, memory, agents):
        """
        # Inputs:
        # memory = shared replay buffer filled with experiences of the agents
        # agents = list of all agents
        #
        # Description
        # - Increase the number of steps
        # - Let all agents learn
        """

        # Increasing the number of steps
        self.t_step += 1

        # Check first, that enough experiences are in the memory and then check, if an update can be performed
        if (len(memory) > self.batch_size) and self.t_step % self.update_step == 0:

            # For loop over all consecutive update steps...
            for i in range(self.consecutive_update_steps):

                # Sample one batch for training the first agent
                experiences = memory.sample()

                # Train the first agent
                self.learn(experiences, agents, own_index = 0, other_index = 1)

                # Sample one batch for training the second agent
                experiences = memory.sample()

                # Train the second agent
                self.learn(experiences, agents, own_index = 1, other_index = 0)

    def act(self, state, add_noise=True, scale=1.0):
        """
        # Inputs:
        # state = current state of the agent
        # add_noise = Boolean to check, if you want to add exploration noise
        # scale = Scaling of the exploration noise
        #
        # Outputs:
        # Predicted action of the online actor neural network + noise
        #
        # Description:
        # - Transform the state to torch
        # - Predict the action of the current state by the online actor neural network
        # (- Add Noise)
        # - Clip the action within the action boundaries
        """

        # Transform the state to a torch-tensor
        state = torch.from_numpy(state).float().to(device)

        # Eval the online actor neural network
        self.actor_local.eval()

        # Do not compute the gradient --> ONLY PREDICTION
        with torch.no_grad():

            # Get the action of the current state
            action = self.actor_local(state).cpu().data.numpy()

        # Complete the prediction step
        self.actor_local.train()

        # If exploration noise is wanted...
        if add_noise:

            # Add the exploration noise to the action
            action += self.noise.sample() * scale

        # Return the cliped action
        return np.clip(action, -1.0, 1.0)

    def reset(self):
        """
        # Inputs:
        # None
        #
        # Outputs:
        # None
        #
        # Description:
        # - Reset the exploration noise
        """
        self.noise.reset()

    def learn(self, experiences, agents, own_index, other_index):
        """
        # Inputs:
        # experiences = random/uniform sampled minibatch of experiences
        # agents = lists of all agents
        # own_index = index of the current agent
        # other_index = index of the other agent
        #
        # Outputs:
        # None
        #
        # Description:
        # - Update policy
        # - Update value parameters
        """


        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, actions) -> Q-value
        """

        # ---------------------------- CURRENT AGENT ---------------------------- #

        # Extract the state of the current agent
        own_states = experiences[own_index][0]

        # Extract the actions of the current agent
        own_actions = experiences[own_index][1]

        # Extract the rewards of the current agent
        own_rewards = experiences[own_index][2]

        # Extract the next state of the current agent
        own_next_states = experiences[own_index][3]

        # Extract the information about terminal state of the current agent
        own_dones = experiences[own_index][4]

        # ---------------------------- OTHER AGENT ---------------------------- #

        # Extract the state of the other agent
        other_states = experiences[other_index][0]

        # Extract the actions of the current agent
        other_actions = experiences[other_index][1]

        # Extract the rewards of the other agent
        other_rewards = experiences[other_index][2]

        # Extract the next state of the other agent
        other_next_states = experiences[other_index][3]


        # ---------------------------- JOINT ---------------------------- #

        # Concatenate states of current and other agent
        all_states=torch.cat((own_states, other_states), dim=1)

        # Concatenate actions of current and other agent
        all_actions=torch.cat((own_actions, other_actions), dim=1)

        # Concatenate next states of current and other agent
        all_next_states=torch.cat((own_next_states, other_next_states), dim=1)


        # Select the current agent
        agent = agents[own_index]
        other_agent = agents[other_index]


        # ---------------------------- UPDATE CRITIC ---------------------------- #

        # Get predicted actions from the current and the other agent from the target neural networks of the next-state
        all_next_actions = torch.cat((agent.actor_target(own_next_states), other_agent.actor_target(other_next_states)),
                                     dim = 1)

        # Get all Q-values of the current agent from its critic neural network with the next state and the corresponding actions
        Q_targets_next = agent.critic_target(all_next_states, all_next_actions)

        # Compute Q-targets for current states (y_i)
        Q_targets = own_rewards + (self.gamma * Q_targets_next * (1 - own_dones))

        # Get the predicted Q-values of the current state from the current online actor
        Q_expected = agent.critic_local(all_states, all_actions)

        # Compute the loss
        critic_loss = F.mse_loss(Q_expected, Q_targets)

        # Minimize the loss
        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm(agent.critic_local.parameters(), 1)
        agent.critic_optimizer.step()


        # ---------------------------- UPDATE ACTOR ---------------------------- #

        # Get the predicted actions from the current and the other agent from the online neural networks of the the current state
        all_actions_pred = torch.cat((agent.actor_local(own_states), agent.actor_local(other_states).detach()),
                                     dim = 1)

        # Compute the loss over the minibatch
        actor_loss = -agent.critic_local(all_states, all_actions_pred).mean()

        # Minimize the loss
        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        agent.actor_optimizer.step()

        # ----------------------- UPDATE THE TARGET NETWORKS ----------------------- #
        agent.soft_update(agent.critic_local, agent.critic_target)
        agent.soft_update(agent.actor_local, agent.actor_target)



    def soft_update(self, local_model, target_model):
        """
        # Inputs:
        # local_model = the online neural network
        # target_model = the target neural network
        #
        # Outputs:
        # None
        #
        # Description:
        # - Get all parameters of the online neural network
        # - Add a fraction of these parameters to the target neural network
        """


        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        """

        # Go though all parameters in the target and online neural network
        for target_param, local_param in zip(target_model.parameters(),
                                             local_model.parameters()):

            # Assign the new value to the target neural network by taking a fraction of the online neural network
            target_param.data.copy_(
                    self.tau * local_param.data + (1.0 - self.tau) * target_param.data)


class OUNoise:
    """Ornstein-Uhlenbeck process.
        Params
        ======
            size (int) : size of action space
            target_model: PyTorch model (weights will be copied to)
            mu (float) :  Ornstein-Uhlenbeck noise parameter
            theta (float) :  Ornstein-Uhlenbeck noise parameter
            sigma (flmoat) : Ornstein-Uhlenbeck noise parameter
        """
    def __init__(self, size, seed, mu=0, theta=0.15, sigma=0.2):
        """
        # Inputs:
        # size = size of the action space
        # seed = random seed for reproduceability
        # mu = Ornstein-Uhlenbeck noise parameter
        # theta = Ornstein-Uhlenbeck noise parameter
        # sigma = Ornstein-Uhlenbeck noise parameter
        #
        # Description:
        # - Performing Ornstein-Uhlenbeck process
        """

        # Transfer all variables to the class
        self.size=size
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.seed = random.seed(seed)
        self.reset()

    def reset(self):
        """
        # Inputs:
        # None
        #
        # Outputs:
        # None
        #
        # Description:
        # - Reset the process to the origin
        """
        self.state = copy.copy(self.mu)

    def sample(self):
        """
        # Inputs:
        # None
        #
        # Outputs:
        # Current state (noise) of the Ornstein-Uhlenbeck process
        #
        # Description.
        # - Get the current state
        # - Perform the Ornstein-Uhlenbeck process on the current state
        """

        # Save current state
        x = self.state

        # Perform Ornstein-Uhlenbeck process with normal distribution
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.size)

        # Add fraction
        self.state = x + dx

        # Return current state
        return self.state


class UniformNoise:
    def __init__(self, size, bound, seed, min_bound = 0.01, exploration_steps = 2000):
        """
        # Inputs:
        # size = size of the action space
        # bound = initial boundary of the uniform distribution
        # min_bound = Lowest value of the boundary for the uniform distribution
        # exploration_steps = number of steps needed to reach the lowest boundary value
        #
        # Description:
        # - Sample noise uniformly between the boundaries
        """

        # Transfer variables to the class
        self.bound = bound
        self.max_bound = bound
        self.min_bound = min_bound
        self.exploration_steps = exploration_steps
        self.size = size
        np.random.seed(seed)

        # Initialize step counter
        self.step = 0

    def sample(self):
        """
        # Inputs:
        # None
        #
        # Outputs:
        # - Uniformly sampled noise of the given size
        #
        # Description:
        # - Sample uniformly the noise
        # - Decrease the boundary of the uniform distribution by the current step
        """

        # Sample uniformly noise between the boundaries
        x = np.random.uniform(-self.bound, self.bound, size = self.size)

        # Decrease the noise by ((y_max - y_min) / x_max) * (x_max - x_current)
        self.bound = max(self.min_bound, ((self.max_bound-self.min_bound)/self.exploration_steps)*(self.exploration_steps-self.step))

        # Increase the current step by 1 (x_current)
        self.step += 1

        # Return the noise
        return x

    # Reset not needed
    def reset(self):
        pass



class ReplayBuffer:
    def __init__(self, params):
        """
        # Inputs:
        # params = parameter dictionary
        #
        # Description:
        # - Initialize a memory/buffer with a fixed size
        # - Construct functions to add new experiences and sample minibatches of experiences
        """

        # Transfer variables to class
        self.buffer_size = int(params['buffer_size'])
        self.batch_size = params['batch_size']
        self.num_agents = params['num_agents']
        random.seed(params['buffer_seed'])

        # Create the fixed size memory
        self.memory = deque(maxlen=int(params['buffer_size']))

        # Create a namedtuple to easily add / sample
        self.experience = namedtuple("Experience",
                                     field_names=["states", "actions", "rewards",
                                                  "next_states", "dones"])

    def add(self, states, actions, rewards, next_states, dones):
        """
        # Inputs:
        # states = current states of the agents
        # actions = performed actions of the agents
        # rewards = received rewards by taking these actions
        # next_states = new states after taking the actions
        # dones = information if a terminal state is reached
        #
        # Outputs:
        # None
        #
        # Description:
        # - Add the current experience to the memory
        """

        # Create the experience with all observations
        e = self.experience(states, actions, rewards, next_states, dones)

        # Add the experience to the memory
        self.memory.append(e)

    def sample(self):
        """
        # Inputs:
        # None
        #
        # Outputs:
        # One minibatch with random experiences
        #
        # Description:
        # - Uniformly sample one minibatch of experiences from the memory
        # - Create a tuple with agent correspondencies
        """

        # Sample one minibatch of experiences
        experiences = random.sample(self.memory, k=self.batch_size)

        # Create a tuple with the range of number of agents
        agent_experiences = dict.fromkeys(range(self.num_agents))

        # For each agent...
        for no_agent in range(self.num_agents):

            # Get the states batch
            states_batch = np.vstack(
                    [experience.states[no_agent] for experience in experiences])

            # Get the action batch
            actions_batch = np.vstack(
                    [experience.actions[no_agent] for experience in experiences])

            # Get the reward batch
            rewards_batch = np.vstack(
                    [experience.rewards[no_agent] for experience in experiences])

            # Get the next states batch
            next_states_batch = np.vstack(
                    [experience.next_states[no_agent] for experience in experiences])

            # Get the batch with the information about terminal states
            dones_batch = np.vstack(
                    [experience.dones[no_agent] for experience in experiences]).astype(
                np.uint8)

            # Create tuple for the corresponding agent
            agent_experiences[no_agent] = (states_batch,
                                           actions_batch,
                                           rewards_batch,
                                           next_states_batch,
                                           dones_batch)

            # Transfer it to a torch-tensor
            agent_experiences[no_agent] = tuple(torch.from_numpy(batch).float().to(device)
                                                for batch in agent_experiences[no_agent])

        # Return the minibatch
        return agent_experiences

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
