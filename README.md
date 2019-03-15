# Reinforcement Learning Projects
This repository serves as a collection of various solutions for reinforcement learning problems.

## Overview
In every single folder of this directory you will find the solutions to the named problems with a detailed instruction:
* [Deep Deterministic Policy Gradients (DDPG) for a Cartpole Up-Swing](DDPG_CartPole_Upswing)
    - Modified [OpenAI Gym Cartpole Environment](https://gym.openai.com/envs/CartPole-v1/) (start position of the pendulum is directed downwards + continuous reward function)
    - [Deep Q-Network (DQN)](https://storage.googleapis.com/deepmind-media/dqn/DQNNaturePaper.pdf) / [Actor-Critic Network](https://arxiv.org/pdf/1509.02971.pdf) implemented in [Tensorflow 1.13](https://www.tensorflow.org/)
* [Multi-Agent Deep Deterministic Policy Gradients (MADDPG) for competitive playing tennis](MADDPG_Tennis)
    - Modified Tennis Environment from [Unity](https://github.com/Unity-Technologies/ml-agents) (current state of the agents contains the three most recent states)
    - [Multi-Agent Actor-Critic Neural Network](https://arxiv.org/pdf/1706.02275.pdf) implemented in [PyTorch](https://pytorch.org)
