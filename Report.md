[//]: # (Image References)

[image1]: https://github.com/GCCFeli/drl_tennis/blob/master/Rewards.png?raw=true "Rewards" 
[image2]: https://github.com/GCCFeli/drl_tennis/blob/master/Demo.gif?raw=true "Result"  

# Report

## 1. Getting started

Please follow the instructions in README.md to setup the environment.

## 2. Learning Algorithm

### 2.1 DDPG in multi-agent environment

This RL problem's state space is continuous, and action space is also continuous. Despite it's a multi-agent problem, its two agents are actually `identical`, receiving their own local observations. Their actions make no side effects. So we choose the well known DDPG([Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971 "Deep Deterministic Policy Gradient")) which perfectly matches the problem.

DDPG is a kind of off-policy Actor-Critic method. The Actor uses DNN to directly map the continuous state space to action space. The Critic estimates the Q value of state-action pair, sharing the same DNN body with the Actor. DDPG uses the ε-greedy method to do exploration, in practics, OU Noise is added to action output. Replay buffer and soft-update which are successful in DQN, are also used in DDPG.

In this problem, we choose a simple 2-layer neural network. Layer sizes are 128, 64. Activation function is ReLU. Loss function is MSE. Optimizer is Adam. Replay buffer and soft update are used to make learning stable. Alghough there are 2 agents in the environment, just one learner is used. Experiences of 2 agents are stored in a shared replay buffer.

### 2.2 Prioritized Replay Buffer

My first a few attemps utilizing the standard DDPG with experience replay buffer failed. My observation is, at start the agents' actions are random, so hitting the ball is rare. If we sample experiences uniformly, the positive cases are very unlikely to be sampled, causing the agents learn from all negative cases.

So I implemented [Prioritized Replay Buffer](https://arxiv.org/pdf/1706.02275.pdf) to overcome this issue. In last project `Reacher`, Prioritized Replay Buffer can effectively boost training efficiency. In this project, Prioritized Replay Buffer make DDPG success.

### 2.3 Gradient Truncation

Large gradients may lead to unstable learning, causing DDPG to fail or oscillate. In critic update, the gradients are truncated to 1, to make our algorithm more stable.

### 2.4 Hyperparameters

| Hyperparameter | Value | Desctiption |
| -------------- | ----- | ----------- |
| minibatch size | 512 | Number of training cases over each stochastic gradient descent (SGD) update is computed. |
| replay buffer size | 100000 | SGD updates are sampled from this number of most recent frames. |
| soft update target paramater | 0.001 | Soft update target parameter  τ used to lerp between local network and target network. |
| discount factor | 0.99 | Discount factor gamma used in the Q-learning update. |
| learning rate - Actor | 0.0005 | The init learning rate of Actor used by Adam. |
| learning rate - Critic | 0.001 | The init learning rate of Critic used by Adam. |
| weight decay | 0 | The weight decay parameter of Critic used by Adam. |
| sigma init | 0.2 | The initial sigma of OU Noise. |
| sigma decay | 0.95 | The sigma decay of OU Noise. |
| sigma min | 0.005 | The minimal sigma of OU Noise. |
| mu | 0 | Mu in OU Noise. |
| theta | 0.15 | Theta in OU Noise. |
| update frequency | 5 | Network update frequency. |
| update times | 10 | Update times in one learning process. |

## 3. Plot of Rewards

Training can be done within 1000 episodes.

![Rewards][image1]

## 4. Result

Watch the video below to see the performance of a trained agent.

![Result][image2]

## 5. Ideas for Future Work

The original DDPG algorithm is enough to solve this problem. However, some methods may achieve better performance:
* [Distributed Distributional Deterministic Policy Gradients (D4PG)](https://arxiv.org/abs/1804.08617 "Distributed Distributional Deterministic Policy Gradients (D4PG)"): N-step bootstrap, action distribution estimation and distributed training are proved effective in this paper.
* [Multi-Agent DDPG (MADDPG)](https://arxiv.org/pdf/1706.02275.pdf "Multi-Agent DDPG (MADDPG)"): Actually MADDPG algorithm is tried but not work (See maddpg.py). Maybe hyperparameter tuning can make it work.

Hyperparameters are manually chosen for this problem. Grid search will be helpful to choose a better hyperparameter set.
