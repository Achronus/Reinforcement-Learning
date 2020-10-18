# Reinforcement Learning (RL)

RL is a subfield of Machine Learning (ML) that focuses on agents (AI) taking actions in an environment and receives a reward as feedback.

It is a framework for solving control tasks, also known as decision problems.

## Contents

1. [RL Process](#RL-Process)
2. [Core Concepts](#Core-Concepts)
3. [Types of Tasks](#Types-of-Tasks)
4. [Exploration vs Exploitation](#Exploration-vs-Exploitation)
5. [Method Types](#Method-Types)
   - [Model-Free vs Model-Based](#Model-Free-vs-Model-Based)
   - [Value-Based vs Policy-Based](#Value-Based-vs-Policy-Based)
   - [On-Policy vs Off-Policy](#On-Policy-vs-Off-Policy)
6. [Value Functions](#Value-Functions)
   - [State-Value Function](#State-Value-Function)
   - [Action-Value Function](#Action-Value-Function)
7. [References](#References)

## RL Process

A basic example of the RL process can be seen below:

![Communication Channels](/imgs/comm-channels.jpg)

1. An agent receives a _state_, S<sub>t</sub>, from the environment.
2. From that state, the agent takes an _action_, A<sub>t</sub>.
3. Environment transitions to a _new state_, S<sub>t</sub>.
4. The environment provides some _reward_, R<sub>t</sub>, to the agent.
5. Repeat steps 1 to 4 until wins or loses, ending the _episode_.

## Core Concepts

__Reward Hypothesis__: the goal of all RL algorithms is to _maximize the expected cumulative reward_, or more commonly referred to as the _expected return_.

__Observation/State Space__: pieces of information that the environment provides the agent to explain what is happening around it. State and observation have a slight difference:

- __State__ (_s_): is a _complete description of the state_ of the world in a fully observed environment.
- __Observation__ (_o_): is a _partial description of the state_ in a partially observed environment.

__Agent__: something which interacts with an environment by: executing certain actions, taking observations and receiving rewards.

__Environment__: this relates to everything outside of an agent. Agents communicate with the environment via actions, where the environment returns observations and rewards in response to those agents actions.

__Action Space__: things an agent can do in an environment. There are two types: discrete and continuous.

- __Discrete__: a finite set of mutually exclusive things an agent can do. E.g. moving left or right.
- __Continuous__: a value attached to a given action. E.g. left speed in mph.

__Reward__: a scalar value obtained periodically by an agent from its environment. This can be positive or negative and tells the agent how well it has behaved.

__Markov Property__: a core component to the _Markov Process (MP)_. This implies that the agent needs only the current state to make its decision, not the history of all the states and actions it has taken before.

__Markov Process (MP)__: known as a _Markov Chain_, this consists of states that can only be observed. A Markov system can switch between states according to a specified law of dynamics.

All possible states form a set of states called the _state space_. A Markov Processes state space must be discete but can be extremely large in size. The formal definition has:

- A set of states (_S_) that a system can be in.
- Uses a transition matrix (_T_), with transition probabilities, that define the system dynamics.

__Transition Matrix__: used to capture transition probabilities within a Markov Property system. This is a square matrix of `NxN`, where `N` is the number of states in the model.

Every cell in a row, `i`, and a column, `j`, in the matrix contains the probability of the system transitioning from state `i` to `j`.

- For example:

  |         |Sunny|Rainy|
  |---------|-----|-----|
  |__Sunny__| 0.8 | 0.2 |
  |__Rainy__| 0.1 | 0.9 |

  If sunny, there is an 80% chance that the next day will be sunny.

__Markov Reward Process (MRP)__: for every episode, define the return at time (_t_), using a discount factor (<img src="https://render.githubusercontent.com/render/math?math={\gamma}">) between 0 and 1, and return the cumulative reward.

![MRP stretched calculation](/imgs/mrp-one.jpg)

The above is equivalent to:

![MRP sum calculation](/imgs/mrp-two.jpg)

The larger <img src="https://render.githubusercontent.com/render/math?math={\gamma}"> the smaller the discount, making the agent care more about its long term reward.

__Markov Decision Process (MDP)__: an MRP with decisions that has an environment where all its states are Markov Processes.

Adds an additional dimension, for actions, to the _transition matrix_, turning it into a cube. This allows the agent to actively choose an action at every timestep.

The transition matrix contains:

- __Depth dimension (_k_)__: the action the agent choose from.
- __Height dimension (_i_)__: the source (current) state.
- __Width dimension (_j_):__ the target state.

When choosing an action, the agent can affect the probabilities of target states.

__Policy__ (<img src="https://render.githubusercontent.com/render/math?math={\pi}">): a set of rules that controls the agent's behaviour (the brain of the agent). A function that informs us what _action to take given the state we are in_.

The policy is the _function_ that needs to be learned, finding the optimal policy we can maximize the _expected return_. This is found through training.

There are two approaches to training the agent to find the optimal policy <img src="https://render.githubusercontent.com/render/math?math={\pi}">*:

- __Directly__: by teaching the agent to learn which action to take. This is a _Policy-Based Method_.
- __Indirectly__: by teaching the agent to learn which state is more valuable and then take the action that leads to the more valuable states. This is a _Value-Based Method_.

__Reward Boundary__: this is the average reward that the agent should gain during a set amount of consecutive episodes (e.g. 100) that are used to solve the environment.

__Episodes__: all states that come in between an initial-state and a terminal-state. The agent must start over at the end of each episode.

__History__: a chain (or sequence) of states that observations form over time.

## Types of Tasks

There are two types of tasks: _episodic_ and _continuous_.

__Episodic__: consist of a starting point and an ending point (terminal state). This creates an episode which has: a list of states, actions, rewards and new states.

__Continuous__: these are tasks that run forever (has no terminal state). The agent has to learn how to choose the best actions and simultaneously interact with its environment.

## Exploration vs Exploitation

Exploration and exploitation are two powerful techniques that an agent can follow. To build a successful RL algorithm there needs to be a balance between the two.

__Exploration__: an agent tries _random actions_ in order to find more information about the environment.

__Exploitation__: an agent uses _known information_ to maximize its reward.

If the agent only focuses on exploitation it can fall into a trap where it only goes for the nearest source of rewards, even though there could be a larger reward further within the environment. This trap must be avoided where possible.

## Method Types

Each RL algorithm consists of three components that are split into two sub-categories each:

- _Model-free_ vs _model-based_
- _Value-based_ vs _policy-based_
- _On-policy_ vs _off-policy_


### Model-Free vs Model-Based

__Model-free__: method doesn't build a model of the environment or give the agent rewards. This type of model _directly connects_ observations to actions (or values) to its related actions.

__Model-based__: methods that try to predict the next observation and/or reward. These allow agents to try to choose the best possible action to take into the future. These are commonly used in deterministic environments.

### Value-Based vs Policy-Based

__Value-based__: trains the agent to learn which state is _more valuable_ and takes the action that has the highest value. Uses a value function.

__Policy-based__: methods that approximate the policy of an agent by learning a policy function directly.

Comes in two forms:

- __Deterministic__:  _a = <img src="https://render.githubusercontent.com/render/math?math={\pi}">(s)_, at a given state, always return the same action.
- __Stochastic__: _<img src="https://render.githubusercontent.com/render/math?math={\pi}">(a|s)  = P[A<sub>t</sub> = a|S<sub>t</sub> = s]_, outputs a probability distribution over the set of possible actions at that state.

### On-Policy vs Off-Policy

__On-policy__: methods that require fresh data obtained from an environment.

__Off-policy__: methods that learn through old historial data. These can be obtained by: a previous version of the agent, recorded by human demonstration or seen by the same agent several episodes ago.

## Value Functions

Often denoted as `V(s)`, represents how good the state is that the agent is in. Value functions describe the _expected total reward_ for a given state. These depend on the policy by which the agent picks actions to perform.

There are several types of value functions, some examples:

- State-value function
- Action-value function

These allow an agent to identify the quality of its current state, rather than waiting for a long-term result. The return is not immediately available and it can be random through a stochastic policy and through the dynamics of its environment.

### State-Value Function

A state value function is defined using a specific policy, making the _expected return_ depend on the policy:

V<sub><img src="https://render.githubusercontent.com/render/math?math={\pi}"></sub>(s) = <img src="https://render.githubusercontent.com/render/math?math={\mathbb E}"><sub><img src="https://render.githubusercontent.com/render/math?math={\pi}"></sub>[G<sub>t</sub>|S<sub>t</sub> = s]

Key:

- <img src="https://render.githubusercontent.com/render/math?math={\mathbb E}"> = expected value
- <img src="https://render.githubusercontent.com/render/math?math={\pi}"> = dependency on the policy
- G<sub>t</sub> = expected return from a given state
- S<sub>t</sub> = state at a given time step

### Action-Value Function

An action-value of a state is the _expected return_ of an agents choosen action, according to a policy:

q<sub><img src="https://render.githubusercontent.com/render/math?math={\pi}"></sub>(s, a) = <img src="https://render.githubusercontent.com/render/math?math={\mathbb E}"><sub><img src="https://render.githubusercontent.com/render/math?math={\pi}"></sub>[G<sub>t</sub>|S<sub>t</sub> = s, A<sub>t</sub> = a]

## References

- [Maxim Lapan - Deep Reinforcement Learning Hands-On (Second Edition) | Book](https://www.packtpub.com/product/deep-reinforcement-learning-hands-on-second-edition/9781838826994)
- [Thomas Simonini - Deep Reinforcement Learning Course v2.0 | YouTube](https://www.youtube.com/playlist?list=PLQLZ37V8CnUTqDCCGfjgYss_7lhj2ugcH)
- [Thomas Simonini - Deep Reinforcement Learning Course v2.0 | GitHub](https://medium.com/@thomassimonini/an-introduction-to-deep-reinforcement-learning-17a565999c0c)
- [Deep Reinforcement Learning Demysitifed (Episode 2) — Policy Iteration, Value Iteration and Q-learning | Medium](https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa)
- [Reinforcement Learning: Value Function and Policy | Medium](https://medium.com/analytics-vidhya/reinforcement-learning-value-function-and-policy-c22f5bd1d1b0)