# reinforcement_Learning

Basic Teminalogies of RL :

1> Agent:
The agent is the decision-maker in the RL setup. It interacts with the environment by taking actions and learning from the outcomes to achieve a specific goal. The agent's objective is to maximize cumulative rewards over time.

2> Environment:
The environment is everything the agent interacts with and learns from. It provides the context in which the agent operates and defines the rules of interaction. The environment responds to the agent's actions and provides feedback in the form of rewards and new observations.

3>Action:
Actions are the set of all possible moves the agent can make. At each step, the agent selects an action based on its policy (strategy). The action taken affects the state of the environment.

Action Space: This can be discrete (a finite set of actions) or continuous (an infinite range of possible actions).

4>Observation
Observations are the data the agent receives from the environment. They represent the current state or partial information about the state of the environment.

*   State: The complete description of the environment's status at a given time. In fully observable environments, the agent can see the full state. In partially observable environments, the agent only sees observations that provide partial information about the state.


5>Reward
The reward is a scalar value received by the agent after taking an action. It indicates how good or bad the action was concerning achieving the agent's goal.

*   Reward Function: This defines how rewards are assigned to different actions and states. The agent aims to maximize the cumulative reward over time, known as the return.

6>Policy
A policy is the strategy the agent uses to decide which actions to take based on the current state or observation.

Deterministic Policy: Maps each state to a specific action.
Stochastic Policy: Maps each state to a probability distribution over actions, meaning the agent can choose different actions with certain probabilities.


7>Value Function
The value function estimates the expected cumulative reward (return) from a given state or state-action pair.

*   State Value Function (V): The expected return starting from a state and following a particular policy.
*   Action Value Function (Q): The expected return starting from a state, taking a particular action, and following a policy thereafter.

8>Model
In model-based RL, the agent uses a model of the environment to predict the next state and reward, which helps in planning future actions. In model-free RL, the agent learns to act directly without explicitly modeling the environment.

9>Exploration vs. Exploitation
The balance between exploring new actions to discover their effects and exploiting known actions that yield high rewards is crucial in RL.

*   Exploration: Trying new actions to gather more information about the environment.
*   Exploitation: Using known actions that maximize rewards based on current knowledge.

10>Episode
An episode is a sequence of states, actions, and rewards, starting from an initial state and ending in a terminal state. The goal of the agent is often to maximize the total reward within an episode.

11>Common Algorithms
Some common RL algorithms include:

*   Q-Learning: A model-free algorithm where the agent learns the value of state-action pairs.
*   Deep Q-Networks (DQN): Uses neural networks to approximate the Q-value function.
*   Policy Gradient Methods: Directly parameterize the policy and optimize it using gradient ascent techniques.
*   Actor-Critic Methods: Combine value-based and policy-based approaches, where the actor updates the policy and the critic evaluates the action.