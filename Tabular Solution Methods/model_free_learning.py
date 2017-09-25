### Episode model free learning using Q-learning and SARSA

# Do not change the arguments and output types of any of the functions provided! You may debug in Main and elsewhere.

import numpy as np
import gym
import time
from lake_envs import *

def learn_Q_QLearning(env, num_episodes=2000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
  """Learn state-action values using the Q-learning algorithm with epsilon-greedy exploration strategy.
  Update Q at the end of every episode.

  Parameters
  ----------
  env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
  num_episodes: int 
    Number of episodes of training.
  gamma: float
    Discount factor. Number in range [0, 1)
  learning_rate: float
    Learning rate. Number in range [0, 1)
  e: float
    Epsilon value used in the epsilon-greedy method. 
  decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)

  Returns
  -------
  np.array
    An array of shape [env.nS x env.nA] representing state, action values
  """
  Q = np.zeros((env.nS, env.nA))

  for episode in xrange(num_episodes):
    print "Current Episode:{}".format(episode)
    state = env.reset()
    done = False
    while not done:
      prob = np.random.random()
      if prob > e:
        action = np.argmax(Q[state])
      else:
        action = np.random.randint(0, env.nA)
      next_state, reward, done, _ = env.step(action)
      if done:
        Q[state, action] = lr * reward + (1-lr) * Q[next_state, action]
      else:
        Q[state, action] = lr * (reward + gamma * np.max(Q[next_state])) + (1-lr) * Q[state, action]
      state = next_state
  return Q

def learn_Q_SARSA(env, num_episodes=2000, gamma=0.95, lr=0.1, e=0.8, decay_rate=0.99):
  """Learn state-action values using the SARSA algorithm with epsilon-greedy exploration strategy
  Update Q at the end of every episode.

  Parameters
  ----------
  env: gym.core.Environment
    Environment to compute Q function for. Must have nS, nA, and P as
    attributes.
  num_episodes: int 
    Number of episodes of training.
  gamma: float
    Discount factor. Number in range [0, 1)
  learning_rate: float
    Learning rate. Number in range [0, 1)
  e: float
    Epsilon value used in the epsilon-greedy method. 
  decay_rate: float
    Rate at which epsilon falls. Number in range [0, 1)

  Returns
  -------
  np.array
    An array of shape [env.nS x env.nA] representing state-action values
  """
  Q = np.zeros((env.nS, env.nA))
  for episode in xrange(num_episodes):
    print "Current Episode:{}".format(episode)
    state = env.reset()
    done = False
    while not done:
      prob = np.random.random()
      if prob > e:
        action = np.argmax(Q[state])
      else:
        action = np.random.randint(0, env.nA)
      next_state, reward, done, _ = env.step(action)

      if done:
        Q[state, action] = lr * reward + (1-lr) * Q[state, action]
      else:
        prob = np.random.random()
        if prob > e:
          next_action = np.argmax(Q[next_state])
        else:
          next_action = np.random.randint(0, env.nA)
        Q[state, action] = lr * (reward + gamma * Q[next_state, next_action]) + (1-lr) * Q[state, action]
      state = next_state
  return Q

def render_single_Q(env, Q):
  """Renders Q function once on environment. Watch your agent play!

    Parameters
    ----------
    env: gym.core.Environment
      Environment to play Q function on. Must have nS, nA, and P as
      attributes.
    Q: np.array of shape [env.nS x env.nA]
      state-action values.
  """

  episode_reward = 0
  state = env.reset()
  done = False
  while not done:
    env.render()
    time.sleep(0.5) # Seconds between frames. Modify as you wish.
    action = np.argmax(Q[state])
    state, reward, done, _ = env.step(action)
    episode_reward += reward

  print "Episode reward: %f" % episode_reward

# Feel free to run your own debug code in main!
def main():
  env = gym.make('Stochastic-4x4-FrozenLake-v0')
  Q = learn_Q_QLearning(env)
  #Q = learn_Q_SARSA(env)
  render_single_Q(env, Q)

if __name__ == '__main__':
    main()
