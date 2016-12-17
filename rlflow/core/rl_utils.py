import numpy as np
import tensorflow as tf

#
# Reward manipulation
#

def discount_rewards(rewards, gamma=0.9):
    """
    Take 1D float array of rewards and compute the sum of discounted rewards
    at each timestep
    """
    discounted_r = np.zeros_like(rewards)
    for i in range(rewards.size):
        rew_sum = 0.0
        for j in range(i, rewards.size):
            rew_sum += rewards[j] * gamma ** j
        discounted_r[i] = rew_sum
    return discounted_r


def standardize_rewards(rewards):
    """
    Subtract the mean and divide by the stddev of the given rewards
    """
    rew_mean = np.mean(rewards)
    rew_std = np.std(rewards)
    rewards -= rew_mean
    if rew_std != 0:
        rewards /= rew_std
    return rewards


def apply_prediction_preprocessors(policy, obs):
    if policy.prediction_preprocessors is not None:
        for preprocessor in policy.prediction_preprocessors:
            obs = preprocessor(obs)
    return obs


def apply_prediction_postprocessors(policy, action):
    if policy.prediction_postprocessors is not None:
        for postprocessor in policy.prediction_postprocessors:
            action = postprocessor(action)
    return action


def run_test_episode(env, policy, episode_len=np.inf, render=False):
    """
    Run an episode and return the reward
    """
    episode_itr = 0
    total_reward = 0.0
    done = False
    obs = env.reset()
    while not done and episode_itr < episode_len:
        if render:
            env.render()

        obs = apply_prediction_preprocessors(policy, obs)
        action = policy.predict(obs)
        action = apply_prediction_postprocessors(policy, action)

        obs, reward, done, _ = env.step(action)

        total_reward += reward
        episode_itr += 1

    return total_reward


def cast_int(x):
    return int(x)
