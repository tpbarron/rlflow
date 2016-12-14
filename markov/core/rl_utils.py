import numpy as np

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
