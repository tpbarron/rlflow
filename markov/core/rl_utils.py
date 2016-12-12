import numpy as np
from markov.policies.policy import Policy


#
# Environment rollouts
#

def apply_prediction_preprocessors(policy, obs):
    # print (policy.prediction_preprocessors)
    if policy.prediction_preprocessors is not None:
        for preprocessor in policy.prediction_preprocessors:
            obs = preprocessor(obs)
    return obs


def apply_prediction_postprocessors(policy, action):
    # print (policy.prediction_postprocessors)
    if policy.prediction_postprocessors is not None:
        for postprocessor in policy.prediction_postprocessors:
            action = postprocessor(action)
    return action


def run_test_episode(env, policy, episode_len=np.inf, render=False):
    """
    Run an episode and return the reward, changes mode to Policy.TEST
    """
    init_mode = policy.mode
    policy.mode = Policy.TEST

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

    policy.mode = init_mode
    return total_reward


def average_test_episodes(env, policy, n_episodes, episode_len=np.inf):
    """
    Run n_episodes of the environment with the given policy and return the
    average reward over the episodes
    """
    total = 0.0
    for _ in range(n_episodes):
        total += run_test_episode(env, policy, episode_len=episode_len)
    return total / n_episodes


# def rollout_env_with_policy(env, policy, episode_len=np.inf, render=True, verbose=False):
#     """
#     Runs environment to completion and returns reward under given policy
#     Returns the sequence of rewards, states, raw actions (direct from the policy),
#         and processed actions (actions sent to the env)
#     """
#     ep_rewards = []
#     ep_states = []
#     ep_raw_actions = []
#     ep_processed_actions = []
#
#     # episode_reward = 0.0
#     done = False
#     obs = env.reset()
#     episode_itr = 0
#     while not done and episode_itr < episode_len:
#         if render:
#             env.render()
#
#         obs = apply_prediction_preprocessors(policy, obs)
#         ep_states.append(obs)
#         action = policy.predict(obs)
#         ep_raw_actions.append(action)
#         action = apply_prediction_postprocessors(policy, action)
#         ep_processed_actions.append(action)
#
#         obs, reward, done, _ = env.step(action)
#         ep_rewards.append(reward)
#
#         episode_itr += 1
#
#     if verbose:
#         print ('Game finished, reward: %f' % (sum(ep_rewards)))
#
#     return ep_states, ep_raw_actions, ep_processed_actions, ep_rewards


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


#
# prediction pre processors
#

# def prepro(I):
#     """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
#     I = I[35:195] # crop
#     I = I[::2, ::2, 0] # downsample by factor of 2
#     I[I == 144] = 0 # erase background (background type 1)
#     I[I == 109] = 0 # erase background (background type 2)
#     I[I != 0] = 1 # everything else (paddles, ball) just set to 1
#     I=I[:,:,np.newaxis]
#     # print (I.shape)
#     # import sys
#     # sys.exit()
#     return I
#     # return I.astype(np.float).ravel()
