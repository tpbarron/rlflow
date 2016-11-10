import numpy as np


def run_test_episode(env, policy, episode_len=np.inf):
    episode_itr = 0
    total_reward = 0.0
    done = False
    obs = env.reset()
    while not done and episode_itr < episode_len:
        env.render()
        if (policy.prediction_preprocessors is not None):
            for preprocessor in policy.prediction_preprocessors:
                obs = preprocessor(obs)
        action = policy.predict(obs)
        if (policy.prediction_postprocessors is not None):
            for postprocessor in policy.prediction_postprocessors:
                action = postprocessor(action)

        obs, reward, done, info = env.step(action)

        total_reward += reward
        episode_itr += 1
    return total_reward


def rollout_env_with_policy(env, policy, episode_len=np.inf, render=True, verbose=True):
    """
    Runs environment to completion and returns reward under given policy
    """
    ep_rewards = []
    ep_states = []
    ep_raw_actions = []
    ep_processed_actions = []

    # episode_reward = 0.0
    done = False
    obs = env.reset()
    episode_itr = 0
    while not done and episode_itr < episode_len:
        if (render): env.render()

        if (policy.prediction_preprocessors is not None):
            for preprocessor in policy.prediction_preprocessors:
                obs = preprocessor(obs)

        action = policy.predict(obs)
        ep_states.append(obs)
        ep_raw_actions.append(action)

        if (policy.prediction_postprocessors is not None):
            for postprocessor in policy.prediction_postprocessors:
                action = postprocessor(action)

        ep_processed_actions.append(action)
        obs, reward, done, info = env.step(action)
        ep_rewards.append(reward)

        episode_itr += 1

    if (verbose):
        print ('Game finished, reward: %f' % (sum(ep_rewards)))

    return ep_states, ep_raw_actions, ep_processed_actions, ep_rewards



def discount_rewards(r, gamma=0.99):
    """ take 1D float array of rewards and compute discounted reward """
    discounted_r = np.zeros_like(r)
    for t in xrange(r.size):
        discounted_r[t] = r[t] * gamma ** t
    return discounted_r

    # running_add = 0
    # for t in reversed(xrange(0, r.size)):
    #     if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
    #     running_add = running_add * gamma + r[t]
    #     discounted_r[t] = running_add
    # return discounted_r

#
# prediction pre processors
#

def prepro(I):
  """ prepro 210x160x3 uint8 frame into 6400 (80x80) 1D float vector """
  I = I[35:195] # crop
  I = I[::2,::2,0] # downsample by factor of 2
  I[I == 144] = 0 # erase background (background type 1)
  I[I == 109] = 0 # erase background (background type 2)
  I[I != 0] = 1 # everything else (paddles, ball) just set to 1
  return I.astype(np.float).ravel()


#
# prediction post processors
#

def sign(x):
    """
    Take in a float and return 0 if negative and 1 otherwise

    >>> sign(0.1)
    >>> 1
    >>> sign(-0.5)
    >>> 0
    """
    assert (type(x) == np.ndarray and len(x) == 1) or type(x) == float
    if (x[0] < 0):
        return 0
    else:
        return 1


def cast_int(x):
    return int(x)




#
# Probability operators
#

def prob(x):
    assert (type(x) == np.ndarray and len(x) == 1) or type(x) == float
    return 0 if np.random.uniform() < x else 1


def sample(x):
    assert (type(x) == np.ndarray)
    x = np.squeeze(x)
    assert x.ndim == 1
    # print (x)
    # renormalize to avoid 'does not sum to 1 errors'
    # x /= x.sum()
    # print (np.random.choice(len(x), 1, p=x/x.sum()))
    return np.random.choice(len(x), 1, p=x/x.sum())


def sample_outputs(x):
    """
    Given array [x1, x2, x3, x4, x5]

    Returns an array of the same shape of 0 or 1 where the entries are 0 with
    probability x_i/1.0 and 1 and probability 1-(x_i/1.0). The outputs are assumed
    to be between 0 and 1
    """
    assert type(x) == np.ndarray
    prob_vec = np.vectorize(prob, otypes=[np.int32])
    sampled = prob_vec(x)
    return sampled


def pong_outputs(x):
    assert type(x) == int
    if (x == 0):
        return 2
    else:
        return 3
