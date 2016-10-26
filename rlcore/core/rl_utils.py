import numpy as np

def rollout_env_with_policy(env, policy, episode_len=np.inf):
    """
    Runs environment to completion and returns reward under given policy
    """
    episode_reward = 0.0
    done = False
    obs = env.reset()
    episode_itr = 0
    while not done and episode_itr < episode_len:
        env.render()
        action = policy.predict(obs)
        obs, reward, done, info = env.step(action)
        episode_reward += reward
        episode_itr += 1
    return episode_reward


#
# prediction post processors
#

def sign(x):
    if (x[0] < 0):
        return 0
    else:
        return 1
