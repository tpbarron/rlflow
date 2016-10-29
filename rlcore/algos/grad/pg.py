from __future__ import print_function

from rlcore.core import rl_utils
import numpy as np


class PolicyGradient:
    """
    Basic stochastic policy gradient implementation based on Keras network
    """

    def __init__(self, env):
        self.env = env


    def discount_rewards(self, r, gamma=0.9):
        """ take 1D float array of rewards and compute discounted reward """
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(xrange(0, r.size)):
            if r[t] != 0: running_add = 0 # reset the sum, since this was a game boundary (pong specific!)
            running_add = running_add * gamma + r[t]
            discounted_r[t] = running_add
        return discounted_r


    def optimize(self, policy, episode_len=np.inf):
        ep_states, ep_raw_actions, ep_processed_actions, ep_rewards = rl_utils.rollout_env_with_policy(self.env,
                                                                                                       policy,
                                                                                                       episode_len=episode_len)

        ep_dlogps = []

        for i in range(len(ep_states)):
            y = 1 if ep_processed_actions[i] == 2 else 0 # a "fake label"
            ep_dlogps.append(y - ep_raw_actions[i])

        # stack together all inputs, action gradients, and rewards for this episode
        epx = np.vstack(ep_states)
        epdlogp = np.vstack(ep_dlogps)
        epr = np.vstack(ep_rewards)

        # compute the discounted reward backwards through time
        discounted_epr = self.discount_rewards(epr)
        # standardize the rewards to be unit normal (helps control the gradient estimator variance)
        discounted_epr -= np.mean(discounted_epr)
        discounted_epr /= np.std(discounted_epr)

        epdlogp *= discounted_epr # modulate the gradient with advantage (PG magic happens right here.)

        policy.keras_model.fit(epx, epdlogp, nb_epoch=1, verbose=0, shuffle=True)
