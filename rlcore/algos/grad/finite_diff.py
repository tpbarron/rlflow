from __future__ import print_function
import numpy as np


class FiniteDifference:

    def __init__(self, env, num_passes=2):
        self.env = env
        #self.policy = policy
        #self.num_weights = self.policy.get_num_weights()
        self.num_passes = num_passes


    def optimize(self, policy, num_variations=None, episode_len=np.inf):
        self.num_weights = policy.get_num_weights()
        if (num_variations == None):
            num_variations = self.num_passes * self.num_weights

        # run episode with initial weights to get J_ref
        J_ref = self.env.rollout_with_policy(policy, episode_len)

        deltaJs = np.empty((self.num_passes*self.num_weights,))
        deltaTs = np.empty((self.num_passes*self.num_weights, self.num_weights))

        for i in range(num_variations):
            # adjust policy
            policy_variation, deltas = policy.get_weight_variation()

            # run one episode with new policy
            total_reward = self.env.rollout_with_policy(policy_variation, episode_len)
            print ("FD variation", i, "of", (num_variations), ", reward =", total_reward)
            deltaJs[i] = total_reward - J_ref
            deltaTs[i,:] = deltas.reshape((self.num_weights,))

        # TODO: may need to check for nans/infs here
        grad = np.dot(np.dot(np.linalg.pinv(np.nan_to_num(np.dot(deltaTs.T, deltaTs))), deltaTs.T), deltaJs)
        return grad
