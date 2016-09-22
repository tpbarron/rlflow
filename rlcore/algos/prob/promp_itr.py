from __future__ import print_function
import numpy as np


class ProMPIteration:

    def __init__(self, env):
        self.env = env

    def optimize(self, policy, num_samples=100, episode_len=np.inf):
        #num_samples = policy.promp.num_bases * policy.promp.num_dof
        samples = np.empty((num_samples, policy.Uw.size))
        rewards = np.empty((num_samples,))
        for i in range(num_samples):
            w_sample = policy.sample()
            samples[i,:] = w_sample.flatten()

            # run one episode with new policy
            total_reward = self.env.rollout_with_policy(policy, episode_len)
            rewards[i] = total_reward

            print ("ProMP sample", i, "of", (num_samples), ", reward =", total_reward)

        # print ("Samples size=",samples.shape, rewards.shape)
        # compute weighted sum of means
        Uw_new = np.average(samples, axis=0, weights=rewards)
        Uw_new = Uw_new.reshape(policy.Uw.shape)

        # compute new cov matrix
        # print (policy.Ew.shape)
        # print (np.diag(rewards).shape)
        # Ew_new = np.dot(np.dot(policy.Ew, np.diag(rewards)), policy.Ew.T)
        Ew_new = np.cov(samples, rowvar=False) #data is in columns
        #print (Ew_new.shape)

        return Uw_new, Ew_new
