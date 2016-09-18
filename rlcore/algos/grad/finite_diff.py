import numpy as np


class FiniteDifference:

    def __init__(self, env, num_passes=2):
        self.env = env
        #self.policy = policy
        #self.num_weights = self.policy.get_num_weights()
        self.num_passes = num_passes


    def optimize(self, policy, episode_len=np.inf):
        # run episode with initial weights to get J_ref
        self.num_weights = policy.get_num_weights()

        J_ref = self.env.rollout_with_policy(policy, episode_len)

        deltaJs = np.empty((self.num_passes*self.num_weights,))
        deltaTs = np.empty((self.num_passes*self.num_weights, self.num_weights))

        for p in range(self.num_passes):
            for i in range(self.num_weights):
                # adjust policy
                #print ("Getting policy variation")
                policy_variation, deltas = policy.get_weight_variation()

                # run one episode with new policy
                #print ("rollout_with_policy variation")
                total_reward = self.env.rollout_with_policy(policy_variation, episode_len)

                deltaJs[p*self.num_weights + i] = total_reward - J_ref
                deltaTs[p*self.num_weights,:] = deltas.reshape((self.num_weights,))

        # #print ("Has inf: ", np.isinf(np.sum(deltaTs)))
        # #print (deltaTs)
        # deltaTs_tmp = np.dot(deltaTs.T, deltaTs)
        # ## check for nans and infs
        # #print ("Has nan: ", np.isnan(np.sum(deltaTs_tmp)))
        # #print ("Has inf: ", np.isinf(np.sum(deltaTs_tmp)))
        # deltaTs_tmp = np.linalg.pinv(deltaTs_tmp)
        # deltaTs_tmp = np.dot(deltaTs_tmp, deltaTs.T)
        # grad = np.dot(deltaTs_tmp, deltaJs)
        grad = np.dot(np.dot(np.linalg.pinv(np.dot(deltaTs.T, deltaTs)), deltaTs.T), deltaJs)
        #self.policy.update(grad)
        return grad
