
import gym
from keras.models import Sequential
from keras.layers import Dense, Activation
import numpy as np


class FiniteDifference:

    def __init__(self, num_passes=2, lr=0.01, max_itr=100):
        self.env = env
        self.model = model
        self.num_passes = num_passes
        self.num_weights = self.model.layers[0].get_weights()[0].shape[0]
        self.lr = lr
        self.max_itr = max_itr


    def optimize(self):
        # Assume only one layer for now
        # The weights are a np array of (output, input) shape
        initial_weights, bias = self.model.layers[0].get_weights()

        # run episode with initial weights to get J_ref
        J_ref = self.rollout([initial_weights, bias])

        deltaJs = np.empty((self.num_passes*self.num_weights,))
        # deltaTs = np.empty((self.num_passes*self.num_weights, 1))
        deltaTs = np.empty((self.num_passes*self.num_weights, self.num_weights))

        print "Running passes"
        for p in range(self.num_passes):
            for i in range(initial_weights.shape[0]):
                # adjust policy weight i
                policy_variation, deltas = self.get_weight_variation(initial_weights, i)
                #self.model.layers[0].set_weights([policy_variation, bias])

                # run one episode with new policy
                total_reward = self.rollout([policy_variation, bias])

                deltaJs[p*self.num_weights + i] = total_reward
                deltaTs[p*self.num_weights,:] = deltas.reshape((self.num_weights,))
                #deltaTs[p*self.num_weights + i][0] = delta

        print "Compute gradient update"
        deltaTs_tmp = np.dot(deltaTs.T, deltaTs)
        deltaTs_tmp = np.linalg.pinv(deltaTs_tmp)
        deltaTs_tmp = np.dot(deltaTs_tmp, deltaTs.T)
        grad = np.dot(deltaTs_tmp, deltaJs)
        print "grad: ", grad
        initial_weights[:,0] -= self.lr * grad


    def rollout(self, weights):
        '''
        Run an episode with the given parameters and return reward
        '''
        self.model.layers[0].set_weights(weights)

        # run one episode with new policy
        total_reward = 0.0
        observation = self.env.reset()
        done = False
        while not done:
            env.render()
            action = int(np.round(self.get_action(observation)))
            if action < 0:
                action = 0
            elif action > 1:
                action = 1

            observation, reward, done, info = self.env.step(action)
            total_reward += reward
        return total_reward


    def get_weight_variation(self, weights_to_vary, dist='gaussian'):
        # define distribution based on given weights
        mu, sigma = 0.0, np.std(weights_to_vary)
        deltas = np.random.normal(mu, sigma, weights_to_vary.shape)
        varied_weights = np.add(np.copy(weights_to_vary),deltas)
        return varied_weights, deltas


    def get_action(self, obs):
        return self.model.predict(obs[np.newaxis,:])



env = gym.make('CartPole-v0')
model = Sequential()
model.add(Dense(1, input_dim=env.state.shape[0], init='uniform'))
model.add(Activation('tanh'))
#model.add(Dense(1, init='uniform'))

fd = FiniteDifference(env, model)
fd.optimize()
