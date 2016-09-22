import numpy as np
from iprims.iprims import InteractionPrimitive
from .f_approximator import FunctionApproximator

class MovementPrimitivesApproximator(FunctionApproximator):

    def __init__(self, promp, lr=0.01):
        super(MovementPrimitivesApproximator, self).__init__()
        self.lr = lr
        self.t = 0
        self.promp = promp
        self.Uw = self.promp.Uw
        self.Ew = self.promp.Ew

        self.w = self.promp.get_weights_from_dist(self.promp.Uw)


    def get_num_weights(self):
        return self.w.size


    def sample(self):
        sample = np.random.multivariate_normal(self.Uw, self.Ew)
        # print ("Originial w=", self.w)
        self.w = self.promp.get_weights_from_dist(sample)
        # print ("Sampled w=", self.w)
        return self.w


    def get_weight_variation(self, stdev_coef=0.1):
        # define distribution based on given weights
        mu, sigma = 0.0, stdev_coef*np.std(self.w)
        deltas = np.random.normal(mu, sigma, self.w.shape)
        varied_weights = np.add(np.copy(self.w), deltas)
        policy_variation = MovementPrimitivesApproximator(self.promp, lr=self.lr)
        policy_variation.w = varied_weights
        return policy_variation, deltas


    def update(self, gradient):
        self.w += self.lr * gradient.reshape((self.promp.num_bases, self.promp.num_dof))


    # def update(self, Uw, Ew):
    #     self.Uw = Uw
    #     self.Ew = Ew


    def predict(self, input):
        # get the t-th timestep
        prediction = self.promp.get_trajectory_from_weights(self.w)[self.t]
        # prediction = self.traj[self.t]
        self.t = self.t+1 if self.t < self.promp.timesteps-1 else 0
        #if (self.t == self.promp.timesteps):
        #    self.t = 0
        return prediction
