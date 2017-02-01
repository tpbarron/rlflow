from __future__ import print_function

import tensorflow as tf
from rlflow.policies.policy import Policy
from rlflow.core import tf_utils
from rlflow.core.output import output_processors


class FunctionApproximator(Policy):

    """
    TYPE_PG: the output represents actions probabilities
    TYPE_DQN: the output represents q-values for a certain number of discrete actions
    """
    TYPE_PG, TYPE_DQN, DEFAULT = range(3)


    def __init__(self, inputs, model, pol_type=DEFAULT, mode=Policy.TRAIN):
        super(FunctionApproximator, self).__init__(mode)
        # self.input_tensor = tf.get_collection(tf.GraphKeys.INPUTS)[0]
        self.sess = tf_utils.get_tf_session()
        self.inputs = inputs
        self.model = model
        self.output = self.model.outputs
        self.pol_type = pol_type

        # build prediction_model, adds some operators (that may not be differentiable)
        # to the network to convert network output to the environment input
        self.prediction = self.build_prediction_model(self.output)


    def build_prediction_model(self, model):
        prediction_model = None
        if self.pol_type == FunctionApproximator.TYPE_PG:
            prediction_model = output_processors.pg_sample(model)
        elif self.pol_type == FunctionApproximator.TYPE_DQN:
            prediction_model = output_processors.max_q_value(model)
        else:
            print ("Default or invalid policy type, setting prediction_model to be same as model")
            prediction_model = model
        return prediction_model


    def get_params(self):
        return self.model.all_params


    def predict(self, obs):
        raise NotImplementedError
