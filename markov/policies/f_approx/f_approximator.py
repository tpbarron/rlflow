from __future__ import print_function

from markov.policies.policy import Policy
from markov.core.output import output_processors


class FunctionApproximator(Policy):

    TYPE_PG, TYPE_DQN = range(2)

    def __init__(self, input_tensor, model, session, pol_type, **kwargs):
        super(FunctionApproximator, self).__init__(**kwargs)
        self.input_tensor = input_tensor
        self.model = model
        self.prediction_model = None
        self.sess = session
        self.pol_type = pol_type
        self.set_prediction_model()


    def set_prediction_model(self):
        if self.pol_type == FunctionApproximator.TYPE_PG:
            print ("Setting prediction_model for type pg")
            self.prediction_model = output_processors.pg_sample(self.model)
        elif self.pol_type == FunctionApproximator.TYPE_DQN:
            self.prediction_model = output_processors.max_q_value(self.model)


    def get_num_weights(self):
        raise NotImplementedError


    def get_weight_variation(self):
        raise NotImplementedError


    def update(self, gradient):
        raise NotImplementedError


    def predict(self, obs):
        raise NotImplementedError
