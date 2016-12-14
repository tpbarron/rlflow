from __future__ import print_function

import tensorflow as tf
from markov.policies.policy import Policy
from markov.core.output import output_processors


class FunctionApproximator(Policy):

    TYPE_PG, TYPE_DQN = range(2)

    def __init__(self, input_tensor, model, session, pol_type, use_clone_net=False, **kwargs):
        super(FunctionApproximator, self).__init__(**kwargs)
        self.input_tensor = input_tensor
        self.model = model
        self.sess = session
        self.pol_type = pol_type

        # build prediction_model, adds some operators (that may not be differentiable)
        # to the network to convert network output to to the policy input
        self.prediction_model = self.build_prediction_model(self.model)

        if use_clone_net:
            self.clone_graph = tf.Graph()
            self.clone_model = None
            self.clone_prediction_model = None

            self.build_clone_model()
            self.clone() # copy initial state


    def build_prediction_model(self, model):
        prediction_model = None
        if self.pol_type == FunctionApproximator.TYPE_PG:
            prediction_model = output_processors.pg_sample(model)
        elif self.pol_type == FunctionApproximator.TYPE_DQN:
            prediction_model = output_processors.max_q_value(model)
        else:
            raise ValueError("Invalid policy type")
        return prediction_model


    def build_clone_model(self):
        # copy variables
        tf_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        for var in tf_vars:
            tf.contrib.copy_graph.copy_variable_to_graph(var, self.clone_graph, scope='clone')

        # copy graph structure
        with self.clone_graph.as_default():
            self.clone_model = tf.contrib.copy_graph.copy_op_to_graph(self.model,
                                                                      self.clone_graph,
                                                                      tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='clone'),
                                                                      scope='clone')

            self.clone_prediction_model = tf.contrib.copy_graph.copy_op_to_graph(self.prediction_model,
                                                                                 self.clone_graph,
                                                                                 tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='clone'),
                                                                                 scope='clone')


    def clone(self):
        # assign variables
        with self.clone_graph.as_default():
            temp_sess = tf.Session()
            temp_sess.run(tf.global_variables_initializer())
            for var1, var2 in zip(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='clone'), tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)):
                temp_sess.run(var1.assign(var2))
            temp_sess.close()
            

    def update(self, gradient):
        raise NotImplementedError


    def predict(self, obs):
        raise NotImplementedError
