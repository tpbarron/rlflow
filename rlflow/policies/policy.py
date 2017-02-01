from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


class Policy(object):

    TRAIN, TEST = range(2)

    def __init__(self, mode):
        # self.prediction_preprocessors = None
        # self.prediction_postprocessors = None
        # if ('prediction_preprocessors' in kwargs):
        #     self.prediction_preprocessors = kwargs['prediction_preprocessors']
        # if ('prediction_postprocessors' in kwargs):
        #     self.prediction_postprocessors = kwargs['prediction_postprocessors']

        if mode != Policy.TRAIN and mode != Policy.TEST:
            raise Exception('Invalid mode: must be Policy.TRAIN or Policy.TEST')


    def predict(self, x):
        raise NotImplementedError
