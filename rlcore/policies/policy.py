from __future__ import print_function
from __future__ import division
from __future__ import absolute_import


class Policy(object):

    TRAIN, TEST = range(2)

    def __init__(self, **kwargs):
        self.prediction_preprocessors = None
        self.prediction_postprocessors = None
        if ('prediction_preprocessors' in kwargs):
            self.prediction_preprocessors = kwargs['prediction_preprocessors']
        if ('prediction_postprocessors' in kwargs):
            self.prediction_postprocessors = kwargs['prediction_postprocessors']

        self.mode = Policy.TRAIN
        if ('mode' in kwargs):
            if kwargs['mode'] != Policy.TRAIN and kwargs['mode'] != Policy.TEST:
                raise KeyError('Invalid mode: must be Policy.TRAIN or Policy.TEST')
            self.mode = kwargs['mode']


    @property
    def is_deterministic(self):
        raise NotImplementedError


    def predict(self, input):
        raise NotImplementedError
