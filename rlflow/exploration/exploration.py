
class Exploration(object):

    ADDITIVE_EXPLORATION, CONDITIONAL_EXPLORATION = range(2)

    def __init__(self, exp_type):
        self.exp_type = exp_type


    def explore(self, x):
        """
        Do operation on x and return new value
        """
        raise NotImplementedError
