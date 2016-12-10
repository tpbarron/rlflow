

class InputStreamProcessor(object):
    """
    This class represents a stateful input processor

    One can define a function such that the input to the agent at time t
    is not only the observation at time t but some function of the previous
    observations as well.
    """

    def __init__(self):
        pass


    def process_observation(self, obs):
        """
        Take in the current observation, do any necessary processing and return
        the processed observation.
        """
        return obs
