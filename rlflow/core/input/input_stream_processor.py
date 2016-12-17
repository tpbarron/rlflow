

class InputStreamProcessor(object):
    """
    This class represents a stateful input processor

    One can define a function such that the input to the agent at time t
    is not only the observation at time t but some function of the previous
    observations as well.

    This takes a list of input processors and applies them to the observation
    IN THE ORDER they are given in the constructor
    """

    def __init__(self, processor_list=None):
        self.processor_list = processor_list


    def reset(self):
        """
        This method is called at the start of every episode, use to reset state
        if necessary
        """
        if self.processor_list is None:
            return

        for p in self.processor_list:
            p.reset()


    def process_observation(self, obs):
        """
        Take in the current observation, do any necessary processing and return
        the processed observation.
        """
        if self.processor_list is None:
            return obs

        for p in self.processor_list:
            obs = p.process_observation(obs)
            # if a processor doesn't have the necessary information to return an observation yet
            # it returns None
            if obs is None:
                return None

        return obs
