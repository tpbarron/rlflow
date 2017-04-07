
import numpy as np

from .input_stream_processor import InputStreamProcessor

class InputStreamSequentialProcessor(InputStreamProcessor):

    def __init__(self, observations=4):
        self.observations = observations
        self.current_sequence = []


    def reset(self):
        """
        This method is called at the start of every episode, use to reset state
        if necessary
        """
        self.current_sequence = []


    def process_observation(self, obs):
        """
        Take in the current observation, do any necessary processing and return
        the processed observation.

        A return value of None indicates that there is no observation yet. A
        random action will be taken.
        """
        self.current_sequence.append(obs)
        l = len(self.current_sequence)

        if l < self.observations:
            # task the last obs and repeat self.observations-l+1 times
            # 4-1+1 = 4
            # 4-2+1 = 3
            # ...
            extra = np.repeat(obs.reshape(tuple(obs.shape)+(1,)), self.observations-l, axis=len(obs.shape))
            concat = np.concatenate((np.stack(self.current_sequence, axis=len(obs.shape)), extra), axis=len(obs.shape))
            # print (concat.shape)
            return concat

        if l > self.observations:
            self.current_sequence.pop(0)

        # convert current sequence to input
        # stacking essentially adds a single axis, want it to be after
        obs_seq = np.stack(self.current_sequence, axis=len(obs.shape))
        return obs_seq
