
import numpy as np

from .input_stream_processor import InputStreamProcessor

class InputStreamSequentialProcessor(InputStreamProcessor):

    def __init__(self, frames=4):
        self.frames = frames
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
        if len(self.current_sequence) < self.frames:
            return None

        if len(self.current_sequence) > self.frames:
            self.current_sequence.pop()

        # convert current sequence to input
        # stacking essentially adds a single axis, want it to be after
        obs_seq = np.stack(self.current_sequence, axis=len(obs.shape))
        return obs_seq
