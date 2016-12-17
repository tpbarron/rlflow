
import numpy as np
import PIL
from .input_stream_processor import InputStreamProcessor

class InputStreamDownsamplerProcessor(InputStreamProcessor):

    def __init__(self, size, gray=False):
        """
        The size to downsample to and whether to convert to grayscales
        """
        self.size = size
        self.gray = gray


    def reset(self):
        """
        This method is called at the start of every episode, use to reset state
        if necessary
        """
        return


    def process_observation(self, obs):
        """
        Take in the current observation, do any necessary processing and return
        the processed observation.
        """
        assert len(obs.shape) == 2 or len(obs.shape) == 3

        I = PIL.Image.fromarray(obs)
        if len(obs.shape) == 3 and self.gray:
            I = I.convert('L')

        I = I.resize(self.size, PIL.Image.NEAREST)
        obs = np.array(I)
        return obs
