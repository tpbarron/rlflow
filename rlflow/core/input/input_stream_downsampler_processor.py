
import numpy as np
import PIL
from .input_stream_processor import InputStreamProcessor

class InputStreamDownsamplerProcessor(InputStreamProcessor):

    def __init__(self, size, gray=False, scale=True):
        """
        The size to downsample to and whether to convert to grayscales
          size: a tuple representing the size of downsample to
          gray: boolean convert rgb to gray
          scale: boolean divides all pixels by 255.
        """
        self.size = size
        self.gray = gray
        self.scale = scale



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
        assert obs.ndim == 2 or obs.ndim == 3

        I = PIL.Image.fromarray(obs)
        if obs.ndim == 3 and self.gray:
            I = I.convert('L')

        I = I.resize(self.size, PIL.Image.BILINEAR)
        obs = np.array(I, dtype=np.float32)

        if self.scale:
            obs /= 255.

        return obs
