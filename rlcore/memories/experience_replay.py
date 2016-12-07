import collections
import random

class ExperienceReplay(object):

    SARS = collections.namedtuple('SARS', ['S1', 'A', 'R', 'S2'])

    def __init__(self,
                 size=1000000):
        self.size = size
        self.memory = []


    def add_element(self, e):
        """
        Add an element to the back of the memory
        Removes an element from the front if full
        """
        # check if at max size
        if (len(self.memory) ==  self.size):
            self.memory.pop(0)

        sars = ExperienceReplay.SARS(e[0], e[1], e[2], e[3])
        self.memory.append(sars)


    def sample(self, n):
        """
        Sample n elements uniformly from the memory
        """
        sample_elements = []
        for _ in range(n):
            sample_elements.append(self.memory[random.randint(0, len(self.memory)-1)])

        return sample_elements
