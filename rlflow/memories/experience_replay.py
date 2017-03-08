import collections
import random
import numpy as np

class ExperienceReplay(object):

    SARS = collections.namedtuple('SARS', ['S1', 'A', 'R', 'S2', 'T'])

    def __init__(self, state_shape=(1,), max_size=1000000):
        self.max_size = max_size
        self.cur_size = 0
        self.next_ind = 0

        self.S1 = np.empty([max_size]+list(state_shape))
        self.A = np.empty((max_size,))
        self.R = np.empty((max_size,))
        self.S2 = np.empty([max_size]+list(state_shape))
        self.T = np.empty((max_size,))
        # self.memory = []


    def add_element(self, s1, a, r, s2, t):
        """
        Add an element to the back of the memory
        Removes an element from the front if full
        """

        # add element at next_ind
        self.S1[self.next_ind] = s1
        self.A[self.next_ind] = a
        self.R[self.next_ind] = r
        self.S2[self.next_ind] = s2
        self.T[self.next_ind] = int(t)

        self.next_ind += 1
        if self.next_ind == self.max_size:
            self.next_ind = 0

        if self.cur_size < self.max_size:
            self.cur_size += 1


    def sample(self, n):
        """
        Sample n elements uniformly from the memory
        """
        indices = np.random.choice(self.cur_size, n, replace=False)

        s1 = np.take(self.S1, indices, axis=0)
        a = np.take(self.A, indices)
        r = np.take(self.R, indices)
        s2 = np.take(self.S2, indices, axis=0)
        t = np.take(self.T, indices)

        return s1, a, r, s2, t
        # sample_elements = []
        # for _ in range(n):
        #     sample_elements.append(self.memory[random.randint(0, len(self.memory)-1)])
        #
        # return sample_elements


    def size(self):
        return self.cur_size
