
import gym

from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Flatten, Reshape

from rlcore.memories.experience_replay import ExperienceReplay

class DQN(object):

    def __init__(self):
        self.env = gym.make('Breakout-v0')

        self.experience_replay = ExperienceReplay()

        self.discount = 0.99

        # define model
        self.model = self.build_model()
        self.target_model = self.build_model()
        self.copy_weights_to_target_model()


    def build_model(self):
        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4, 4), border_mode='same', input_shape=env.observation_space.shape, activation='relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2, 2), border_mode='same', activation='relu'))
        model.add(Convolution2D(64, 3, 3, subsample=(1, 1), border_mode='same', activation='relu'))
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(Dense(env.action_space.n, activation='sigmoid'))
        return model


    def copy_weights_to_target_model(self):
        self.target_model.set_weights(self.model.get_weights())


    def optimize(self):
        pass
