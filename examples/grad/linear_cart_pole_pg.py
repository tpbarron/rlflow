from __future__ import print_function

import gym
import tensorflow as tf
import tflearn

from rlcore.core import rl_utils
from rlcore.policies.f_approx import LinearApproximator
from rlcore.algos.grad import PolicyGradient

if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    with tf.Session() as sess:

        in_dimen = env.observation_space.shape[0]
        out_dimen = env.action_space.n

        input_tensor = tflearn.input_data(shape=[None, in_dimen])
        linear = tflearn.fully_connected(input_tensor, out_dimen, activation='linear')
        linear = tflearn.softmax(linear) # use softmax since we want probabilities for outputs

        lin_approx = LinearApproximator(input_tensor,
                                        linear,
                                        sess,
                                        prediction_postprocessors=[rl_utils.sample, rl_utils.cast_int])
        pg = PolicyGradient(env,
                            lin_approx,
                            sess,
                            episode_len=100,
                            discount=0.9,
                            optimizer='adam')

        pg.train(max_iterations=1000, gym_record=False)

        average_reward = rl_utils.average_test_episodes(env,
                                                        lin_approx,
                                                        10,
                                                        episode_len=pg.episode_len)
        print ("Average: ", average_reward)
