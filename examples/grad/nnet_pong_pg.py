from __future__ import print_function

import gym
import tensorflow as tf
import tflearn
from markov.core import rl_utils
from markov.core import tf_utils
from markov.policies.f_approx import Network
from markov.algos.grad import PolicyGradient

if __name__ == "__main__":
    env = gym.make("Pong-v0")

    with tf.Session() as sess:
        # Build neural network
        input_tensor = tflearn.input_data(shape=[None, 80, 80, 1]) #tf_utils.get_input_tensor_shape(env))
        net = tflearn.conv_2d(input_tensor, 16, 4, strides=2, activation='relu')
        # net = tflearn.conv_2d(input_tensor, 64, 4, strides=2, activation='relu')
        net = tflearn.flatten(net)
        net = tflearn.fully_connected(net, 16, activation='sigmoid')
        net = tflearn.fully_connected(net, env.action_space.n, activation='softmax')

        # initialize policy with network
        policy = Network(input_tensor,
                         net,
                         sess,
                         prediction_preprocessors=[rl_utils.prepro],
                         prediction_postprocessors=[rl_utils.sample, rl_utils.cast_int])

        # initialize algorithm with env, policy, session and other params
        pg = PolicyGradient(env,
                            policy,
                            session=sess,
                            episode_len=1000,
                            discount=True,
                            optimizer='adam')

        # start the training process
        pg.train(max_iterations=5000, gym_record=False)

        # see what we have learned
        average_reward = rl_utils.average_test_episodes(env,
                                                        policy,
                                                        10,
                                                        episode_len=pg.episode_len)
        print ("Average: ", average_reward)
