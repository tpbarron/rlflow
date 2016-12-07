from __future__ import print_function

import gym
import tensorflow as tf
import tflearn

from rlcore.core import rl_utils
from rlcore.core import tf_utils
from rlcore.policies.f_approx import Network
from rlcore.algos.grad import PolicyGradient

if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    with tf.Session() as sess:
        # Build neural network
        input_tensor = tflearn.input_data(shape=tf_utils.get_input_tensor_shape(env))
        net = tflearn.fully_connected(input_tensor, 4, activation='sigmoid')
        net = tflearn.fully_connected(net, env.action_space.n, activation='softmax')

        policy = Network(input_tensor,
                         net,
                         sess,
                         prediction_postprocessors=[rl_utils.sample, rl_utils.cast_int])
        pg = PolicyGradient(env,
                            policy,
                            session=sess,
                            episode_len=1000,
                            discount=True,
                            optimizer='adam')

        pg.train(max_iterations=5000, gym_record=False)

        average_reward = rl_utils.average_test_episodes(env,
                                                        policy,
                                                        10,
                                                        episode_len=pg.episode_len)
        print ("Average: ", average_reward)
