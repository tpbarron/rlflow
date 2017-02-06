from __future__ import print_function

import gym
import tensorflow as tf
import tensorlayer as tl

from rlflow.policies.f_approx import LinearApproximator
from rlflow.algos.grad import PolicyGradient


if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    in_dimen = env.observation_space.shape[0]
    out_dimen = env.action_space.n

    name_scope = 'lin_approx'
    with tf.name_scope(name_scope) as scope:
        input_tensor = tf.placeholder(tf.float32, shape=[None, in_dimen], name='policy_input_'+name_scope)
        linear = tl.layers.InputLayer(input_tensor, name='input1_'+name_scope)
        linear = tl.layers.DenseLayer(linear, out_dimen, act=tf.nn.softmax, name='dense1_'+name_scope)

    lin_approx = LinearApproximator([input_tensor],
                                    linear,
                                    LinearApproximator.TYPE_PG)

    pg = PolicyGradient(env,
                        lin_approx,
                        episode_len=1000,
                        discount=0.99,
                        optimizer=tf.train.AdamOptimizer(learning_rate=0.01))

    pg.train(max_episodes=10000)

    rewards = pg.test(10)
    average_reward = float(sum(rewards) / len(rewards))
    print ("Average: ", average_reward)
