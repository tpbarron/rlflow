from __future__ import print_function

import gym
import numpy as np
import tensorflow as tf

from rlflow.core import tf_utils
from rlflow.policies.f_approx import Network
from rlflow.algos.grad import PolicyGradient


def build_network(name_scope, env):
    w_init_dense = tf.truncated_normal_initializer() #contrib.layers.xavier_initializer()
    b_init = tf.constant_initializer(value=0.0)

    with tf.variable_scope(name_scope):
        input_tensor = tf.placeholder(tf.float32,
                                      shape=tf_utils.get_input_tensor_shape(env),
                                      name='policy_input_'+name_scope)
        net = tf.contrib.layers.fully_connected(input_tensor,
                                                32, #env.action_space.n, #32,
                                                activation_fn=tf.nn.tanh, #sigmoid,
                                                weights_initializer=w_init_dense,
                                                biases_initializer=b_init,
                                                scope='dense1_'+name_scope)
        net = tf.contrib.layers.fully_connected(net,
                                                env.action_space.n,
                                                weights_initializer=w_init_dense,
                                                biases_initializer=b_init,
                                                scope='dense2_'+name_scope)
        net = tf.contrib.layers.softmax(net)

    return [input_tensor], [net]



if __name__ == "__main__":
    env = gym.make("CartPole-v1")

    inputs, outputs = build_network("train_policy", env)
    policy = Network(inputs, outputs, scope="train_policy")

    pg = PolicyGradient(env,
                        policy,
                        episode_len=np.inf,
                        discount=0.99,
                        optimizer=tf.train.RMSPropOptimizer(learning_rate=0.01))

    pg.train(max_episodes=5000,
             save_frequency=10,
             render_train=True)

    # pg.restore(ckpt_file="/tmp/rlflow/model.ckpt-320")

    rewards = pg.test(episodes=10,
                      record_experience=True)
    print ("Average: ", float(sum(rewards)) / len(rewards))
