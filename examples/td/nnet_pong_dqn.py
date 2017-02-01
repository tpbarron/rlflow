from __future__ import print_function

import gym
import tensorflow as tf
import tensorlayer as tl
#import tflearn


from rlflow.policies.f_approx import Network
from rlflow.algos.td import DQN
from rlflow.memories import ExperienceReplay
from rlflow.exploration.egreedy import EpsilonGreedy
from rlflow.core.input import InputStreamDownsamplerProcessor, InputStreamSequentialProcessor, InputStreamProcessor


def build_network(name_scope):
    w_init = tf.truncated_normal_initializer(stddev=0.01)
    b_init = tf.constant_initializer(value=0.0)

    with tf.name_scope(name_scope) as scope:
        input_tensor = tf.placeholder(tf.float32, shape=[None, 84, 84, 4], name='policy_input_'+name_scope)
        net = tl.layers.InputLayer(input_tensor, name='input1_'+name_scope)
        net = tl.layers.Conv2d(net, 16, (8, 8), (4, 4), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv1_'+name_scope)
        net = tl.layers.Conv2d(net, 32, (4, 4), (2, 2), act=tf.nn.relu, padding='SAME', W_init=w_init, b_init=b_init, name='conv2_'+name_scope)
        net = tl.layers.FlattenLayer(net, name='flatten1_'+name_scope)
        net = tl.layers.DenseLayer(net, 1024, act=tf.nn.relu, name='dense1_'+name_scope)
        net = tl.layers.DenseLayer(net, env.action_space.n, act=tf.identity, name='dense2_'+name_scope)

    return input_tensor, net



if __name__ == "__main__":
    env = gym.make("Pong-v0")

    input_tensor, net = build_network('train_net')
    network = Network([input_tensor],
                      net,
                      Network.TYPE_DQN)
    # net.print_layers()

    clone_input_tensor, clone_net = build_network('clone_net')
    clone_network = Network([clone_input_tensor],
                            clone_net,
                            Network.TYPE_DQN)
    # clone_net.print_layers()

    memory = ExperienceReplay(max_size=1000000)
    egreedy = EpsilonGreedy(0.9, 0.1, 100000)

    downsampler = InputStreamDownsamplerProcessor((84, 84), gray=True)
    sequential = InputStreamSequentialProcessor(observations=4)
    input_processor = InputStreamProcessor(processor_list=[downsampler, sequential])

    opt = tf.train.RMSPropOptimizer(learning_rate=0.001, momentum=0.95, epsilon=0.01)
    # opt = tflearn.optimizers.RMSProp(learning_rate=0.001, momentum=0.95, epsilon=0.01)

    dqn = DQN(env,
              network,
              clone_network,
              memory,
              egreedy,
              input_processor=input_processor,
              discount=0.99,
              learning_rate=0.001,
              optimizer=opt,
              memory_init_size=500,
              clip_gradients=(-10.0, 10.0),
              clone_frequency=10000)

    dqn.train(max_episodes=100000000, save_frequency=100)

    rewards = dqn.test(episodes=10)
    print ("Avg test reward: ", float(sum(rewards)) / len(rewards))
