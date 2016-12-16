from __future__ import print_function

import gym
import tensorflow as tf
import tflearn

from markov.policies.f_approx import LinearApproximator
from markov.algos.grad import PolicyGradient

if __name__ == "__main__":
    env = gym.make("CartPole-v0")

    with tf.Session() as sess:
        in_dimen = env.observation_space.shape[0]
        out_dimen = env.action_space.n

        input_tensor = tflearn.input_data(shape=[None, in_dimen])
        linear = tflearn.fully_connected(input_tensor, out_dimen, activation='linear')
        linear = tflearn.softmax(linear) # use softmax since we want probabilities for outputs

        lin_approx = LinearApproximator(linear,
                                        sess,
                                        LinearApproximator.TYPE_PG)

        pg = PolicyGradient(env,
                            lin_approx,
                            sess,
                            episode_len=100,
                            discount=0.9,
                            optimizer='adam')

        pg.train(max_episodes=1000)

        rewards = pg.test(10)
        average_reward = float(sum(rewards) / len(rewards))
        print ("Average: ", average_reward)
