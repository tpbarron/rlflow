=====
Getting Started
=====

To use RLFlow in a project:

.. code-block:: python

    import gym
    import tensorflow as tf
    import tflearn

    from rlflow.policies.f_approx import LinearApproximator
    from rlflow.algos.grad import PolicyGradient

    if __name__ == "__main__":

        # Create the desired environment
        env = gym.make("CartPole-v0")

        with tf.Session() as sess:
            in_dimen = env.observation_space.shape[0]
            out_dimen = env.action_space.n

            # Set up the network we want to use. In this case it is a simple
            # linear model but can be an arbitrary structure, just be sure the
            # inputs and outputs are proper. Here we use softmax outputs since
            # we want to sample from them as probabilities.
            input_tensor = tflearn.input_data(shape=[None, in_dimen])
            linear = tflearn.fully_connected(input_tensor, out_dimen, activation='linear')
            linear = tflearn.softmax(linear)

            # Create the approximator object. Note that you can specify different output
            # processing. In this case TYPE_PG indicates the outputs should be sampled.
            # TYPE_DQN indicates the outputs are Q values.
            lin_approx = LinearApproximator(linear,
                                            sess,
                                            LinearApproximator.TYPE_PG)

            # Now instantiate our algorithm, a basic policy gradient implementation.
            pg = PolicyGradient(env,
                                lin_approx,
                                sess,
                                episode_len=100,
                                discount=0.9,
                                optimizer='adam')

            # Run the algorithm for a desired number of episodes. In this call
            # one can also specify whether to record data to upload to the
            # OpenAI gym evaluation system.
            pg.train(max_episodes=1000)

            # Now just test what we have learned!
            rewards = pg.test(10)
            average_reward = float(sum(rewards) / len(rewards))
            print ("Average: ", average_reward)
