=====
Getting Started
=====

To use RLFlow in a project:

.. code-block:: python

    from __future__ import print_function

    import gym
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
        # Create the desired environment
        env = gym.make("CartPole-v0")

        # Set up the network we want to use. In this case it is a simple
        # linear model but can be an arbitrary structure, just be sure the
        # inputs and outputs are proper. Here we use softmax outputs since
        # we want to sample from them as probabilities.
        inputs, outputs = build_network("train_policy", env)

        # Create the approximator object. This is just and abstraction of the
        # model structure
        policy = Network(inputs, outputs, scope="train_policy")

        # Now instantiate our algorithm, a basic policy gradient implementation.
        pg = PolicyGradient(env,
                            policy,
                            episode_len=500,
                            discount=0.99,
                            optimizer=tf.train.AdamOptimizer(learning_rate=0.005))

        # Run the algorithm for a desired number of episodes. In this call
        # one can also specify whether to record data to upload to the
        # OpenAI gym evaluation system.
        pg.train(max_episodes=5000,
                 save_frequency=10,
                 render_train=True)

        # We could restore a previous model if desired
        # pg.restore(ckpt_file="/tmp/rlflow/model.ckpt-###")

        # Now just test what we have learned!
        rewards = pg.test(episodes=10,
                          record_experience=True)
        print ("Average: ", float(sum(rewards)) / len(rewards))
