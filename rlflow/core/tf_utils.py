from __future__ import print_function

import tensorflow as tf

TF_SESSION = None

def get_tf_session():
    global TF_SESSION
    if TF_SESSION is None:
        TF_SESSION = tf.Session()
    return TF_SESSION


def set_tf_session(sess):
    global TF_SESSION
    TF_SESSION = sess


def build_policy_copy_ops(policy1, policy2):
    """
    Build the ops to copy weights from policy1 to policy2
    """
    clone_ops = [var2.assign(var1) for var1, var2 in zip(policy1.get_params(), policy2.get_params())]
    return clone_ops

#
# Mathematical convenience function
#

def stddev(x):
    x = tf.to_float(x)
    return tf.sqrt(tf.reduce_mean(tf.square(tf.abs
        (tf.sub(x, tf.fill(x.get_shape(), tf.reduce_mean(x)))))))


#
# Get input tensor shape from env obs space
#
def get_input_tensor_shape(env):
    obs_shape = tuple([None]+list(env.observation_space.shape))
    return obs_shape


#
# Tensorflow decorators
#
def wrap_session(f):
    def enclosed(*args, **kwargs):
        if (tf.get_default_session() != None):
            # if there is already a session open, use the default
            with tf.get_default_session() as sess:
                f(*args, **kwargs)
        else:
            # otherwise create a new one
            with tf.Session() as sess:
                f(*args, **kwargs)
    return enclosed


if __name__ == "__main__":
    x = tf.constant([2,4,4,4,5,5,7,9])
    print (x)
    print (type(x))
    sess = tf.Session()
    print (sess.run(tf.fill(x.get_shape(), tf.reduce_mean(x))))
    print (sess.run(stddev(x)))
