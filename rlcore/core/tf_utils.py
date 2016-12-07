import tensorflow as tf


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
    print type(x)
    sess = tf.Session()
    print sess.run(tf.fill(x.get_shape(), tf.reduce_mean(x)))
    print sess.run(stddev(x))
