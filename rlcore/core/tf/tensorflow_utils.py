import tensorflow as tf


#
# Mathematical convenience function
#

def stddev(x):
    x = tf.to_float(x)
    return tf.sqrt(tf.reduce_mean(tf.square(tf.abs
        (tf.sub(x, tf.fill(x.get_shape(), tf.reduce_mean(x)))))))


#
# Neural network layers
#

def fc():
    """
    Fully connected layer
    """
    pass


def conv():
    """
    Convolutional layer
    """
    pass


#
# Network activations
#

def relu():
    """
    Rectified Linear activation
    """
    pass


def softmax():
    """
    Softmax activation
    """
    pass


def tanh():
    """
    Hyperbolic tangent activation
    """
    pass


#
# Layer normalization
#

def batch_normalization():
    """

    """
    pass
    

if __name__ == "__main__":
    x = tf.constant([2,4,4,4,5,5,7,9])
    print (x)
    print type(x)
    sess = tf.Session()
    print sess.run(tf.fill(x.get_shape(), tf.reduce_mean(x)))
    print sess.run(stddev(x))
