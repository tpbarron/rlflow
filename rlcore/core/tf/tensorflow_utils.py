import tensorflow as tf


#
# Mathematical convenience function
#

def stddev(x):
    x = tf.to_float(x)
    return tf.sqrt(tf.reduce_mean(tf.square(tf.abs
        (tf.sub(x, tf.fill(x.get_shape(), tf.reduce_mean(x)))))))


optimizer_map = {
    'sgd': tf.train.GradientDescentOptimizer(0.01),
    'adadelta': tf.train.AdadeltaOptimizer(),
    'adagrad': tf.train.AdagradOptimizer(0.01),
    'adagradda': tf.train.AdagradDAOptimizer(0.01),
    'momentum': tf.train.MomentumOptimizer(0.01, 0.01),
    'adam': tf.train.AdamOptimizer(),
    'ftrl': tf.train.FtrlOptimizer(0.01),
    'rmsprop': tf.train.RMSPromOptimizer(0.001),
}

def optimizer_from_str(o):
    """
    Return a TF optimizer given its name

    o = [sgd|adadelta|adagrad|adagradda|momentum|adam|ftrl|rmsprop]
    """
    if (o in optimizer_map):
        return optimizer_map[o]
    raise KeyError('Invalid optimizer type: ' + str(o))


if __name__ == "__main__":
    x = tf.constant([2,4,4,4,5,5,7,9])
    print (x)
    print type(x)
    sess = tf.Session()
    print sess.run(tf.fill(x.get_shape(), tf.reduce_mean(x)))
    print sess.run(stddev(x))
