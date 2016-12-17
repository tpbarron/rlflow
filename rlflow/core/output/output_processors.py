"""
prediction post processors
"""

import tensorflow as tf
import numpy as np


def cast_int(x):
    """
    Cast the output of x to an int64
    """
    return tf.to_int64(x)


# Q learning DQN style
def max_q_value(x, d=1):
    """
    Return the index of the max, which would correspond to the
    action to be taken.
    """
    x = cast_int(tf.squeeze(tf.argmax(x, axis=d)))
    return x


def sample(x):
    return tf.multinomial(tf.log(x), 1)


def pg_sample(x):
    """
    Sample from the outputs as probabilities,
    then return int of index chosen
    """
    x = cast_int(tf.squeeze(sample(x)))
    return x





def sign(x):
    """
    Take in a float and return 0 if negative and 1 otherwise

    >>> sign(0.1)
    >>> 1
    >>> sign(-0.5)
    >>> 0
    """
    assert (isinstance(x, np.ndarray) and len(x) == 1) or isinstance(x, np.float32) or isinstance(x, float)
    x = float(x)
    if x < 0:
        return 0
    else:
        return 1


def prob(x):
    assert (isinstance(x, np.ndarray) and len(x) == 1) or isinstance(x, float)
    return 0 if np.random.uniform() < x else 1
