"""
prediction post processors
"""

import tensorflow as tf
import numpy as np

# Q learning DQN style
def max_q_value(x, d=1):
    """
    Return the index of the max, which would correspond to the
    action to be taken.
    """
    x = cast_int(tf.squeeze(tf.argmax(x, axis=d)))
    return x


def cast_int(x):
    """
    Cast the output of x to an int64
    """
    return tf.to_int64(x)


def sample(x):
    return tf.multinomial(tf.log(x), 1)


def pg_sample(x):
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


# def cast_int(x):
#     """
#     Cast the output of x to an int64
#     """
#     return tf.to_int64(x)


def prob(x):
    assert (isinstance(x, np.ndarray) and len(x) == 1) or isinstance(x, float)
    return 0 if np.random.uniform() < x else 1


# def sample(x):
#     assert isinstance(x, np.ndarray)
#     x = np.squeeze(x)
#     assert x.ndim == 1
#     # renormalize to avoid 'does not sum to 1 errors'
#     return np.random.choice(len(x), 1, p=x/x.sum())

#
# def sample_outputs(x):
#     """
#     Given array [x1, x2, x3, x4, x5]
#
#     Returns an array of the same shape of 0 or 1 where the entries are 0 with
#     probability x_i/1.0 and 1 and probability 1-(x_i/1.0). The outputs are assumed
#     to be between 0 and 1
#     """
#     assert isinstance(x, np.ndarray)
#     prob_vec = np.vectorize(prob, otypes=[np.int32])
#     sampled = prob_vec(x)
#     return sampled


def pong_outputs(x):
    assert isinstance(x, int)
    if x == 0:
        return 2
    else:
        return 3
