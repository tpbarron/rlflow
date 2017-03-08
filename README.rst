===============================
RLFlow
===============================


.. image:: https://img.shields.io/pypi/v/rlflow.svg
        :target: https://pypi.python.org/pypi/rlflow

.. image:: https://img.shields.io/travis/tpbarron/rlflow.svg
        :target: https://travis-ci.org/tpbarron/rlflow

.. image:: https://readthedocs.org/projects/rlflow/badge/?version=latest
        :target: https://rlflow.readthedocs.io/en/latest/?badge=latest
        :alt: Documentation Status

A framework for learning about and experimenting with reinforcement learning algorithms.
It is built on top of TensorFlow and interfaces with OpenAI gym (universe should work, too).
It aims to be as modular as possible so that new algorithms and ideas can easily be tested.
I started it to gain a better understanding of core RL algorithms and maybe it can be
useful for others as well.

Features
--------

Algorithms (future algorithms italicized):

  - MDP algorithms

      + Value iteration
      + Policy iteration

  - Temporal Difference Learning

      + SARSA
      + Deep Q-Learning
      + *Policy gradient Q-learning*

  - Gradient algorithms

      + Vanilla policy gradient
      + *Deterministic policy gradient*
      + *Natural policy gradient*

  - Gradient-Free algorithms

      + *Cross entropy method*

Function approximators (defined by TFLearn model):

  - Linear
  - Neural network
  - *RBF*

Works with any OpenAI gym environment.


Future Enhancements
-------------------

* Improved TensorBoard logging
* Improved model snapshotting to include exploration states, memories, etc.
* Any suggestions?


Fixes
------------------
* Errors / warnings on TensorFlow session save


License
------------------

* Free software: MIT license
* Documentation: https://rlflow.readthedocs.io.
