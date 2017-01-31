=======
History
=======

0.1.4 (????-??-??)
------------------
* Either a TFLearn optimizer object or a string key can be passed to an algorithm
* Fix replay memory bug where size would only reach max size - 1
* Improve discount_rewards efficiency with scipy.signal.lfilter

* TODO: improve session management so user doesn't have to create session
* TODO: implement policy duplicate function, where the duplicated ops are in the same
  graph with a different scope


0.1.2/3 (2016-12-17)
------------------
* Improving meta data and fixing __init__ scripts to load subpackages properly


0.1.0 (2016-12-16)
------------------
* First release on PyPI.
