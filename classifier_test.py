import unittest
import numpy
import copy

import classifier


class TestNNet(unittest.TestCase):

  def __init__(self, *args, **kwargs):
    super(TestNNet, self).__init__(*args, **kwargs)

    self._vocabulary_size = 3
    self._embedding_dimension = 10
    self._context_window_size = 3
    self._num_hidden_layer = 5
    self._learning_rate = 0.001

    self._input = [0, 1, 2]

  def tearDown(self):
    self.assertTrue(len(self._input) == self._context_window_size)
    self.assertTrue(self._context_window_size % 2 == 1)
    nnet = classifier.NNet(vocabulary_size=self._vocabulary_size,
                           embedding_dimension=self._embedding_dimension,
                           context_window_size=self._context_window_size,
                           num_hidden_layer=self._num_hidden_layer,
                           learning_rate=self._learning_rate)

    input = numpy.array(self._input, dtype=numpy.int32)
    mutations = self._GetMutations()
    loss = nnet.Train(input, mutations)
    self.assertGreaterEqual(loss, 0)

  def test_DefaultShouldWork(self):
    pass

  def _GetMutations(self):
    mutations = []
    for i in range(self._vocabulary_size):
      mutation = copy.copy(self._input)
      mutation[self._context_window_size / 2] = i
      mutations.append(mutation)

    return numpy.array(mutations, dtype=numpy.int32)


if __name__ == '__main__':
  unittest.main()
