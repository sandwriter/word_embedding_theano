import unittest
import numpy
import copy

import classifier


class TestNNet(unittest.TestCase):

  def __init__(self, *args, **kwargs):
    super(TestNNet, self).__init__(*args, **kwargs)

    self._vocabulary_size = 6
    self._embedding_dimension = 10
    self._context_window_size = 3
    self._num_hidden_layer = 5
    self._learning_rate = 0.001

  def test_MiniBatchContainsOneInput(self):
    input = [0, 1, 2]
    nnet = classifier.NNet(vocabulary_size=self._vocabulary_size,
                           embedding_dimension=self._embedding_dimension,
                           context_window_size=self._context_window_size,
                           num_hidden_layer=self._num_hidden_layer,
                           learning_rate=self._learning_rate)

    input_tensor = numpy.array(
        [[input for i in range(self._vocabulary_size)]],
        dtype=numpy.int32)
    mutations = self._GetMutations(input)
    mutation_tensor = numpy.array([mutations], dtype=numpy.int32)
    loss = nnet.Train(input_tensor, mutation_tensor)
    self.assertGreaterEqual(loss, 0)

  def test_MiniBatchContainsMultipleInputs(self):
    inputs = [[0, 1, 2], [3, 4, 5]]
    nnet = classifier.NNet(vocabulary_size=self._vocabulary_size,
                           embedding_dimension=self._embedding_dimension,
                           context_window_size=self._context_window_size,
                           num_hidden_layer=self._num_hidden_layer,
                           learning_rate=self._learning_rate)

    input_tensor = numpy.array(
        [[input for i in range(self._vocabulary_size)] for input in inputs],
        dtype=numpy.int32)
    mutation_tensor = numpy.array(
        [self._GetMutations(input) for input in inputs],
        dtype=numpy.int32)
    loss = nnet.Train(input_tensor, mutation_tensor)
    self.assertGreaterEqual(loss, 0)

  def _GetMutations(self, input):
    mutations = []
    for i in range(self._vocabulary_size):
      mutation = copy.copy(input)
      mutation[self._context_window_size / 2] = i
      mutations.append(mutation)

    return mutations


if __name__ == '__main__':
  unittest.main()
