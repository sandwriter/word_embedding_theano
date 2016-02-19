from mock import patch
import unittest

import load


class TestLoad(unittest.TestCase):

  def __init__(self, *args, **kwargs):
    super(TestLoad, self).__init__(*args, **kwargs)

    self._train_set = None
    self._test_set = None
    self._dicts = None

    self._patchers = []
    self._mock_cpickle_load = None

  def setUp(self):
    patcher = patch('load.os.path.isfile')
    patcher.start()
    self._patchers.append(patcher)
    patcher = patch('load.os.system')
    patcher.start()
    self._patchers.append(patcher)
    patcher = patch('__builtin__.open')
    patcher.start()
    self._patchers.append(patcher)
    patcher = patch('load.cPickle.load')
    self._mock_cpickle_load = patcher.start()
    self._patchers.append(patcher)

  def tearDown(self):
    for patcher in self._patchers:
      patcher.stop()

  def test_GetAtisData(self):
    expected_train_word_seq_seq = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    train_set = (expected_train_word_seq_seq, None, None)
    word_idx_dict = {'a': 0,
                     'b': 1,
                     'c': 2,
                     'd': 3,
                     'e': 4,
                     'f': 5,
                     'g': 6,
                     'h': 7,
                     'i': 8,
                     'j': 9}
    dicts = {'words2idx': word_idx_dict}

    self._mock_cpickle_load.return_value = (train_set, None, dicts)
    actual_vocabulary_size, actual_idx_word_dict, actual_train_word_seq_seq = load.GetAtisData(
    )

    expected_vocabulary_size = 10
    expected_idx_word_dict = {0: 'a',
                              1: 'b',
                              2: 'c',
                              3: 'd',
                              4: 'e',
                              5: 'f',
                              6: 'g',
                              7: 'h',
                              8: 'i',
                              9: 'j'}

    self.assertEqual(expected_vocabulary_size, actual_vocabulary_size)
    self.assertEqual(expected_idx_word_dict, actual_idx_word_dict)
    self.assertEqual(expected_train_word_seq_seq, actual_train_word_seq_seq)

  def test_ContextWindowNormal(self):
    word_seq = [1, 2, 3, 4, 5]
    expected_contexts = [[1, 2, 3], [2, 3, 4], [3, 4, 5]]

    self.assertEqual(expected_contexts, load.ContextWindow(word_seq, 3))

  def test_ContextWindowTooLarge(self):
    word_seq = [1, 2, 3, 4, 5]
    expected_contexts = []

    self.assertEqual(expected_contexts, load.ContextWindow(word_seq, 10))

  def test_ContextWindowSameLength(self):
    word_seq = [1, 2, 3, 4, 5]
    expected_contexts = [[1, 2, 3, 4, 5]]

    self.assertEqual(expected_contexts, load.ContextWindow(word_seq, 5))

  def test_GetTrainSetIllegalContextWindowSize(self):
    self.assertRaises(AssertionError, load.GetTrainSet, 2)

  def test_GetTrainSetNormal(self):
    expected_train_word_seq_seq = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]
    train_set = (expected_train_word_seq_seq, None, None)
    word_idx_dict = {'a': 0,
                     'b': 1,
                     'c': 2,
                     'd': 3,
                     'e': 4,
                     'f': 5,
                     'g': 6,
                     'h': 7,
                     'i': 8,
                     'j': 9}
    dicts = {'words2idx': word_idx_dict}

    self._mock_cpickle_load.return_value = (train_set, None, dicts)

    actual_vocabulary_size, actual_idx_word_dict, actual_train_set = load.GetTrainSet(
        3)
    expected_vocabulary_size = 10
    expected_idx_word_dict = {0: 'a',
                              1: 'b',
                              2: 'c',
                              3: 'd',
                              4: 'e',
                              5: 'f',
                              6: 'g',
                              7: 'h',
                              8: 'i',
                              9: 'j'}

    expected_train_set = [[0, 1, 2], [1, 2, 3], [2, 3, 4], [5, 6, 7], [6, 7, 8],
                          [7, 8, 9]]

    self.assertEqual(expected_vocabulary_size, actual_vocabulary_size)
    self.assertEqual(expected_idx_word_dict, actual_idx_word_dict)
    self.assertEqual(expected_train_set, actual_train_set)


if __name__ == '__main__':
  unittest.main()
