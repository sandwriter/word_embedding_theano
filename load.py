import os
import gzip
import cPickle

DATA_DUMP = 'atis.pkl'


def GetAtisData():
  if not os.path.isfile(DATA_DUMP):
    os.system(
        'wget -O %s https://www.dropbox.com/s/3lxl9jsbw0j7h8a/atis.pkl?dl=0' %
        DATA_DUMP)
  f = open(DATA_DUMP, 'rb')
  train_set, test_set, dicts = cPickle.load(f)

  word_idx_dict = dicts['words2idx']
  idx_word_dict = dict((v, k) for k, v in word_idx_dict.iteritems())
  vocabulary_size = len(word_idx_dict)
  train_word_seq_seq, _, _ = train_set

  return vocabulary_size, idx_word_dict, train_word_seq_seq


def ContextWindow(word_seq, context_window_size):
  contexts = []
  for i in range(max(0, len(word_seq) - context_window_size + 1)):
    contexts.append(word_seq[i:i + context_window_size])

  return contexts


def GetTrainSet(context_window_size):
  assert context_window_size % 2 == 1, 'context_window_size should be odd number.'
  vocabulary_size, idx_word_dict, word_seq_seq = GetAtisData()
  train_set = []
  for word_seq in word_seq_seq:
    train_set.extend(ContextWindow(word_seq, context_window_size))
  return vocabulary_size, idx_word_dict, train_set
