import time
import sys
import math
import numpy
import classifier
import load
import copy


def TrainWordEmbedding():
  context_window_size = 3
  embedding_dimension = 100
  num_hidden_layer = 10
  learning_rate = 0.01
  num_epoch = 100
  # TODO: This is psedo minibatch. The real matrix batch size
  # is really vocabulary_size. Change that to actually
  # configurable.
  batch_size = 50
  vocabulary_size, idx_word_dict, train_set = load.GetTrainSet(
      context_window_size)

  print('vocabulary_size: %d' % vocabulary_size)
  print('number of sentences: %d' % len(train_set))

  nnet = classifier.NNet(vocabulary_size=vocabulary_size,
                         embedding_dimension=embedding_dimension,
                         context_window_size=context_window_size,
                         num_hidden_layer=num_hidden_layer,
                         learning_rate=learning_rate)
  batch_num = int(math.ceil(len(train_set) / float(batch_size)))
  train_batches = [train_set[i * batch_size:(i + 1) * batch_size]
                   for i in range(batch_num)]
  for i in range(num_epoch):
    loss = 0
    sentence_num = 0
    start_time = time.time()
    for j in range(len(train_batches)):
      batch = train_batches[j]
      input_tensor = numpy.array(
          [[input for i in range(vocabulary_size)] for input in batch],
          dtype=numpy.int32)
      mutation_tensor = numpy.array(
          [_GetMutations(input, vocabulary_size, context_window_size)
           for input in batch],
          dtype=numpy.int32)

      loss += nnet.Train(input_tensor, mutation_tensor)
      sentence_num += len(batch)

      sys.stdout.write(
          '[learning] epoch %i >> %2.2f%% completed in %.2f (sec) << ===> loss: %.2f%s%s'
          % (i, (j + 1) * 100. / batch_num, time.time() - start_time,
             loss / sentence_num / vocabulary_size, ' ' * 20, '\n' if
             j == len(train_batches) - 1 else '\r'))
      sys.stdout.flush()


def _GetMutations(input, vocabulary_size, context_window_size):
  mutations = []
  for i in range(vocabulary_size):
    mutation = copy.copy(input)
    mutation[context_window_size / 2] = i
    mutations.append(mutation)

  return mutations


def Main():
  TrainWordEmbedding()


if __name__ == '__main__':
  Main()
