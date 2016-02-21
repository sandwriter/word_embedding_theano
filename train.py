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
  vocabulary_size, idx_word_dict, train_set = load.GetTrainSet(
      context_window_size)

  print('vocabulary_size: %d' % vocabulary_size)
  print('number of sentences: %d' % len(train_set))

  nnet = classifier.NNet(vocabulary_size=vocabulary_size,
                         embedding_dimension=embedding_dimension,
                         context_window_size=context_window_size,
                         num_hidden_layer=num_hidden_layer,
                         learning_rate=learning_rate)
  for i in range(num_epoch):
    loss = 0
    for word_seq in train_set:
      input = numpy.array(word_seq, dtype=numpy.int32)
      mutations = []
      for j in range(vocabulary_size):
        mutation = copy.copy(word_seq)
        mutation[context_window_size / 2] = j
        mutations.append(mutation)

      mutation_array = numpy.array(mutations, dtype=numpy.int32)
      loss += nnet.Train(input, mutation_array)

    print('epoch %d: loss: %f' % (i, loss))


def Main():
  TrainWordEmbedding()


if __name__ == '__main__':
  Main()
