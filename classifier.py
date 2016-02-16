import sys
import numpy
import cPickle
import theano
import theano.tensor as T


class NNet(object):

  def __init__(self, vocabulary_size, embedding_dimension, context_window_size,
               num_hidden_layer):
    self.x = theano.imatrix('x')

    self.embedding = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (
        vocabulary_size + 1, embedding_dimension)).astype(theano.config.floatX))

    self.w_input = theano.shared(0.2 * numpy.random.uniform(
        -1.0, 1.0, (context_window_size * embedding_dimension,
                    num_hidden_layer)).astype(theano.config.floatX))
    self.b_input = theano.shared(numpy.random.uniform(-1.0, 1.0, (
        num_hidden_layer,)).astype(theano.config.floatX))

    self.w_classifier = theano.shared(0.2 * numpy.random.uniform(
        -1.0, 1.0, (num_hidden_layer)).astype(theano.config.floatX))
    self.b_classifier = theano.shared(numpy.random.uniform(-1.0, 1.0).astype(
        theano.config.floatX))

    self.input = embedding[x].reshape(x.shape[0],
                                      context_window_size * embedding_dimension)
    self.activation = T.tanh(T.dot(input, w_input) + b_input)
    self.output = T.tanh(T.dot(activation, w_classifier) + b_classifier)

    # mutated x.
