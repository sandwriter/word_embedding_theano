import sys
# import pydot
import numpy
import theano
import theano.tensor as T

# theano.config.optimizer = 'None'
# theano.config.exception_verbosity = 'high'
# theano.config.linker = 'py'


class NNet(object):

  def __init__(self, vocabulary_size, embedding_dimension, context_window_size,
               num_hidden_layer, learning_rate):
    self.vocabulary_size = vocabulary_size
    self.embedding_dimension = embedding_dimension
    self.context_window_size = context_window_size
    self.num_hidden_layer = num_hidden_layer
    self.learning_rate = learning_rate

    self.embedding = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (
        vocabulary_size, embedding_dimension)).astype(theano.config.floatX),
                                   name='embedding')

    self.w_input = theano.shared(0.2 * numpy.random.uniform(
        -1.0, 1.0, (context_window_size * embedding_dimension,
                    num_hidden_layer)).astype(theano.config.floatX),
                                 name='w_input')
    self.b_input = theano.shared(
        numpy.random.uniform(-1.0, 1.0,
                             (num_hidden_layer,)).astype(theano.config.floatX),
        name='b_input')

    self.w_classifier = theano.shared(0.2 * numpy.random.uniform(
        -1.0, 1.0, (num_hidden_layer)).astype(theano.config.floatX),
                                      name='w_classifier')
    self.b_classifier = theano.shared(
        numpy.random.uniform(-1.0, 1.0),
        name='b_classifier')

    self.params = [self.embedding, self.w_input, self.b_input,
                   self.w_classifier, self.b_classifier]

    self.classifier = self.GetClassifier()

  def F(self, x):
    """
    The scalar output of neural network.
    """
    # self.embedding[x] expands each element of x to be the row: embedding[element]
    # TODO: It only takes a single context window. Change to do batch processing.
    input = self.embedding[x].reshape((x.shape[
        0], self.context_window_size * self.embedding_dimension))
    activation = T.tanh(T.dot(input, self.w_input) + self.b_input)
    output = T.dot(activation, self.w_classifier) + self.b_classifier
    # TODO: Use another Tanh layer to see whether there is performance boost.

    return output

  def Train(self, input, mutations):
    return self.classifier(input, mutations)

  def GetClassifier(self):

    def PairwiseLoss(x, mutation):
      """
      This function takes two matrix and return a vector by first
      calculating the loss for each row and then take element-wise
      maximum for each row.
      """
      return T.maximum(0., 1. - self.F(x) + self.F(mutation)).sum()

    inputs = T.tensor3(name='input', dtype='int32')
    mutations = T.tensor3(name='mutations', dtype='int32')

    components, updates = theano.scan(fn=PairwiseLoss,
                                      outputs_info=None,
                                      sequences=[inputs, mutations])
    loss = components.sum()

    gparams = [T.grad(loss, param) for param in self.params]

    updates = [(param, param - self.learning_rate * gparam)
               for param, gparam in zip(self.params, gparams)]

    return theano.function(inputs=[inputs, mutations],
                           outputs=loss,
                           updates=updates)
