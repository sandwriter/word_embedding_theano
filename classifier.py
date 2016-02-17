import sys
import numpy
import theano
import theano.tensor as T


class NNet(object):

  def __init__(self, vocabulary_size, embedding_dimension, context_window_size,
               num_hidden_layer, learning_rate):
    self.vocabulary_size = vocabulary_size
    self.embedding_dimension = embedding_dimension
    self.context_window_size = context_window_size
    self.num_hidden_layer = num_hidden_layer
    self.learning_rate = learning_rate

    self.embedding = theano.shared(0.2 * numpy.random.uniform(-1.0, 1.0, (
        vocabulary_size + 1, embedding_dimension)).astype(theano.config.floatX))

    self.w_input = theano.shared(0.2 * numpy.random.uniform(
        -1.0, 1.0, (context_window_size * embedding_dimension,
                    num_hidden_layer)).astype(theano.config.floatX))
    self.b_input = theano.shared(numpy.random.uniform(-1.0, 1.0, (
        num_hidden_layer,)).astype(theano.config.floatX))

    self.w_classifier = theano.shared(0.2 * numpy.random.uniform(
        -1.0, 1.0, (num_hidden_layer)).astype(theano.config.floatX))
    self.b_classifier = theano.shared(numpy.random.uniform(-1.0, 1.0))

    self.params = [self.embedding, self.w_input, self.b_input,
                   self.w_classifier, self.b_classifier]

    self.classifier = self.GetClassifier()

  def F(self, x):
    """
    The scalar output of neural network.
    """
    input = self.embedding[x].reshape((x.shape[0], self.context_window_size *
                                       self.embedding_dimension))
    activation = T.tanh(T.dot(input, self.w_input) + self.b_input)
    output = T.tanh(T.dot(activation, self.w_classifier) + self.b_classifier)
    return output

  def Train(self, input, mutations):
    print(input)
    print(mutations)
    return self.classifier(input, mutations)

  def GetClassifier(self):

    def PairwiseLoss(mutation, x):
      return T.maximum(0., 1. - self.F(x) + self.F(mutation))

    input = T.ivector('input')
    mutations = T.imatrix('mutation')

    components, updates = theano.scan(fn=PairwiseLoss,
                                      sequences=[mutations],
                                      non_sequences=input)
    loss = components.sum()

    gparams = [T.grad(loss, param) for param in self.params]

    updates = [(param, self.learning_rate * gparam)
               for param, gparam in zip(self.params, gparams)]

    return theano.function(inputs=[input, mutations],
                           outputs=loss,
                           updates=updates)
