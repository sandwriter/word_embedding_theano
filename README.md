![alt text](https://travis-ci.org/wenjiesha/word_embedding.svg?branch=master "Continuous test status")
# Word Embedding
Theano implementation of Paper ["Natural Language Processing (almost) from Scratch"](http://arxiv.org/pdf/1103.0398v1.pdf)

## Status
There are still quite a few important items to finish, but it seems like learning the embedding.
First few lines of output of the training process. (each epoch takes ~3 hours on my laptop... Need to use GPU perhaps...)
*******************************************
~/Workspace/word_embedding$ python train.py

vocabulary_size: 572

number of sentences: 46635

/usr/local/lib/python2.7/dist-packages/Theano-0.7.0-py2.7.egg/theano/scan_module/scan_perform_ext.py:133: RuntimeWarning: numpy.ndarray size changed, may indicate binary incompatibility

  from scan_perform.scan_perform import *

epoch 0: loss: 144234975.912176

epoch 1: loss: 137165925.422383

epoch 2: loss: 130193645.065526
*******************************************

## Input
ATIS Data. Contains 46635 sentences, with 572 words.

## Output
Word embedding for each word.

## How to run
```shell
python train.py
```

You might need to install at least Theano 0.7+ and numpy to run the program.

## TODO
1. cPickle the embedding at the end of each epoch.
2. Use validation set to avoid over fitting.
3. Hyper param tuning.
4. GPU perhaps, cos it is really slow right now.
5. Normalizing the embedding?


