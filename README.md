![alt text](https://travis-ci.org/wenjiesha/word_embedding.svg?branch=master "Continuous test status")
# Word Embedding
Theano implementation of Paper ["Natural Language Processing (almost) from Scratch"](http://arxiv.org/pdf/1103.0398v1.pdf)

## Status
There are still quite a few important items to finish, but it seems like learning the embedding.

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

## Useful links
[A nice blog about Python Internal](https://jakevdp.github.io/blog/2014/05/09/why-python-is-slow/)
