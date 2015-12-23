'''
Module to test theano grads and random streams normal
'''

import numpy as np
import random

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


trng = RandomStreams(random.randint(0, 10000))
X = T.matrix('X', dtype=theano.config.floatX)

S = theano.shared(np.random.normal(0, 1, size=(11, 13)).astype('float32'))
W = theano.shared(np.random.normal(0, 1, size=(21, 13)).astype('float32'))

Y = trng.normal(avg=s, std=T.exp(T.dot(X, W)), size=S.shape)
grads = theano.grad(Y.mean(), wrt=W)

f = theano.function([X], grads)

x = np.random.randint(0, 10, size=(11, 21)).astype('float32')

g = f(x)

print g.mean(), g.min(), g.max()