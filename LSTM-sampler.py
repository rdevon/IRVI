'''
Sampling and inference with LSTM models
'''

import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time

from gru import GRU
from gru import GenStochasticGRU
import mnist
import optimization as op
from tools import itemlist

floatX = theano.config.floatX


def test(dim_h=100, n_steps=7, batch_size=1, inference_steps=21, l=0.1):
    learning_rate = 0.001
    optimizer = 'sgd'

    dataset = mnist.mnist_iterator(batch_size=2 * batch_size)

    X0 = T.matrix('X0', dtype=floatX)
    XT = T.matrix('XT', dtype=floatX)
    inps = [X0, XT]

    print 'Making the model'
    rnn = GenStochasticGRU(dataset.dim, dim_h)
    tparams = rnn.set_tparams()
    params = rnn.get_non_seqs()

    outs, updates = rnn(x0=X0, xT=XT, n_steps=n_steps)
    x = outs['x']
    z = outs['z']
    t0 = time.time()

    print 'Dumb looping'
    for i in xrange(inference_steps):
        x, z, h, p = rnn.step_infer(x, z, 0.1, rnn.HX, rnn.bx, rnn.sigmas, *params)
    #(x, z), updates_1 = rnn.inference(x, z, 0.1, inference_steps)
    updates += [(rnn.sigmas, (h - h.mean(axis=(0, 1))**2).mean(axis=(0, 1)))]
    t1 = time.time()
    print 'Time:', t1 - t0

    print 'Calculating cost'
    cost = (- x * T.log(p + 1e-7) - (1. - x) * T.log(1. - p + 1e-7)).mean()
    #f_cost = theano.function(inps, cost, updates=updates)

    (x0, xT), _ = dataset.next()
    print 'Compiling grad'

    f_cost = theano.function(inps, p, updates=updates)
    print f_cost(x0.reshape(batch_size, dataset.dim),
                 xT.reshape(batch_size, dataset.dim))

    return

    grads = T.grad(cost, wrt=itemlist(tparams), consider_constant=[x, z])

    f_grad = theano.function(inps, grads, updates=updates)
    print f_grad(x0, xT)

    lr = T.scalar(name='lr')

    f_grad_shared, f_grad_updates = eval('op.' + optimizer)(
        lr, tparams, grads, inps, cost,
        extra_ups=updates)

    t2 = time.time()
    print 'Time:', t2 - t1


    print 'Actually running'
    c = f_grad_shared(x0, xT)
    print c
    f_grad_updates(learning_rate)
    #x, z = fn(x0.reshape(1, dataset.dim), xT.reshape(1, dataset.dim))
    t3 = time.time()
    print 'Time:', t3 - t2

    dataset.save_images(x, '/Users/devon/tmp/naive_sampler.png')

if __name__ == '__main__':
    test()