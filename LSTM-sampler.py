'''
Sampling and inference with LSTM models
'''

import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time

from gru import GRU
from gru import GenStochasticGRU
import mnist
import op
from tools import itemlist

floatX = theano.config.floatX


def test(dim_h=500, n_steps=4, batch_size=1, inference_steps=20, l=0.01):
    learning_rate = 0.0001
    epochs = 2000
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
        x, z, p, h, lp, z_, zp, x_ = rnn.step_infer(x, z, l, rnn.HX, rnn.bx, rnn.sigmas, *params)
    updates += [(rnn.sigmas, T.sqrt(((h - h.mean(axis=(0, 1)))**2).mean(axis=(0, 1))))]
    t1 = time.time()
    print 'Time:', t1 - t0
    '''
    fn = theano.function(inps, [x, z, p, h, lp], updates=updates)
    outs = fn(x0, xT)
    for k, out in zip(['x', 'z', 'p', 'h', 'lp'], outs):
        if np.any(np.isnan(out)):
            print k, 'nan'
        elif np.any(np.isinf(out)):
            print k, 'inf'
        else:
            print k, 'clean'
    return
    '''

    print 'Calculating cost'
    #p0 = T.zeros_like(p) + p
    cost = -lp
    #f_cost = theano.function(inps, cost, updates=updates)

    print 'Compiling grad'

    #f_cost = theano.function(inps, p, updates=updates)
    #print f_cost(x0.reshape(batch_size, dataset.dim),
    #             xT.reshape(batch_size, dataset.dim))

    #return

    grads = T.grad(cost, wrt=itemlist(tparams), consider_constant=[z, z_, zp, x_])

    #f_grad = theano.function(inps, grads, updates=updates)
    #print f_grad(x0, xT)

    lr = T.scalar(name='lr')

    f_grad_shared, f_grad_updates = eval('op.' + optimizer)(
        lr, tparams, grads, inps, cost,
        extra_ups=updates,
        extra_outs=[x, z, p, h, lp, z_, zp, x_])

    t2 = time.time()
    print 'Time:', t2 - t1

    print 'Actually running'
    for e in xrange(epochs):
        (x0, xT), _ = dataset.next()
        x0 = x0.reshape((1, x0.shape[0]))
        xT = xT.reshape((1, xT.shape[0]))
        rval = f_grad_shared(x0, xT)
        r = False
        for k, out in zip(['cost', 'x', 'z', 'p', 'h', 'lp', 'z_', 'zp', 'x_'], rval):
            if np.any(np.isnan(out)):
                print k, 'nan'
                r = True
            elif np.any(np.isinf(out)):
                print k, 'inf'
                r = True
        if r:
            return
        if e % 10 == 0:
            print e, rval[0], rnn.sigmas.mean().eval()

        f_grad_updates(learning_rate)
        #x, z = fn(x0.reshape(1, dataset.dim), xT.reshape(1, dataset.dim))
        t3 = time.time()
        #print 'Time:', t3 - t2
    print 'sampling'
    (x0, xT), _ = dataset.next()
    x0 = x0.reshape((1, x0.shape[0]))
    xT = xT.reshape((1, xT.shape[0]))
    fn = theano.function(inps, x, updates=updates)
    sample = fn(x0, xT)

    print 'Saving'
    dataset.save_images(sample, '/Users/devon/tmp/naive_sampler.png')

if __name__ == '__main__':
    test()