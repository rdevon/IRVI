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
from gru import SimpleInferGRU
from mnist import mnist_iterator
import op
from tools import itemlist

floatX = theano.config.floatX


def test(batch_size=20, dim_h=200, l=.01, n_inference_steps=30):
    train = mnist_iterator(batch_size=2*batch_size, mode='train', inf=True)
    dim_in = train.dim

    X = T.tensor3('x', dtype=floatX)

    trng = RandomStreams(6 * 23 * 2015)
    rnn = SimpleInferGRU(dim_in, dim_h, trng=trng)
    exclude_params = rnn.get_excludes()
    tparams = rnn.set_tparams()
    mask = T.alloc(1., 2).astype('float32')
    (x_hats, energies), updates = rnn.inference(
        X, mask, l, n_inference_steps=n_inference_steps)

    energy = energies[-1]
    thresholded_energy = (T.sort(energy)[:energy.shape[0] / 10]).mean()

    exclude_params = [tparams[ep] for ep in exclude_params]
    grads = T.grad(thresholded_energy, wrt=itemlist(tparams))

    chain, updates_s = rnn.sample(X[0])
    updates.update(updates_s)

    lr = T.scalar(name='lr')
    optimizer = 'rmsprop'
    f_grad_shared, f_grad_updates = eval('op.' + optimizer)(
        lr, tparams, grads, [X], thresholded_energy,
        extra_ups=updates,
        extra_outs=[x_hats, chain],
        exclude_params=exclude_params)

    print 'Actually running'
    learning_rate = 0.001
    for e in xrange(10000):
        x, _ = train.next()
        x = x.reshape(2, batch_size, x.shape[1]).astype(floatX)
        rval = f_grad_shared(x)
        r = False
        for k, out in zip(['energy'], rval):
            if np.any(np.isnan(out)):
                print k, 'nan'
                r = True
            elif np.any(np.isinf(out)):
                print k, 'inf'
                r = True
        if r:
            return
        if e % 10 == 0:
            print e, rval[0], np.exp(-rval[0])
        if e % 100 == 0:
            sample = rval[1][:, :, 0]
            sample[0] = x[:, 0, :]
            train.save_images(sample, '/Users/devon/tmp/grad_sampler.png')
            sample_chain = rval[2]
            train.save_images(sample_chain, '/Users/devon/tmp/grad_chain.png')

        f_grad_updates(learning_rate)

if __name__ == '__main__':
    test()