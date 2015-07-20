'''
SFFN experiment
'''

from collections import OrderedDict
import numpy as np
import os
from os import path
import sys
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time

from mnist import mnist_iterator
import op
from sffn import SFFN
from sffn import SFFN_2Layer
from tools import itemlist


floatX = theano.config.floatX

def test(batch_size=10, dim_h=256, l=0.1, n_inference_steps=30, out_path=''):

    train = mnist_iterator(batch_size=batch_size, mode='train', inf=True, repeat=1)
    valid = mnist_iterator(batch_size=batch_size, mode='valid', inf=True, repeat=1)

    dim_in = train.dim / 2
    dim_out = train.dim / 2
    D = T.matrix('x', dtype=floatX)
    X = D[:, :dim_in]
    Y = D[:, dim_in:]

    trng = RandomStreams(6 * 23 * 2015)
    sffn = SFFN(dim_in, dim_h, dim_out, trng=trng, h_mode='noise', noise=0.1)
    tparams = sffn.set_tparams()

    (y_hats, d_hats, ps, dps, zs, energy), updates = sffn.inference(
        X, Y, l, n_inference_steps=n_inference_steps, m=100, noise_mode='noise_all')

    y_hat, y_energy, dp, d_hat = sffn(X, Y)
    f_d_hat = theano.function([X, Y], [dp, d_hat, y_energy])

    consider_constant = [zs, y_hats]

    cost = energy.mean()

    if sffn.h_mode == 'ffn':
        raise NotImplementedError('There\'s a problem! h is constant!')
        print 'Using a ffn h with inputs x0 x1'
        h = zs[0]
        ht = zs[-1]
        ht_c = T.zeros_like(ht) + ht
        h_cost = ((ht_c - h)**2).sum(axis=1).mean()
        cost += h_cost
        consider_constant.append(ht_c)
    elif sffn.h_mode == 'x':
        z0 = T.dot(X, sffn.W0) + sffn.b0
        zt = zs[-1]
        z_cost = ((zt - z0)**2).sum(axis=1).mean()
        cost += z_cost
    else:
        z_cost = T.constant(0.)

    tparams = OrderedDict((k, v)
        for k, v in tparams.iteritems()
        if (v not in updates.keys()))

    grads = T.grad(cost, wrt=itemlist(tparams),
                   consider_constant=consider_constant)

    print 'Building optimizer'
    lr = T.scalar(name='lr')
    optimizer = 'rmsprop'
    f_grad_shared, f_grad_updates = eval('op.' + optimizer)(
        lr, tparams, grads, [D], cost,
        extra_ups=updates,
        extra_outs=[energy.mean(), z_cost, dps])

    print 'Actually running'
    learning_rate = 0.001

    try:
        for e in xrange(100000):
            x, _ = train.next()
            rval = f_grad_shared(x)
            r = False
            for k, out in zip(['cost', 'energy', 'z_cost', 'ps', 'd_hats'], rval):
                if np.any(np.isnan(out)):
                    print k, 'nan'
                    r = True
                elif np.any(np.isinf(out)):
                    print k, 'inf'
                    r = True
            if r:
                return
            if e % 10 == 0:
                d_v, _ = valid.next()
                x_v = d_v[:, :dim_in]
                y_v = d_v[:, dim_in:]
                p_samples, samples, y_en = f_d_hat(x_v, y_v)

                print ('%d: cost: %.5f | energy: %.5f | v_energy: %.5f | prob: %.5f | z_cost: %.5f '
                       % (e, rval[0], rval[1],
                          y_en,
                          np.exp(-rval[1]),
                          rval[2]))

                idx = np.random.randint(rval[3].shape[1])
                inference = rval[3][:, idx]
                train.save_images(inference, path.join(out_path, 'sffn_inference.png'))

                samples = np.concatenate([d_v[None, :, :],
                                          samples[None, :, :],
                                          p_samples[None, :, :]], axis=0)
                samples = samples[:, :min(10, samples.shape[1] - 1)]
                train.save_images(samples, path.join(out_path, 'sffn_samples.png'))

            f_grad_updates(learning_rate)
    except KeyboardInterrupt:
        print 'Training interrupted'

    outfile = path.join(out_path,
                        'rnn_grad_model_{}.npz'.format(int(time.time())))

    print 'Saving'
    np.savez(outfile, **dict((k, v.get_value()) for k, v in tparams.items()))
    print 'Done saving. Bye bye.'

if __name__ == '__main__':
    out_path = sys.argv[1]
    test(out_path=out_path)