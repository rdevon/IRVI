'''
SFFN experiment
'''

import argparse
from collections import OrderedDict
from glob import glob
from monitor import SimpleMonitor
import numpy as np
import os
from os import path
import sys
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time

from layers import MLP
from mnist import mnist_iterator
import op
from sffn import SFFN_2Layer
from tools import itemlist
from tools import load_model


floatX = theano.config.floatX

def train_model(batch_size=100,
          dim_h=200,
          l=.1,
          learning_rate = 0.0001,
          min_lr = 0.00001,
          lr_decay = False,
          n_inference_steps=5,
          inference_decay=.99,
          second_sffn=False,
          out_path='',
          load_last=False,
          model_to_load=None):

    train = mnist_iterator(batch_size=batch_size, mode='train', inf=True, repeat=1)
    valid = mnist_iterator(batch_size=batch_size, mode='valid', inf=True, repeat=1)
    test = mnist_iterator(batch_size=20000, mode='test', inf=True, repeat=1)

    dim_in = train.dim / 2
    dim_out = train.dim / 2
    D = T.matrix('x', dtype=floatX)
    X = D[:, :dim_in]
    Y = D[:, dim_in:]

    trng = RandomStreams(6 * 23 * 2015)

    sffn = SFFN_2Layer(
        dim_in, dim_h, dim_out, trng=trng, x_init='sample',
        inference_noise=None, noise_amount=0., inference_rate=l,
        n_inference_steps=n_inference_steps, inference_decay=inference_decay)

    if model_to_load is not None:
        sffn.cond_to_h1 = load_model(sffn.cond_to_h1, model_to_load)
        sffn.cond_to_h2 = load_model(sffn.cond_to_h2, model_to_load)
        sffn.cond_from_h2 = load_model(sffn.cond_from_h2, model_to_load)
    elif load_last:
        model_file = glob(path.join(out_path, '*.npz'))[-1]
        sffn.cond_to_h1 = load_model(sffn.cond_to_h1, model_file)
        sffn.cond_to_h2 = load_model(sffn.cond_to_h2, model_file)
        sffn.cond_from_h2 = load_model(sffn.cond_from_h2, model_file)

    tparams = sffn.set_tparams()

    (z1s, z2s, y_hats, d_hats, pds, h1_energy, h2_energy, y_energy, i_energy), updates = sffn.inference(
        X, Y, m=20)

    y_hat_s, _, y_energy_s, pd_s, d_hat_s = sffn(X, Y, from_z=False)
    f_d_hat = theano.function([X, Y], [y_energy_s, pd_s, d_hat_s])

    consider_constant = [X, Y, z1s, z2s, y_hats]
    cost = h1_energy + h2_energy + y_energy
    extra_outs = [h1_energy, h2_energy, y_energy, i_energy, pds, d_hats]

    tparams = OrderedDict((k, v)
        for k, v in tparams.iteritems()
        if (v not in updates.keys()))

    grads = T.grad(cost, wrt=itemlist(tparams),
                   consider_constant=consider_constant)

    print 'Building optimizer'
    lr = T.scalar(name='lr')
    optimizer = 'adam'
    f_grad_shared, f_grad_updates = eval('op.' + optimizer)(
        lr, tparams, grads, [D], cost,
        extra_ups=updates,
        extra_outs=extra_outs)

    monitor = SimpleMonitor()

    print 'Actually running'

    best_cost = float('inf')
    bestfile = path.join(out_path,
                        'rnn_grad_model_best.npz')

    try:
        for e in xrange(100000):
            x, _ = train.next()
            rval = f_grad_shared(x)
            r = False
            for k, out in zip(['cost',
                               'h1_energy',
                               'h2_energy',
                               'y_energy',
                               'inference energy',
                               'pds',
                               'd_hats'], rval):
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
                ye_v, pd_v, d_hat_v = f_d_hat(x_v, y_v)
                ytt, _, _ = f_d_hat(x[:, :dim_in], x[:, dim_in:])
                monitor.update(
                    **OrderedDict({
                        'cost': rval[0],
                        'h1 energy': rval[1],
                        'h2 energy': rval[2],
                        'train y energy': rval[3],
                        'train energy at test': ytt,
                        'valid y energy': ye_v,
                        'inference energy': rval[4]
                    })
                )

                if ye_v < best_cost:
                    best_cost = ye_v
                    np.savez(bestfile, **dict((k, v.get_value()) for k, v in tparams.items()))

                monitor.display(e * batch_size)
                monitor.save(path.join(out_path, 'sffn_monitor.png'))

                idx = np.random.randint(rval[5].shape[1])
                inference = rval[5][:, idx]
                i_samples = rval[6][:, idx]
                inference = np.concatenate([inference[:, None, :],
                                            i_samples[:, None, :]], axis=1)
                train.save_images(inference, path.join(out_path, 'sffn_inference.png'))

                samples = np.concatenate([d_v[None, :, :],
                                          pd_v,
                                          d_hat_v[None, :, :]], axis=0)
                samples = samples[:, :min(10, samples.shape[1] - 1)]
                train.save_images(samples, path.join(out_path, 'sffn_samples.png'))

            f_grad_updates(learning_rate)

            if lr_decay:
                learning_rate = max(learning_rate * lr_decay, min_lr)

    except KeyboardInterrupt:
        print 'Training interrupted'

    d_t, _ = test.next()
    x_t = d_t[:, :dim_in]
    y_t = d_t[:, dim_in:]
    ye_t, _, _ = f_d_hat(x_t, y_t)
    print 'End test: %.5f' % ye_t

    outfile = path.join(out_path,
                        'sffn_{}.npz'.format(int(time.time())))

    print 'Saving'
    np.savez(outfile, **dict((k, v.get_value()) for k, v in tparams.items()))
    print 'Done saving. Bye bye.'

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_path', default=None)
    parser.add_argument('-l', '--load_last', action='store_true')
    parser.add_argument('-r', '--load_model', default=None)
    return parser

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()

    out_path = args.out_path
    if path.isfile(out_path):
        raise ValueError()
    elif not path.isdir(out_path):
        os.mkdir(path.abspath(out_path))

    train_model(out_path=args.out_path, load_last=args.load_last,
                model_to_load=args.load_model)