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
from sbn import SBN
from tools import check_bad_nums
from tools import itemlist
from tools import load_model


floatX = theano.config.floatX

def train_model(batch_size=101,
                dim_h=203,
                depth=2,
                ls=[0.01, 0.01],
                learning_rate=0.01,
                min_lr=0.001,
                lr_decay=False,
                n_inference_steps=100,
                inference_decay=0.99,
                inference_samples=20,
                z_init=None,
                out_path='',
                load_last=False,
                model_to_load=None,
                load_from_simple=None,
                inference_method='momentum',
                save_images=False,
                optimizer='adam'):

    print 'Setting up data'
    train = mnist_iterator(batch_size=batch_size, mode='train', inf=True, repeat=1)
    valid = mnist_iterator(batch_size=batch_size, mode='valid', inf=True, repeat=1)
    test = mnist_iterator(batch_size=500, mode='test', inf=True, repeat=1)

    print 'Setting model'
    dim_out = train.dim
    X = T.matrix('x', dtype=floatX)

    trng = RandomStreams(6 * 23 * 2015)

    sbn = SBN(dim_h, dim_out,
               trng=trng,
               noise_amount=0.,
               inference_rates=ls,
               n_inference_steps=n_inference_steps,
               inference_decay=inference_decay,
               inference_method=inference_method)


    if model_to_load is not None:
        raise NotImplementedError()
        sffn.cond_to_h1 = load_model(sffn.cond_to_h1, model_to_load)
        sffn.cond_to_h2 = load_model(sffn.cond_to_h2, model_to_load)
        sffn.cond_from_h2 = load_model(sffn.cond_from_h2, model_to_load)
    elif load_last:
        raise NotImplementedError()
        model_file = glob(path.join(out_path, '*.npz'))[-1]
        sffn.cond_to_h1 = load_model(sffn.cond_to_h1, model_file)
        sffn.cond_to_h2 = load_model(sffn.cond_to_h2, model_file)
        sffn.cond_from_h2 = load_model(sffn.cond_from_h2, model_file)

    tparams = sbn.set_tparams()

    (xs, zs, hs, energies, i_energy), updates = sbn.inference(
        X, n_samples=inference_samples)

    mu = T.nnet.sigmoid(zs[-1])
    h = sbn.h_conds[-2].sample(mu)
    px = sbn.h_conds[-1](h2)
    x_hat = sbn.h_conds[-1].sample(px)

    px_s, x_energy_s = sbn(X)
    x_hat_x = sbn.h_conds[-1].sample(px_s)
    f_x_hat = theano.function([X], [x_energy_s, px_s, x_hat_s])

    consider_constant = [xs] + zs + hs
    cost = sum(energies)

    extra_outs = energies + [i_energy]
    vis_outs = [pd_i, d_hat_i]

    extra_outs_names = ['cost'] + ['h%d energy' % l for l in xrange(sbn.depth)] + ['train x energy', 'inference energy']
    vis_outs_names = ['pxs', 'x_hats']

    tparams = OrderedDict((k, v)
        for k, v in tparams.iteritems()
        if (v not in updates.keys()))

    grads = T.grad(cost, wrt=itemlist(tparams),
                   consider_constant=consider_constant)

    print 'Building optimizer'
    lr = T.scalar(name='lr')
    f_grad_shared, f_grad_updates = eval('op.' + optimizer)(
        lr, tparams, grads, [D], cost,
        extra_ups=updates,
        extra_outs=extra_outs+vis_outs)

    monitor = SimpleMonitor()

    print 'Actually running'

    best_cost = float('inf')
    bestfile = path.join(out_path, 'sbn_best.npz')

    try:
        for e in xrange(100000):
            x, _ = train.next()
            rval = f_grad_shared(x)
            if check_bad_nums(rval, extra_outs_names+vis_outs_names):
                return

            if e % 10 == 0:
                x_v, _ = valid.next()

                xe_v, px_v, x_hat_v = f_d_hat(x_v)
                xe_t, _, _ = f_d_hat(x)

                outs = OrderedDict((k, v)
                    for k, v in zip(extra_outs_names,
                                    rval[:len(extra_outs_names)]))
                outs.update(**{
                    'train energy at test': xe_t,
                    'valid x energy': xe_v}
                )
                monitor.update(**outs)

                if ye_v < best_cost:
                    best_cost = ye_v
                    np.savez(bestfile, **dict((k, v.get_value()) for k, v in tparams.items()))

                monitor.display(e * batch_size)

                if save_images:
                    monitor.save(path.join(out_path, 'sbn_monitor.png'))

                    px_i, x_hat_i = rval[len(extra_outs_names):]

                    idx = np.random.randint(px_i.shape[1])
                    px_i = px_i[:, idx]
                    x_hat_i = x_hat_i[:, idx]
                    x_hat_i = np.concatenate([px_i[:, None, :],
                                              x_hat_i[:, None, :]], axis=1)
                    train.save_images(x_hat_i, path.join(out_path, 'sbn_inference.png'))
                    x_hat_s = np.concatenate([px_v[:10],
                                              x_hat_v[1][None, :, :]], axis=0)
                    x_hat_s = x_hat_s[:, :min(10, x_hat_s.shape[1] - 1)]
                    train.save_images(x_hat_s, path.join(out_path, 'sbn_samples.png'))

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
                        'sbn_{}.npz'.format(int(time.time())))

    print 'Saving'
    np.savez(outfile, **dict((k, v.get_value()) for k, v in tparams.items()))
    print 'Done saving. Bye bye.'

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_path', default=None,
                        help='Output path for stuff')
    parser.add_argument('-l', '--load_last', action='store_true')
    parser.add_argument('-r', '--load_model', default=None)
    parser.add_argument('-i', '--save_images', action='store_true')
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
                model_to_load=args.load_model, save_images=args.save_images)