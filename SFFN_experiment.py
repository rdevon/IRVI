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
from sffn import SFFN
from tools import check_bad_nums
from tools import itemlist
from tools import load_model


floatX = theano.config.floatX

def concatenate_inputs(model, x, y, py):
    y_hat = model.cond_from_h.sample(py)

    py = T.concatenate([y[None, :, :], py], axis=0)
    y = T.concatenate([y[None, :, :], y_hat], axis=0)

    x = T.alloc(0, py.shape[0], x.shape[0], x.shape[1]) + x[None, :, :]

    pd = T.concatenate([x, py], axis=2)
    d_hat = T.concatenate([x, y], axis=2)

    return pd, d_hat

def train_model(batch_size=100,
                dim_h=200,
                l=.1,
                learning_rate = 0.1,
                min_lr = 0.01,
                lr_decay = False,
                n_inference_steps=50,
                inference_decay=1.0,
                inference_samples=20,
                second_sffn=True,
                out_path='',
                load_last=False,
                model_to_load=None,
                inference_method='momentum',
                save_images=False,
                optimizer='adam'):
    if out_path is not None:
        out_path = path.abspath(out_path)

    print 'Setting up data'
    train = mnist_iterator(batch_size=batch_size, mode='train', inf=True, repeat=1)
    valid = mnist_iterator(batch_size=batch_size, mode='valid', inf=True, repeat=1)
    test = mnist_iterator(batch_size=20000, mode='test', inf=True, repeat=1)

    print 'Setting model'
    dim_in = train.dim / 2
    dim_out = train.dim / 2
    D = T.matrix('x', dtype=floatX)
    X = D[:, :dim_in]
    Y = D[:, dim_in:]

    trng = RandomStreams(6 * 23 * 2015)

    cond_to_h = MLP(dim_in, dim_h, dim_h, 2,
                    h_act='T.nnet.sigmoid',
                    out_act='T.nnet.sigmoid')

    sffn = SFFN(dim_in, dim_h, dim_out, trng=trng,
                cond_to_h=cond_to_h,
                noise_amount=0.,
                inference_rate=l, n_inference_steps=n_inference_steps,
                inference_decay=inference_decay,
                inference_method=inference_method)

    if model_to_load is not None:
        sffn.cond_to_h = load_model(sffn.cond_to_h, model_to_load)
        sffn.cond_from_h = load_model(sffn.cond_from_h, model_to_load)
    elif load_last:
        model_file = glob(path.join(out_path, '*.npz'))[-1]
        sffn.cond_to_h = load_model(sffn.cond_to_h, model_file)
        sffn.cond_from_h = load_model(sffn.cond_from_h, model_file)

    tparams = sffn.set_tparams()

    (xs, ys, zs, h_energy, y_energy, i_energy), updates = sffn.inference(
        X, Y, n_samples=inference_samples)

    mu = T.nnet.sigmoid(zs)
    py = sffn.cond_from_h(mu)

    pd_i, d_hat_i = concatenate_inputs(sffn, xs[0], ys[0], py)

    py_s, y_energy_s = sffn(X, Y, from_z=False)
    pd_s, d_hat_s = concatenate_inputs(sffn, X, Y, py_s)
    f_d_hat = theano.function([X, Y], [y_energy_s, pd_s, d_hat_s])

    consider_constant = [xs, ys, zs]
    cost = h_energy + y_energy

    extra_outs = [h_energy, y_energy, i_energy]
    vis_outs = [pd_i, d_hat_i]

    extra_outs_names = ['cost', 'h energy', 'train y energy', 'inference energy']
    vis_outs_names = ['pds', 'd_hats']

    '''
    i_energy_2 = None
    if second_sffn:
        sffn2 = SFFN(dim_in, dim_h, dim_h, trng=trng,
                     x_init='sample_x', inference_noise='x', name='sffn2',
                     inference_rate=0.5, n_inference_steps=50, inference_decay=0.99)

        if load_model is not None:
            sffn2.cond_to_h = load_model(sffn2.cond_to_h, model_to_load)
            sffn2.cond_from_h = load_model(sffn2.cond_from_h, model_to_load)
        elif load_last:
            model_file = glob(path.join(out_path, '*.npz'))[-1]
            sffn2.cond_to_h = load_model(sffn2.cond_to_h, model_file)
            sffn2.cond_from_h = load_model(sffn2.cond_from_h, model_file)

        tparams.update(sffn2.set_tparams())

        mu = T.nnet.sigmoid(zs[-1])
        (z2s, mu_hats, _, _, h2_energy, mu_energy, i_energy_2), updates2 = sffn2.inference(
            X, mu, m=20)
        updates.update(updates2)

        z_cost = h2_energy + mu_energy
        cost += z_cost
        consider_constant += [z2s, mu, mu_hats]
        _, mu_hat, _, _, _ = sffn2(X, mu, from_z=False)

        _, _, y_energy_t, _, _ = sffn(X, Y, ph=mu_hat, from_z=False)
        f_d_hat2 = theano.function([X, Y], y_energy_t)

        #z_i = T.log(mu / (1. - mu))
        #sffn.initialize_z(z_i)
    elif sffn.z_init == 'xy':
        print 'Using a ffn h with inputs x y'
        z0 = T.dot(X, sffn.W0) + T.dot(Y, sffn.U0) + sffn.b0
        zt = zs[-1]
        z_cost = ((zt - z0)**2).sum(axis=1).mean()
        cost += z_cost
    elif sffn.z_init == 'x':
        z0 = T.dot(X, sffn.W0) + sffn.b0
        zt = zs[-1]
        z_cost = ((zt - z0)**2).sum(axis=1).mean()
        cost += z_cost
    elif sffn.learn_z:
        print 'Learning z with a ffn'
        zh = T.tanh(T.dot(X, sffn.W0) + sffn.b0)
        z0 = T.dot(zh, sffn.W1) + sffn.b1
        zt = zs[-1]
        z_cost = ((zt - z0)**2).sum(axis=1).mean()
        cost += z_cost

        (y_energy_t, pd_t, d_hat_t), updates_s2 = sffn(X, Y, from_z=True)
        updates.update(updates_s2)
        f_d_hat2 = theano.function([X, Y], [y_energy_t, pd_t, d_hat_t])
    else:
        z_cost = T.constant(0.)
        f_d_hat2 = None

    if second_sffn:
        extra_outs.append(i_energy_2)
    '''

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
    if out_path is not None:
        bestfile = path.join(out_path, 'sffn_best.npz')

    try:
        for e in xrange(100000):
            x, _ = train.next()
            rval = f_grad_shared(x)
            if check_bad_nums(rval, extra_outs_names+vis_outs_names):
                return

            if e % 10 == 0:
                d_v, _ = valid.next()
                x_v, y_v = d_v[:, :dim_in], d_v[:, dim_in:]

                ye_v, pd_v, d_hat_v = f_d_hat(x_v, y_v)
                ye_t, _, _ = f_d_hat(x[:, :dim_in], x[:, dim_in:])

                outs = OrderedDict((k, v)
                    for k, v in zip(extra_outs_names,
                                    rval[:len(extra_outs_names)]))
                outs.update(**{
                    'train energy at test': ye_t,
                    'valid y energy': ye_v}
                )
                monitor.update(**outs)

                if ye_v < best_cost:
                    best_cost = ye_v
                    np.savez(bestfile, **dict((k, v.get_value()) for k, v in tparams.items()))

                monitor.display(e * batch_size)

                if save_images:
                    monitor.save(path.join(out_path, 'sffn_monitor.png'))

                    pd_i, d_hat_i = rval[len(extra_outs_names):]

                    idx = np.random.randint(pd_i.shape[1])
                    pd_i = pd_i[:, idx]
                    d_hat_i = d_hat_i[:, idx]
                    d_hat_i = np.concatenate([pd_i[:, None, :],
                                              d_hat_i[:, None, :]], axis=1)
                    train.save_images(d_hat_i, path.join(out_path, 'sffn_inference.png'))
                    d_hat_s = np.concatenate([pd_v[:10],
                                              d_hat_v[1][None, :, :]], axis=0)
                    d_hat_s = d_hat_s[:, :min(10, d_hat_s.shape[1] - 1)]
                    train.save_images(d_hat_s, path.join(out_path, 'sffn_samples.png'))

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
    if out_path is not None:
        if path.isfile(out_path):
            raise ValueError()
        elif not path.isdir(out_path):
            os.mkdir(path.abspath(out_path))

    train_model(out_path=args.out_path, load_last=args.load_last,
                model_to_load=args.load_model, save_images=args.save_images)