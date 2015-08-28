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
import pprint
import random
import sys
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time
import yaml

from gradient_inference import load_inference
from layers import MLP
from mnist import mnist_iterator
import op
from sffn import SFFN_MultiLayer
from tools import check_bad_nums
from tools import itemlist
from tools import load_model
from tools import load_experiment


floatX = theano.config.floatX

def unpack(batch_size=None,
           dim_h=None,
           optimizer=None,
           mode=None,
           learning_rate=None,
           x_cond=None,
           steps=None,
           dataset='horses',
           dataset_args=None,
           h_init=None,
           **model_args):

    dataset_args = dataset_args[()]
    x_cond = x_cond[()]
    models = []

    trng = RandomStreams(random.randint(0, 100000))

    if dataset == 'mnist':
        dim_in = 28 * 28
    elif dataset == 'horses':
        dims = dataset_args['dims']
        dim_in = dims[0] * dims[1]
    else:
        raise ValueError()

    if x_cond is not None:
        print 'Found xcond: \n%s' % pprint.pformat(x_cond)
        condition_on_x  = MLP(dim_in=dim_in, dim_h=x_cond['dim_h'],
                              dim_out=dim_in, n_layers=x_cond['n_layers'],
                              h_act=x_cond['h_act'], name='xcond')
        models.append(condition_on_x)
    else:
        condition_on_x = None

    if mode == 'gru':
        C = GenGRU
    elif mode == 'rnn':
        C = GenRNN
    else:
        raise ValueError('Mode %s not recognized' % mode)

    rnn = C(dim_in, dim_h, trng=trng, condition_on_x=condition_on_x)
    models.append(rnn)

    if h_init == 'average':
        averager = Averager((batch_size, dim_h))
        models.append(averager)
    elif h_init == 'mlp':
        mlp = MLP(dim_in, dim_h, dim_h, 1, out_act='T.tanh')
        models.append(mlp)

    return models, model_args, dict(
        batch_size=batch_size,
        optimizer=optimizer,
        mode=mode,
        dataset=dataset,
        learning_rate=learning_rate,
        condition_on_x=condition_on_x,
        steps=steps,
        h_init=h_init,
        dataset_args=dataset_args
    )


def translate_load(model, model_file):
    pretrained_model = np.load(model_file)

    print pretrained_model.keys()

    l1 = len([k for k in pretrained_model.keys() if 'cond_to_h' in k])
    l2 = len([k for k in pretrained_model.keys() if 'cond_from_h' in k])

    cond_to_h_params = [pretrained_model['sffn_cond_to_h_%s' % k]
                        for l in xrange(l1 // 2)
                        for k in ['W_%d' % l, 'b_%d' % l]]

    cond_from_h_params = [pretrained_model['sffn_cond_from_h_%s' % k]
                        for l in xrange(l2 // 2)
                        for k in ['W_%d' % l, 'b_%d' % l]]

    pretrained_params = cond_to_h_params + cond_from_h_params

    for m in [model.cond_to_h1, model.cond_to_h2, model.cond_from_h2]:
        for k in ['W_0', 'b_0']:
            p = pretrained_params.pop(0)
            assert p.shape == m.params[k].shape, (p.shape, m.params[k].shape)
            m.params[k] = p

    return model

def concatenate_inputs(model, x, y, py):
    y_hat = model.layers[-1].sample(py)

    py = T.concatenate([y[None, :, :], py], axis=0)
    y = T.concatenate([y[None, :, :], y_hat], axis=0)

    x = T.alloc(0, py.shape[0], x.shape[0], x.shape[1]) + x[None, :, :]

    pd = T.concatenate([x, py], axis=2)
    d_hat = T.concatenate([x, y], axis=2)

    return pd, d_hat

def train_model(batch_size=100,
                dim_h=200,
                n_layers=2,
                layers=None,
                learning_rate=0.1,
                min_lr=0.001,
                lr_decay=False,
                out_path='',
                source=None,
                load_last=False, model_to_load=None,
                inference_procedure=None,
                inference_samples=20,
                save_images=False,
                optimizer='adam',
                name='',
                dataset='mnist'):

    print 'Setting up data'
    if dataset == 'mnist':
        train = mnist_iterator(batch_size=batch_size, mode='train', source=source, inf=True, repeat=1)
        valid = mnist_iterator(batch_size=batch_size, mode='valid', source=source, inf=True, repeat=1)
        test = mnist_iterator(batch_size=500, mode='test', source=source, inf=True, repeat=1)
    else:
        raise ValueError()

    print 'Setting up inference method %s' % pprint.pformat(inference_procedure)
    inference_procedure = load_inference(n_layers - 1, **inference_procedure)

    print 'Setting up model'
    trng = RandomStreams(random.randint(0, 100000))
    dim_in = train.dim / 2
    dim_out = train.dim / 2
    D = T.matrix('x', dtype=floatX)
    X = D[:, :dim_in]
    Y = D[:, dim_in:]

    if model_to_load is not None:
        models, _ = load_model(model_to_load, unpack)
    elif load_last:
        model_file = glob(path.join(out_path, '*last.npz'))[0]
        models, _ = load_model(model_file, unpack)
    else:
        sffn = SFFN_MultiLayer(dim_in, dim_h, dim_out, n_layers,
                               layers=layers,
                               inference_procedure=inference_procedure)

    tparams = sffn.set_tparams()

    (xs, ys, zs, hs, energies, i_energy), updates = sffn.inference(
        X, Y, n_samples=inference_samples)

    mu_f = T.nnet.sigmoid(zs[-1])
    h_f = sffn.layers[-2].sample(mu_f)
    py = sffn.layers[-1](h_f)

    pd_i, d_hat_i = concatenate_inputs(sffn, xs[0], ys[0], py)

    py_s, y_energy_s = sffn(X, Y, from_z=False)
    pd_s, d_hat_s = concatenate_inputs(sffn, X, Y, py_s)
    f_d_hat = theano.function([X, Y], [y_energy_s, pd_s, d_hat_s])

    consider_constant = [xs, ys] + zs + hs
    cost = sum(energies)

    extra_outs = energies + [i_energy]
    vis_outs = [pd_i, d_hat_i]

    extra_outs_names = ['cost'] + ['train h%d energy' % l
                                   for l in xrange(sffn.n_layers - 1)] + ['train y energy', 'inference energy']
    vis_outs_names = ['pds', 'd_hats']

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
    bestfile = path.join(out_path, '{name}_best.npz'.format(name=name))

    try:
        t0 = time.time()
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
                t1 = time.time()
                outs.update(**{
                    'train energy at test': ye_t,
                    'valid y energy': ye_v,
                    'elapsed_time': t1 - t0}
                )
                monitor.update(**outs)
                t0 = time.time()

                if ye_v < best_cost:
                    best_cost = ye_v
                    np.savez(bestfile, **dict((k, v.get_value()) for k, v in tparams.items()))

                monitor.display(e * batch_size)

                if save_images:
                    monitor.save(path.join(out_path, 'monitor.png'))

                    pd_i, d_hat_i = rval[len(extra_outs_names):]

                    idx = np.random.randint(pd_i.shape[1])
                    pd_i = pd_i[:, idx]
                    d_hat_i = d_hat_i[:, idx]
                    d_hat_i = np.concatenate([pd_i[:, None, :],
                                              d_hat_i[:, None, :]], axis=1)
                    train.save_images(d_hat_i, path.join(out_path, 'inference.png'), x_limit=10)
                    d_hat_s = np.concatenate([pd_v[:10],
                                              d_hat_v[1][None, :, :]], axis=0)
                    d_hat_s = d_hat_s[:, :min(10, d_hat_s.shape[1] - 1)]
                    train.save_images(d_hat_s, path.join(out_path, 'samples.png'))

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
                        '{name}_{t}.npz'.format(name=name, t=int(time.time())))

    print 'Saving'
    d = dict((k, v.get_value()) for k, v in tparams.items())
    d.update(
        dim_h=dim_h,
        n_layers=n_layers,
        layers=layers,
    )
    np.savez(outfile, **d)
    print 'Done saving. Bye bye.'

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', default=None)
    parser.add_argument('-o', '--out_path', default=None,
                        help='Output path for stuff')
    parser.add_argument('-l', '--load_last', action='store_true')
    parser.add_argument('-r', '--load_model', default=None)
    parser.add_argument('-i', '--save_images', action='store_true')
    return parser

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()

    exp_dict = load_experiment(path.abspath(args.experiment))
    out_path = path.join(args.out_path, exp_dict['name'])

    if path.isfile(out_path):
        raise ValueError()
    elif not path.isdir(out_path):
        os.mkdir(path.abspath(out_path))

    train_model(out_path=out_path, load_last=args.load_last,
                model_to_load=args.load_model, save_images=args.save_images,
                **exp_dict)