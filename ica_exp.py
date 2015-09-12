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
import shutil
import sys
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time

from layers import MLP
from mnist import mnist_iterator
import op
from sffn_ica import SFFN
from tools import check_bad_nums
from tools import itemlist
from tools import load_model
from tools import load_experiment


floatX = theano.config.floatX

def concatenate_inputs(model, y, py):
    '''
    Function to concatenate ground truth to samples and probabilities.
    '''
    y_hat = model.cond_from_h.sample(py)

    py = T.concatenate([y[None, :, :], py], axis=0)
    y = T.concatenate([y[None, :, :], y_hat], axis=0)

    return py, y

def load_mlp(name, dim_in, dim_out, dim_h=None, n_layers=None, **kwargs):
    mlp = MLP(dim_in, dim_h, dim_out, n_layers, name=name, **kwargs)
    return mlp

def unpack(dim_h=None,
           z_init=None,
           recognition_net=None,
           generation_net=None,
           dataset=None,
           dataset_args=None,
           noise_amount=None,
           n_inference_steps=None,
           inference_decay=None,
           inference_method=None,
           inference_rate=None,
           x_noise_mode=None,
           y_noise_mode=None,
           **model_args):
    '''
    Function to unpack pretrained model into fresh SFFN class.
    '''

    dataset_args = dataset_args[()]

    if dataset == 'mnist':
        dim_in = 28 * 28
        dim_out = dim_in
    else:
        raise ValueError()

    models = []
    if recognition_net is not None:
        recognition_net = recognition_net[()]
        cond_to_h = load_mlp('cond_to_h', dim_in, dim_h,
                             out_act='T.nnet.sigmoid',
                             **recognition_net)
        models.append(cond_to_h)
    else:
        cond_to_h = None

    if generation_net is not None:
        generation_net = generation_net[()]
        cond_from_h = load_mlp('cond_from_h', dim_h, dim_in,
                               out_act='T.nnet.sigmoid',
                               **generation_net)
        models.append(cond_from_h)

    sffn = SFFN(dim_in, dim_h, dim_out,
                cond_from_h=cond_from_h,
                cond_to_h=cond_to_h,
                noise_amount=0.,
                z_init=z_init,
                inference_rate=inference_rate,
                n_inference_steps=n_inference_steps,
                inference_decay=inference_decay,
                inference_method=inference_method,
                x_noise_mode=x_noise_mode,
                y_noise_mode=y_noise_mode)
    models.append(sffn)

    return models, model_args, dict(
        z_init=z_init,
        dataset=dataset,
        dataset_args=dataset_args
    )

def train_model(
    out_path='', name='',
    load_last=False, model_to_load=None, save_images=True,
    source=None,
    learning_rate=0.1, optimizer='adam', batch_size=100, epochs=100,
    dim_h=300,
    x_noise_mode=None, y_noise_mode=None, noise_amout=0.1,
    generation_net=None, recognition_net=None,
    inference_method='momentum',
    inference_rate=.01, n_inference_steps=100,
    inference_decay=1.0, inference_samples=20,
    n_inference_steps_eval=0,
    z_init='recognition_net',
    dataset=None, dataset_args=None,
    model_save_freq=100, show_freq=10
    ):

    print 'Dataset args: %s' % pprint.pformat(dataset_args)

    print 'Setting up data'
    if dataset == 'mnist':
        train = mnist_iterator(batch_size=batch_size, mode='train', **dataset_args)
        valid = mnist_iterator(batch_size=batch_size, mode='valid', **dataset_args)
        test = mnist_iterator(batch_size=2000, mode='test', **dataset_args)
    else:
        raise ValueError()

    print 'Setting model'
    dim_in = train.dim
    dim_out = train.dim
    D = T.matrix('x', dtype=floatX)
    X = D.copy()
    Y = X.copy()

    trng = RandomStreams(random.randint(0, 1000000))

    if model_to_load is not None:
        models, _ = load_model(model_to_load, unpack,
                               inference_rate=inference_rate,
                               n_inference_steps=n_inference_steps,
                               inference_decay=inference_decay,
                               inference_method=inference_method)
    elif load_last:
        model_file = glob(path.join(out_path, '*last.npz'))[0]
        models, _ = load_model(model_file, unpack,
                               inference_rate=inference_rate,
                               n_inference_steps=n_inference_steps,
                               inference_decay=inference_decay,
                               inference_method=inference_method)
    else:
        # The recognition net is a MLP with 2 layers. The intermediate layer is
        # deterministic.
        if recognition_net is not None:
            cond_to_h = load_mlp('cond_to_h', dim_in, dim_h,
                                 out_act='T.nnet.sigmoid',
                                 **recognition_net)
        else:
            cond_to_h = None

        # The generation net has much of the same structure as the recognition net,
        # with roles reversed.
        if generation_net is not None:
            cond_from_h = load_mlp('cond_from_h', dim_h, dim_in,
                                   out_act='T.nnet.sigmoid',
                                   **generation_net)
        else:
            cond_from_h = None

        sffn = SFFN(dim_in, dim_h, dim_out, trng=trng,
                    cond_from_h=cond_from_h,
                    cond_to_h=cond_to_h,
                    noise_amount=0.,
                    z_init=z_init,
                    inference_rate=inference_rate,
                    n_inference_steps=n_inference_steps,
                    inference_decay=inference_decay,
                    inference_method=inference_method,
                    x_noise_mode=x_noise_mode,
                    y_noise_mode=y_noise_mode)

        models = OrderedDict()
        models[sffn.name] = sffn

    print 'Getting params'
    sffn = models['sffn']
    tparams = sffn.set_tparams()

    print 'Getting cost'
    (xs, ys, zs, h_energy, y_energy, i_energy), updates = sffn.inference(
        X, Y, n_samples=inference_samples)

    mu = T.nnet.sigmoid(zs)
    py = sffn.cond_from_h(mu)

    pd_i, d_hat_i = concatenate_inputs(sffn, ys[0], py)

    (py_s, y_energy_s), updates_s = sffn(X, Y, from_z=False,
                            n_inference_steps=n_inference_steps_eval)
    updates.update(updates_s)
    pd_s, d_hat_s = concatenate_inputs(sffn, Y, py_s)
    f_d_hat = theano.function([X, Y], [y_energy_s, pd_s, d_hat_s])

    py_p = sffn.sample_from_prior()
    f_py_p = theano.function([], py_p)

    consider_constant = [xs, ys, zs]
    cost = h_energy + y_energy

    extra_outs = [h_energy, y_energy, i_energy]
    vis_outs = [pd_i, d_hat_i]

    extra_outs_names = ['cost', 'h energy', 'train y energy', 'inference energy']
    vis_outs_names = ['pds', 'd_hats']

    # Remove the parameters found in updates from the ones we will take
    # gradients of.
    tparams = OrderedDict((k, v)
        for k, v in tparams.iteritems()
        if (v not in updates.keys()))

    print 'Getting gradients.'
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
        t0 = time.time()
        s = 0
        e = 0
        while True:
            try:
                x, _ = train.next()
            except StopIteration:
                e += 1
                print 'Epoch {epoch}'.format(epoch=epoch)
                continue

            if e > epochs:
                break

            rval = f_grad_shared(x)
            if check_bad_nums(rval, extra_outs_names+vis_outs_names):
                return

            if s % show_freq == 0:
                d_v, _ = valid.next()
                x_v, y_v = d_v, d_v

                ye_v, pd_v, d_hat_v = f_d_hat(x_v, y_v)
                ye_t, _, _ = f_d_hat(x, x)

                outs = OrderedDict((k, v)
                    for k, v in zip(extra_outs_names,
                                    rval[:len(extra_outs_names)]))

                t1 = time.time()
                outs.update(**{
                    'train energy at test': ye_t,
                    'valid y energy': ye_v,
                    'elapsed_time': t1-t0}
                )
                monitor.update(**outs)
                t0 = time.time()

                if ye_v < best_cost:
                    best_cost = ye_v
                    if out_path is not None:
                        d = dict((k, v.get_value()) for k, v in tparams.items())

                        d.update(
                            dim_h=dim_h,
                            x_noise_mode=x_noise_mode,
                            y_noise_mode=y_noise_mode,
                            noise_amout=noise_amout,
                            generation_net=generation_net,
                            recognition_net=recognition_net,
                            dataset=dataset, dataset_args=dataset_args
                        )

                        np.savez(bestfile, **d)

                monitor.display(e, s)

                if save_images and s % model_save_freq == 0:
                    monitor.save(path.join(
                        out_path, '{name}_monitor.png').format(name=name))

                    pd_i, d_hat_i = rval[len(extra_outs_names):]

                    idx = np.random.randint(pd_i.shape[1])
                    pd_i = pd_i[:, idx]
                    d_hat_i = d_hat_i[:, idx]
                    d_hat_i = np.concatenate([pd_i[:, None, :],
                                              d_hat_i[:, None, :]], axis=1)
                    train.save_images(
                        d_hat_i, path.join(
                            out_path, '{name}_inference.png').format(name=name))
                    d_hat_s = np.concatenate([pd_v[:10],
                                              d_hat_v[1][None, :, :]], axis=0)
                    d_hat_s = d_hat_s[:, :min(10, d_hat_s.shape[1] - 1)]
                    train.save_images(d_hat_s, path.join(
                        out_path, '{name}_samples.png'.format(name=name)))

                    pd_p = f_py_p()
                    train.save_images(
                        pd_p[:, None], path.join(
                            out_path,
                            '{name}_samples_from_prior.png'.format(name=name)),
                        x_limit=10
                    )

            f_grad_updates(learning_rate)

            s += 1

    except KeyboardInterrupt:
        print 'Training interrupted'

    print 'Quick test, please wait...'
    d_t, _ = test.next()
    x_t = d_t.copy()
    y_t = d_t.copy()
    ye_t, _, _ = f_d_hat(x_t, y_t)
    print 'End test: %.5f' % ye_t

    if out_path is not None:
        outfile = path.join(out_path, '{name}_{t}.npz'.format(name=name, t=int(time.time())))
        last_outfile = path.join(out_path, '{name}_last.npz'.format(name=name))

        d = dict((k, v.get_value()) for k, v in tparams.items())

        d.update(
            dim_h=dim_h,
            x_noise_mode=x_noise_mode, y_noise_mode=y_noise_mode,
            noise_amout=noise_amout,
            generation_net=generation_net, recognition_net=recognition_net,
            dataset=dataset, dataset_args=dataset_args
        )

        print 'Saving'
        np.savez(outfile, **d)
        np.savez(last_outfile, **d)
        print 'Done saving.'
    print 'Bye bye!'

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

    if out_path is not None:
        if path.isfile(out_path):
            raise ValueError()
        elif not path.isdir(out_path):
            os.mkdir(path.abspath(out_path))

    shutil.copy(path.abspath(args.experiment), path.abspath(out_path))

    train_model(out_path=out_path, load_last=args.load_last,
                model_to_load=args.load_model, save_images=args.save_images,
                **exp_dict)
