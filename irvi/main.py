'''
SFFN experiment
'''

import argparse
from collections import OrderedDict
from glob import glob
import numpy as np
import os
from os import path
import pprint
from progressbar import ProgressBar
import random
import shutil
import sys
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time

from datasets.mnist import MNIST
from models.gbn import GaussianBeliefNet as GBN
from models.layers import MLP
from models.sbn import SigmoidBeliefNetwork as SBN
from models.sbn import load_mlp
from utils.monitor import SimpleMonitor
import utils.op
from utils.tools import (
    check_bad_nums,
    itemlist,
    load_model,
    load_experiment,
    _slice
)

floatX = theano.config.floatX

def concatenate_inputs(model, y, py):
    '''
    Function to concatenate ground truth to samples and probabilities.
    '''
    y_hat = model.conditional.sample(py)

    py = T.concatenate([y[None, :, :], py], axis=0)
    y = T.concatenate([y[None, :, :], y_hat], axis=0)

    return py, y

def load_data(dataset,
              train_batch_size,
              valid_batch_size,
              test_batch_size,
              **dataset_args):
    if dataset == 'mnist':
        if train_batch_size is not None:
            train = MNIST(batch_size=train_batch_size,
                          mode='train',
                          inf=False,
                          **dataset_args)
        else:
            train = None
        if valid_batch_size is not None:
            valid = MNIST(batch_size=valid_batch_size,
                          mode='valid',
                          inf=False,
                          **dataset_args)
        else:
            valid = None
        if test_batch_size is not None:
            test = MNIST(batch_size=test_batch_size,
                         mode='test',
                         inf=False,
                         **dataset_args)
        else:
            test = None
    else:
        raise ValueError()

    return train, valid, test

def train_model(
    out_path='', name='', load_last=False, model_to_load=None, save_images=True,

    learning_rate=0.0001, optimizer='rmsprop', optimizer_args=dict(),
    batch_size=100, valid_batch_size=100, test_batch_size=1000,
    max_valid=10000,
    epochs=100,

    dim_h=300, prior='logistic',
    input_mode=None,
    generation_net=None, recognition_net=None,
    excludes=['log_sigma'],
    center_input=True,

    z_init=None,
    inference_method='momentum',
    inference_rate=.01,
    n_inference_steps=100,
    n_inference_steps_test=0,
    inference_decay=1.0,
    n_inference_samples=20,
    inference_scaling=None,
    entropy_scale=1.0,
    alpha=7,
    extra_inference_args=dict(),

    n_mcmc_samples=20,
    n_mcmc_samples_test=20,
    importance_sampling=False,

    dataset=None, dataset_args=None,
    model_save_freq=1000, show_freq=100
    ):

    kwargs = dict(
        z_init=z_init,
        inference_method=inference_method,
        inference_rate=inference_rate,
        n_inference_samples=n_inference_samples,
        extra_inference_args=extra_inference_args
    )

    # ========================================================================
    print 'Dataset args: %s' % pprint.pformat(dataset_args)
    print 'Model args: %s' % pprint.pformat(kwargs)

    # ========================================================================
    print 'Setting up data'
    train, valid, test = load_data(dataset, batch_size,
                                   valid_batch_size, test_batch_size,
                                   **dataset_args)

    # ========================================================================
    print 'Setting model and variables'
    dim_in = train.dim
    X = T.matrix('x', dtype=floatX)
    X.tag.test_value = np.zeros((batch_size, dim_in), dtype=X.dtype)
    trng = RandomStreams(random.randint(0, 1000000))

    if input_mode == 'sample':
        print 'Sampling datapoints'
        X = trng.binomial(p=X, size=X.shape, n=1, dtype=X.dtype)
    elif input_mode == 'noise':
        print 'Adding noise to data points'
        X = X * trng.binomial(p=0.1, size=X.shape, n=1, dtype=X.dtype)

    if center_input:
        print 'Centering input with train dataset mean image'
        X_mean = theano.shared(train.mean_image.astype(floatX), name='X_mean')
        X_i = X - X_mean
    else:
        X_i = X

    # ========================================================================
    print 'Loading model and forming graph'

    if model_to_load is not None:
        models, _ = load_model(model_to_load, unpack, **kwargs)
    elif load_last:
        model_file = glob(path.join(out_path, '*last.npz'))[0]
        models, _ = load_model(model_file, unpack, **kwargs)
    else:
        if prior == 'logistic':
            out_act = 'T.nnet.sigmoid'
        elif prior == 'gaussian':
            out_act = 'lambda x: x'
        else:
            raise ValueError('%s prior not known' % prior)

        if recognition_net is not None:
            posterior = load_mlp('posterior', dim_in, dim_h,
                                 out_act=out_act,
                                 **recognition_net)
        else:
            posterior = None

        if generation_net is not None:
            conditional = load_mlp('conditional', dim_h, dim_in,
                                   out_act='T.nnet.sigmoid',
                                   **generation_net)
        else:
            conditional = None

        if prior == 'logistic':
            C = SBN
        elif prior == 'gaussian':
            C = GBN
        else:
            raise ValueError()
        model = C(dim_in, dim_h, trng=trng,
                conditional=conditional,
                posterior=posterior,
                **kwargs)

        models = OrderedDict()
        models[model.name] = model

    if prior == 'logistic':
        model = models['sbn']
    elif prior == 'gaussian':
        model = models['gbn']

    tparams = model.set_tparams(excludes=excludes)

    # ========================================================================
    print 'Getting cost'
    (z, prior_energy, h_energy, y_energy, entropy), updates, constants = model.inference(
        X_i, X, n_inference_steps=n_inference_steps, n_samples=n_mcmc_samples)

    cost = prior_energy + h_energy + y_energy

    extra_outs = [prior_energy, h_energy, y_energy, entropy]
    extra_outs_names = ['cost', 'prior_energy', 'h energy',
                        'train y energy', 'entropy']

    # ========================================================================
    print 'Extra functions'

    # Test function with sampling
    rval, updates_s = model(
        X_i, X, n_samples=n_mcmc_samples_test, n_inference_steps=n_inference_steps_test)

    py_s = rval['py']
    lower_bound = rval['lower_bound']
    pd_s, d_hat_s = concatenate_inputs(model, X, py_s)

    outs_s = [lower_bound, pd_s, d_hat_s]

    if 'inference_cost' in rval.keys():
        outs_s.append(rval['inference_cost'])

    f_test = theano.function([X], outs_s, updates=updates_s)

    # Sample from prior
    py_p = model.sample_from_prior()
    f_prior = theano.function([], py_p)

    # ========================================================================
    print 'Setting final tparams and save function'

    all_params = OrderedDict((k, v) for k, v in tparams.iteritems())

    tparams = OrderedDict((k, v)
        for k, v in tparams.iteritems()
        if (v not in updates.keys()))

    print 'Learned model params: %s' % tparams.keys()
    print 'Saved params: %s' % all_params.keys()

    def save(tparams, outfile):
        d = dict((k, v.get_value()) for k, v in all_params.items())

        d.update(
            dim_in=dim_in,
            dim_h=dim_h,
            input_mode=input_mode,
            prior=prior,
            generation_net=generation_net, recognition_net=recognition_net,
            dataset=dataset, dataset_args=dataset_args
        )
        np.savez(outfile, **d)

    # ========================================================================
    print 'Getting gradients.'
    grads = T.grad(cost, wrt=itemlist(tparams),
                   consider_constant=constants)

    # ========================================================================
    print 'Building optimizer'
    lr = T.scalar(name='lr')
    f_grad_shared, f_grad_updates = eval('op.' + optimizer)(
        lr, tparams, grads, [X], cost,
        extra_ups=updates,
        extra_outs=extra_outs, **optimizer_args)

    monitor = SimpleMonitor()

    # ========================================================================
    print 'Actually running (main loop)'

    best_cost = float('inf')
    best_epoch = 0

    valid_lbs = []
    train_lbs = []

    if out_path is not None:
        bestfile = path.join(out_path, '{name}_best.npz'.format(name=name))

    try:
        t0 = time.time()
        s = 0
        e = 0
        while True:
            try:
                x, _ = train.next()
            except StopIteration:
                print 'End Epoch {epoch} ({name})'.format(epoch=e, name=name)
                print '=' * 100
                valid.reset()

                lb_vs = []
                lb_ts = []

                print 'Validating'
                pbar = ProgressBar(maxval=min(max_valid, valid.n)).start()
                while True:
                    try:
                        if valid.pos != -1:
                            pbar.update(valid.pos)

                        x_v, _ = valid.next()
                        x_t, _ = train.next()
                        if valid.pos >= max_valid:
                            raise StopIteration

                        lb_v = f_test(x_v)[0]
                        lb_t = f_test(x_t)[0]

                        lb_vs.append(lb_v)
                        lb_ts.append(lb_t)

                    except StopIteration:
                        break

                lb_v = np.mean(lb_vs)
                lb_t = np.mean(lb_ts)

                print 'Train / Valid lower bound at end of epoch: %.2f / %.2f' % (lb_t, lb_v)

                if lb_v < best_cost:
                    print 'Found best: %.2f' % lb_v
                    best_cost = lb_v
                    best_epoch = e
                    if out_path is not None:
                        print 'Saving best to %s' % bestfile
                        save(tparams, bestfile)
                else:
                    print 'Best (%.2f) at epoch %d' % (best_cost, best_epoch)

                valid_lbs.append(lb_v)
                train_lbs.append(lb_t)

                if out_path is not None:
                    print 'Saving lower bounds in %s' % out_path
                    np.save(path.join(out_path, 'valid_lbs.npy'), valid_lbs)
                    np.save(path.join(out_path, 'train_lbs.npy'), train_lbs)

                e += 1

                print '=' * 100
                print 'Epoch {epoch} ({name})'.format(epoch=e, name=name)

                valid.reset()
                train.reset()
                continue

            if e > epochs:
                break

            rval = f_grad_shared(x)

            if check_bad_nums(rval, extra_outs_names):
                return

            if s % show_freq == 0:
                try:
                    x_v, _ = valid.next()
                except StopIteration:
                    x_v, _ = valid.next()

                outs_v = f_test(x_v)
                outs_t = f_test(x)

                lb_v, pd_v, d_hat_v = outs_v[:3]
                lb_t = outs_t[0]

                outs = OrderedDict((k, v)
                    for k, v in zip(extra_outs_names,
                                    rval[:len(extra_outs_names)]))

                t1 = time.time()
                outs.update(**{
                    'train lower bound': lb_t,
                    'valid lower bound': lb_v,
                    'elapsed_time': t1-t0}
                )

                try:
                    i_cost = outs_v[3]
                    outs.update(inference_cost=i_cost)
                except IndexError:
                    pass

                monitor.update(**outs)
                t0 = time.time()

                monitor.display(e, s)

                if save_images and s % model_save_freq == 0:
                    monitor.save(path.join(
                        out_path, '{name}_monitor.png').format(name=name))
                    if archive_every and s % archive_every == 0:
                        monitor.save(path.join(
                            out_path, '{name}_monitor({s})'.format(name=name, s=s))
                        )

                    d_hat_s = np.concatenate([pd_v[:10],
                                              d_hat_v[1][None, :, :]], axis=0)
                    d_hat_s = d_hat_s[:, :min(10, d_hat_s.shape[1] - 1)]
                    train.save_images(d_hat_s, path.join(
                        out_path, '{name}_samples.png'.format(name=name)))

                    pd_p = f_prior()
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

    if out_path is not None:
        outfile = path.join(out_path, '{name}_{t}.npz'.format(name=name, t=int(time.time())))
        last_outfile = path.join(out_path, '{name}_last.npz'.format(name=name))

        print 'Saving'
        save(tparams, outfile)
        save(tparams, last_outfile)
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
    parser.add_argument('-n', '--name', default=None)
    return parser

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()

    exp_dict = load_experiment(path.abspath(args.experiment))
    if args.name is not None:
        exp_dict['name'] = args.name
    out_path = path.join(args.out_path, exp_dict['name'])

    if out_path is not None:
        print 'Saving to %s' % out_path
        if path.isfile(out_path):
            raise ValueError()
        elif not path.isdir(out_path):
            os.mkdir(path.abspath(out_path))

    shutil.copy(path.abspath(args.experiment), path.abspath(out_path))

    train_model(out_path=out_path, load_last=args.load_last,
                model_to_load=args.load_model, save_images=args.save_images,
                **exp_dict)
