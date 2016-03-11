'''
Module for comparing models
'''

import argparse
from collections import OrderedDict
import csv
from glob import glob
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import numpy as np
import os
from os import path
import pprint
from progressbar import Bar, ProgressBar, Timer
import sys
from tabulate import tabulate
import theano
from theano import tensor as T
import time

from datasets.caltech import CALTECH
from datasets.mnist import MNIST
from datasets.uci import UCI
from inference import resolve as resolve_inference
from models.dsbn import unpack as unpack_dsbn
from models.gbn import unpack as unpack_gbn
from models.mlp import MLP
from models.sbn import unpack as unpack_sbn
from utils import floatX
from utils.tools import (
    check_bad_nums,
    itemlist,
    load_model,
    load_experiment,
    log_sum_exp,
    update_dict_of_lists,
    _slice
)


def unpack_model_and_data(model_dir):
    name         = model_dir.split('/')[-2]
    yaml         = glob(path.join(model_dir, '*.yaml'))[0]
    model_file   = glob(path.join(model_dir, '*best*npz'))[0]
    exp_dict     = load_experiment(path.abspath(yaml))
    dataset_args = exp_dict['dataset_args']
    dataset      = dataset_args['dataset']

    def filter(prior=None, dim_hs=None, dim_h=None,
               inference_args=None, learning_args=None,
               **kwargs):
        if dim_h is not None:
            dim_hs = [dim_h]
        return OrderedDict(
            prior=prior,
            dim_hs=dim_hs,
            inference_args=inference_args,
            learning_args=learning_args
        )

    exp_dict = filter(**exp_dict)
    prior    = exp_dict['prior']
    deep     = len(exp_dict['dim_hs']) > 1

    if dataset == 'mnist':
        print 'Loading MNIST'
        train_iter = MNIST(mode='train', batch_size=10, **dataset_args)
        data_iter = MNIST(batch_size=10, mode='test', **dataset_args)
    elif dataset == 'caltech':
        print 'Loading Caltech 101 Silhouettes'
        train_iter = CALTECH(mode='train', batch_size=10, **dataset_args)
        data_iter = CALTECH(batch_size=10, mode='test', **dataset_args)
    elif dataset == 'uci':
        print 'Loading the %s UCI dataset' % dataset
        train_iter = UCI(mode='train', batch_size=10, **dataset_args)
        data_iter = UCI(batch_size=10, mode='test', **dataset_args)
    mean_image = train_iter.mean_image.astype(floatX)

    if prior == 'gaussian':
        unpack = unpack_gbn
        model_name = 'gbn'
        inference_method = 'momentum'
    elif prior in ['binomial', 'darn'] and not deep:
        unpack = unpack_sbn
        model_name = 'sbn'
        inference_method = 'air'
    elif prior == 'binomial' and deep:
        unpack = unpack_dsbn
        model_name = 'sbn'
        inference_method = 'air'
    else:
        raise ValueError(prior)

    models, _ = load_model(model_file, unpack,
                           distributions=data_iter.distributions,
                           dims=data_iter.dims)

    models['main'] = models[model_name]

    return models, data_iter, name, exp_dict, mean_image, deep, inference_method

def sample_from_prior(models, data_iter, name, out_dir):
    model = models['main']
    tparams = model.set_tparams()

    py_p, updates = model.sample_from_prior()
    f_prior = theano.function([], py_p, updates=updates)
    samples = f_prior()
    data_iter.save_images(
        samples[:, None],
        path.join(out_dir, name + '_samples_from_prior.png'),
        x_limit=10)

def calculate_true_likelihood(models, data_iter):
    model = models['main']
    H = T.matrix('H', dtype=floatX)
    Y = T.matrix('Y', dtype=floatX)
    log_ph = -model.prior.neg_log_prob(H)
    py = model.conditional(H)
    log_py_h = -model.conditional.neg_log_prob(Y, py)
    log_p = log_ph + log_py_h
    log_px = log_sum_exp(log_p, axis=0)

    f_test = theano.function([Y, H], log_px)

    dim = model.dim_h
    h_i = np.array([[i] for i in range(2 ** dim)], dtype=np.uint8)
    h = np.unpackbits(h_i, axis=1)[:, -dim:].astype(floatX)

    vals = []
    widgets = ['Calculating LL %s:' % name, Timer(), Bar()]
    pbar = ProgressBar(maxval=data_iter.n).start()
    while True:
        try:
            y, _ = data_iter.next(batch_size=dx)
        except StopIteration:
            break
        r = f_test(y, h)
        vals.append(r)

        if data_iter.pos == -1:
            pbar.update(data_iter.n)
        else:
            pbar.update(data_iter.pos)
    print
    print 'LL: ', np.mean(vals)

def test(models, data_iter, name, mean_image, deep=False,
         data_samples=10000, n_posterior_samples=1000,
         inference_args=None, inference_method=None,
         dx=100, calculate_true_likelihood=False,
         center_input=True, **extra_kwargs):

    model = models['main']
    tparams = model.set_tparams()
    data_iter.reset()

    X = T.matrix('x', dtype=floatX)

    if center_input:
        print 'Centering input with train dataset mean image'
        X_mean = theano.shared(mean_image, name='X_mean')
        X_i = X - X_mean
    else:
        X_i = X.copy()

    inference = resolve_inference(model, deep=deep,
                                  inference_method=inference_method,
                                  **inference_args)

    if inference_method == 'momentum':
        if prior == 'binomial':
            raise NotImplementedError()
        results, samples, full_results, updates = inference(
            X_i, X,
            n_posterior_samples=n_posterior_samples)
    elif inference_method == 'air':
        results, samples, full_results, updates = inference(
            X_i, X, n_posterior_samples=n_posterior_samples)
    else:
        raise ValueError(inference_method)

    f_test_keys  = results.keys()
    f_test       = theano.function([X], results.values(), updates=updates)
    widgets = ['Testing %s:' % name, Timer(), Bar()]
    pbar = ProgressBar(maxval=data_iter.n).start()
    rs = OrderedDict()
    while True:
        try:
            y = data_iter.next(batch_size=dx)[data_iter.name]
        except StopIteration:
            break
        r = f_test(y)
        rs_i = dict((k, v) for k, v in zip(f_test_keys, r))
        update_dict_of_lists(rs, **rs_i)

        if data_iter.pos == -1:
            pbar.update(data_iter.n)
        else:
            pbar.update(data_iter.pos)
    print

    def summarize(d):
        for k, v in d.iteritems():
            d[k] = np.mean(v)

    summarize(rs)

    return rs

def compare(model_dirs,
            out_path,
            name=None,
            by_training_time=False,
            omit_deltas=True,
            **test_args):

    model_results = OrderedDict()
    valid_results = OrderedDict()

    for model_dir in model_dirs:
        n = model_dir.split('/')[-1]
        if n == '':
            n = model_dir.split('/')[-2]
        result_file = path.join(model_dir,
                                '{name}_monitor.npz'.format(name=n))
        params = np.load(result_file)
        d = dict(params)
        update_dict_of_lists(model_results, **d)

        valid_file = path.join(model_dir,
                               '{name}_monitor_valid.npz'.format(name=n))
        params_valid = np.load(valid_file)
        d_valid = dict(params_valid)
        update_dict_of_lists(valid_results, **d_valid)
        update_dict_of_lists(model_results, name=n)

    if omit_deltas:
        model_results = OrderedDict(
            (k, v) for k, v in model_results.iteritems() if not k.startswith('d_'))

    model_results.pop('dt_epoch')
    names = model_results.pop('name')
    training_times = model_results.pop('training_time')

    if name is None:
        name = '.'.join(names)

    out_dir = path.join(out_path, 'compare.' + name)
    if path.isfile(out_dir):
        raise ValueError()
    elif not path.isdir(out_dir):
        os.mkdir(path.abspath(out_dir))

    plt.clf()
    x = 3
    y = ((len(model_results) - 1) // x) + 1

    fig, axes = plt.subplots(y, x)
    fig.set_size_inches(15, 10)

    if by_training_time:
        xlabel = 'seconds'
        us = [tt - tt[0] for tt in training_times]
    else:
        us = [range(tt.shape[0]) for tt in training_times]
        xlabel = 'epochs'

    for j, (k, vs) in enumerate(model_results.iteritems()):
        ax = axes[j // x, j % x]
        for n, u, v in zip(names, us, vs):
            ax.plot(u[10:], v[10:], label=n)

        if k in valid_results.keys():
            for n, u, v in zip(names, us, valid_results[k]):
                ax.plot(u[10:], v[10:], label=n + '(v)')

        ax.set_ylabel(k)
        ax.set_xlabel(xlabel)
        ax.legend()
        ax.patch.set_alpha(0.5)

    plt.tight_layout()
    plt.savefig(path.join(out_dir, 'results.png'))
    plt.close()

    print 'Sampling from priors'

    results = OrderedDict()
    hps = OrderedDict()
    for model_dir in model_dirs:
        models, data_iter, name, exp_dict, mean_image, deep, inference_method = unpack_model_and_data(model_dir)
        sample_from_prior(models, data_iter, name, out_dir)
        rs = test(models, data_iter, name, mean_image, deep=deep,
                  inference_method=inference_method, **test_args)
        update_dict_of_lists(results, **rs)
        update_dict_of_lists(hps, **exp_dict)

    columns = ['Stat'] + names
    data = [[k] + v for k, v in hps.iteritems()]
    data += [[k] + v for k, v in results.iteritems() if not k.startswith('d_')]

    with open(path.join(out_dir, 'summary.txt'), 'w+') as f:
        print >>f, tabulate(data, headers=columns)

    print tabulate(data, headers=columns)

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-l', '--model_list', nargs='+', default=None)
    parser.add_argument('-D', '--by_dir', default=None)
    parser.add_argument('-t', '--by_time', action='store_true')
    parser.add_argument('-o', '--out_path', required=True)
    parser.add_argument('-d', '--see_deltas', action='store_true')
    parser.add_argument('-p', '--n_posterior_samples', default=1000, type=int)
    parser.add_argument('-i', '--n_inference_samples', default=100, type=int)
    parser.add_argument('-s', '--n_inference_steps', default=100, type=int)
    parser.add_argument('-b', '--batch_size', default=100, type=int)
    parser.add_argument('-r', '--inference_rate', default=0.1, type=float)
    return parser

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()

    if args.model_list is not None:
        models = args.model_list
        name = None
    elif args.by_dir is not None:
        models = glob(path.join(args.by_dir, '*'))
        name = os.path.basename(os.path.normpath(args.by_dir))

    inference_args = OrderedDict(
        inference_rate=args.inference_rate,
        n_inference_samples=args.n_inference_samples,
        n_inference_steps=args.n_inference_steps,
    )

    compare(models, args.out_path, name=name,
            omit_deltas=not(args.see_deltas),
            n_posterior_samples=args.n_posterior_samples,
            inference_args=inference_args,
            dx=args.batch_size,
            by_training_time=args.by_time)
