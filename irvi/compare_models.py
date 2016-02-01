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

from datasets.mnist import MNIST
from models.dsbn import unpack as unpack_dsbn
from models.gbn import GaussianBeliefNet as GBN
from models.mlp import MLP
from models.sbn import SigmoidBeliefNetwork as SBN
from models.sbn import unpack as unpack_sbn
from utils.tools import (
    check_bad_nums,
    floatX,
    itemlist,
    load_model,
    load_experiment,
    update_dict_of_lists,
    _slice
)


def unpack_model_and_data(model_dir):

    model_args = dict(
        inference_method='adaptive',
        inference_rate=0.1,
    )

    name       = model_dir.split('/')[-2]
    model_file = glob(path.join(model_dir, '*best*npz'))[0]

    try:
        models, model_args = load_model(model_file, unpack_sbn, **model_args)
    except:
        models, model_args = load_model(model_file, unpack_dsbn, **model_args)

    dataset = model_args['dataset']
    dataset_args = model_args['dataset_args']
    data_iter = MNIST(batch_size=10, mode='test', **dataset_args)

    yaml = glob(path.join(model_dir, '*.yaml'))[0]
    exp_dict = load_experiment(path.abspath(yaml))

    def filter(dim_hs=None, dim_h=None, inference_method=None, inference_rate=None,
               n_mcmc_samples=None, n_inference_steps=None, prior=None,
               n_inference_samples=None, epochs=None,
               **kwargs):
        if dim_h is not None:
            dim_hs = [dim_h]
        return OrderedDict(epochs=epochs,
            dim_hs=dim_hs, inference_method=inference_method,
            inference_rate=inference_rate,
            n_mcmc_samples=n_mcmc_samples,
            n_inference_steps=n_inference_steps,
            prior=prior, n_inference_samples=n_inference_samples
        )

    exp_dict = filter(**exp_dict)

    return models, data_iter, name, exp_dict

def sample_from_prior(models, data_iter, name, out_dir):
    model = models['sbn']
    tparams = model.set_tparams()

    py_p, updates = model.sample_from_prior()
    f_prior = theano.function([], py_p, updates=updates)
    samples = f_prior()
    data_iter.save_images(
        samples[:, None],
        path.join(out_dir, name + '_samples_from_prior.png'),
        x_limit=10)

def test(models, data_iter, name, n_inference_steps=100, n_inference_samples=100,
         data_samples=10000, posterior_samples=1000, dx=100,
         center_input=True, **extra_kwargs):

    model = models['sbn']
    tparams = model.set_tparams()
    data_iter.reset()

    X = T.matrix('x', dtype=floatX)
    if center_input:
        print 'Centering input with train dataset mean image'
        X_mean = theano.shared(data_iter.mean_image.astype(floatX), name='X_mean')
        X_i = X - X_mean
    else:
        X_i = X

    results, _, _, updates = model(
        X_i, X,
        n_inference_steps=n_inference_steps,
        n_samples=posterior_samples,
        n_inference_samples=n_inference_samples,
        stride=0)

    f_test_keys  = results.keys()
    f_test       = theano.function([X], results.values(), updates=updates)

    data_samples = min(data_samples, data_iter.n)
    x, _         = data_iter.next(batch_size=data_samples)
    xs           = [x[i: (i + dx)] for i in range(0, data_samples, dx)]
    N            = data_samples // dx

    widgets = ['Testing %s:' % name, Timer(), Bar()]
    pbar = ProgressBar(maxval=len(xs)).start()
    rs = OrderedDict()
    for i, y in enumerate(xs):
        r = f_test(y)
        rs_i = dict((k, v) for k, v in zip(f_test_keys, r))
        update_dict_of_lists(rs, **rs_i)
        pbar.update(i+1)
    print

    def summarize(d):
        for k, v in d.iteritems():
            d[k] = np.mean(v)

    summarize(rs)
    return rs

def compare(model_dirs,
            out_path,
            by_training_time=False,
            omit_deltas=True,
            **test_args):

    model_results = OrderedDict()
    valid_results = OrderedDict()

    for model_dir in model_dirs:
        name = model_dir.split('/')[-2]
        result_file = path.join(model_dir,
                                '{name}_monitor.npz'.format(name=name))
        params = np.load(result_file)
        d = dict(params)
        update_dict_of_lists(model_results, **d)

        valid_file = path.join(model_dir,
                               '{name}_monitor_valid.npz'.format(name=name))
        params_valid = np.load(valid_file)
        d_valid = dict(params_valid)
        update_dict_of_lists(valid_results, **d_valid)

        update_dict_of_lists(model_results, name=name)

    if omit_deltas:
        model_results = OrderedDict(
            (k, v) for k, v in model_results.iteritems() if not k.startswith('d_'))

    model_results.pop('dt_epoch')
    names = model_results.pop('name')
    training_times = model_results.pop('training_time')

    out_dir = path.join(out_path, 'compare.' + '.'.join(names))
    if path.isfile(out_dir):
        raise ValueError()
    elif not path.isdir(out_dir):
        os.mkdir(path.abspath(out_dir))

    plt.clf()
    x = 3
    y = ((len(model_results) - 1) // x) + 1

    fig, axes = plt.subplots(y, x)
    fig.set_size_inches(20, 20)

    if by_training_time:
        xlabel = 'seconds'
        us = [tt - tt[0] for tt in training_times]
    else:
        us = [range(tt.shape[0]) for tt in training_times]
        xlabel = 'epochs'

    for j, (k, vs) in enumerate(model_results.iteritems()):
        ax = axes[j // x, j % x]
        for name, u, v in zip(names, us, vs):
            ax.plot(u, v, label=name)

        if k in valid_results.keys():
            for name, u, v in zip(names, us, valid_results[k]):
                ax.plot(u, v, label=name + '(v)')

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
        models, data_iter, name, exp_dict = unpack_model_and_data(model_dir)
        sample_from_prior(models, data_iter, name, out_dir)
        rs = test(models, data_iter, name, **test_args)
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
    parser.add_argument('models', nargs='+')
    parser.add_argument('-t', '--by_time', action='store_true')
    parser.add_argument('-o', '--out_path', required=True)
    parser.add_argument('-d', '--see_deltas', action='store_true')
    parser.add_argument('-p', '--posterior_samples', default=1000, type=int)
    parser.add_argument('-i', '--inference_samples', default=100, type=int)
    parser.add_argument('-s', '--inference_steps', default=100, type=int)
    parser.add_argument('-b', '--batch_size', default=100, type=int)
    return parser

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()
    compare(args.models, args.out_path,
            omit_deltas=not(args.see_deltas),
            posterior_samples=args.posterior_samples,
            n_inference_samples=args.inference_samples,
            n_inference_steps=args.inference_steps,
            dx=args.batch_size,
            by_training_time=args.by_time)
