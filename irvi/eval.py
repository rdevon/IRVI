'''
Module for evaluating SBN/GBN
'''

import argparse
from collections import OrderedDict
from glob import glob
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import numpy as np
import os
from os import path
from progressbar import ProgressBar
import theano
from theano import tensor as T
import time

from inference import resolve as resolve_inference
from main import load_data
from models.gbn import (
    GBN,
    unpack as unpack_gbn
)
from models.mlp import MLP
from models.sbn import (
    SBN,
    unpack as unpack_sbn
)
from utils.monitor import SimpleMonitor
from utils import (
    floatX,
    op
)
from utils.tools import (
    check_bad_nums,
    itemlist,
    load_model,
    load_experiment,
    _slice
)


def eval_model(
    model_file,
    dim_h=None,
    dim_hs=None,
    transpose=False,
    n_posterior_samples=100,
    drop_units=0,
    metric='likelihood',
    steps=20,
    data_samples=10000,
    center_input=True,
    inference_args=dict(),
    out_path=None,
    optimizer=None,
    optimizer_args=dict(),
    batch_size=100,
    mode='valid',
    prior='binomial',
    dataset_args=None,
    **kwargs):

    if dim_h is None:
        raise NotImplementedError('This evaluation script does not yet'
                                  'cover deeper latent models yet.')
    else:
        assert dim_hs is None

    # ========================================================================
    print 'Loading Data'
    train_iter, valid_iter, test_iter = load_data(train_batch_size=batch_size,
                                                  valid_batch_size=batch_size,
                                                  test_batch_size=batch_size,
                                                  **dataset_args)

    if mode == 'train':
        data_iter = train_tier
    elif mode == 'valid':
        data_iter = valid_iter
    elif mode == 'test':
        data_iter = test_iter
    else:
        raise ValueError(mode)

    # ========================================================================
    print 'Loading Model'

    if prior == 'gaussian':
        unpack = unpack_gbn
        model_name = 'gbn'
        inference_method = 'momentum'
    elif prior in ['binomial', 'darn']:
        unpack = unpack_sbn
        model_name = 'sbn'
        inference_method = 'air'
    else:
        raise ValueError(prior)

    models, _ = load_model(model_file, unpack,
                           distributions=data_iter.distributions,
                           dims=data_iter.dims)

    inference_args['inference_method'] = inference_method

    model = models[model_name]
    tparams = model.set_tparams()

    # ========================================================================
    print 'Evaluating Refinement'

    X = T.matrix('x', dtype=floatX)

    if center_input:
        print 'Centering input with train dataset mean image'
        X_mean = theano.shared(train_iter.mean_image.astype(floatX), name='X_mean')
        X_i = X - X_mean
    else:
        X_i = X

    x = data_iter.next()[data_iter.name]
    if drop_units:
        print 'Dropping %.2f%% of units' % (100 * drop_units)
        q0 = model.posterior(X_i)
        r = model.trng.binomial(p=1-drop_units, size=q0.shape, dtype=floatX)
        q0 = q0 * r + 0.5 * (1. - r)
    else:
        q0 = None

    inference = resolve_inference(model, **inference_args)

    if inference_method == 'momentum':
        if prior == 'binomial':
            raise NotImplementedError()
        results, samples, full_results, updates_s = inference(
            X_i, X,
            n_posterior_samples=n_posterior_samples)
    elif inference_method == 'air':
        results, samples, full_results, updates_s = inference(
            X_i, X, n_posterior_samples=n_posterior_samples)
    else:
        raise ValueError(inference_method)

    print 'Saving sampling from posterior'
    x_test = x
    b_py = samples['py'][0]
    py = samples['py'][-1]

    if metric == 'likelihood':
        energies0 = samples['batch_energies'][0]
        energiesk = samples['batch_energies'][-1]
        distance = energiesk - energies0
    elif metric == 'cosine':
        q0 = samples['q'][0]
        qk = samples['q'][-1]
        distance = (q0 * qk).sum(axis=1) / (q0.norm(L=2) * qk.norm(L=2))
    elif metric == 'manhattan':
        q0 = samples['q'][0]
        qk = samples['q'][-1]
        distance = -(abs(q0 - qk)).sum(axis=1)
    else:
        raise ValueError(metric)

    best_idx = distance.argsort()[:1000].astype('int64')
    p_best = T.concatenate([X[best_idx][None, :, :],
                            b_py[:, best_idx].mean(axis=0)[None, :, :],
                            py[:, best_idx].mean(axis=0)[None, :, :]])
    f_best = theano.function([X], p_best, updates=updates_s)
    py_best = f_best(x_test)

    data_iter.save_images(
        py_best,
        path.join(out_path, 'samples_from_post_best.png'),
        transpose=transpose
    )

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir',
                        help='Location of the experiment (directory)')
    parser.add_argument('-m', '--mode', default='valid',
                        help='Dataset mode: valid, test, or train')
    parser.add_argument('-p', '--n_posterior_samples', default=20, type=int,
                        help='Number of posterior during eval')
    parser.add_argument('-i', '--n_inference_samples', default=20, type=int,
                        help='Number of inference samples for IRVI')
    parser.add_argument('-s', '--n_inference_steps', default=20, type=int,
                        help='Number of inference steps for irvi')
    parser.add_argument('-r', '--inference_rate', default=0.01, type=float,
                        help='Inference rate for IRVI')
    parser.add_argument('-d', '--data_samples', default=10000, type=int,
                        help='Number of data samples for eavluation')
    parser.add_argument('-M', '--metric', default='likelihood',
                        help='Metric for evaluating posterior refinement on '
                        'reconstruction')
    parser.add_argument('-D', '--drop_units', default=0, type=float,
                        help='Drop latent units before refinement')
    parser.add_argument('-t', '--transpose', action='store_true',
                        help='Transpose the reconstruction images')
    return parser

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()

    exp_dir = path.abspath(args.experiment_dir)
    out_path = path.join(exp_dir, 'results')
    if not path.isdir(out_path):
        os.mkdir(out_path)

    try:
        yaml = glob(path.join(exp_dir, '*.yaml'))[0]
        print 'Found yaml %s' % yaml
    except:
        raise ValueError('Best file not found')

    exp_dict = load_experiment(path.abspath(yaml))

    if args.mode not in ['valid', 'test', 'train']:
        raise ValueError('mode must be `train`, `valid`, or `test`. Got %s' % args.mode)

    try:
        model_file = glob(path.join(exp_dir, '*best*npz'))[0]
        print 'Found best in %s' % model_file
    except:
        raise ValueError()

    inference_args = OrderedDict(
        inference_rate=args.inference_rate,
        n_inference_samples=args.n_inference_samples,
        n_inference_steps=args.n_inference_steps
    )

    exp_dict.pop('inference_args')

    eval_model(model_file, metric=args.metric, mode=args.mode, out_path=out_path,
               transpose=args.transpose,
               inference_args=inference_args,
               n_posterior_samples=args.n_posterior_samples,
               data_samples=args.data_samples,
               drop_units=args.drop_units,
               **exp_dict)
