'''
Module for evaluating SBN/GBN
'''

import argparse
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

from datasets.mnist import MNIST
from main import load_data
from models.gbn import GaussianBeliefNet as GBN
from models.mlp import MLP
from models.sbn import SigmoidBeliefNetwork as SBN
from models.sbn import unpack
from utils.monitor import SimpleMonitor
from utils import op
from utils.tools import (
    check_bad_nums,
    floatX,
    itemlist,
    load_model,
    load_experiment,
    _slice
)


def eval_model(
    model_file, 
    transpose=False,
    drop_units=0,
    metric='likelihood',
    steps=20,
    data_samples=10000,
    out_path=None,
    optimizer=None,
    optimizer_args=dict(),
    batch_size=100,
    mode='valid',
    prior='logistic',
    center_input=True,
    z_init='recognition_net',
    inference_method='momentum',
    inference_rate=.01,
    n_mcmc_samples=20,
    posterior_samples=20,
    inference_samples=20,
    dataset=None,
    dataset_args=None,
    extra_inference_args=dict(),
    **kwargs):

    model_args = dict(
        prior=prior,
        z_init=z_init,
        inference_method=inference_method,
        inference_rate=inference_rate,
        extra_inference_args=extra_inference_args
    )

    models, _ = load_model(model_file, unpack, **model_args)

    if prior == 'logistic' or prior == 'darn':
        model = models['sbn']
    elif prior == 'gaussian':
        model = models['gbn']

    tparams = model.set_tparams()

    train_iter, valid_iter, test_iter = load_data(dataset, batch_size, batch_size, batch_size,
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
    print 'Setting up Theano graph for lower bound'

    X = T.matrix('x', dtype=floatX)

    if center_input:
        print 'Centering input with train dataset mean image'
        X_mean = theano.shared(train_iter.mean_image.astype(floatX), name='X_mean')
        X_i = X - X_mean
    else:
        X_i = X

    x, _ = data_iter.next()
    if drop_units:
        print 'Dropping %.2f%% of units' % (100 * drop_units)
        q0 = model.posterior(X_i)
        r = model.trng.binomial(p=1-drop_units, size=q0.shape, dtype=floatX)
        q0 = q0 * r + 0.5 * (1. - r)
    else:
        q0 = None
    results, samples, full_results, updates_s = model(X_i, X, n_inference_steps=steps, n_samples=posterior_samples, stride=0, q0=q0, n_inference_samples=inference_samples)

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
    '''
    f = theano.function([X], [q0.shape, qk.shape, T.tensordot(q0, qk, axis=1).shape, metric.shape], updates=updates_s)
    print f(x_test)
    assert False
    '''

    best_idx = distance.argsort()[:1000].astype('int64')
    #worst_idx = (b_energies - energies).argsort().astype('int64')
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
    parser.add_argument('experiment_dir')
    parser.add_argument('-m', '--mode', default='valid',
                        help='Dataset mode: valid, test, or train')
    parser.add_argument('-p', '--posterior_samples', default=20, type=int,
                        help='Number of posterior during eval')
    parser.add_argument('-i', '--inference_samples', default=20, type=int)
    parser.add_argument('-s', '--inference_steps', default=20, type=int)
    parser.add_argument('-d', '--data_samples', default=10000, type=int)
    parser.add_argument('-M', '--metric', default='likelihood')
    parser.add_argument('-D', '--drop_units', default=0, type=float)
    parser.add_argument('-r', '--inference_rate', default=0.1, type=float)
    parser.add_argument('-t', '--transpose', action='store_true')
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
        raise ValueError()

    exp_dict = load_experiment(path.abspath(yaml))

    if args.mode not in ['valid', 'test', 'train']:
        raise ValueError('mode must be `train`, `valid`, or `test`. Got %s' % args.mode)

    try:
        model_file = glob(path.join(exp_dir, '*best*npz'))[0]
        print 'Found best in %s' % model_file
    except:
        raise ValueError()

    exp_dict.pop('inference_rate')
    eval_model(model_file, metric=args.metric, mode=args.mode, out_path=out_path,
               transpose=args.transpose,
               inference_rate=args.inference_rate,
               posterior_samples=args.posterior_samples,
               inference_samples=args.inference_samples,
               data_samples=args.data_samples,
               steps=args.inference_steps,
               drop_units=args.drop_units,
               **exp_dict)
