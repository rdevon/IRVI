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
    model_file, steps=20,
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

    '''
    np.set_printoptions(precision=2)
    sam, _ = model.sample_from_prior(2)
    f = theano.function([], sam)
    print f()
    assert False
    '''


    if dataset == 'mnist':
        train_iter = MNIST(mode='train', **dataset_args)
        data_iter = MNIST(batch_size=data_samples, mode=mode, inf=False, **dataset_args)
    else:
        raise ValueError()

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

    dx = 100
    data_samples = min(data_samples, data_iter.n)
    xs = [x[i: (i + dx)] for i in range(0, data_samples, dx)]
    N = data_samples // dx

    print ('Calculating final lower bound and marginal with %d data samples, %d posterior samples '
           'with %d inference steps' % (N * dx, posterior_samples, steps))


    results, samples, full_results, updates_s = model(X_i, X, n_inference_steps=steps, n_samples=posterior_samples, stride=0)

    print 'Saving sampling from posterior'
    x_test = x
    b_energies = samples['batch_energies'][0]
    energies = samples['batch_energies'][-1]
    b_py = samples['py'][0]
    py = samples['py'][-1]
    '''
    f = theano.function([X], [b_energies.shape, energies.shape, b_py.shape, py.shape], updates=updates_s)
    assert False, f(x_test)
    '''
    best_idx = (energies - b_energies).argsort()[:1000].astype('int64')
    #worst_idx = (b_energies - energies).argsort().astype('int64')
    p_best = T.concatenate([X[best_idx][None, :, :],
                            b_py[:, best_idx].mean(axis=0)[None, :, :],
                            py[:, best_idx].mean(axis=0)[None, :, :]])
    f_best = theano.function([X], p_best, updates=updates_s)
    py_best = f_best(x_test)
    data_iter.save_images(
        py_best,
        path.join(out_path, 'samples_from_post_best.png')
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


    eval_model(model_file, mode=args.mode, out_path=out_path,
               posterior_samples=args.posterior_samples,
               inference_samples=args.inference_samples,
               data_samples=args.data_samples,
               steps=args.inference_steps,
               **exp_dict)
