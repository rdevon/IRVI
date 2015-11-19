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

from ica_exp import load_data
from ica_exp import unpack
from mnist import MNIST
import op
from tools import itemlist, load_experiment, load_model


floatX = theano.config.floatX

def eval_model(
    model_file, steps=50,
    data_samples=10000,
    out_path=None,
    optimizer=None,
    optimizer_args=dict(),
    batch_size=100,
    valid_scores=None,
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
        n_inference_samples=inference_samples
    )

    models, _ = load_model(model_file, unpack, **model_args)

    if dataset == 'mnist':
        data_iter = MNIST(batch_size=data_samples, mode=mode, inf=False, **dataset_args)
        valid_iter = MNIST(batch_size=500, mode='valid', inf=False, **dataset_args)
    else:
        raise ValueError()

    if prior == 'logistic':
        model = models['sbn']
    elif prior == 'gaussian':
        model = models['gbn']

    tparams = model.set_tparams()

    # ========================================================================
    print 'Setting up Theano graph for lower bound'

    X = T.matrix('x', dtype=floatX)

    if center_input:
        print 'Centering input with train dataset mean image'
        X_mean = theano.shared(data_iter.mean_image.astype(floatX), name='X_mean')
        X_i = X - X_mean
    else:
        X_i = X

    x, _ = data_iter.next()
    x_v, _ = valid_iter.next()

<<<<<<< HEAD
    xs = [x[i: (i + 100)] for i in range(0, n_samples, 100)]

    N = len(range(0, n_samples, 100))
    lb_t = 0.
    nll_t = 0.

    #pbar = ProgressBar(maxval=len(xs)).start()
    for i, x in enumerate(xs):
        lb, nll = f_lower_bound(x)
        lb_t += lb
        nll_t += nll
        #pbar.update(i)

    lb = lb_t / N
    nll = nll_t / N
    lbs = [lb]
    nlls = [nll]

    print 'number of inference steps: 0'
    print 'lower bound: %.2f, nll: %.2f' % (lb, nll)

    # ========================================================================
    print 'Calculating lower bound curve (on 500 samples)'

    if rs is None:
        rs = range(inference_stride, inference_stride * 10 + 1, inference_stride)
    #rs = [50]

    best_r = 0
    best_lb = lb
    lbs = []
    nlls = []
    try:
        for r in rs:
            print 'number of inference steps: %d' % r
            outs_s, updates_s = model(X_i, X, n_inference_steps=r,
                                      n_samples=posterior_samples,
                                      calculate_log_marginal=True)
            f_lower_bound = theano.function([X], [outs_s['lower_bound'],
                                                  outs_s['nll']],
                updates=updates_s)

            lb, nll = f_lower_bound(x[:500])

            lb_v, nll_v = f_lower_bound(x_v)

            if lb_v < best_lb:
                best_lb = lb_v
                best_r = r

            lbs.append(lb)
            nlls.append(nll)
            print 'lower bound: %.2f, nll: %.2f' % (lb, nll)
    except MemoryError:
        print 'Memory Error. Stopped early.'

    fig = plt.figure()
    plt.plot(range(len(lbs)), lbs)
    plt.plot(range(len(lbs)), nlls)
=======
    dx = 100
    xs = [x[i: (i + dx)] for i in range(0, data_samples, dx)]
    N = data_samples // dx
>>>>>>> 21185271c8644b6d04d055fa99f68786588fb4ee

    print ('Calculating final lower bound and marginal with %d data samples, %d posterior samples '
           'with %d inference steps' % (N * dx, posterior_samples, steps))

    outs_s, updates_s = model(X_i, X, n_inference_steps=steps, n_samples=posterior_samples, calculate_log_marginal=True)
    f_lower_bound = theano.function([X], [outs_s['lower_bound'], outs_s['nll']] + outs_s['lower_bounds'] + outs_s['nlls'], updates=updates_s)
    lb_t = []
    nll_t = []
    nlls_t = []
    lbs_t = []

    pbar = ProgressBar(maxval=len(xs)).start()
    for i, x in enumerate(xs):
        outs = f_lower_bound(x)
        lb, nll = outs[:2]
        outs = outs[2:]
        lbs = outs[:len(outs)/2]
        nlls = outs[len(outs)/2:]
        lbs_t.append(lbs)
        nlls_t.append(nlls)
        lb_t.append(lb)
        nll_t.append(nll)
        pbar.update(i)

    lb_t = np.mean(lb_t)
    nll_t = np.mean(nll_t)
    lbs_t = np.mean(lbs_t, axis=0).tolist()
    nlls_t = np.mean(nlls_t, axis=0).tolist()
    print 'Final lower bound and NLL: %.2f and %.2f' % (lb_t, nll_t)
    print lbs_t
    print nlls_t

    if out_path is not None:
        plt.savefig(out_path)
        print 'Sampling from the prior'

        np.save(path.join(out_path, 'lbs.npy'), lbs_t)
        np.save(path.join(out_path, 'nlls.npy'), nlls_t)

        py_p = model.sample_from_prior()
        f_prior = theano.function([], py_p)

        samples = f_prior()
        data_iter.save_images(
            samples[:, None],
            path.join(out_path, 'samples_from_prior.png'),
            x_limit=10)

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment_dir')
    parser.add_argument('-m', '--mode', default='valid',
                        help='Dataset mode: valid, test, or train')
    parser.add_argument('-p', '--posterior_samples', default=1000, type=int,
                        help='Number of posterior during eval')
    parser.add_argument('-i', '--inference_samples', default=1000, type=int)
    parser.add_argument('-s', '--inference_steps', default=50, type=int)
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

    valid_file = path.join(exp_dir, 'valid_lbs.npy')
    valid_scores = np.load(valid_file)

    eval_model(model_file, mode=args.mode, out_path=out_path,
               valid_scores=valid_scores,
               posterior_samples=args.posterior_samples,
               inference_samples=args.inference_samples,
               data_samples=args.data_samples,
               steps=args.inference_steps,
               **exp_dict)
