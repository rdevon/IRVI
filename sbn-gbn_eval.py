'''
Module for evaluating SBN/GBN
'''

import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import os
from os import path
from progressbar import ProgressBar
import theano
from theano import tensor as T

from ica_exp import load_data
from ica_exp import unpack
from mnist import MNIST
from tools import load_experiment, load_model


floatX = theano.config.floatX

def lower_bound_curve(
    model_file, rs=None, n_samples=10000,
    out_path=None,
    mode='valid',
    prior='logistic',
    center_input=True,
    center_latent=False,
    z_init='recognition_net',
    inference_method='momentum',
    inference_rate=.01,
    inference_decay=1.0,
    n_inference_samples=20,
    entropy_scale=1.0,
    inference_scaling=None,
    alpha=7,
    n_mcmc_samples_test=20,
    dataset=None,
    dataset_args=None,
    **kwargs):

    model_args = dict(
        z_init=z_init,
        inference_method=inference_method,
        inference_rate=inference_rate,
        n_inference_samples=n_inference_samples,
        inference_decay=inference_decay,
        entropy_scale=entropy_scale,
        inference_scaling=inference_scaling,
        n_mcmc_samples_test=n_mcmc_samples_test,
        alpha=alpha,
        center_latent=center_latent
    )

    models, _ = load_model(model_file, unpack, **model_args)
    n_mcmc_samples_test = 50

    if dataset == 'mnist':
        data_iter = MNIST(batch_size=10000, mode=mode, inf=False, **dataset_args)
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

    outs_s, updates_s = model(X_i, X, n_inference_steps=0, n_samples=n_mcmc_samples_test, calculate_log_marginal=True)

    f_lower_bound = theano.function([X], [outs_s['lower_bound'], outs_s['nll']], updates=updates_s)

    # ========================================================================
    print 'Getting initial lower bound'

    x, _ = data_iter.next()
    lb, nll = f_lower_bound(x)
    lbs = [lb]
    nlls = [nll]

    print 'number of inference steps: 0'
    print 'lower bound: %.2f, nll: %.2f' % (lb, nll)

    R = T.scalar('r', dtype='int64')

    outs_s, updates_s = model(X_i, X, n_inference_steps=R, n_samples=n_mcmc_samples_test, calculate_log_marginal=True)

    f_lower_bound = theano.function([X, R], [outs_s['lower_bound'], outs_s['nll']], updates=updates_s)

    # ========================================================================
    print 'Calculating lower bound curve (on 1000 samples)'

    if rs is None:
        rs = range(5, 50, 5)

    try:
        for r in rs:
            print 'number of inference steps: %d' % r
            lb, nll = f_lower_bound(x[:1000], r)
            lbs.append(lb)
            nlls.append(nll)
            print 'lower bound: %.2f, nll: %.2f' % (lb, nll)
    except MemoryError:
        print 'Memory Error. Stopped early.'

    fig = plt.figure()
    plt.plot(lbs)

    print 'Calculating final lower bound and marginal with %d posterior samples' % x.shape[0]

    outs_s, updates_s = model(X_i, X, n_inference_steps=rs[-1], n_samples=n_mcmc_samples_test, calculate_log_marginal=True)

    f_lower_bound = theano.function([X], [outs_s['lower_bound'], outs_s['nll']], updates=updates_s)

    xs = [x[i: (i + 100)] for i in range(0, n_samples, 100)]

    N = len(range(0, n_samples, 100))
    lb_t = 0.
    nll_t = 0.

    pbar = ProgressBar(maxval=len(xs)).start()
    for i, x in enumerate(xs):
        lb, nll = f_lower_bound(x)
        lb_t += lb
        nll_t += nll
        pbar.update(i)

    lb_t /= N
    nll_t /= N
    print 'Final lower bound and NLL: %.2f and %.2f' % (lb_t, nll_t)

    if out_path is not None:
        plt.savefig(out_path)
        print 'Sampling from the prior'

        py_p = model.sample_from_prior()
        f_prior = theano.function([], py_p)

        samples = f_prior()
        data_iter.save_images(
            pd_p[:, None],
            path.join(out_path, 'samples_from_prior.png'),
            x_limit=10)

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('experiment')
    parser.add_argument('-m', '--mode', default='valid',
                        help='Dataset mode: valid, test, or train')
    parser.add_argument('-o', '--out_path', default=None,
                        help='Output path for stuff')
    return parser

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()

    exp_dict = load_experiment(path.abspath(args.experiment))
    out_path = args.out_path

    if out_path is not None:
        print 'Saving to %s' % out_path
        if path.isfile(out_path):
            raise ValueError()
        elif not path.isdir(out_path):
            os.mkdir(path.abspath(out_path))

    if args.mode not in ['valid', 'test', 'train']:
        raise ValueError('mode must be `train`, `valid`, or `test`. Got %s' % args.mode)

    lower_bound_curve(args.model, mode=args.mode, out_path=out_path, **exp_dict)
