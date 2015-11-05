'''
Module for evaluating SBN/GBN
'''

import argparse
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import os
from os import path
import theano
from theano import tensor as T

from ica_exp import load_data
from ica_exp import unpack
from tools import load_experiment
from tools import load_model


floatX = theano.config.floatX

def lower_bound_curve(
    model_file, rs=None, n_samples=1000,
    out_path=None,
    prior='logistic',
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
        alpha=alpha
    )

    models, _ = load_model(model_file, unpack, **model_args)
    n_mcmc_samples_test = 50

    _, _, test = load_data(dataset,
                           None,
                           None,
                           n_samples,
                           **dataset_args)

    if prior == 'logistic':
        model = models['sbn']
    elif prior == 'gaussian':
        model = models['gbn']

    tparams = model.set_tparams()

    # ========================================================================
    print 'Setting up Theano graph for lower bound'

    X = T.matrix('x', dtype=floatX)

    outs_s, updates_s = model(X, n_inference_steps=0, n_samples=n_mcmc_samples_test, calculate_log_marginal=True)

    f_lower_bound = theano.function([X], [outs_s['lower_bound'], outs_s['nll']], updates=updates_s)

    # ========================================================================
    print 'Getting initial lower bound'

    x_t, _ = test.next()
    lb, nll = f_lower_bound(x_t)
    lbs = [lb]
    nlls = [nll]

    R = T.scalar('r', dtype='int64')

    outs_s, updates_s = model(X, n_inference_steps=R, n_samples=n_mcmc_samples_test, calculate_log_marginal=True)

    f_lower_bound = theano.function([X, R], [outs_s['lower_bound'], outs_s['nll']], updates=updates_s)

    # ========================================================================
    print 'Calculating lower bounds'

    if rs is None:
        rs = range(5, 100, 5)

    for r in rs:
        print 'number of inference steps: %d' % r
        lb, nll = f_lower_bound(x_t, r)
        lbs.append(lb)
        nlls.append(nll)
        print 'lower bound: %.2f, nll: %.2f' % (lb, nll)

    fig = plt.figure()
    plt.plot(lbs)

    print 'Calculating final lower bound and marginal with 1000 posterior samples'

    outs_s, updates_s = model(X, n_inference_steps=rs[-1], n_samples=n_mcmc_samples_test, calculate_log_marginal=True)

    f_lower_bound = theano.function([X], [outs_s['lower_bound'], outs_s['nll']], updates=updates_s)

    xs = [x_t[i: (i + 100)] for i in range(0, n_samples, 100)]

    N = len(range(0, n_samples, 100))
    lb_t = 0.
    nll_t = 0.
    for x in xs:
        lb, nll = f_lower_bound(x)
        lb_t += lb
        nll_t += nll

    lb_t /= N
    nll_t /= N
    print 'Final lower bound and NLL: %.2f and %.2f' % (lb_t, nll_t)

    if out_path is not None:
        plt.savefig(out_path)
    else:
        plt.show()

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('model')
    parser.add_argument('experiment')
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

    lower_bound_curve(args.model, out_path=out_path, **exp_dict)
