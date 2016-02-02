'''
Module for building a classifier with and without refinement
'''

import argparse
from collections import OrderedDict
from copy import deepcopy
import csv
from glob import glob
import matplotlib
matplotlib.use('Agg')
from matplotlib import pylab as plt
import numpy as np
import os
from os import path
import pprint
from progressbar import (
    Bar,
    Percentage,
    ProgressBar,
    RotatingMarker,
    SimpleProgress,
    Timer
)
import random
import sys
from tabulate import tabulate
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time

from datasets.mnist import MNIST
from main import load_data
from models.distributions import _softmax
from models.dsbn import unpack as unpack_dsbn
from models.gbn import GaussianBeliefNet as GBN
from models.mlp import MLP
from models.sbn import SigmoidBeliefNetwork as SBN
from models.sbn import unpack as unpack_sbn
from utils.monitor import SimpleMonitor
from utils import op
from utils.tools import (
    check_bad_nums,
    floatX,
    itemlist,
    load_model,
    load_experiment,
    print_profile,
    update_dict_of_lists,
    _slice
)


def classify(model_dir,
             n_inference_steps=20, n_inference_samples=20,
             dim_hs=[100], h_act='T.nnet.softplus',
             learning_rate=0.0001, learning_rate_schedule=None,
             dropout=0.1, batch_size=100, l2_decay=0.002,
             epochs=100,
             optimizer='rmsprop', optimizer_args=dict(),
             center_input=True, name='classifier'):
    out_path = model_dir

    inference_args = dict(
        inference_method='adaptive',
        inference_rate=0.1,
    )

    # ========================================================================
    print 'Loading model'

    model_file = glob(path.join(model_dir, '*best*npz'))[0]

    models, model_args = load_model(model_file, unpack_sbn, **inference_args)

    model = models['sbn']
    model.set_tparams()

    dataset = model_args['dataset']
    dataset_args = model_args['dataset_args']
    if dataset == 'mnist':
        dataset_args['binarize'] = True
        dataset_args['source'] = '/export/mialab/users/dhjelm/data/mnist.pkl.gz'

    train, valid, test = load_data(dataset, batch_size, batch_size, batch_size,
                                   **dataset_args)

    mlp_args = dict(
        dim_hs=dim_hs,
        h_act=h_act,
        dropout=dropout,
        out_act=train.acts['label']
    )

    X = T.matrix('x', dtype=floatX)
    Y = T.matrix('y', dtype=floatX)
    trng = RandomStreams(random.randint(0, 1000000))

    if center_input:
        print 'Centering input with train dataset mean image'
        X_mean = theano.shared(train.mean_image.astype(floatX), name='X_mean')
        X_i = X - X_mean
    else:
        X_i = X

    # ========================================================================
    print 'Loading MLP and forming graph'

    (qs, i_costs), _, updates = model.infer_q(
            X_i, X, n_inference_steps, n_inference_samples=n_inference_samples)

    q0 = qs[0]
    qk = qs[-1]

    constants = [q0, qk]
    dim_in = model.dim_h
    dim_out = train.dims['label']

    mlp0_args = deepcopy(mlp_args)
    mlp0 = MLP(dim_in, dim_out, name='classifier_0', **mlp0_args)
    mlpk_args = deepcopy(mlp_args)
    mlpk = MLP(dim_in, dim_out, name='classifier_k', **mlpk_args)
    mlpx_args = deepcopy(mlp_args)
    mlpx = MLP(train.dims[str(dataset)], dim_out, name='classifier_x', **mlpx_args)
    tparams = mlp0.set_tparams()
    tparams.update(**mlpk.set_tparams())
    tparams.update(**mlpx.set_tparams())

    print_profile(tparams)

    p0 = mlp0(q0)
    pk = mlpk(qk)
    px = mlpx(X_i)

    # ========================================================================
    print 'Getting cost'

    cost0 = mlp0.neg_log_prob(Y, p0).sum(axis=0)
    costk = mlpk.neg_log_prob(Y, pk).sum(axis=0)
    costx = mlpx.neg_log_prob(Y, px).sum(axis=0)

    cost = cost0 + costk + costx
    extra_outs = []
    extra_outs_names = ['cost']

    if l2_decay > 0.:
        print 'Adding %.5f L2 weight decay' % l2_decay
        mlp0_l2_cost = mlp0.get_L2_weight_cost(l2_decay)
        mlpk_l2_cost = mlpk.get_L2_weight_cost(l2_decay)
        mlpx_l2_cost = mlpx.get_L2_weight_cost(l2_decay)
        cost += mlp0_l2_cost + mlpk_l2_cost + mlpx_l2_cost
        extra_outs += [mlp0_l2_cost, mlpk_l2_cost, mlpx_l2_cost]
        extra_outs_names += ['MLP0 L2 cost', 'MLPk L2 cost', 'MLPx L2 cost']

    # ========================================================================
    print 'Extra functions'
    error0 = (Y * (1 - p0)).sum(1).mean()
    errork = (Y * (1 - pk)).sum(1).mean()
    errorx = (Y * (1 - px)).sum(1).mean()
    
    f_test_keys = ['Error 0', 'Error k', 'Error x', 'Cost 0', 'Cost k', 'Cost x']
    f_test = theano.function([X, Y], [error0, errork, errorx, cost0, costk, costx])
    
    # ========================================================================
    print 'Setting final tparams and save function'

    all_params = OrderedDict((k, v) for k, v in tparams.iteritems())

    tparams = OrderedDict((k, v)
        for k, v in tparams.iteritems()
        if (v not in updates.keys() or v not in excludes))

    print 'Learned model params: %s' % tparams.keys()
    print 'Saved params: %s' % all_params.keys()

    def save(tparams, outfile):
        d = dict((k, v.get_value()) for k, v in all_params.items())

        d.update(
            dim_in=dim_in,
            dim_out=dim_out,
            dataset=dataset, dataset_args=dataset_args,
            **mlp_args
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
        lr, tparams, grads, [X, Y], cost, extra_ups=updates,
        extra_outs=extra_outs, **optimizer_args)

    monitor = SimpleMonitor()

    try:
        epoch_t0 = time.time()
        s = 0
        e = 0

        widgets = ['Epoch {epoch} ({name}, '.format(epoch=e, name=name),
                   Timer(), '): ', Bar()]
        epoch_pbar = ProgressBar(widgets=widgets, maxval=train.n).start()
        training_time = 0

        while True:
            try:
                x, y = train.next()
                
                if train.pos == -1:
                    epoch_pbar.update(train.n)
                else:
                    epoch_pbar.update(train.pos)

            except StopIteration:
                print
                epoch_t1 = time.time()
                training_time += (epoch_t1 - epoch_t0)
                valid.reset()

                widgets = ['Validating: ',
                          Percentage(), ' (', Timer(), ')']
                pbar    = ProgressBar(widgets=widgets, maxval=valid.n).start()
                results_train = OrderedDict()
                results_valid = OrderedDict()
                while True:
                    try:
                        x_valid, y_valid = valid.next()
                        x_train, y_train = train.next()

                        r_train = f_test(x_train, y_train)
                        r_valid = f_test(x_valid, y_valid)
                        results_i_train = dict((k, v) for k, v in zip(f_test_keys, r_train))
                        results_i_valid = dict((k, v) for k, v in zip(f_test_keys, r_valid))
                        update_dict_of_lists(results_train, **results_i_train)
                        update_dict_of_lists(results_valid, **results_i_valid)

                        if valid.pos == -1:
                            pbar.update(valid.n)
                        else:
                            pbar.update(valid.pos)

                    except StopIteration:
                        print
                        break

                def summarize(d):
                    for k, v in d.iteritems():
                        d[k] = np.mean(v)

                summarize(results_train)
                summarize(results_valid)

                monitor.update(**results_train)
                monitor.update(dt_epoch=(epoch_t1-epoch_t0),
                               training_time=training_time)
                monitor.update_valid(**results_valid)
                monitor.display()

                monitor.save(path.join(
                    out_path, '{name}_monitor.png').format(name=name))
                monitor.save_stats(path.join(
                    out_path, '{name}_monitor.npz').format(name=name))
                monitor.save_stats_valid(path.join(
                    out_path, '{name}_monitor_valid.npz').format(name=name))

                e += 1
                epoch_t0 = time.time()

                valid.reset()
                train.reset()

                if learning_rate_schedule is not None:
                    if e in learning_rate_schedule.keys():
                        lr = learning_rate_schedule[e]
                        print 'Changing learning rate to %.5f' % lr
                        learning_rate = lr

                widgets = ['Epoch {epoch} ({name}, '.format(epoch=e, name=name),
                           Timer(), '): ', Bar()]
                epoch_pbar = ProgressBar(widgets=widgets, maxval=train.n).start()

                continue

            if e > epochs:
                break

            rval = f_grad_shared(x, y)

            if check_bad_nums(rval, extra_outs_names):
                print rval
                print np.any(np.isnan(mlpk.W0.get_value()))
                print np.any(np.isnan(mlpk.b0.get_value()))
                print np.any(np.isnan(mlpk.W1.get_value()))
                print np.any(np.isnan(mlpk.b1.get_value()))
                raise ValueError('Bad number!')

            f_grad_updates(learning_rate)
            s += 1

    except KeyboardInterrupt:
        print 'Training interrupted'

    test.reset()

    widgets = ['Testing: ',
               Percentage(), ' (', Timer(), ')']
    pbar    = ProgressBar(widgets=widgets, maxval=test.n).start()
    results_test = OrderedDict()
    while True:
        try:
            x_test, y_test = test.next()
            r_test = f_test(x_test, y_test)
            results_i_test = dict((k, v) for k, v in zip(f_test_keys, r_test))
            update_dict_of_lists(results_test, **results_i_test)
            if test.pos == -1:
                pbar.update(test.n)
            else:
                pbar.update(test.pos)

        except StopIteration:
            print
            break

    def summarize(d):
        for k, v in d.iteritems():
            d[k] = np.mean(v)

    summarize(results_test)
    print 'Test results:'
    monitor.simple_display(results_test)

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
    parser.add_argument('model_dir', default=None)
    return parser

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()
    classify(args.model_dir)
