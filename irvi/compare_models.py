'''
Module for comparing models
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

from datasets.mnist import MNIST
from models.gbn import GaussianBeliefNet as GBN
from models.mlp import MLP
from models.sbn import SigmoidBeliefNetwork as SBN
from models.sbn import unpack
from utils.tools import (
    check_bad_nums,
    floatX,
    itemlist,
    load_model,
    load_experiment,
    update_dict_of_lists,
    _slice
)


def sample_from_prior(model_dir, out_path):
    name       = model_dir.split('/')[-2]
    model_file = glob(path.join(model_dir, '*best*npz'))[0]
    models, model_args  = load_model(model_file, unpack)

    model = models['sbn']
    tparams = model.set_tparams()

    dataset = model_args['dataset']
    dataset_args = model_args['dataset_args']
    data_iter = MNIST(batch_size=10, **dataset_args)

    py_p, updates = model.sample_from_prior()
    f_prior = theano.function([], py_p, updates=updates)
    samples = f_prior()
    data_iter.save_images(
        samples[:, None],
        path.join(out_path, name + '_samples_from_prior.png'),
        x_limit=10)

def compare(model_dirs,
            out_path,
            by_training_time=False,
            omit_deltas=True):

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

    out_dir = path.join(out_path, 'compare.' + '|'.join(names))
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
        ax.legend(loc=3)
        ax.patch.set_alpha(0.5)

    plt.tight_layout()
    plt.savefig(path.join(out_dir, 'results.png'))
    plt.close()

    print 'Sampling from priors'

    for model_dir in model_dirs:
        sample_from_prior(model_dir, out_dir)

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+')
    parser.add_argument('-t', '--by_time', action='store_true')
    parser.add_argument('-o', '--out_path', required=True)
    parser.add_argument('-d', '--see_deltas', action='store_true')
    return parser

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()
    compare(args.models, args.out_path, omit_deltas=not(args.see_deltas),
            by_training_time=args.by_time)