'''
Sampling and inference with LSTM models
'''

import argparse
from collections import OrderedDict
from glob import glob
import matplotlib
from matplotlib import animation
from matplotlib import pylab as plt
import numpy as np
import os
from os import path
import pprint
import random
import shutil
import sys
from sys import stdout
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time
import yaml

from GSN import likelihood_estimation_parzen as lep
from gru import GenGRU
from rbm import RBM
from rnn import GenRNN
from horses import Horses
from horses import SimpleHorses
from layers import Averager
from layers import BaselineWithInput
from layers import MLP
from layers import ParzenEstimator
from mnist import mnist_iterator
from mnist import MNIST_Chains
import op
import tools
from tools import check_bad_nums
from tools import itemlist
from tools import load_model
from tools import log_mean_exp
from tools import parzen_estimation


floatX = theano.config.floatX

def unpack(dim_h=None,
           dataset='mnist',
           dataset_args=None,
           **model_args):

    dim_h = int(dim_h)

    if dataset == 'mnist':
        dim_in = 28 * 28
    elif dataset == 'horses':
        dims = dataset_args['dims']
        dim_in = dims[0] * dims[1]
    else:
        raise ValueError()

    rbm = RBM(dim_in, dim_h)
    models = [rbm]

    return models, model_args, dict(
        dataset_args=dataset_args
    )

def load_model_for_sampling(model_file):
    models, kwargs = load_model(model_file, unpack)
    dataset_args = kwargs['dataset_args']
    dataset = kwargs['dataset']

    if dataset == 'mnist':
        train = MNIST_Chains(batch_size=1, mode='train', **dataset_args)
        test = MNIST_Chains(batch_size=1, mode='test', **dataset_args)
    elif dataset == 'horses':
        train = Horses(batch_size=1, crop_image=True, **dataset_args)
    else:
        raise ValueError()

    rbm = models['rbm']

    tparams = rbm.set_tparams()
    train.set_f_energy(energy_function, rnn)

    return rbm, train, test

def get_sample_cross_correlation(model_file, n_steps=100):
    rbm, dataset, test = load_model_for_sampling(model_file)
    samples = generate_samples(model_file, n_steps=n_steps, n_samples=1)[1:, 0]

    c = np.corrcoef(samples, samples)[:n_steps, n_steps:]
    plt.imshow(c)
    plt.colorbar()
    plt.show()

def generate(model_file, n_steps=20, n_samples=40, out_path=None):
    rbm, train, test = load_model_for_sampling(model_file)
    params = rbm.get_params()

    X = T.matrix('x', dtype=floatX)
    x_s, h_s, p_s, q_s = rbm._step(X, *params)
    f_sam = theano.function([X], [x_s, h_s, p_s, q_s])

    x = rnn.rng.binomial(p=0.5, size=(n_samples, rbm.dim_in), n=1).astype(floatX)
    ps = [x]
    for s in xrange(n_steps):
        x, h, p, q = f_sam(x)
        ps.append(p)

    if out_path is not None:
        train.save_images(np.array(ps), path.join(out_path, 'generation_samples.png'))
    else:
        return ps[-1]

def visualize(model_file, out_path=None, interval=1, n_samples=-1,
              save_movie=True, use_data_every=50, use_data_in=False,
              save_hiddens=False):
    rbm, train, test = load_model_for_sampling(model_file)
    params = rbm.get_params()

    X = T.matrix('x', dtype=floatX)
    x_s, h_s, p_s, q_s = rnn._step(X, *params)
    f_sam = theano.function([X], [x_s, h_s, p_s, q_s])
    ps = []
    xs = []
    hs = []

    try:
        x = train.X[:1]
        s = 0
        while True:
            stdout.write('\rSampling (%d): Press ^c to stop' % s)
            stdout.flush()
            x, h, p, q = f_sam(x)
            hs.append(h)
            xs.append(x)
            if use_data_every > 0 and s % use_data_every == 0:
                x_n = train.next_simple(20)
                energies, _, h_p = train.f_energy(x_n, x, h)
                energies = energies[0]
                x = x_n[np.argmin(energies)][None, :]
                if use_data_in:
                    ps.append(x)
                else:
                    ps.append(p)
            else:
                ps.append(p)

            s += 1
            if n_samples != -1 and s > n_samples:
                raise KeyboardInterrupt()
    except KeyboardInterrupt:
        print 'Finishing'

    if out_path is not None:
        train.save_images(np.array(ps), path.join(out_path, 'vis_samples.png'), x_limit=100)
        if save_hiddens:
            np.save(path.join(out_path, 'hiddens.npy'), np.array(hs))

    fig = plt.figure()
    data = np.zeros(train.dims)
    im = plt.imshow(data, vmin=0, vmax=1, cmap='Greys_r')

    def init():
        im.set_data(np.zeros(train.dims))

    def animate(i):
        data = ps[i].reshape(train.dims)
        im.set_data(data)
        return im

    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=s,
                                   interval=interval)

    if out_path is not None and save_movie:
        print 'Saving movie'
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Devon Hjelm'), bitrate=1800)
        anim.save(path.join(out_path, 'vis_movie.mp4'), writer=writer)
    else:
        print 'Showing movie'
        plt.show()

    if out_path is not None and save_movie:
        train.next()
        fig = plt.figure()
        data = np.zeros(train.dims)
        X_tr = train._load_chains()
        im = plt.imshow(data, vmin=0, vmax=1, cmap='Greys_r')

        def animate_training_examples(i):
            data = X_tr[i, 0].reshape(train.dims)
            im.set_data(data)
            return im

        def init():
            im.set_data(np.zeros(train.dims))

        anim = animation.FuncAnimation(fig, animate_training_examples,
                                       init_func=init, frames=X_tr.shape[0],
                                       interval=interval)

        print 'Saving data movie'
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=15, metadata=dict(artist='Devon Hjelm'), bitrate=1800)
        anim.save(path.join(out_path, 'vis_train_movie.mp4'), writer=writer)

def energy_function(model):
    x = T.matrix('x', dtype=floatX)
    x_p = T.matrix('x_p', dtype=floatX)
    h_p = T.matrix('h_p', dtype=floatX)
    x_e = T.alloc(0., x_p.shape[0], x.shape[0], x.shape[1]).astype(floatX) + x[None, :, :]

    params = model.get_params()
    x_s, h, p, q = model._step(x_p, *params)

    p = T.alloc(0., p.shape[0], x.shape[0], x.shape[1]).astype(floatX) + p[:, None, :]

    energy = -(x_e * T.log(p + 1e-7) + (1 - x_e) * T.log(1 - p + 1e-7)).sum(axis=2)

    return theano.function([x, x_p, h_p], [energy, x_s, h])

def euclidean_distance(model):
    '''
    h_p are dummy variables to keep it working for dataset chain generators.
    '''

    x = T.matrix('x', dtype=floatX)
    x_p = T.matrix('x_p', dtype=floatX)
    h_p = T.matrix('h_p', dtype=floatX)
    x_e = T.alloc(0., x_p.shape[0], x.shape[0], x.shape[1]).astype(floatX) + x[None, :, :]
    x_pe = T.alloc(0., x_p.shape[0], x.shape[0], x_p.shape[1]).astype(floatX) + x_p[:, None, :]

    params = model.get_sample_params()
    distance = (x_e - x_pe) ** 2
    distance = distance.sum(axis=2)
    return theano.function([x, x_p, h_p], [distance, x, h_p])

def train_model(save_graphs=False, out_path='', name='',
                load_last=False, model_to_load=None, save_images=True,
                source=None,
                learning_rate=0.01, optimizer='adam', batch_size=10, steps=1000,
                metric='energy',
                dim_h=500,
                dataset=None, dataset_args=None,
                noise_input=True, sample=True,
                model_save_freq=100, show_freq=10):

    print 'Dataset args: %s' % pprint.pformat(dataset_args)
    window = dataset_args['window']
    stride = min(window, dataset_args['chain_stride'])
    out_path = path.abspath(out_path)

    if dataset == 'mnist':
        train = MNIST_Chains(batch_size=batch_size, out_path=out_path, **dataset_args)
    elif dataset == 'horses':
        train = Horses(batch_size=batch_size, out_path=out_path, crop_image=True, **dataset_args)
    else:
        raise ValueError()

    dim_in = train.dim
    X = T.tensor3('x', dtype=floatX)
    trng = RandomStreams(random.randint(0, 100000))

    print 'Forming model'

    if model_to_load is not None:
        models, _ = load_model(model_to_load, unpack)
    elif load_last:
        model_file = glob(path.join(out_path, '*last.npz'))[0]
        models, _ = load_model(model_file, unpack)
    else:
        mlps = {}

        rbm = RBM(dim_in, dim_h, trng=trng)
        models = OrderedDict()
        models[rbm.name] = rbm

    print 'Getting params...'
    rbm = models['rbm']
    tparams = rbm.set_tparams()

    X = trng.binomial(p=X, size=X.shape, n=1, dtype=X.dtype)
    X_s = X[:-1]
    updates = theano.OrderedUpdates()
    if noise_input:
        X_s = X_s * (1 - trng.binomial(p=0.1, size=X_s.shape, n=1, dtype=X_s.dtype))

    print 'Model params: %s' % tparams.keys()
    if metric == 'energy':
        print 'Energy-based metric'
        train.set_f_energy(energy_function, rnn)
    elif metric in ['euclidean', 'euclidean_then_energy']:
        print 'Euclidean-based metic'
        train.set_f_energy(euclidean_distance, rnn)
    else:
        raise ValueError(metric)

    outs, updates_1 = rbm(X_s)
    h = outs['h']
    p = outs['p']
    x = outs['y']
    updates.update(updates_1)

    energy = -(X[1:] * T.log(p + 1e-7) + (1 - X[1:]) * T.log(1 - p + 1e-7)).sum(axis=(0, 2))
    cost = energy.mean()
    consider_constant = [x]

    extra_outs = [energy.mean(), h, p]

    if sample:
        print 'Setting up sampler'
        out_s, updates_s = rbm.sample(x0=X[:, 0], n_samples=10, n_steps=10)
        f_sample = theano.function([X], out_s['p'], updates=updates_s)

    grad_tparams = OrderedDict((k, v)
        for k, v in tparams.iteritems()
        if (v not in updates.keys()))

    grads = T.grad(cost, wrt=itemlist(grad_tparams),
                   consider_constant=consider_constant)

    print 'Building optimizer'
    lr = T.scalar(name='lr')
    f_grad_shared, f_grad_updates = eval('op.' + optimizer)(
        lr, tparams, grads, [X], cost,
        extra_ups=updates,
        extra_outs=extra_outs)

    print 'Actually running'

    try:
        e = 0
        for s in xrange(steps):
            try:
                x, _ = train.next()
            except StopIteration:
                e += 1
                print 'Epoch {epoch}'.format(epoch=e)
                if metric == 'euclidean_then_energy' and e == 2:
                    print 'Switching to model energy'
                    train.set_f_energy(energy_function, rnn)
                continue
            rval = f_grad_shared(x)

            if check_bad_nums(rval, ['cost', 'energy', 'h', 'x', 'p']):
                return

            if s % show_freq == 0:
                print ('%d: cost: %.5f | energy: %.2f | prob: %.2f'
                       % (e, rval[0], rval[1], np.exp(-rval[1])))
            if s % show_freq == 0:
                idx = np.random.randint(rval[3].shape[1])
                samples = np.concatenate([x[1:, idx, :][None, :, :],
                                        rval[3][:, idx, :][None, :, :]], axis=0)
                train.save_images(
                    samples,
                    path.join(
                        out_path,
                        '{name}_inference_chain.png'.format(name=name)))
                train.save_images(
                    x, path.join(
                        out_path, '{name}_input_samples.png'.format(name=name)))
                if sample:
                    sample_chain = f_sample(x)
                    train.save_images(
                        sample_chain,
                        path.join(
                            out_path, '{name}_samples.png'.format(name=name)))
            if s % model_save_freq == 0:
                temp_file = path.join(
                    out_path, '{name}_temp.npz'.format(name=name))
                d = dict((k, v.get_value()) for k, v in tparams.items())
                d.update(mode=mode,
                         dim_h=dim_h,
                         h_init=h_init,
                         mlp_a=mlp_a, mlp_b=mlp_b, mlp_o=mlp_o, mlp_c=mlp_c,
                         dataset=dataset, dataset_args=dataset_args)
                np.savez(temp_file, **d)

            f_grad_updates(learning_rate)
    except KeyboardInterrupt:
        print 'Training interrupted'

    outfile = os.path.join(
        out_path, '{name}_{t}.npz'.format(name=name, t=int(time.time())))
    last_outfile = path.join(out_path, '{name}_last.npz'.format(name=name))

    print 'Saving the following params: %s' % tparams.keys()
    d = dict((k, v.get_value()) for k, v in tparams.items())
    d.update(mode=mode,
             dim_h=dim_h,
             h_init=h_init,
             mlp_a=mlp_a, mlp_b=mlp_b, mlp_o=mlp_o, mlp_c=mlp_c,
             dataset=dataset, dataset_args=dataset_args)

    np.savez(outfile, **d)
    np.savez(last_outfile,  **d)
    print 'Done saving. Bye bye.'

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('experiment', default=None)
    parser.add_argument('-o', '--out_path', default=None,
                        help='Output path for stuff')
    parser.add_argument('-l', '--load_last', action='store_true')
    parser.add_argument('-r', '--load_model', default=None)
    parser.add_argument('-i', '--save_images', action='store_true')
    return parser

def load_experiment(experiment_yaml):
    print('Loading experiment from %s' % experiment_yaml)
    exp_dict = yaml.load(open(experiment_yaml))
    print('Experiment hyperparams: %s' % pprint.pformat(exp_dict))
    return exp_dict

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()

    exp_dict = load_experiment(path.abspath(args.experiment))
    out_path = path.join(args.out_path, exp_dict['name'])

    if out_path is not None:
        if path.isfile(out_path):
            raise ValueError()
        elif not path.isdir(out_path):
            os.mkdir(path.abspath(out_path))

    shutil.copy(path.abspath(args.experiment), path.abspath(out_path))

    train_model(out_path=out_path, load_last=args.load_last,
                model_to_load=args.load_model, save_images=args.save_images,
                **exp_dict)