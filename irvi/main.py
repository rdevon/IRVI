'''
SFFN experiment
'''

import argparse
from collections import OrderedDict
from glob import glob
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
import shutil
import sys
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time

from datasets.cifar import CIFAR
from datasets.mnist import MNIST
from models.darn import (
    AutoRegressor
)
from models.distributions import (
    Bernoulli,
    Gaussian
)
from models.gbn import GaussianBeliefNet as GBN
from models.mlp import MLP
from models.sbn import SigmoidBeliefNetwork as SBN
from models.sbn import unpack
from utils.monitor import SimpleMonitor
from utils import op
from utils.tools import (
    check_bad_nums,
    debug_shape,
    itemlist,
    load_model,
    load_experiment,
    print_profile,
    update_dict_of_lists,
    _slice
)


floatX = theano.config.floatX

def concatenate_inputs(model, y, py):
    '''
    Function to concatenate ground truth to samples and probabilities.
    '''
    y_hat, updates = model.conditional.sample(py[0], 1)

    py = T.concatenate([y[None, :, :], model.get_center(py)], axis=0)
    y = T.concatenate([y[None, :, :], y_hat], axis=0)

    return (py, y), updates

def load_data(dataset,
              train_batch_size,
              valid_batch_size,
              test_batch_size,
              **dataset_args):
    if dataset == 'mnist':
        C = MNIST
    elif dataset == 'cifar':
        C = CIFAR

    if train_batch_size is not None:
        train = C(batch_size=train_batch_size,
                  mode='train',
                  inf=False,
                  **dataset_args)
    else:
        train = None
    if valid_batch_size is not None:
        valid = C(batch_size=valid_batch_size,
                  mode='valid',
                  inf=False,
                  **dataset_args)
    else:
        valid = None
    if test_batch_size is not None:
        test = C(batch_size=test_batch_size,
                 mode='test',
                 inf=False,
                 **dataset_args)
    else:
        test = None


    return train, valid, test

def train_model(
    out_path='', name='', load_last=False, model_to_load=None, save_images=True,

    learning_rate=0.0001, optimizer='rmsprop', optimizer_args=dict(),
    learning_rate_schedule=None,
    batch_size=100, valid_batch_size=100, test_batch_size=1000,
    max_valid=10000,
    epochs=100,

    dim_h=300, prior='logistic', pass_gradients=False,
    l2_decay=0.,

    input_mode=None,
    generation_net=None, recognition_net=None,
    excludes=['gaussian.log_sigma'],
    center_input=True,

    z_init=None,
    inference_method='momentum',
    inference_rate=.01,
    n_inference_steps=20,
    n_inference_steps_test=20,
    n_inference_samples=20,
    extra_inference_args=dict(),

    n_mcmc_samples=20,
    n_mcmc_samples_test=20,

    dataset=None, dataset_args=None,
    model_save_freq=1000, show_freq=100
    ):

    kwargs = dict(
        z_init=z_init,
        inference_method=inference_method,
        inference_rate=inference_rate,
        n_inference_samples=n_inference_samples,
        extra_inference_args=extra_inference_args
    )

    # ========================================================================
    print 'Dataset args: %s' % pprint.pformat(dataset_args)
    print 'Model args: %s' % pprint.pformat(kwargs)

    # ========================================================================
    print 'Setting up data'
    train, valid, test = load_data(dataset,
                                   batch_size,
                                   valid_batch_size,
                                   test_batch_size,
                                   **dataset_args)

    # ========================================================================
    print 'Setting model and variables'
    dim_in = train.dims[dataset]
    X = T.matrix('x', dtype=floatX)
    X.tag.test_value = np.zeros((batch_size, dim_in), dtype=X.dtype)
    trng = RandomStreams(random.randint(0, 1000000))

    if input_mode == 'sample':
        print 'Sampling datapoints'
        X = trng.binomial(p=X, size=X.shape, n=1, dtype=X.dtype)
    elif input_mode == 'noise':
        print 'Adding noise to data points'
        X = X * trng.binomial(p=0.1, size=X.shape, n=1, dtype=X.dtype)

    if center_input:
        print 'Centering input with train dataset mean image'
        X_mean = theano.shared(train.mean_image.astype(floatX), name='X_mean')
        X_i = X - X_mean
    else:
        X_i = X

    # ========================================================================
    print 'Loading model and forming graph'

    if model_to_load is not None:
        models, _ = load_model(model_to_load, unpack, **kwargs)
    elif load_last:
        model_file = glob(path.join(out_path, '*last.npz'))[0]
        models, _ = load_model(model_file, unpack, **kwargs)
    else:
        if prior == 'logistic':
            out_act = 'T.nnet.sigmoid'
            prior_model = Bernoulli(dim_h)
        elif prior == 'darn':
            out_act = 'T.nnet.sigmoid'
            prior_model = AutoRegressor(dim_h)
        elif prior == 'gaussian':
            out_act = 'lambda x: x'
            prior_model = Gaussian(dim_h)
        else:
            raise ValueError('%s prior not known' % prior)
        print prior, prior_model

        if recognition_net is not None:
            input_layer = recognition_net.pop('input_layer')
            recognition_net['dim_in'] = train.dims[input_layer]
            recognition_net['dim_out'] = dim_h
            recognition_net['out_act'] = out_act
        if generation_net is not None:
            generation_net['dim_in'] = dim_h
            t = generation_net.get('type', None)
            if t is None or t == 'darn':
                generation_net['dim_out'] = train.dims[generation_net['output']]
                generation_net['out_act'] = train.acts[generation_net['output']]
            elif t == 'MMMLP':
                generation_net['graph']['outs'] = dict()
                for out in generation_net['graph']['outputs']:
                    generation_net['graph']['outs'][out] = dict(
                        dim=train.dims[out],
                        act=train.acts[out]
                    )
            else:
                raise ValueError()

        mlps = SBN.mlp_factory(recognition_net=recognition_net,
                               generation_net=generation_net)

        if prior == 'logistic' or prior == 'darn':
            C = SBN
        elif prior == 'gaussian':
            C = GBN
        else:
            raise ValueError()

        kwargs.update(**mlps)
        model = C(recognition_net['dim_in'], dim_h, trng=trng, prior=prior_model, **kwargs)

        models = OrderedDict()
        models[model.name] = model

    if prior == 'logistic' or prior == 'darn':
        model = models['sbn']
    elif prior == 'gaussian':
        model = models['gbn']

    tparams = model.set_tparams(excludes=[])
    print_profile(tparams)

    # ========================================================================
    print 'Getting cost'
    results, updates, constants = model.inference(
        X_i, X, n_inference_steps=n_inference_steps, n_samples=n_mcmc_samples,
        pass_gradients=pass_gradients)

    cost = results.pop('cost')
    extra_outs = []
    extra_outs_names = ['cost']

    if l2_decay > 0.:
        print 'Adding %.5f L2 weight decay' % l2_decay
        rec_l2_cost = model.posterior.get_L2_weight_cost(l2_decay)
        gen_l2_cost = model.conditional.get_L2_weight_cost(l2_decay)
        cost += rec_l2_cost + gen_l2_cost
        extra_outs += [rec_l2_cost, gen_l2_cost]
        extra_outs_names += ['Rec net L2 cost', 'Gen net L2 cost']
        if prior == 'darn':
            print 'Adding autoregressor weight decay'
            ar_l2_cost = model.prior.get_L2_weight_cost(l2_decay)
            cost += ar_l2_cost
            extra_outs += [ar_l2_cost]
            extra_outs_names += ['AR L2 cost']

    # ========================================================================
    print 'Extra functions'
    # Test function with sampling
    results_s, samples, full_results, updates_s = model(
        X_i, X, n_samples=n_mcmc_samples_test,
        n_inference_steps=n_inference_steps_test)

    f_test_keys = results_s.keys()
    f_test = theano.function([X], results_s.values(), updates=updates_s)

    py_s = samples['py'][-1]
    (pd_s, d_hat_s), updates_c = concatenate_inputs(model, X, py_s)
    updates_s.update(updates_c)

    f_sample = theano.function([X], [pd_s, d_hat_s], updates=updates_s)

    # Sample from prior
    py_p, updates_p = model.sample_from_prior()
    f_prior = theano.function([], py_p, updates=updates_p)

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
            dim_h=dim_h,
            input_mode=input_mode,
            prior=prior,
            generation_net=generation_net, recognition_net=recognition_net,
            dataset=dataset, dataset_args=dataset_args
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
        lr, tparams, grads, [X], cost, extra_ups=updates,
        extra_outs=extra_outs, **optimizer_args)

    monitor = SimpleMonitor()

    # ========================================================================
    print 'Actually running (main loop)'

    best_cost = float('inf')
    best_epoch = 0

    if out_path is not None:
        bestfile = path.join(out_path, '{name}_best.npz'.format(name=name))

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
                x, _ = train.next()
                if train.pos == -1:
                    epoch_pbar.update(train.n)
                else:
                    epoch_pbar.update(train.pos)

            except StopIteration:
                print
                epoch_t1 = time.time()
                training_time += epoch_t1
                valid.reset()
                maxvalid = min(max_valid, valid.n)

                widgets =['Validating: (%d posterior samples)' % n_mcmc_samples_test,
                          Percentage(), ' (', Timer(), ')']
                pbar    = ProgressBar(widgets=widgets, maxval=maxvalid).start()
                results_train = OrderedDict()
                results_valid = OrderedDict()
                while True:
                    try:
                        x_valid, _ = valid.next()
                        if valid.pos > max_valid:
                            raise StopIteration
                        x_train, _ = train.next()

                        r_train = f_test(x_train)
                        r_valid = f_test(x_valid)
                        results_i_train = dict((k, v) for k, v in zip(f_test_keys, r_train))
                        results_i_valid = dict((k, v) for k, v in zip(f_test_keys, r_valid))
                        update_dict_of_lists(results_train, **results_i_train)
                        update_dict_of_lists(results_valid, **results_i_valid)

                        if valid.pos == -1:
                            pbar.update(maxvalid)
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
                lower_bound = results_valid['lower_bound']

                if lower_bound < best_cost:
                    print 'Found best: %.2f' % lower_bound
                    best_cost = lower_bound
                    best_epoch = e
                    if out_path is not None:
                        print 'Saving best to %s' % bestfile
                        save(tparams, bestfile)
                else:
                    print 'Best (%.2f) at epoch %d' % (best_cost, best_epoch)

                monitor.update(**results_train)
                monitor.update(dt_epoch=(epoch_t1-epoch_t0),
                               training_time=training_time)
                monitor.update_valid(**results_valid)
                monitor.display()

                monitor.save(path.join(
                    out_path, '{name}_monitor.png').format(name=name))
                monitor.save_stats(path.join(
                    out_path, '{name}_monitor.npz').format(name=name))

                prior_file = path.join(out_path, 'samples_from_prior.png')
                print 'Saving posterior samples'
                samples = f_prior()
                train.save_images(samples[:, None], prior_file, x_limit=10)

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

            rval = f_grad_shared(x)

            if check_bad_nums(rval, extra_outs_names):
                raise ValueError('Bad number!')

            if save_images and s % model_save_freq == 0:
                try:
                    x_v, _ = valid.next()
                except StopIteration:
                    x_v, _ = valid.next()

                pd_v, d_hat_v = f_sample(x_v)
                d_hat_s = np.concatenate([pd_v[:10],
                                          d_hat_v[1][None, :, :]], axis=0)
                d_hat_s = d_hat_s[:, :min(10, d_hat_s.shape[1] - 1)]
                train.save_images(d_hat_s, path.join(
                    out_path, '{name}_samples.png'.format(name=name)))

                pd_p = f_prior()
                train.save_images(
                    pd_p[:, None], path.join(
                        out_path,
                        '{name}_samples_from_prior.png'.format(name=name)),
                    x_limit=10
                )

            f_grad_updates(learning_rate)
            s += 1

    except KeyboardInterrupt:
        print 'Training interrupted'

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
    parser.add_argument('experiment', default=None)
    parser.add_argument('-o', '--out_path', default=None,
                        help='Output path for stuff')
    parser.add_argument('-l', '--load_last', action='store_true')
    parser.add_argument('-r', '--load_model', default=None)
    parser.add_argument('-i', '--save_images', action='store_true')
    parser.add_argument('-n', '--name', default=None)
    return parser

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()

    exp_dict = load_experiment(path.abspath(args.experiment))
    if args.name is not None:
        exp_dict['name'] = args.name
    out_path = path.join(args.out_path, exp_dict['name'])

    if out_path is not None:
        print 'Saving to %s' % out_path
        if path.isfile(out_path):
            raise ValueError()
        elif not path.isdir(out_path):
            os.mkdir(path.abspath(out_path))

    shutil.copy(path.abspath(args.experiment), path.abspath(out_path))

    train_model(out_path=out_path, load_last=args.load_last,
                model_to_load=args.load_model, save_images=args.save_images,
                **exp_dict)
