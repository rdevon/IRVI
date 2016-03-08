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
import shutil
import sys
import theano
from theano import tensor as T
import time

from datasets import load_data
from inference import resolve as resolve_inference
from models.distributions import (
    Binomial,
    Gaussian
)
from models.darn import AutoRegressor
from models.dsbn import (
    DeepSBN,
    unpack as unpack_deepsbn
)
from models.gbn import (
    GBN,
    unpack as unpack_gbn
)
from models.sbn import (
    SBN,
    unpack as unpack_sbn
)
from utils.monitor import SimpleMonitor
from utils import floatX
from utils import op
from utils.tools import (
    check_bad_nums,
    get_trng,
    itemlist,
    load_experiment,
    load_model,
    print_profile,
    print_section,
    resolve_path,
    update_dict_of_lists
)

def concatenate_inputs(model, y, py):
    '''
    Function to concatenate ground truth to samples and probabilities.
    '''
    y_hat, updates = model.conditional.sample(py[0], 1)

    py = T.concatenate([y[None, :, :], model.get_center(py)], axis=0)
    y = T.concatenate([y[None, :, :], y_hat], axis=0)

    return (py, y), updates

def init_learning_args(
    learning_rate=0.0001,
    l2_decay=0.,
    optimizer='rmsprop',
    optimizer_args=dict(),
    learning_rate_schedule=None,
    batch_size=100,
    valid_batch_size=100,
    epochs=100,
    n_posterior_samples=20,
    n_posterior_samples_test=20,
    valid_key='lower_bound',
    valid_sign='-',
    excludes=['gaussian_log_sigma', 'gaussian_mu']):
    return locals()

def init_inference_args(
    init_inference='recognition_network',
    inference_method=None,
    inference_rate=0.01,
    n_inference_steps=0,
    n_inference_samples=0,
    pass_gradients=True):
    return locals()


def train(
    out_path='', name='', model_to_load=None, save_images=True,
    dim_h=None, dim_hs=None, center_input=True, prior='binomial',
    recognition_net=None, generation_net=None,

    learning_args=dict(),
    inference_args=dict(),
    inference_args_test=dict(),
    dataset_args=None):

    if dim_h is None:
        assert dim_hs is not None
        deep = True
    else:
        assert dim_hs is None
        deep = False

    # ========================================================================
    learning_args = init_learning_args(**learning_args)
    inference_args = init_inference_args(**inference_args)
    inference_args_test = init_inference_args(**inference_args_test)

    print 'Dataset args: %s' % pprint.pformat(dataset_args)
    print 'Learning args: %s' % pprint.pformat(learning_args)
    print 'Inference args: %s' % pprint.pformat(inference_args)
    print 'Inference args (test): %s' % pprint.pformat(inference_args_test)

    # ========================================================================
    print_section('Setting up data')
    train, valid, test = load_data(
        train_batch_size=learning_args['batch_size'],
        valid_batch_size=learning_args['valid_batch_size'],
        **dataset_args)

    # ========================================================================
    print_section('Setting model and variables')
    dim_in = train.dims[train.name]
    batch_size = learning_args['batch_size']

    X = T.matrix('x', dtype=floatX)
    X.tag.test_value = np.zeros((batch_size, dim_in), dtype=X.dtype)
    trng = get_trng()

    if center_input:
        print 'Centering input with train dataset mean image'
        X_mean = theano.shared(train.mean_image.astype(floatX), name='X_mean')
        X_i = X - X_mean
    else:
        X_i = X

    # ========================================================================
    print_section('Loading model and forming graph')

    if prior == 'gaussian':
        if deep:
            raise NotImplementedError()
        C = GBN
        PC = Gaussian
        unpack = unpack_gbn
        model_name = 'gbn'
    elif prior == 'binomial':
        if deep:
            C = DeepSBN
            unpack = unpack_deepsbn
        else:
            C = SBN
            unpack = unpack_sbn
        PC = Binomial
        model_name = 'sbn'
    elif prior == 'darn':
        if deep:
            raise NotImplementedError()
        C = SBN
        PC = AutoRegressor
        unpack = unpack_sbn
        model_name = 'sbn'
    else:
        raise ValueError(prior)

    if model_to_load is not None:
        models, _ = load_model(model_to_load, unpack,
                               distributions=train.distributions, dims=train.dims)
    else:
        if deep:
            model = C(dim_in, dim_hs, trng=trng)
        else:
            prior_model = PC(dim_h)
            mlps = C.mlp_factory(
                dim_h, train.dims, train.distributions,
                recognition_net=recognition_net,
                generation_net=generation_net)

            model = C(dim_in, dim_h, trng=trng, prior=prior_model, **mlps)

        models = OrderedDict()
        models[model.name] = model

    model = models[model_name]
    tparams = model.set_tparams(excludes=[])
    print_profile(tparams)

    # ==========================================================================
    print_section('Getting cost')

    inference_method = inference_args['inference_method']

    if inference_method is not None:
        inference = resolve_inference(model, deep=deep, **inference_args)
    else:
        inference = None

    if inference_method == 'momentum':
        if prior == 'binomial':
            raise NotImplementedError()
        i_results, constants, updates = inference.inference(X_i, X)
        qk = i_results['qk']
        results, samples, constants_m = model(
            X_i, X, qk, pass_gradients=inference_args['pass_gradients'],
            n_posterior_samples=learning_args['n_posterior_samples'])
        constants += constants_m
    elif inference_method == 'rws':
        results, _, constants = inference(
            X_i, X, n_posterior_samples=learning_args['n_posterior_samples'])
        updates = theano.OrderedUpdates()
    elif inference_method == 'air':
        if prior == 'gaussian':
            raise NotImplementedError()
        i_results, constants, updates = inference.inference(X_i, X)
        qk = i_results['qk']
        results, _, _ = model(
            X_i, X, qk, n_posterior_samples=learning_args['n_posterior_samples'])
    elif inference_method is None:
        if prior != 'gaussian':
            raise NotImplementedError()
        qk = None
        constants = []
        updates = theano.OrderedUpdates()
        results, samples, constants_m = model(
            X_i, X, qk, pass_gradients=inference_args['pass_gradients'],
            n_posterior_samples=learning_args['n_posterior_samples'])
        constants += constants_m
    else:
        raise ValueError(inference_method)

    cost = results.pop('cost')
    extra_outs = []
    extra_outs_keys = ['cost']

    l2_decay = learning_args['l2_decay']
    if l2_decay > 0.:
        print 'Adding %.5f L2 weight decay' % l2_decay
        l2_rval = model.l2_decay(l2_decay)
        cost += l2_rval.pop('cost')
        extra_outs += l2_rval.values()
        extra_outs_keys += l2_rval.keys()

    # ==========================================================================
    print_section('Test functions')
    # Test function with sampling
    inference_method_test = inference_args_test['inference_method']
    if inference_method_test is not None:
        inference = resolve_inference(model, deep=deep, **inference_args_test)
    else:
        inference = None

    if inference_method_test == 'momentum':
        if prior == 'binomial':
            raise NotImplementedError()
        results, samples, full_results, updates_s = inference(
            X_i, X,
            n_posterior_samples=learning_args['n_posterior_samples_test'])
        py = samples['py'][-1]
    elif inference_method_test == 'rws':
        results, samples,  = inference(
            X_i, X, n_posterior_samples=learning_args['n_posterior_samples_test'])
        updates_s = theano.OrderedUpdates()
        py = samples['py']
    elif inference_method_test == 'air':
        results, samples, full_results, updates_s = inference(
            X_i, X, n_posterior_samples=learning_args['n_posterior_samples_test'])
        py = samples['py'][-1]
    elif inference_method_test is None:
        updates_s = theano.OrderedUpdates()
        py = samples['py']
    else:
        raise ValueError(inference_method_test)

    f_test_keys = results.keys()
    f_test = theano.function([X], results.values(), updates=updates_s)
    f_icost = theano.function([X], full_results['i_cost'], updates=updates_s)

    # ========================================================================
    print_section('Setting final tparams and save function')

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
            prior=prior,
            center_input=center_input,
            generation_net=generation_net,
            recognition_net=recognition_net,
            dataset_args=dataset_args
        )
        np.savez(outfile, **d)

    # ========================================================================
    print_section('Getting gradients.')
    grads = T.grad(cost, wrt=itemlist(tparams),
                   consider_constant=constants)

    # ========================================================================
    print_section('Building optimizer')
    lr = T.scalar(name='lr')
    optimizer = learning_args['optimizer']
    optimizer_args = learning_args['optimizer_args']
    f_grad_shared, f_grad_updates = eval('op.' + optimizer)(
        lr, tparams, grads, [X], cost, extra_ups=updates,
        extra_outs=extra_outs, **optimizer_args)

    monitor = SimpleMonitor()

    # ========================================================================
    print_section('Actually running (main loop)')

    best_cost = float('inf')
    best_epoch = 0

    if out_path is not None:
        bestfile = path.join(out_path, '{name}_best.npz'.format(name=name))

    epochs = learning_args['epochs']
    learning_rate = learning_args['learning_rate']
    learning_rate_schedule = learning_args['learning_rate_schedule']
    valid_key = learning_args['valid_key']
    valid_sign = learning_args['valid_sign']
    try:
        epoch_t0 = time.time()
        s = 0
        e = 0

        widgets = ['Epoch {epoch} (training {name}, '.format(epoch=e, name=name),
                   Timer(), '): ', Bar()]
        epoch_pbar = ProgressBar(widgets=widgets, maxval=train.n).start()
        training_time = 0
        while True:
            try:
                x = train.next()[train.name]
                if train.pos == -1:
                    epoch_pbar.update(train.n)
                else:
                    epoch_pbar.update(train.pos)

            except StopIteration:
                print
                epoch_t1 = time.time()
                training_time += (epoch_t1 - epoch_t0)
                valid.reset()
                maxvalid = valid.n

                widgets = ['Validating: (%d posterior samples) '
                           % learning_args['n_posterior_samples_test'],
                           Percentage(), ' (', Timer(), ')']
                pbar    = ProgressBar(widgets=widgets, maxval=maxvalid).start()
                results_train = OrderedDict()
                results_valid = OrderedDict()
                while True:
                    try:
                        x_valid = valid.next()[train.name]
                        if valid.pos > maxvalid:
                            raise StopIteration
                        x_train = train.next()[train.name]
                        #print f_icost(x_valid)
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
                valid_value = results_valid[valid_key]
                if valid_sign == '-':
                    valid_value *= -1

                if valid_value < best_cost:
                    print 'Found best %s: %.2f' % (valid_key, valid_value)
                    best_cost = valid_value
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
                monitor.save_stats_valid(path.join(
                    out_path, '{name}_monitor_valid.npz').format(name=name))

                e += 1
                epoch_t0 = time.time()

                valid.reset()
                train.reset()

                if learning_rate_schedule is not None:
                    if 'decay' in learning_rate_schedule.keys():
                        learning_rate /= learning_rate_schedule['decay']
                        print 'Changing learning rate to %.5f' % learning_rate
                    elif e in learning_rate_schedule.keys():
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
            check_bad_nums(rval, extra_outs_keys)
            if check_bad_nums(rval[:1], extra_outs_keys[:1]):
                print zip(extra_outs_keys, rval)
                print 'Dying, found bad cost... Sorry (bleh)'
                exit()
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
    parser.add_argument('-r', '--load_last', action='store_true')
    parser.add_argument('-l', '--load_model', default=None)
    parser.add_argument('-i', '--save_images', action='store_true')
    parser.add_argument('-n', '--name', default=None)
    return parser

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()

    exp_dict = load_experiment(path.abspath(args.experiment))
    if args.name is not None:
        exp_dict['name'] = args.name

    if args.out_path is None:
        out_path = resolve_path('$irvi_outs')
    else:
        out_path = args.out_path
    out_path = path.join(out_path, exp_dict['name'])

    print 'Saving to %s' % out_path
    if path.isfile(out_path):
        raise ValueError()
    elif not path.isdir(out_path):
        os.mkdir(path.abspath(out_path))

    shutil.copy(path.abspath(args.experiment), path.abspath(out_path))

    if args.load_model is not None:
        model_to_load = args.load_model
    elif args.load_last:
        model_to_load = glob(path.join(args.out_path, '*last.npz'))
    else:
        model_to_load = None

    train(out_path=out_path,
          model_to_load=model_to_load,
          save_images=args.save_images,
          **exp_dict)
