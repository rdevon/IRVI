'''
SFFN experiment
'''

import argparse
from collections import OrderedDict
from glob import glob
from monitor import SimpleMonitor
import numpy as np
import os
from os import path
import pprint
import random
import shutil
import sys
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time

from layers import MLP
from mnist import MNIST
import op
from sffn_ica import GaussianBeliefNet as GBN
from sffn_ica import SigmoidBeliefNetwork as SBN
from tools import check_bad_nums
from tools import itemlist
from tools import load_model
from tools import load_experiment
from tools import _slice

floatX = theano.config.floatX

def lower_bound_curve(
    model_file, rs=None, n_samples=10000,
    inference_method='momentum',
    inference_rate=.01, n_inference_steps=100,
    inference_decay=1.0, inference_samples=20,
    ):

    models, kwargs = load_model(model_file, unpack, inference_rate=inference_rate,
                               n_inference_steps=n_inference_steps,
                               inference_decay=inference_decay,
                               inference_method=inference_method)
    dataset_args = kwargs['dataset_args']
    dataset = kwargs['dataset']

    if dataset == 'mnist':
        test = MNIST(batch_size=n_samples, mode='test', **dataset_args)
    else:
        raise ValueError()

    model = models['sbn']
    model.set_tparams()

    if rs is None:
        rs = range(5, 100, 5)

    x_t, _ = test.next()

    X = T.matrix('x', dtype=floatX)
    Y = T.matrix('y', dtype=floatX)

    (py_s, y_energy_s), updates_s = model(X, Y, end_with_inference=False)

    f_ll = theano.function([X, Y], y_energy_s)

    lls = [f_ll(x_t, x_t)]

    R = T.scalar('r', dtype='int64')

    (py_s, y_energy_s), updates_s = model(X, Y, n_inference_steps=R)

    f_ll = theano.function([X, Y, R], y_energy_s)

    for r in rs:
        print 'number of inference steps: %d' % r
        ll = f_ll(x_t, x_t, r)
        lls.append(ll)
        print 'lower bound %.2f' % ll

    return lls

def concatenate_inputs(model, y, py):
    '''
    Function to concatenate ground truth to samples and probabilities.
    '''
    y_hat = model.conditional.sample(py)

    py = T.concatenate([y[None, :, :], py], axis=0)
    y = T.concatenate([y[None, :, :], y_hat], axis=0)

    return py, y

def load_mlp(name, dim_in, dim_out, dim_h=None, n_layers=None, **kwargs):
    mlp = MLP(dim_in, dim_h, dim_out, n_layers, name=name, **kwargs)
    return mlp

def unpack(dim_h=None,
           z_init=None,
           recognition_net=None,
           generation_net=None,
           prior=None,
           dataset=None,
           dataset_args=None,
           noise_amount=None,
           n_inference_steps=None,
           inference_decay=None,
           inference_method=None,
           inference_rate=None,
           inference_scaling=None,
           importance_sampling=None,
           entropy_scale=None,
           x_noise_mode=None,
           y_noise_mode=None,
           **model_args):
    '''
    Function to unpack pretrained model into fresh SFFN class.
    '''

    kwargs = dict(
        prior=prior,
        inference_method=inference_method,
        inference_rate=inference_rate,
        n_inference_steps=n_inference_steps,
        inference_decay=inference_decay,
        z_init=z_init,
        entropy_scale=entropy_scale,
        inference_scaling=inference_scaling,
        importance_sampling=importance_sampling
    )

    dim_h = int(dim_h)
    dataset_args = dataset_args[()]

    if dataset == 'mnist':
        dim_in = 28 * 28
        dim_out = dim_in
    else:
        raise ValueError()

    if prior == 'logistic':
        out_act = 'T.nnet.sigmoid'
    elif prior == 'gaussian':
        out_act = 'lambda x: x'
    else:
        raise ValueError('%s prior not known' % prior)

    models = []
    if recognition_net is not None:
        recognition_net = recognition_net[()]
        posterior = load_mlp('posterior', dim_in, dim_h,
                             out_act=out_act,
                             **recognition_net)
        models.append(posterior)
    else:
        posterior = None

    if generation_net is not None:
        generation_net = generation_net[()]
        conditional = load_mlp('conditional', dim_h, dim_in,
                               out_act='T.nnet.sigmoid',
                               **generation_net)
        models.append(conditional)

    if prior == 'logistic':
        C = SBN
    elif prior == 'gaussian':
        C = GBN
    else:
        raise ValueError('%s prior not known' % prior)

    model = C(dim_in, dim_h, dim_out,
                conditional=conditional,
                posterior=posterior,
                noise_amount=noise_amount,
                x_noise_mode=x_noise_mode,
                y_noise_mode=y_noise_mode,
                **kwargs)
    models.append(model)

    return models, model_args, dict(
        z_init=z_init,
        dataset=dataset,
        dataset_args=dataset_args
    )

def train_model(
    out_path='', name='', load_last=False, model_to_load=None, save_images=True,

    learning_rate=0.1, optimizer='adam', batch_size=100, epochs=100,

    dim_h=300, prior='logistic', learn_prior=True,

    x_noise_mode=None, y_noise_mode=None, noise_amout=0.1,

    generation_net=None, recognition_net=None,

    z_init=None,
    inference_method='momentum',
    inference_rate=.01,
    n_inference_steps=100,
    n_inference_steps_test=0,
    inference_decay=1.0,
    n_inference_samples=20,
    inference_scaling=None,
    entropy_scale=1.0,

    n_mcmc_samples=20,
    n_mcmc_samples_test=20,
    importance_sampling=False,

    dataset=None, dataset_args=None,
    model_save_freq=10, show_freq=10, archive_every=0
    ):

    kwargs = dict(
        z_init=z_init,
        inference_method=inference_method,
        inference_rate=inference_rate,
        n_inference_samples=n_inference_samples,
        inference_decay=inference_decay,
        entropy_scale=entropy_scale,
        inference_scaling=inference_scaling,
        importance_sampling=importance_sampling
    )

    # ========================================================================
    print 'Dataset args: %s' % pprint.pformat(dataset_args)
    print 'Model args: %s' % pprint.pformat(kwargs)

    # ========================================================================
    print 'Setting up data'
    if dataset == 'mnist':
        train = MNIST(batch_size=batch_size, mode='train', inf=False,
                               **dataset_args)
        valid = MNIST(batch_size=batch_size, mode='valid', inf=True,
                               **dataset_args)
        test = MNIST(batch_size=2000, mode='test', inf=True,
                              **dataset_args)
    else:
        raise ValueError()

    # ========================================================================
    print 'Setting model and variables'
    dim_in = train.dim
    dim_out = train.dim
    D = T.matrix('x', dtype=floatX)
    X = D.copy()
    Y = X.copy()

    trng = RandomStreams(random.randint(0, 1000000))

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
        elif prior == 'gaussian':
            out_act = 'lambda x: x'
        else:
            raise ValueError('%s prior not known' % prior)

        if recognition_net is not None:
            posterior = load_mlp('posterior', dim_in, dim_h,
                                 out_act=out_act,
                                 **recognition_net)
        else:
            posterior = None

        if generation_net is not None:
            conditional = load_mlp('conditional', dim_h, dim_in,
                                   out_act='T.nnet.sigmoid',
                                   **generation_net)
        else:
            conditional = None

        if prior == 'logistic':
            C = SBN
        elif prior == 'gaussian':
            C = GBN
        else:
            raise ValueError()
        model = C(dim_in, dim_h, dim_out, trng=trng,
                conditional=conditional,
                posterior=posterior,
                noise_amount=0.,
                x_noise_mode=x_noise_mode,
                y_noise_mode=y_noise_mode,
                **kwargs)

        models = OrderedDict()
        models[model.name] = model

    if prior == 'logistic':
        model = models['sbn']
    elif prior == 'gaussian':
        model = models['gbn']

    if not learn_prior:
        excludes = ['z']
    else:
        excludes = []
    tparams = model.set_tparams(excludes=excludes)

    # ========================================================================
    print 'Getting cost'
    (xs, ys, zs,
     prior_energy, h_energy, y_energy,
     y_energy_approx, entropy), updates, constants = model.inference(
        X, Y, n_inference_steps=n_inference_steps, n_samples=n_mcmc_samples)

    if learn_prior:
        cost = prior_energy + h_energy + y_energy
    else:
        cost = h_energy + y_energy

    # ========================================================================
    print 'Extra functions'

    if prior == 'logistic':
        mu = T.nnet.sigmoid(zs)
        ph = mu
    elif prior == 'gaussian':
        ph = zs
        mu = _slice(zs, 0, model.dim_h)
        print 'Not learning sigma for gaussian prior'
        constants.append(model.log_sigma)
    else:
        raise ValueError()

    py = model.conditional(mu)
    pd_i, d_hat_i = concatenate_inputs(model, ys[0], py)

    # Test function with sampling
    rval, updates_s = model(
        X, Y, n_samples=n_mcmc_samples_test, n_inference_steps=n_inference_steps_test)
    py_s = rval['py']
    lower_bound = rval['lower_bound']
    pd_s, d_hat_s = concatenate_inputs(model, Y, py_s)

    outs_s = [lower_bound, pd_s, d_hat_s]
    if prior == 'logistic':
        conditionals_approx = rval['c_a']
        conditionals_mc = rval['c_mc']
        kl_terms = rval['kl']
        outs_s += [conditionals_approx, conditionals_mc, kl_terms]

    f_test = theano.function([X, Y], outs_s, updates=updates_s)

    # Sample from prior
    py_p = model.sample_from_prior()
    f_prior = theano.function([], py_p)

    # ========================================================================

    extra_outs = [prior_energy, h_energy, y_energy, y_energy_approx,
                  y_energy / y_energy_approx, entropy]
    vis_outs = [pd_i, d_hat_i]

    extra_outs_names = ['cost', 'prior_energy', 'h energy',
                        'train y energy', 'approx train y energy',
                        'y to y approx ratio', 'entropy']
    vis_outs_names = ['pds', 'd_hats']

    # ========================================================================
    print 'Setting final tparams and save function'

    all_params = OrderedDict((k, v) for k, v in tparams.iteritems())

    tparams = OrderedDict((k, v)
        for k, v in tparams.iteritems()
        if (v not in updates.keys()))

    print 'Learned model params: %s' % tparams.keys()
    print 'Saved params: %s' % all_params.keys()

    def save(tparams, outfile):
        d = dict((k, v.get_value()) for k, v in all_params.items())

        d.update(
            dim_h=dim_h,
            x_noise_mode=x_noise_mode, y_noise_mode=y_noise_mode,
            noise_amout=noise_amout,
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
        lr, tparams, grads, [D], cost,
        extra_ups=updates,
        extra_outs=extra_outs+vis_outs)

    monitor = SimpleMonitor()

    # ========================================================================
    print 'Actually running'

    best_cost = float('inf')
    if out_path is not None:
        bestfile = path.join(out_path, '{name}_best.npz'.format(name=name))

    try:
        t0 = time.time()
        s = 0
        e = 0
        while True:
            try:
                x, _ = train.next()
            except StopIteration:
                e += 1
                print 'Epoch {epoch} ({name})'.format(epoch=e, name=name)
                continue

            if e > epochs:
                break

            rval = f_grad_shared(x)

            if check_bad_nums(rval, extra_outs_names+vis_outs_names):
                return

            if s % show_freq == 0:
                try:
                    d_v, _ = valid.next()
                except StopIteration:
                    d_v, _ = valid.next()
                x_v, y_v = d_v, d_v

                outs_v = f_test(x_v, y_v)
                outs_t = f_test(x, x)

                lb_v, pd_v, d_hat_v = outs_v[:3]
                lb_t = outs_t[0]

                outs = OrderedDict((k, v)
                    for k, v in zip(extra_outs_names,
                                    rval[:len(extra_outs_names)]))

                t1 = time.time()
                outs.update(**{
                    'train lower bound': lb_t,
                    'valid lower bound': lb_v,
                    'elapsed_time': t1-t0}
                )
                monitor.update(**outs)
                t0 = time.time()

                if lb_v < best_cost:
                    best_cost = lb_v
                    if out_path is not None:
                        save(tparams, bestfile)

                if prior == 'logistic':
                    ca_v, cm_v, kl_v = outs_v[3:]
                    outs_adds = OrderedDict({
                        'conditionals approx': ca_v,
                        'conditionals mc': cm_v,
                        'kls': kl_v
                    })
                    monitor.add(**outs_adds)

                monitor.display(e, s)

                if save_images and s % model_save_freq == 0:
                    monitor.save(path.join(
                        out_path, '{name}_monitor.png').format(name=name))
                    if archive_every and s % archive_every == 0:
                        monitor.save(path.join(
                            out_path, '{name}_monitor({s})'.format(name=name, s=s))
                        )

                    pd_i, d_hat_i = rval[len(extra_outs_names):]

                    idx = np.random.randint(pd_i.shape[1])
                    pd_i = pd_i[:, idx]
                    d_hat_i = d_hat_i[:, idx]
                    d_hat_i = np.concatenate([pd_i[:, None, :],
                                              d_hat_i[:, None, :]], axis=1)
                    train.save_images(
                        d_hat_i, path.join(
                            out_path, '{name}_inference.png').format(name=name))
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

    try:
        print 'Quick test, please wait...'
        d_t, _ = test.next()
        x_t = d_t.copy()
        y_t = d_t.copy()
        outs_test = f_test(x_t, y_t)
        print 'End test: %.5f' % outs_test[0]
    except KeyboardInterrupt:
        print 'Aborting test'

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
