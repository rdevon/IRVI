"""
Helper module for NMT
"""

from collections import OrderedDict
from ConfigParser import ConfigParser
import numpy as np
import os
import pprint
import random
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import warnings
import yaml


floatX = theano.config.floatX
pi = theano.shared(np.pi).astype(floatX)
e = theano.shared(np.e).astype(floatX)

random_seed = random.randint(0, 10000)
rng_ = np.random.RandomState(random_seed)

profile = False

f_clip = lambda x, y, z: T.clip(x, y, 1.)

_, _columns = os.popen('stty size', 'r').read().split()
_columns = int(_columns)

def print_section(s):
    print ('-' * 3) + s + ('-' * (_columns - 3 - len(s)))

def get_paths():
    d = os.path.abspath(os.path.dirname(os.path.realpath(__file__)) + '/..')
    config_file = os.path.join(d, 'irvi.conf')
    config = ConfigParser()
    config.read(config_file)
    path_dict = config._sections['PATHS']
    path_dict.pop('__name__')
    return path_dict

def resolve_path(p):
    path_dict = get_paths()
    for k, v in path_dict.iteritems():
        p = p.replace(k, v)
    return p

def get_trng():
    trng = RandomStreams(random.randint(0, 1000000))
    return trng

def warn_kwargs(c, **kwargs):
    if len(kwargs) > 0:
        warnings.warn('Class instance %s has leftover kwargs %s'
                       % (type(c), kwargs), RuntimeWarning)

def update_dict_of_lists(d_to_update, **d):
    for k, v in d.iteritems():
        if k in d_to_update.keys():
            d_to_update[k].append(v)
        else:
            d_to_update[k] = [v]

def debug_shape(X, x, t_out, updates=None):
    f = theano.function([X], t_out, updates=updates)
    out = f(x)
    print out.shape
    assert False

def print_profile(tparams):
    print 'Print profile for tparams (name, shape)'
    for (k, v) in tparams.iteritems():
        print k, v.get_value().shape

def shuffle_columns(x, srng):
    def step_shuffle(m, perm):
        return m[perm]

    perm_mat = srng.permutation(n=x.shape[0], size=(x.shape[1],))
    y, _ = scan(
        step_shuffle, [x.transpose(1, 0, 2), perm_mat], [None], [], x.shape[1],
        name='shuffle', strict=False)
    return y.transpose(1, 0, 2)

def scan(f_scan, seqs, outputs_info, non_seqs, n_steps, name='scan', strict=False):
    return theano.scan(
        f_scan,
        sequences=seqs,
        outputs_info=outputs_info,
        non_sequences=non_seqs,
        name=name,
        n_steps=n_steps,
        profile=profile,
        strict=strict
    )

def init_weights(model, weight_noise=False, weight_scale=0.001, dropout=False, **kwargs):
    model.weight_noise = weight_noise
    model.weight_scale = weight_scale
    model.dropout = dropout
    return kwargs

def init_rngs(model, rng=None, trng=None, **kwargs):
    if rng is None:
        rng = rng_
    model.rng = rng
    if trng is None:
        model.trng = RandomStreams(random.randint(0, 10000))
    else:
        model.trng = trng
    return kwargs

def gaussian(x, mu, s):
    return T.exp(-(x - mu)**2 / (2 * s)**2) / (s * T.sqrt(2 * pi)).astype('float32')

def log_gaussian(x, mu, s):
    return -(x - mu)**2 / (2 * s**2) - T.log(s + 1e-7) - T.sqrt(2 * pi).astype('float32')

def logit(z):
    z = T.clip(z, 1e-7, 1.0 - 1e-7)
    return T.log(z) - T.log(1 - z)

def _slice(_x, n, dim):
    if _x.ndim == 1:
        return _x[n*dim:(n+1)*dim]
    elif _x.ndim == 2:
        return _x[:, n*dim:(n+1)*dim]
    elif _x.ndim == 3:
        return _x[:, :, n*dim:(n+1)*dim]
    elif _x.ndim == 4:
        return _x[:, :, :, n*dim:(n+1)*dim]
    else:
        raise ValueError('Number of dims (%d) not supported'
                         ' (but can add easily here)' % _x.ndim)

def _slice2(_x, start, end):
    if _x.ndim == 1:
        return _x[start:end]
    elif _x.ndim == 2:
        return _x[:, start:end]
    elif _x.ndim == 3:
        return _x[:, :, start:end]
    elif _x.ndim == 4:
        return _x[:, :, :, start:end]
    else:
        raise ValueError('Number of dims (%d) not supported'
                         ' (but can add easily here)' % _x.ndim)

def load_experiment(experiment_yaml):
    print('Loading experiment from %s' % experiment_yaml)
    exp_dict = yaml.load(open(experiment_yaml))
    print('Experiment hyperparams: %s' % pprint.pformat(exp_dict))
    return exp_dict

def load_model(model_file, f_unpack=None, **extra_args):
    '''
    Loads pretrained model.
    '''

    print 'Loading model from %s' % model_file
    params = np.load(model_file)
    d = dict()
    for k in params.keys():
        try:
            d[k] = params[k].item()
        except ValueError:
            d[k] = params[k]

    d.update(**extra_args)
    models, pretrained_kwargs, kwargs = f_unpack(**d)

    print('Pretrained model(s) has the following parameters: \n%s'
          % pprint.pformat(pretrained_kwargs.keys()))

    model_dict = OrderedDict()

    for model in models:
        print '--- Loading params for %s' % model.name
        for k, v in model.params.iteritems():
            try:
                param_key = '{name}_{key}'.format(name=model.name, key=k)
                pretrained_v = pretrained_kwargs.pop(param_key)
                print 'Found %s for %s %s' % (k, model.name, pretrained_v.shape)
                assert model.params[k].shape == pretrained_v.shape, (
                    'Sizes do not match: %s vs %s'
                    % (model.params[k].shape, pretrained_v.shape)
                )
                model.params[k] = pretrained_v
            except KeyError:
                try:
                    param_key = '{key}'.format(key=k)
                    pretrained_v = pretrained_kwargs[param_key]
                    print 'Found %s, but name is ambiguous' % (k)
                    assert model.params[k].shape == pretrained_v.shape, (
                        'Sizes do not match: %s vs %s'
                        % (model.params[k].shape, pretrained_v.shape)
                    )
                    model.params[k] = pretrained_v
                except KeyError:
                    print '{} not found'.format(k)
        model_dict[model.name] = model

    if len(pretrained_kwargs) > 0:
        raise ValueError('ERROR: Leftover params: %s' %
                         pprint.pformat(pretrained_kwargs.keys()))

    return model_dict, kwargs

def check_bad_nums(rvals, names):
    found = False
    for k, out in zip(names, rvals):
        if np.any(np.isnan(out)):
            print 'Found nan num ', k, '(nan)'
            found = True
        elif np.any(np.isinf(out)):
            print 'Found inf ', k, '(inf)'
            found = True
    return found

def flatten_dict(d):
    rval = OrderedDict()
    for k, v in d.iteritems():
        if isinstance(v, OrderedDict):
            new_d = flatten_dict(v)
            for k_, v_ in new_d.iteritems():
                rval[k + '_' + k_] = v_
        else:
            rval[k] = v
    return rval

# push parameters to Theano shared variables
def zipp(params, tparams):
    for kk, vv in params.iteritems():
        tparams[kk].set_value(vv)

# pull parameters from Theano shared variables
def unzip(zipped):
    new_params = OrderedDict()
    for kk, vv in zipped.iteritems():
        new_params[kk] = vv.get_value()
    return new_params

# get the list of parameters: Note that tparams must be OrderedDict
def itemlist(tparams):
    return tparams.values()

# make prefix-appended name
def _p(pp, name):
    return '%s_%s'%(pp, name)

def ortho_weight(ndim, rng=None):
    if not rng:
        rng = rng_
    W = rng.randn(ndim, ndim)
    u, s, v = np.linalg.svd(W)
    return u.astype('float32')

def norm_weight(nin, nout=None, scale=0.01, ortho=True, rng=None):
    if not rng:
        rng = rng_
    if nout is None:
        nout = nin
    if nout == nin and ortho:
        W = ortho_weight(nin, rng=rng)
    else:
        W = scale * rng.randn(nin, nout)
    return W.astype('float32')

def parzen_estimation(samples, tests, h=1.0):
    log_p = 0.
    d = samples.shape[-1]
    z = d * np.log(h * np.sqrt(2 * np.pi))
    for test in tests:
        d_s = (samples - test[None, :]) / h
        e = log_mean_exp((-.5 * d_s ** 2).sum(axis=1), as_numpy=True, axis=0)
        log_p += e - z
    return log_p / float(tests.shape[0])

def log_mean_exp(x, axis=None, as_numpy=False):
    if as_numpy:
        Te = np
    else:
        Te = T
    x_max = Te.max(x, axis=axis, keepdims=True)
    return Te.log(Te.mean(Te.exp(x - x_max), axis=axis, keepdims=True)) + x_max

def log_sum_exp(x, axis=None):
    '''
    Numerically stable log( sum( exp(A) ) ).
    '''
    x_max = T.max(x, axis=axis, keepdims=True)
    y = T.log(T.sum(T.exp(x - x_max), axis=axis, keepdims=True)) + x_max
    y = T.sum(y, axis=axis)
    return y

def concatenate(tensor_list, axis=0):
    """
    Alternative implementation of `theano.T.concatenate`.
    This function does exactly the same thing, but contrary to Theano's own
    implementation, the gradient is implemented on the GPU.
    Backpropagating through `theano.tensor.concatenate` yields slowdowns
    because the inverse operation (splitting) needs to be done on the CPU.
    This implementation does not have that problem.
    :usage:
        >>> x, y = theano.tensor.matrices('x', 'y')
        >>> c = concatenate([x, y], axis=1)
    :parameters:
        - tensor_list : list
            list of Theano tensor expressions that should be concatenated.
        - axis : int
            the tensors will be joined along this axis.
    :returns:
        - out : tensor
            the concatenated tensor expression.
    """
    concat_size = sum(tt.shape[axis] for tt in tensor_list)

    output_shape = ()
    for k in range(axis):
        output_shape += (tensor_list[0].shape[k],)
    output_shape += (concat_size,)
    for k in range(axis + 1, tensor_list[0].ndim):
        output_shape += (tensor_list[0].shape[k],)

    out = T.zeros(output_shape)
    offset = 0
    for tt in tensor_list:
        indices = ()
        for k in range(axis):
            indices += (slice(None),)
        indices += (slice(offset, offset + tt.shape[axis]),)
        for k in range(axis + 1, tensor_list[0].ndim):
            indices += (slice(None),)

        out = T.set_subtensor(out[indices], tt)
        offset += tt.shape[axis]

    return out
