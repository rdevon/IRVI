'''
Module for general layers
'''

from collections import OrderedDict
import numpy as np
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import tensor as T

from utils import tools
from utils.tools import log_mean_exp
from utils.tools import init_rngs
from utils.tools import init_weights
from utils.tools import _slice


floatX = theano.config.floatX
pi = theano.shared(np.pi).astype('float32')
e = theano.shared(np.e).astype('float32')


class Layer(object):
    def __init__(self, name='', learn=True):
        self.name = name
        self.params = None
        self.excludes = []
        self.learn = learn
        self.set_params()

    def set_params(self):
        raise NotImplementedError()

    def set_tparams(self):
        if self.params is None:
            raise ValueError('Params not set yet')
        tparams = OrderedDict()
        for kk, pp in self.params.iteritems():
            tp = theano.shared(self.params[kk], name=kk)
            tparams[tools._p(self.name, kk)] = tp
            self.__dict__[kk] = tp
        return tparams

    def get_excludes(self):
        if self.learn:
            return [tools._p(self.name, e) for e in self.excludes]
        else:
            return [tools._p(self.name, k) for k in self.params.keys()]

    def __call__(self, state_below):
        raise NotImplementedError()


class ParzenEstimator(Layer):
    def __init__(self, dim, name='parzen'):
        self.dim = dim
        super(ParzenEstimator, self).__init__(name=name)

    def set_params(self, samples, x):
        sigma = self.get_sigma(samples, x).astype('float32')
        self.params = OrderedDict(sigma=sigma)

    def get_sigma(self, samples, xs):
        sigma = np.zeros((samples.shape[-1],))
        for x in xs:
            dx = (samples - x[None, :]) ** 2
            sigma += T.sqrt(dx / np.log(np.sqrt(2 * pi))).mean(axis=(0))
        return sigma / float(xs.shape[0])

    def __call__(self, samples, x):
        z = T.log(self.sigma * T.sqrt(2 * pi)).sum()
        d_s = (samples[:, None, :] - x[None, :, :]) / self.sigma[None, None, :]
        e = log_mean_exp((-.5 * d_s ** 2).sum(axis=2), axis=0)
        return (e - z).mean()


class Scheduler(Layer):
    '''
    Class for tensor constant scheduling.
    This can be used to schedule updates to tensor parameters, such as learning rates given
    the epoch.
    '''
    def __init__(self, rate=1, method='lambda x: 2 * x', name='scheduler'):
        self.rate = rate
        self.method = method
        super(Scheduler, self).__init__(name=name)

    def set_params(self):
        counter = 0
        self.params = OrderedDict(counter=counter, switch=switch)

    def __call__(self, x):
        counter = T.switch(T.ge(self.counter, self.rate), 0, self.counter + 1)
        switch = T.switch(T.ge(self.counter, 0), 1, 0)
        x = T.switch(switch, eval(method)(x), x)

        return OrderedDict(x=x), theano.OrderedUpdates([(self.counter, counter)])


class FFN(Layer):
    def __init__(self, nin, nout, ortho=True, dropout=False, trng=None,
                 activ='lambda x: x', name='ff', weight_noise=False):
        self.nin = nin
        self.nout = nout
        self.ortho = ortho
        self.activ = activ
        self.dropout = dropout
        self.weight_noise = weight_noise

        if self.dropout and trng is None:
            self.trng = RandomStreams(1234)
        else:
            self.trng = trng

        super(FFN, self).__init__(name)
        self.set_params()

    def set_params(self):
        W = tools.norm_weight(self.nin, self.nout, scale=0.01, ortho=self.ortho)
        b = np.zeros((self.nout,)).astype(floatX)

        self.params = OrderedDict(
            W=W,
            b=b
        )

        if self.weight_noise:
            W_noise = (W * 0).astype(floatX)
            self.params.update(W_noise=W_noise)

    def __call__(self, state_below):
        W = self.W + self.W_noise if self.weight_noise else self.W
        z = eval(self.activ)(T.dot(state_below, W) + self.b)
        if self.dropout:
            if self.activ == 'T.tanh':
                raise NotImplementedError()
            else:
                z_d = self.trng.binomial(z.shape, p=1-self.dropout, n=1,
                                         dtype=z.dtype)
                z = z * z_d / (1 - self.dropout)
        return OrderedDict(z=z), theano.OrderedUpdates()


class MLP(Layer):
    def __init__(self, dim_in, dim_h, dim_out, n_layers,
                 f_sample=None, f_neg_log_prob=None, f_entropy=None,
                 h_act='T.nnet.sigmoid', out_act='T.nnet.sigmoid',
                 name='MLP',
                 **kwargs):

        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out
        self.n_layers = n_layers
        assert n_layers > 0

        self.h_act = h_act
        self.out_act = out_act

        if out_act == 'T.nnet.sigmoid':
            self.sample = self._binomial
            self.neg_log_prob = self._cross_entropy
            self.entropy = self._binary_entropy
            self.prob = lambda x: x
        elif out_act == 'T.tanh':
            self.sample = self._centered_binomial
        elif out_act == 'lambda x: x':
            self.sample = self._normal
            self.neg_log_prob = self._neg_normal_log_prob
            self.entropy = self._normal_entropy
            self.prob = self._normal_prob
        else:
            assert f_sample is not None
            assert f_neg_log_prob is not None

        if f_sample is not None:
            self.sample = f_sample
        if f_neg_log_prob is not None:
            self.net_log_prob = f_neg_log_prob
        if f_entropy is not None:
            self.entropy = f_entropy

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        assert len(kwargs) == 0, kwargs.keys()
        super(MLP, self).__init__(name=name)

    @staticmethod
    def factory(dim_in=None, dim_h=None, dim_out=None, n_layers=None,
                **kwargs):
        return MLP(dim_in, dim_h, dim_out, n_layers, **kwargs)

    def _binomial(self, p, size=None):
        if size is None:
            size = p.shape
        return self.trng.binomial(p=p, size=size, n=1, dtype=p.dtype)

    def _centered_binomial(mlp, p, size=None):
        if size is None:
            size = p.shape
        return 2 * self.trng.binomial(p=0.5*(p+1), size=size, n=1, dtype=p.dtype) - 1.

    def _cross_entropy(self, x, p, axis=None):
        p = T.clip(p, 1e-7, 1.0 - 1e-7)
        energy = T.nnet.binary_crossentropy(p, x)
        if axis is None:
            axis = energy.ndim - 1
        energy = energy.sum(axis=axis)
        return energy

    def _binary_entropy(self, p, axis=None):
        p_c = T.clip(p, 1e-7, 1.0 - 1e-7)
        entropy = T.nnet.binary_crossentropy(p_c, p)
        if axis is None:
            axis = entropy.ndim - 1
        entropy = entropy.sum(axis=axis)
        return entropy

    def _normal(self, p, size=None):
        mu = _slice(p, 0, mlp.dim_out)
        log_sigma = _slice(p, 1, self.dim_out)
        if size is None:
            size = mu.shape
        return self.trng.normal(avg=mu, std=T.exp(log_sigma), size=size)

    def _normal_prob(mlp, p):
        mu = _slice(p, 0, mlp.dim_out)
        return mu

    def _neg_normal_log_prob(self, x, p, axis=None):
        mu = _slice(p, 0, self.dim_out)
        log_sigma = _slice(p, 1, self.dim_out)
        energy = 0.5 * ((x - mu)**2 / (T.exp(2 * log_sigma)) + 2 * log_sigma + T.log(2 * pi))
        if axis is None:
            axis = energy.ndim - 1
        energy = energy.sum(axis=axis)
        return energy

    def _normal_entropy(self, p, axis=None):
        log_sigma = _slice(p, 1, self.dim_out)
        entropy = 0.5 * T.log(2 * pi * e) + log_sigma
        if axis is None:
            axis = entropy.ndim - 1
        entropy = entropy.sum(axis=axis)
        return entropy

    def get_L2_weight_cost(self, gamma, layers=None):

        if layers is None:
            layers = range(self.n_layers)

        cost = T.constant(0.).astype(floatX)
        for l in layers:
            W = mlp.__dict__['W%d' % l]
            cost += gamma * (W ** 2).sum()

        return cost

    def set_params(self):
        self.params = OrderedDict()

        for l in xrange(self.n_layers):
            if l == 0:
                dim_in = self.dim_in
            else:
                dim_in = self.dim_h

            if l == self.n_layers - 1:
                dim_out = self.dim_out
            else:
                dim_out = self.dim_h

            W = tools.norm_weight(dim_in, dim_out,
                                  scale=self.weight_scale, ortho=False)
            b = np.zeros((dim_out,)).astype(floatX)

            self.params['W%d' % l] = W
            self.params['b%d' % l] = b

        if self.out_act == 'lambda x: x':
            if l == 0:
                dim_in = self.dim_in
            else:
                dim_in = self.dim_h
            Ws = tools.norm_weight(dim_in, self.dim_out,
                                    scale=self.weight_scale, ortho=False)
            bs = np.zeros((self.dim_out,)).astype(floatX)
            self.params.update(Ws=Ws, bs=bs)

    def get_params(self):
        params = []
        for l in xrange(self.n_layers):
            W = self.__dict__['W%d' % l]
            b = self.__dict__['b%d' % l]
            params += [W, b]

        if self.out_act == 'lambda x: x':
            Ws = self.Ws
            bs = self.bs
            params += [Ws, bs]

        return params

    def preact(self, x, *params):
        # Used within scan with `get_params`
        params = list(params)

        for l in xrange(self.n_layers):
            W = params.pop(0)
            b = params.pop(0)

            if self.weight_noise:
                print 'Using weight noise in layer %d for MLP %s' % (l, self.name)
                W += self.trng.normal(avg=0., std=self.weight_noise, size=W.shape)

            if l == self.n_layers - 1:
                if self.out_act == 'lambda x: x':
                    Ws = params.pop(0)
                    bs = params.pop(0)
                    if self.weight_noise:
                        print 'Using weight noise in log sigma layer %d for MLP %s' % (l, self.name)
                        Ws += self.trng.normal(avg=0., std=self.weight_noise, size=W.shape)
                    mu = T.dot(x, W) + b
                    log_sigma = T.dot(x, Ws) + bs
                    x = T.concatenate([mu, log_sigma], axis=mu.ndim-1)
                else:
                    x = T.dot(x, W) + b
            else:
                activ = self.h_act
                x = eval(activ)(T.dot(x, W) + b)

            if self.dropout:
                if activ == 'T.tanh':
                    raise NotImplementedError('dropout for tanh units not implemented yet')
                elif activ in ['T.nnet.sigmoid', 'T.nnet.softplus', 'lambda x: x']:
                    x_d = self.trng.binomial(x.shape, p=1-self.dropout, n=1,
                                             dtype=x.dtype)
                    x = x * x_d / (1 - self.dropout)
                else:
                    raise NotImplementedError('No dropout for %s yet' % activ)

        assert len(params) == 0, params
        return x

    def step_call(self, x, *params):
        x = self.preact(x, *params)
        return eval(self.out_act)(x)

    def __call__(self, x, return_preact=False):
        params = self.get_params()
        if return_preact:
            x = self.preact(x, *params)
        else:
            x = self.step_call(x, *params)

        return x


class MultiModalMLP(Layer):
    def __init__(self, dim_in, graph, name='MLP', **kwargs):


        self.dim_in = dim_in

        graph = dict(
            nodes=dict(
                a=dict(dim=200, act='T.nnet.sigmoid'),
                b=dict(dim=200, act='T.nnet.sigmoid')
            ),
            outs=dict(
                o=dict(
                    dim=500, act='lambda x: x'
                )
            ),
            edges=[
                ('i', 'a'), ('i', 'b'), ('a', 'o'), ('b', 'o')
            ]
        )

        self.nodes = graph['nodes']
        self.edges = graph['edges']
        self.outs = graph['outs']

        if out_act == 'T.nnet.sigmoid':
            self.sample = self._binomial
            self.neg_log_prob = self._cross_entropy
            self.entropy = self._binary_entropy
            self.prob = lambda x: x
        elif out_act == 'T.tanh':
            self.sample = self._centered_binomial
        elif out_act == 'lambda x: x':
            self.sample = self._normal
            self.neg_log_prob = self._neg_normal_log_prob
            self.entropy = self._normal_entropy
            self.prob = self._normal_prob
        else:
            assert f_sample is not None
            assert f_neg_log_prob is not None

        if f_sample is not None:
            self.sample = f_sample
        if f_neg_log_prob is not None:
            self.net_log_prob = f_neg_log_prob
        if f_entropy is not None:
            self.entropy = f_entropy

        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        assert len(kwargs) == 0, kwargs.keys()
        super(MLP, self).__init__(name=name)


class Softmax(Layer):
    def __init__(self, name='softmax'):
        super(Softmax, self).__init__(name)

    def __call__(self, input_):
        if input_.ndim == 3:
            reshape = input_.shape
            input_ = input_.reshape((input_.shape[0] * input_.shape[1], input_.shape[2]))
        else:
            reshape = False
        y_hat = T.nnet.softmax(input_)
        if reshape:
            y_hat = y_hat.reshape(reshape)
        return OrderedDict(y_hat=y_hat), theano.OrderedUpdates()

    def err(self, y_hat, Y):
        err = (Y * (1 - y_hat) + (1 - Y) * y_hat).mean()
        return err

    def zero_one_loss(self, y_hat, Y):
        y_hat = self.take_argmax(y_hat)
        y = self.take_argmax(Y)
        return T.mean(T.neq(y_hat, y))

    def take_argmax(self, var):
        """Takes argmax along 1st axis if tensor variable is matrix."""
        if var.ndim == 2:
            return T.argmax(var, axis=1)
        elif var.ndim == 1:
            return var
        else:
            raise ValueError("Un-recognized shape for Softmax::"
                             "zero-one-loss, taking argmax!")


class Logistic(Layer):
    def __init__(self, name='logistic'):
        super(Logistic, self).__init__(name)

    def __call__(self, input_):
        y_hat = T.nnet.sigmoid(input_)
        return OrderedDict(y_hat=y_hat), theano.OrderedUpdates()

    def err(self, y_hat, Y):
        err = (Y * (1 - y_hat) + (1 - Y) * y_hat).mean()
        return err


class Averager(Layer):
    def __init__(self, shape, name='averager', rate=0.1):
        self.rate = np.float32(rate)
        self.shape = shape
        super(Averager, self).__init__(name)

    def set_params(self):
        m = np.zeros(self.shape).astype(floatX)
        self.params = OrderedDict(m=m)

    def __call__(self, x):
        if x.ndim == 1:
            m = x
        elif x.ndim == 2:
            m = x.mean(axis=0)
        elif x.ndim == 3:
            m = x.mean(axis=(0, 1))
        else:
            raise ValueError()

        new_m = ((1. - self.rate) * self.m + self.rate * m).astype(floatX)
        updates = [(self.m, new_m)]
        return OrderedDict(m=new_m), updates


class Baseline(Layer):
    def __init__(self, name='baseline', rate=0.1):
        self.rate = np.float32(rate)
        super(Baseline, self).__init__(name)

    def set_params(self):
        m = np.float32(0.)
        var = np.float32(0.)

        self.params = OrderedDict(m=m, var=var)

    def __call__(self, input_):
        m = input_.mean()
        v = input_.std()

        new_m = T.switch(T.eq(self.m, 0.),
                         m,
                         (np.float32(1.) - self.rate) * self.m + self.rate * m)
        new_var = T.switch(T.eq(self.var, 0.),
                           v,
                           (np.float32(1.) - self.rate) * self.var + self.rate * v)

        updates = [(self.m, new_m), (self.var, new_var)]

        input_centered = (
            (input_ - new_m) / T.maximum(1., T.sqrt(new_var)))

        input_ = T.zeros_like(input_) + input_

        outs = OrderedDict(
            x=input_,
            x_centered=input_centered,
            m=new_m,
            var=new_var
        )
        return outs, updates

class BaselineWithInput(Baseline):
    def __init__(self, dims_in, dim_out, rate=0.1, name='baseline_with_input'):
        if len(dims_in) < 1:
            raise ValueError('One or more dims_in needed, %d provided'
                             % len(dims_in))
        self.dims_in = dims_in
        self.dim_out = dim_out
        super(BaselineWithInput, self).__init__(name=name, rate=rate)

    def set_params(self):
        super(BaselineWithInput, self).set_params()
        for i, dim_in in enumerate(self.dims_in):
            w = np.zeros((dim_in, self.dim_out)).astype('float32')
            k = 'w%d' % i
            self.params[k] = w

    def __call__(self, input_, update_params, *xs):
        '''
        Maybe unclear: input_ is the variable to be baselined, xs are the
        actual inputs.
        '''
        m = input_.mean()
        v = input_.std()

        new_m = T.switch(T.eq(self.m, 0.), m + self.m,
                         (np.float32(1.) - self.rate) * self.m + self.rate * m)
        new_m.name = 'new_m'
        new_var = T.switch(T.eq(self.var, 0.), v + self.var,
                           (np.float32(1.) - self.rate) * self.var + self.rate * v)

        if update_params:
            updates = [(self.m, new_m), (self.var, new_var)]
        else:
            updates = theano.OrderedUpdates()

        if len(xs) != len(self.dims_in):
            raise ValueError('Number of (external) inputs for baseline must'
                             ' match parameters')

        ws = []
        for i in xrange(len(xs)):
            # Maybe not the most pythonic way...
            ws.append(self.__dict__['w%d' % i])

        idb = T.sum([x.dot(W) for x, W in zip(xs, ws)], axis=0).T
        idb_c = T.zeros_like(idb) + idb
        input_centered = (
            (input_ - idb_c - new_m) / T.maximum(1., T.sqrt(new_var)))
        input_ = T.zeros_like(input_) + input_

        outs = OrderedDict(
            x_c=input_,
            x_centered=input_centered,
            m=new_m,
            var=new_var,
            idb=idb,
            idb_c=idb_c
        )

        return outs, updates

class ScalingWithInput(Layer):
    def __init__(self, dims_in, dim_out, name='scaling_with_input'):
        if len(dims_in) < 1:
            raise ValueError('One or more dims_in needed, %d provided'
                             % len(dims_in))
        self.dims_in = dims_in
        self.dim_out = dim_out
        super(ScalingWithInput, self).__init__(name=name)
        self.set_params()

    def set_params(self):
        for i, dim_in in enumerate(self.dims_in):
            w = np.zeros((dim_in, self.dim_out)).astype('float32')
            k = 'w%d' % i
            self.params[k] = w

    def __call__(self, input_, *xs):
        '''
        Maybe unclear: input_ is the variable to be scaled, xs are the
        actual inputs.
        '''
        updates = theano.OrderedUpdates()

        if len(xs) != len(self.dims_in):
            raise ValueError('Number of (external) inputs for baseline must'
                             ' match parameters')

        ws = []
        for i in xrange(len(xs)):
            # Maybe not the most pythonic way...
            ws.append(self.__dict__['w%d' % i])

        ids = T.sum([x.dot(W) for x, W in zip(xs, ws)], axis=0).T
        ids_c = T.zeros_like(ids) + ids
        input_scaled = input_ / ids_c
        input_ = T.zeros_like(input_) + input_

        outs = OrderedDict(
            x_c=input_,
            x_scaled=input_scaled,
            ids=ids,
            ids_c=ids_c
        )

        return outs, updates