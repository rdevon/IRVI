'''
Module of Stochastic Feed Forward Networks
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from layers import Layer
from layers import MLP
import tools
from tools import init_rngs
from tools import init_weights
from tools import log_mean_exp
from tools import logit
from tools import _slice


norm_weight = tools.norm_weight
ortho_weight = tools.ortho_weight
floatX = 'float32' # theano.config.floatX


def init_momentum_args(model, momentum=0.9, **kwargs):
    model.momentum = momentum
    return kwargs

def init_sgd_args(model, **kwargs):
    return kwargs

def init_inference_args(model,
                        inference_rate=0.1,
                        inference_decay=0.99,
                        entropy_scale=1.0,
                        importance_sampling=False,
                        n_inference_steps=20,
                        inference_scaling=None,
                        inference_method='momentum',
                        sample_from_joint=False,
                        **kwargs):
    model.inference_rate = inference_rate
    model.inference_decay = inference_decay
    model.entropy_scale = entropy_scale
    model.importance_sampling = importance_sampling
    model.n_inference_steps = T.constant(n_inference_steps).astype('int64')
    model.inference_scaling = inference_scaling
    model.sample_from_joint = sample_from_joint

    if inference_method == 'sgd':
        model.step_infer = model._step_sgd
        model.init_infer = model._init_sgd
        model.unpack_infer = model._unpack_sgd
        model.params_infer = model._params_sgd
        kwargs = init_sgd_args(model, **kwargs)
    elif inference_method == 'momentum':
        model.step_infer = model._step_momentum
        model.init_infer = model._init_momentum
        model.unpack_infer = model._unpack_momentum
        model.params_infer = model._params_momentum
        kwargs = init_momentum_args(model, **kwargs)
    else:
        raise ValueError()

    return kwargs


class SigmoidBeliefNetwork(Layer):
    def __init__(self, dim_in, dim_h, dim_out,
                 posterior=None, conditional=None,
                 z_init=None,
                 x_noise_mode=None, y_noise_mode=None, noise_amount=0.1,
                 name='sbn',
                 **kwargs):

        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out

        self.posterior = posterior
        self.conditional = conditional

        self.z_init = z_init

        self.x_mode = x_noise_mode
        self.y_mode = y_noise_mode
        self.noise_amount = noise_amount

        kwargs = init_inference_args(self, **kwargs)
        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        super(SigmoidBeliefNetwork, self).__init__(name=name)

    def set_params(self):
        z = np.zeros((self.dim_h,)).astype(floatX)
        inference_scale_factor = np.float32(1.0)

        self.params = OrderedDict(
            z=z, inference_scale_factor=inference_scale_factor)

        if self.posterior is None:
            self.posterior = MLP(self.dim_in, self.dim_h, self.dim_h, 1,
                                 rng=self.rng, trng=self.trng,
                                 h_act='T.nnet.sigmoid',
                                 out_act='T.nnet.sigmoid')
        if self.conditional is None:
            self.conditional = MLP(self.dim_h, self.dim_out, self.dim_out, 1,
                                   rng=self.rng, trng=self.trng,
                                   h_act='T.nnet.sigmoid',
                                   out_act='T.nnet.sigmoid')

        self.posterior.name = self.name + '_posterior'
        self.conditional.name = self.name + '_conditional'

    def set_tparams(self, excludes=[]):
        excludes.append('inference_scale_factor')
        excludes = ['{name}_{key}'.format(name=self.name, key=key)
                    for key in excludes]
        tparams = super(SigmoidBeliefNetwork, self).set_tparams()
        tparams.update(**self.posterior.set_tparams())
        tparams.update(**self.conditional.set_tparams())
        tparams = OrderedDict((k, v) for k, v in tparams.iteritems()
            if k not in excludes)

        return tparams

    def _sample(self, p, size=None):
        if size is None:
            size = p.shape
        return self.trng.binomial(p=p, size=size, n=1, dtype=p.dtype)

    def _noise(self, x, amount, size):
        return x * (1 - self.trng.binomial(p=amount, size=size, n=1,
                                           dtype=x.dtype))

    def set_input(self, x, mode, size=None):
        if size is None:
            size = x.shape
        if mode == 'sample':
            x = self._sample(x[None, :, :], size=size)
        elif mode == 'noise':
            x = self._sample(x)
            x = self._noise(x[None, :, :], size=size)
        elif mode is None:
            x = self._sample(x, size=x.shape)
            x = T.alloc(0., *size) + x[None, :, :]
        else:
            raise ValueError('% not supported' % mode)
        return x

    def init_inputs(self, x, y, steps=1):
        x_size = (steps, x.shape[0], x.shape[1])
        y_size = (steps, y.shape[0], y.shape[1])

        x = self.set_input(x, self.x_mode, size=x_size)
        y = self.set_input(y, self.y_mode, size=y_size)
        return x, y

    def get_params(self):
        params = [self.z] + self.conditional.get_params() + [self.inference_scale_factor]
        return params

    def p_y_given_h(self, h, *params):
        params = params[1:-1]
        return self.conditional.step_call(h, *params)

    def sample_from_prior(self, n_samples=100):
        p = T.nnet.sigmoid(self.z)
        h = self.posterior.sample(p=p, size=(n_samples, p.shape[0]))
        py = self.conditional(h)
        return py

    def m_step(self, ph, y, z, n_samples=10):
        constants = []
        mu = T.nnet.sigmoid(z)
        prior = T.nnet.sigmoid(self.z)

        if n_samples == 0:
            h = mu[None, :, :]
        else:
            h = self.posterior.sample(
                mu, size=(n_samples, mu.shape[0], mu.shape[1]))

        py = self.conditional(h)
        py_approx = self.conditional(mu)

        prior_energy = self.posterior.neg_log_prob(mu, prior[None, :]).mean()
        h_energy = self.posterior.neg_log_prob(mu, ph).mean()

        y_energy_approx = self.conditional.neg_log_prob(y, py_approx).mean()
        if self.importance_sampling:
            y_energy = self.conditional.neg_log_prob(y[None, :, :], py)
            w = T.exp(-y_energy + self.posterior.neg_log_prob(h, mu))
            w_sum = w.sum(axis=0)
            w_sum = T.clip(w_sum, 1e-7, 1.0 - 1e-7)
            w_tilda = w / w_sum
            y_energy = (w_tilda * y_energy).mean()
            constants += [w_tilda, w]
        elif self.sample_from_joint:
            raise NotImplementedError()
            y_hat = self.conditional.sample(
                py, size=(n_samples, py.shape[0], py.shape[1]))
            ph = self.posterior(y_hat)
            h_energy = self.posterior.neg_log_prob(h).mean()
        else:
            y_energy = self.conditional.neg_log_prob(y[None, :, :], py).mean()

        entropy = self.posterior.entropy(mu).mean()

        return (prior_energy, h_energy, y_energy, y_energy_approx, entropy), constants

    def kl_divergence(self, p, q, entropy_scale=1.0):
        entropy_term = entropy_scale * self.posterior.entropy(p)
        prior_term = self.posterior.neg_log_prob(p, q)
        return -(entropy_term - prior_term)

    def e_step(self, y, z, *params):
        prior = T.nnet.sigmoid(params[0])
        mu = T.nnet.sigmoid(z)
        py = self.p_y_given_h(mu, *params)

        consider_constant = [y, prior]
        cond_term = self.conditional.neg_log_prob(y, py)
        if self.inference_scaling == 'global':
            print 'Using global scaling in inference'
            scale_factor = params[-1]
            cond_term = scale_factor * cond_term
            consider_constant += [scale_factor]
        elif self.inference_scaling == 'inference':
            print 'Calculating scaling during inference'
            h = self.posterior.sample(mu, size=(10, mu.shape[0], mu.shape[1]))
            py_r = self.p_y_given_h(h, *params)
            mc = self.conditional.neg_log_prob(y[None, :, :], py_r).mean(axis=0)
            cond_term_c = T.zeros_like(cond_term) + cond_term
            scale_factor = mc / cond_term_c
            cond_term = scale_factor * cond_term
            consider_constant += [scale_factor, cond_term_c]
        elif self.inference_scaling == 'KL':
            raise NotImplementedError()
            print 'Adding KL term to inference'
            mc = self.conditional.neg_log_prob(y[None, :, :], py_r).mean(axis=0)
        elif self.inference_scaling == 'reweight':
            print 'Reweighting mus'
            pass
        elif self.inference_scaling is not None:
            raise ValueError(self.inference_scaling)
        else:
            print 'No inference scaling'

        kl_term = self.kl_divergence(
            mu, prior[None, :], entropy_scale=self.entropy_scale)

        cost = (cond_term + kl_term).sum(axis=0)

        grad = theano.grad(cost, wrt=z, consider_constant=consider_constant)

        return cost, grad, cond_term.mean(), kl_term.mean()

    def step_infer(self, *params):
        raise NotImplementedError()

    def init_infer(self, z):
        raise NotImplementedError()

    def unpack_infer(self, outs):
        raise NotImplementedError()

    def params_infer(self):
        raise NotImplementedError()

    # SGD
    def _step_sgd(self, y, z, l, *params):
        cost, grad = self.e_step(y, z, *params)
        z = (z - l * grad).astype(floatX)
        l *= self.inference_decay
        return z, l, cost

    def _init_sgd(self, ph, y, z):
        return [self.inference_rate]

    def _unpack_sgd(self, outs):
        zs, ls, costs = outs
        return zs, costs

    def _params_sgd(self):
        return []

    # Momentum
    def _step_momentum(self, y, z, l, dz_, m, *params):
        cost, grad, c_term, kl_term = self.e_step(y, z, *params)
        dz = (-l * grad + m * dz_).astype(floatX)
        z = (z + dz).astype(floatX)
        if self.inference_scaling == 'reweight':
            mu = T.nnet.sigmoid(z)
            prior = T.nnet.sigmoid(params[0])
            h = self.posterior.sample(mu, size=(10, mu.shape[0], mu.shape[1]))

            py_r = self.p_y_given_h(h, *params)
            mc = self.conditional.neg_log_prob(y[None, :, :], py_r)
            kl = self.kl_divergence(h, prior[None, None, :])
            w = mc + kl
            w_tilda = w / w.sum(axis=0)[None, :]
            mu = (w_tilda[:, :, None] * h).mean(axis=0)

            z = logit(mu)
        l *= self.inference_decay
        return z, l, dz, cost, c_term, kl_term

    def _init_momentum(self, ph, y, z):
        return [self.inference_rate, T.zeros_like(z)]

    def _unpack_momentum(self, outs):
        zs, ls, dzs, costs, c_terms, kl_terms = outs
        return zs, costs, c_terms, kl_terms

    def _params_momentum(self):
        return [T.constant(self.momentum).astype('float32')]

    def infer_q(self, x, y, n_inference_steps, z0=None):
        updates = theano.OrderedUpdates()

        xs, ys = self.init_inputs(x, y, steps=self.n_inference_steps)
        ph = self.posterior(xs)
        if z0 is None:
            if self.z_init == 'recognition_net':
                print 'Starting z0 at recognition net'
                z0 = logit(ph[0])
            else:
                z0 = T.alloc(0., x.shape[0], self.dim_h).astype(floatX)

        seqs = [ys]
        outputs_info = [z0] + self.init_infer(ph[0], ys[0], z0) + [None, None, None]
        non_seqs = self.params_infer() + self.get_params()

        outs, updates_2 = theano.scan(
            self.step_infer,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, 'infer'),
            n_steps=n_inference_steps,
            profile=tools.profile,
            strict=True
        )
        updates.update(updates_2)

        zs, i_costs, c_terms, kl_terms = self.unpack_infer(outs)
        zs = T.concatenate([z0[None, :, :], zs], axis=0)

        return (zs, i_costs, ph, xs, ys, c_terms, kl_terms), updates

    # Inference
    def inference(self, x, y, z0=None, n_samples=100):
        n_inference_steps = self.n_inference_steps

        (zs, i_costs, ph, xs, ys, c_terms, kl_terms), updates = self.infer_q(
            x, y, n_inference_steps, z0=z0)

        (prior_energy, h_energy, y_energy,
         y_energy_approx, entropy), m_constants = self.m_step(
            ph[0], ys[0], zs[-1], n_samples=n_samples)
        constants = [xs, ys, zs, entropy] + m_constants

        if self.inference_scaling == 'global':
            updates += [(self.inference_scale_factor,
                         y_energy / y_energy_approx)]
            constants += [self.inference_scale_factor]

        return (xs, ys, zs,
                prior_energy, h_energy, y_energy, y_energy_approx, entropy,
                i_costs[-1], c_terms[-1], kl_terms[-1]), updates, constants

    def __call__(self, x, y, ph=None, n_samples=100,
                 n_inference_steps=0, end_with_inference=True):
        updates = theano.OrderedUpdates()

        x_n = self.trng.binomial(p=x, size=x.shape, n=1, dtype=x.dtype)

        if ph is None:
            ph = self.posterior(x)

        if end_with_inference:
            z0 = logit(ph)
            (zs, i_energy, _, _, _, cs, kls), updates_i = self.infer_q(x_n, y, n_inference_steps, z0=z0)
            updates.update(updates_i)
            ph = T.nnet.sigmoid(zs[-1])

        h = self.posterior.sample(ph, size=(n_samples, ph.shape[0], ph.shape[1]))
        py = self.conditional(h)
        y_energy = self.conditional.neg_log_prob(y[None, :, :], py).mean(axis=0)
        prior = T.nnet.sigmoid(self.z)
        kl_term = self.kl_divergence(ph, prior[None, :])

        return (py, (y_energy + kl_term).mean(axis=0),
                     i_energy[-1], cs[-1], kls[-1]), updates


class GaussianBeliefNet(Layer):
    def __init__(self, dim_in, dim_h, dim_out, rng=None, trng=None,
                 posterior=None, conditional=None,
                 weight_scale=1.0, weight_noise=False,
                 z_init=None, learn_z=False,
                 x_noise_mode=None, y_noise_mode=None, noise_amount=0.1,
                 momentum=0.9, b1=0.9, b2=0.999,
                 inference_rate=0.1, inference_decay=0.99, n_inference_steps=30,
                 inference_step_scheduler=None,
                 update_inference_scale=False,
                 inference_scaling='global',
                 entropy_scale=1.0,
                 use_geometric_mean=False,
                 inference_method='sgd', name='sffn'):

        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out

        self.posterior = posterior
        self.conditional = conditional

        self.weight_noise = weight_noise
        self.weight_scale = weight_scale

        self.momentum = momentum
        self.b1 = b1
        self.b2 = b2

        self.z_init = z_init
        self.learn_z = learn_z

        self.x_mode = x_noise_mode
        self.y_mode = y_noise_mode
        self.noise_amount = noise_amount

        self.inference_rate = inference_rate
        self.inference_decay = inference_decay
        self.update_inference_scale = update_inference_scale
        self.entropy_scale = entropy_scale
        self.use_geometric_mean = use_geometric_mean

        self.n_inference_steps = T.constant(n_inference_steps).astype('int64')
        self.inference_step_scheduler = inference_step_scheduler
        self.inference_scaling = inference_scaling

        if inference_method == 'momentum':
            self.step_infer = self._step_momentum
            self.init_infer = self._init_momentum
            self.unpack_infer = self._unpack_momentum
            self.params_infer = self._params_momentum
        else:
            raise ValueError()

        if rng is None:
            rng = tools.rng_
        self.rng = rng

        if trng is None:
            self.trng = RandomStreams(6 * 10 * 2015)
        else:
            self.trng = trng

        super(GaussianBeliefNet, self).__init__(name=name)

    def set_params(self):
        mu = np.zeros((self.dim_h,)).astype(floatX)
        log_sigma = np.zeros((self.dim_h,)).astype(floatX)
        inference_scale_factor = np.float32(1.0)

        self.params = OrderedDict(
            mu=mu, log_sigma=log_sigma,
            inference_scale_factor=inference_scale_factor)

        if self.posterior is None:
            self.posterior = MLP(self.dim_in, self.dim_h, self.dim_h, 1,
                                 rng=self.rng, trng=self.trng,
                                 h_act='T.nnet.sigmoid',
                                 out_act='lambda x: x')
        if self.conditional is None:
            self.conditional = MLP(self.dim_h, self.dim_out, self.dim_out, 1,
                                   rng=self.rng, trng=self.trng,
                                   h_act='T.nnet.sigmoid',
                                   out_act='T.nnet.sigmoid')

        self.posterior.name = self.name + '_posterior'
        self.conditional.name = self.name + '_conditional'

    def set_tparams(self, excludes=[]):
        excludes = ['{name}_{key}'.format(name=self.name, key=key)
                    for key in excludes]
        tparams = super(GaussianBeliefNet, self).set_tparams()
        tparams.update(**self.posterior.set_tparams())
        tparams.update(**self.conditional.set_tparams())

        tparams = OrderedDict((k, v) for k, v in tparams.iteritems()
            if k not in excludes)

        return tparams

    def init_z(self, x, y):
        z = T.alloc(0., x.shape[0], 2 * self.dim_h).astype(floatX)
        return z

    def _sample(self, p, size=None):
        if size is None:
            size = p.shape
        return self.trng.binomial(p=p, size=size, n=1, dtype=p.dtype)

    def _noise(self, x, amount, size):
        return x * (1 - self.trng.binomial(p=amount, size=size, n=1,
                                           dtype=x.dtype))

    def set_input(self, x, mode, size=None):
        if size is None:
            size = x.shape
        if mode == 'sample':
            x = self._sample(x[None, :, :], size=size)
        elif mode == 'noise':
            x = self._sample(x)
            x = self._noise(x[None, :, :], size=size)
        elif mode is None:
            x = self._sample(x, size=x.shape)
            x = T.alloc(0., *size) + x[None, :, :]
        else:
            raise ValueError('% not supported' % mode)
        return x

    def init_inputs(self, x, y, steps=1):
        x_size = (steps, x.shape[0], x.shape[1])
        y_size = (steps, y.shape[0], y.shape[1])

        x = self.set_input(x, self.x_mode, size=x_size)
        y = self.set_input(y, self.y_mode, size=y_size)
        return x, y

    def get_params(self):
        params = [self.mu, self.log_sigma] + self.conditional.get_params() + [self.inference_scale_factor]
        return params

    def p_y_given_h(self, h, *params):
        params = params[2:-1]
        return self.conditional.step_call(h, *params)

    def sample_from_prior(self, n_samples=100):
        p = T.concatenate([self.mu, self.log_sigma], axis=0)
        h = self.posterior.sample(p=p, size=(n_samples, self.dim_h))
        py = self.conditional(h)
        return py

    def m_step(self, ph, y, z, n_samples=10):
        mu = _slice(z, 0, self.dim_h)
        log_sigma = _slice(z, 1, self.dim_h)

        if n_samples == 0:
            h = mu[None, :, :]
        else:
            h = self.posterior.sample(p=z, size=(n_samples, mu.shape[0], mu.shape[1]))

        py = self.conditional(h)
        py_approx = self.conditional(mu)

        prior_energy = self.kl_divergence(
            mu, log_sigma, self.mu[None, :], self.log_sigma[None, :]).mean()

        mu_h = _slice(ph, 0, self.dim_h)
        log_sigma_h = _slice(ph, 1, self.dim_h)
        h_energy = self.kl_divergence(mu_h, log_sigma_h, mu, log_sigma).mean()

        y_energy = self.conditional.neg_log_prob(y[None, :, :], py).mean()
        y_energy_approx = self.conditional.neg_log_prob(y, py_approx).mean()
        entropy = self.posterior.entropy(z).mean()

        return (prior_energy, h_energy, y_energy, y_energy_approx, entropy)

    def kl_divergence(self, mu, log_sigma, mu_p, log_sigma_p, entropy_scale=1.0):
        kl = log_sigma_p - log_sigma + 0.5 * ((T.exp(2 * log_sigma) + (mu - mu_p) ** 2) / T.exp(2 * log_sigma_p) - 1)
        return kl.sum(axis=kl.ndim-1)

    def e_step(self, y, z, *params):
        mu_p = params[0]
        log_sigma_p = params[1]
        mu = _slice(z, 0, self.dim_h)
        log_sigma = _slice(z, 1, self.dim_h)
        py = self.p_y_given_h(mu, *params)

        consider_constant = [y, mu_p, log_sigma_p]
        cond_term = self.conditional.neg_log_prob(y, py)
        if self.inference_scaling == 'global':
            print 'Using global scaling in inference'
            scale_factor = params[-1]
            cond_term = scale_factor * cond_term
            consider_constant += [scale_factor]
        elif self.inference_scaling == 'inference':
            print 'Calculating scaling during inference'
            h = self.posterior.sample(mu, size=(10, mu.shape[0], mu.shape[1]))
            py_r = self.p_y_given_h(h, *params)
            mc = self.conditional.neg_log_prob(y[None, :, :], py_r).mean(axis=0)
            cond_term_c = T.zeros_like(cond_term) + cond_term
            scale_factor = mc / cond_term_c
            cond_term = scale_factor * cond_term
            consider_constant += [scale_factor, cond_term_c]
        elif self.inference_scaling is not None:
            raise ValueError(self.inference_scaling)

        cond_term = self.conditional.neg_log_prob(y, py)
        kl_term = self.kl_divergence(mu, log_sigma, mu_p[None, :],
                                     log_sigma_p[None, :],
                                     entropy_scale=self.entropy_scale)

        cost = (cond_term + kl_term).sum(axis=0)
        grad = theano.grad(cost, wrt=z, consider_constant=consider_constant)

        return cost, grad, cond_term.mean(), kl_term.mean()

    def step_infer(self, *params):
        raise NotImplementedError()

    def init_infer(self, z):
        raise NotImplementedError()

    def unpack_infer(self, outs):
        raise NotImplementedError()

    def params_infer(self):
        raise NotImplementedError()

    # Momentum
    def _step_momentum(self, y, z, l, dz_, m, *params):
        cost, grad, c_term, kl_term = self.e_step(y, z, *params)
        dz = (-l * grad + m * dz_).astype(floatX)
        z = (z + dz).astype(floatX)
        l *= self.inference_decay
        return z, l, dz, cost, c_term, kl_term

    def _init_momentum(self, ph, y, z):
        return [self.inference_rate, T.zeros_like(z)]

    def _unpack_momentum(self, outs):
        zs, ls, dzs, costs, c_terms, kl_terms = outs
        return zs, costs, c_terms, kl_terms

    def _params_momentum(self):
        return [T.constant(self.momentum).astype('float32')]

    def infer_q(self, x, y, n_inference_steps, z0=None):
        updates = theano.OrderedUpdates()

        xs, ys = self.init_inputs(x, y, steps=self.n_inference_steps)
        ph = self.posterior(xs)
        if z0 is None:
            if self.z_init == 'recognition_net':
                print 'Starting z0 at recognition net'
                z0 = ph[0]
            else:
                z0 = self.init_z(x, y)

        seqs = [ys]
        outputs_info = [z0] + self.init_infer(ph[0], ys[0], z0) + [None, None, None]
        non_seqs = self.params_infer() + self.get_params()

        outs, updates_2 = theano.scan(
            self.step_infer,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, 'infer'),
            n_steps=n_inference_steps,
            profile=tools.profile,
            strict=True
        )
        updates.update(updates_2)

        zs, i_costs, c_terms, kl_terms = self.unpack_infer(outs)
        zs = T.concatenate([z0[None, :, :], zs], axis=0)

        return (zs, i_costs, ph, xs, ys, c_terms, kl_terms), updates

    # Inference
    def inference(self, x, y, z0=None, n_samples=100):
        n_inference_steps = self.n_inference_steps

        (zs, i_costs, ph, xs, ys, c_terms, kl_terms), updates = self.infer_q(
            x, y, n_inference_steps, z0=z0)

        prior_energy, h_energy, y_energy, y_energy_approx, entropy = self.m_step(
            ph[0], ys[0], zs[-1], n_samples=n_samples)

        if self.update_inference_scale:
            updates += [(self.inference_scale_factor,
                         y_energy / y_energy_approx)]

        return (xs, ys, zs,
                prior_energy, h_energy, y_energy, y_energy_approx, entropy,
                i_costs[-1], c_terms[-1], kl_terms[-1]), updates

    def __call__(self, x, y, ph=None, n_samples=100,
                 n_inference_steps=0, end_with_inference=True):

        updates = theano.OrderedUpdates()

        x_n = self.trng.binomial(p=x, size=x.shape, n=1, dtype=x.dtype)

        if ph is None:
            ph = self.posterior(x)

        if end_with_inference:
            z0 = ph
            (zs, i_energy, _, _, _, cs, kls), updates_i = self.infer_q(
                x_n, y, n_inference_steps, z0=z0)
            updates.update(updates_i)

            mu = _slice(zs[-1], 0, self.dim_h)
            log_sigma = _slice(zs[-1], 1, self.dim_h)
        else:
            mu = _slice(ph, 0, self.dim_h)
            log_sigma = _slice(ph, 1, self.dim_h)

        h = self.posterior.sample(ph,
                                  size=(n_samples, mu.shape[0], mu.shape[1]))
        py = self.conditional(h)
        y_energy = self.conditional.neg_log_prob(y[None, :, :], py).mean(axis=0)
        kl_term = self.kl_divergence(mu, log_sigma,
                                     self.mu[None, :], self.log_sigma[None, :])

        return (py, (y_energy + kl_term).mean(axis=0),
                     i_energy[-1], cs[-1], kls[-1]), updates
