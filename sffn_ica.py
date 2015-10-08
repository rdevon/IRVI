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


norm_weight = tools.norm_weight
ortho_weight = tools.ortho_weight
floatX = 'float32' # theano.config.floatX


class SFFN(Layer):
    def __init__(self, dim_in, dim_h, dim_out, rng=None, trng=None,
                 cond_to_h=None, cond_from_h=None,
                 weight_scale=1.0, weight_noise=False,
                 z_init=None, learn_z=False,
                 x_noise_mode=None, y_noise_mode=None, noise_amount=0.1,
                 momentum=0.9, b1=0.9, b2=0.999,
                 inference_rate=0.1, inference_decay=0.99, n_inference_steps=30,
                 inference_step_scheduler=None,
                 update_inference_scale=False,
                 inference_method='sgd', name='sffn'):

        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out

        self.cond_to_h = cond_to_h
        self.cond_from_h = cond_from_h

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

        self.n_inference_steps = T.constant(n_inference_steps).astype('int64')
        self.inference_step_scheduler = inference_step_scheduler

        if inference_method == 'sgd':
            self.step_infer = self._step_sgd
            self.init_infer = self._init_sgd
            self.unpack_infer = self._unpack_sgd
            self.params_infer = self._params_sgd
        elif inference_method == 'momentum':
            self.step_infer = self._step_momentum
            self.init_infer = self._init_momentum
            self.unpack_infer = self._unpack_momentum
            self.params_infer = self._params_momentum
        elif inference_method == 'adam':
            self.step_infer = self._step_adam
            self.init_infer = self._init_adam
            self.unpack_infer = self._unpack_adam
            self.params_infer = self._params_adam
        elif inference_method == 'cg':
            self.step_infer = self._step_cg
            self.init_infer = self._init_cg
            self.unpack_infer = self._unpack_cg
            self.params_infer = self._params_cg
        elif inference_method == 'cg2':
            self.step_infer = self._step_cg2
            self.init_infer = self._init_cg2
            self.unpack_infer = self._unpack_cg2
            self.params_infer = self._params_cg2
        else:
            raise ValueError()

        if rng is None:
            rng = tools.rng_
        self.rng = rng

        if trng is None:
            self.trng = RandomStreams(6 * 10 * 2015)
        else:
            self.trng = trng

        super(SFFN, self).__init__(name=name)

    def set_params(self):
        z = np.zeros((self.dim_h,)).astype(floatX)
        inference_scale_factor = np.float32(1.0)

        self.params = OrderedDict(
            z=z, inference_scale_factor=inference_scale_factor)

        if self.cond_to_h is None:
            self.cond_to_h = MLP(self.dim_in, self.dim_h, self.dim_h, 1,
                                 rng=self.rng, trng=self.trng,
                                 h_act='T.nnet.sigmoid',
                                 out_act='T.nnet.sigmoid')
        if self.cond_from_h is None:
            self.cond_from_h = MLP(self.dim_h, self.dim_out, self.dim_out, 1,
                                   rng=self.rng, trng=self.trng,
                                   h_act='T.nnet.sigmoid',
                                   out_act='T.nnet.sigmoid')

        self.cond_to_h.name = self.name + '_cond_to_h'
        self.cond_from_h.name = self.name + '_cond_from_h'

    def set_tparams(self, excludes=[]):
        excludes = ['{name}_{key}'.format(name=self.name, key=key)
                    for key in excludes]
        tparams = super(SFFN, self).set_tparams()
        tparams.update(**self.cond_to_h.set_tparams())
        tparams.update(**self.cond_from_h.set_tparams())

        tparams = OrderedDict((k, v) for k, v in tparams.iteritems()
            if k not in excludes)

        return tparams

    def init_z(self, x, y):
        z = T.alloc(0., x.shape[0], self.dim_h).astype(floatX)
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
        params = [self.z] + self.cond_from_h.get_params() + [self.inference_scale_factor]
        return params

    def p_y_given_h(self, h, *params):
        params = params[1:-1]
        return self.cond_from_h.step_call(h, *params)

    def sample_from_prior(self, n_samples=100):
        h = self.trng.binomial(p=T.nnet.sigmoid(self.z), size=(n_samples, self.dim_h), n=1,
                               dtype=floatX)
        py = self.cond_from_h(h)
        return py

    def m_step(self, ph, y, z, n_samples=10):
        mu = T.nnet.sigmoid(z)

        if n_samples == 0:
            h = mu[None, :, :]
        else:
            h = self.cond_to_h.sample(
                mu, size=(n_samples, mu.shape[0], mu.shape[1]))

        py = self.cond_from_h(h)
        py_approx = self.cond_from_h(mu)

        prior = T.nnet.sigmoid(self.z)
        prior_energy = self.cond_to_h.neg_log_prob(mu, prior[None, :]).mean()
        h_energy = self.cond_to_h.neg_log_prob(mu, ph).mean()
        y_energy = self.cond_from_h.neg_log_prob(y[None, :, :], py).mean()
        y_energy_approx = self.cond_from_h.neg_log_prob(y, py_approx).mean()

        return (prior_energy, h_energy, y_energy, y_energy_approx)

    def step_infer(self, *params):
        raise NotImplementedError()

    def init_infer(self, z):
        raise NotImplementedError()

    def unpack_infer(self, outs):
        raise NotImplementedError()

    def params_infer(self):
        raise NotImplementedError()

    def e_step(self, ph, y, z, *params):
        prior = T.nnet.sigmoid(params[0])
        mu = T.nnet.sigmoid(z)

        py = self.p_y_given_h(mu, *params)
        scale_factor = params[-1]

        cost = (scale_factor * self.cond_from_h.neg_log_prob(y, py)
                + self.cond_to_h.neg_log_prob(mu, prior)
                - self.cond_to_h.entropy(mu)
                ).sum(axis=0)
        grad = theano.grad(cost, wrt=z, consider_constant=[ph, y])
        return cost, grad

    # SGD
    def _step_sgd(self, ph, y, z, l, *params):
        cost, grad = self.e_step(ph, y, z, *params)
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
    def _step_momentum(self, ph, y, z, l, dz_, m, *params):
        cost, grad = self.e_step(ph, y, z, *params)
        dz = (-l * grad + m * dz_).astype(floatX)
        z = (z + dz).astype(floatX)
        l *= self.inference_decay
        return z, l, dz, cost

    def _init_momentum(self, ph, y, z):
        return [self.inference_rate, T.zeros_like(z)]

    def _unpack_momentum(self, outs):
        zs, ls, dzs, costs = outs
        return zs, costs

    def _params_momentum(self):
        return [T.constant(self.momentum).astype('float32')]

    # Adam
    def _step_adam(self, ph, y, z, m_tm1, v_tm1, cnt, b1, b2, lr, *params):

        b1 = b1 * (1 - 1e-8)**cnt
        cost, grad = self.e_step(ph, y, z, *params)
        m_t = b1 * m_tm1 + (1 - b1) * grad
        v_t = b2 * v_tm1 + (1 - b2) * grad**2
        m_t_hat = m_t / (1. - b1**(cnt + 1))
        v_t_hat = v_t / (1. - b2**(cnt + 1))
        grad_t = m_t_hat / (T.sqrt(v_t_hat) + 1e-8)
        z_t = (z - lr * grad_t).astype(floatX)
        cnt += 1

        return z_t, m_t, v_t, cnt, cost

    def _init_adam(self, ph, y, z):
        return [T.zeros_like(z), T.zeros_like(z), 0]

    def _unpack_adam(self, outs):
        zs, ms, vs, cnts, costs = outs
        return zs, costs

    def _params_adam(self):
        return [T.constant(self.b1).astype('float32'),
                T.constant(self.b2).astype('float32'),
                T.constant(self.inference_rate).astype('float32')]

    def _inference_cost_cg(self, ph, y, z, *params):
        mu = T.nnet.sigmoid(z)
        py = self.p_y_given_h(mu, *params)
        cost = (self.cond_from_h.neg_log_prob(y, py)
                + self.cond_to_h.neg_log_prob(mu, ph)
                - self.cond_to_h.entropy(mu)
                )
        return cost

    # Conjugate gradient with log-grid line search
    def _step_cg(self, ph, y, z, s_, dz_sq_, alphas, *params):
        cost, grad = self.e_step(ph, y, z, *params)
        dz = -grad
        dz_sq = (dz * dz).sum(axis=1)
        beta = dz_sq / (dz_sq_ + 1e-8)
        s = dz + beta[:, None] * s_
        z_alpha = z[None, :, :] + alphas[:, None, None] * s[None, :, :]
        costs = self._inference_cost_cg(
            ph[None, :, :], y[None, :, :], z_alpha, *params)
        idx = costs.argmin(axis=0)
        z = z + alphas[idx][:, None] * s
        return z, s, dz_sq, cost

    def _init_cg(self, ph, y, z):
        params = self.get_params()
        s0 = T.zeros_like(z)
        dz_sq0 = T.alloc(1., z.shape[0]).astype(floatX)
        return [s0, dz_sq0]

    def _unpack_cg(self, outs):
        zs, ss, dz_sqs, costs = outs
        return zs, costs

    def _params_cg(self, ):
        return [(self.inference_rate * 2. ** T.arange(8)).astype(floatX)]

    def infer_q(self, x, y, n_inference_steps, z0=None):
        updates = theano.OrderedUpdates()

        xs, ys = self.init_inputs(x, y, steps=self.n_inference_steps)
        ph = self.cond_to_h(xs)
        if z0 is None:
            if self.z_init == 'recognition_net':
                print 'Starting z0 at recognition net'
                z0 = T.log(ph[0] + 1e-7) - T.log(1 - ph[0] + 1e-7)
            else:
                z0 = self.init_z(x, y)

        seqs = [ph, ys]
        outputs_info = [z0] + self.init_infer(ph[0], ys[0], z0) + [None]
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

        zs, i_costs = self.unpack_infer(outs)

        return (zs, i_costs, ph, xs, ys), updates

    # Inference
    def inference(self, x, y, z0=None, n_samples=100):
        n_inference_steps = self.n_inference_steps

        (zs, i_costs, ph, xs, ys), updates = self.infer_q(
            x, y, n_inference_steps, z0=z0)

        prior_energy, h_energy, y_energy, y_energy_approx = self.m_step(
            ph[0], ys[0], zs[-1], n_samples=n_samples)

        if self.update_inference_scale:
            updates += [(self.inference_scale_factor,
                         y_energy / y_energy_approx)]

        return (xs, ys, zs,
                prior_energy, h_energy, y_energy, y_energy_approx,
                i_costs[-1]), updates

    def __call__(self, x, y, ph=None, n_samples=100, from_z=False,
                 n_inference_steps=0, end_with_inference=True):

        updates = theano.OrderedUpdates()

        x_n = self.trng.binomial(p=x, size=x.shape, n=1, dtype=x.dtype)

        if ph is not None:
            pass
        elif from_z:
            assert self.learn_z
            zh = T.tanh(T.dot(x_n, self.W0) + self.b0)
            z = T.dot(zh, self.W1) + self.b1
            ph = T.nnet.sigmoid(z)
        else:
            ph = self.cond_to_h(x)

        if end_with_inference:
            z0 = T.log(ph + 1e-7) - T.log(1 - ph + 1e-7)
            (zs, _, _, _, _), updates_i = self.infer_q(x_n, y, n_inference_steps, z0=z0)
            updates.update(updates_i)
            ph = T.nnet.sigmoid(zs[-1])

        h = self.cond_to_h.sample(ph, size=(n_samples, ph.shape[0], ph.shape[1]))
        py = self.cond_from_h(h)
        y_energy = self.cond_from_h.neg_log_prob(y[None, :, :], py)#.mean()
        y_energy = -log_mean_exp(-y_energy, axis=0).mean()

        return (py, y_energy), updates
