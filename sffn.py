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
        self.params = OrderedDict()
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

    def set_tparams(self):
        tparams = super(SFFN, self).set_tparams()
        tparams.update(**self.cond_to_h.set_tparams())
        tparams.update(**self.cond_from_h.set_tparams())

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
        elif mode == None:
            x = self._sample(x, size=x.shape)
            x = T.alloc(0., *size) + x[None, :, :]
        else:
            raise ValueError()
        return x

    def init_inputs(self, x, y, steps=1):
        x_size = (steps, x.shape[0], x.shape[1])
        y_size = (steps, y.shape[0], y.shape[1])

        x = self.set_input(x, self.x_mode, size=x_size)
        y = self.set_input(y, self.y_mode, size=y_size)
        return x, y

    def get_params(self):
        return self.cond_from_h.get_params()

    def p_y_given_h(self, h, *params):
        return self.cond_from_h.step_call(h, *params)

    def sample_energy(self, ph, y, z, n_samples=10):
        mu = T.nnet.sigmoid(z)

        if n_samples == 0:
            h = mu
        else:
            h = self.cond_to_h.sample(mu, size=(n_samples,
                                                mu.shape[0], mu.shape[1]))

        py = self.cond_from_h(h)

        h_energy = self.cond_to_h.neg_log_prob(h, ph[None, :, :])
        h_energy = -log_mean_exp(-h_energy, axis=0).mean()
        y_energy = self.cond_from_h.neg_log_prob(y[None, :, :], py)
        y_energy = -log_mean_exp(-y_energy, axis=0).mean()

        return (h_energy, y_energy)

    def step_infer(self, *params):
        raise NotImplementedError()

    def init_infer(self, z):
        raise NotImplementedError()

    def unpack_infer(self, outs):
        raise NotImplementedError()

    def params_infer(self):
        raise NotImplementedError()

    def inference_cost(self, ph, y, z, *params):
        mu = T.nnet.sigmoid(z)
        py = self.p_y_given_h(mu, *params)

        cost = (self.cond_from_h.neg_log_prob(y, py)
                + self.cond_to_h.neg_log_prob(mu, ph)
                - self.cond_to_h.entropy(mu)
                ).sum(axis=0)
        grad = theano.grad(cost, wrt=z, consider_constant=[ph, y])
        return cost, grad

    # SGD
    def _step_sgd(self, ph, y, z, l, *params):
        cost, grad = self.inference_cost(ph, y, z, *params)
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
        cost, grad = self.inference_cost(ph, y, z, *params)
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
        cost, grad = self.inference_cost(ph, y, z, *params)
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
        cost, grad = self.inference_cost(ph, y, z, *params)
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

    # Inference
    def inference(self, x, y, n_samples=100):
        updates = theano.OrderedUpdates()

        xs, ys = self.init_inputs(x, y, steps=self.n_inference_steps)
        ph = self.cond_to_h(xs)
        z0 = self.init_z(xs[0], ys[0])

        if self.inference_step_scheduler is None:
            n_inference_steps = self.n_inference_steps
        else:
            n_inference_steps, updates_c = self.inference_step_scheduler(n_inference_steps)
            updates.update(updates_c)

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
        h_energy, y_energy = self.sample_energy(ph[0], ys[0], zs[-1],
                                                n_samples=n_samples)
        return (xs, ys, zs, h_energy, y_energy, i_costs[-1]), updates

    def __call__(self, x, y, ph=None, n_samples=100, from_z=False):
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

        h = self.cond_to_h.sample(ph, size=(n_samples, ph.shape[0], ph.shape[1]))
        py = self.cond_from_h(h)
        y_energy = self.cond_from_h.neg_log_prob(y[None, :, :], py)
        y_energy = -log_mean_exp(-y_energy, axis=0).mean()
        return py, y_energy


class SFFN_2Layer(Layer):
    def __init__(self, dim_in, dim_h, dim_out,
                 rng=None, trng=None,
                 cond_to_h1=None, cond_to_h2=None, cond_from_h2=None,
                 weight_scale=1.0, weight_noise=False,
                 z_init=None, learn_z=False,
                 x_noise_mode=None, y_noise_mode=None, noise_amount=0.1,
                 momentum=0.9, b1=0.9, b2=0.999,
                 inference_rate_1=0.1, inference_rate_2=0.1,
                 name='sffn_2layer',
                 h1_lme=False, h2_lme=False, y_lme=True,
                 h1_bias=False, h2_bias=True, h2_sample_bias=False,
                 inference_decay=0.99, n_inference_steps=30,
                 inference_step_scheduler=None,
                 inference_method='sgd'):

        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out

        self.cond_to_h1 = cond_to_h1
        self.cond_to_h2 = cond_to_h2
        self.cond_from_h2 = cond_from_h2

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

        self.inference_rate_1 = inference_rate_1
        self.inference_rate_2 = inference_rate_2

        self.n_inference_steps = T.constant(n_inference_steps).astype('int64')
        self.inference_step_scheduler = inference_step_scheduler
        self.inference_decay = inference_decay

        self.h1_lme = h1_lme
        self.h2_lme = h2_lme
        self.y_lme = y_lme

        self.h1_bias = h1_bias
        self.h2_bias = h2_bias
        self.h2_sample_bias = h2_sample_bias

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

        super(SFFN_2Layer, self).__init__(name=name)

    def set_params(self):
        self.params = OrderedDict()
        if self.cond_to_h1 is None:
            self.cond_to_h1 = MLP(self.dim_in, self.dim_h, self.dim_h, 1,
                                  rng=self.rng, trng=self.trng,
                                  h_act='T.nnet.sigmoid',
                                  out_act='T.nnet.sigmoid')
        if self.cond_to_h2 is None:
            self.cond_to_h2 = MLP(self.dim_h, self.dim_h, self.dim_h, 1,
                                  rng=self.rng, trng=self.trng,
                                  h_act='T.nnet.sigmoid',
                                  out_act='T.nnet.sigmoid')
        if self.cond_from_h2 is None:
            self.cond_from_h2 = MLP(self.dim_h, self.dim_out, self.dim_out, 1,
                                    rng=self.rng, trng=self.trng,
                                    h_act='T.nnet.sigmoid',
                                    out_act='T.nnet.sigmoid')

        self.cond_to_h1.name = 'cond_to_h1'
        self.cond_to_h2.name = 'cond_to_h2'
        self.cond_from_h2.name = 'cond_from_h2'

    def set_tparams(self):
        tparams = super(SFFN_2Layer, self).set_tparams()
        tparams.update(**self.cond_to_h1.set_tparams())
        tparams.update(**self.cond_to_h2.set_tparams())
        tparams.update(**self.cond_from_h2.set_tparams())

        return tparams

    def get_params(self):
        return (self.cond_to_h2.get_params()
                + self.cond_from_h2.get_params())

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
        elif mode == None:
            x = self._sample(x, size=x.shape)
            x = T.alloc(0., *size) + x[None, :, :]
        else:
            raise ValueError()
        return x

    def init_inputs(self, x, y, steps=1):
        x_size = (steps, x.shape[0], x.shape[1])
        y_size = (steps, y.shape[0], y.shape[1])

        x = self.set_input(x, self.x_mode, size=x_size)
        y = self.set_input(y, self.y_mode, size=y_size)
        return x, y

    def p_h2_given_h1(self, h1, *params):
        ps = params[:len(self.cond_to_h2.get_params())]
        return self.cond_to_h2.step_call(h1, *ps)

    def preact_h1_to_h2(self, h1, *params):
        ps = params[:len(self.cond_to_h2.get_params())]
        return self.cond_to_h2.preact(h1, *ps)

    def p_y_given_h2(self, h2, *params):
        ps = params[len(self.cond_to_h2.get_params()):]
        return self.cond_from_h2.step_call(h2, *ps)

    def sample_energy(self, ph1, preact_h1, y, z1, z2, n_samples=10):
        if self.h1_bias:
            mu1 = eval(self.cond_to_h1.out_act)(preact_h1 + z1)
        else:
            mu1 = T.nnet.sigmoid(z1)

        if n_samples == 0:
            h1 = mu1
        else:
            h1 = self._sample(p=mu1, size=(n_samples, mu1.shape[0], mu1.shape[1]))

        if self.h2_bias:
            if self.h2_sample_bias:
                h1_signal = h1
            else:
                h1_signal = T.zeros_like(h1) + mu1[None, :, :]
            mu2 = eval(self.cond_to_h2.out_act)(
                self.cond_to_h2(h1_signal, return_preact=True) + z2[None, :, :])
        else:
            mu2 = T.nnet.sigmoid(z2)

        if n_samples == 0:
            h2 = mu2
        else:
            h2 = self._sample(p=mu2)

        ph2 = self.cond_to_h2(h1)
        py = self.cond_from_h2(h2)

        h1_energy = self.cond_to_h1.neg_log_prob(h1, ph1[None, :, :])
        if self.h1_lme:
            h1_energy = -log_mean_exp(-h1_energy, axis=0).mean()
        else:
            h1_energy = h1_energy.mean()

        h2_energy = self.cond_to_h2.neg_log_prob(h2, ph2, axis=2)
        if self.h2_lme:
            h2_energy = -log_mean_exp(-h2_energy, axis=0).mean()
        else:
            h2_energy = h2_energy.mean()

        y_energy = self.cond_from_h2.neg_log_prob(y[None, :, :], py)
        if self.y_lme:
            y_energy = -log_mean_exp(-y_energy, axis=0).mean()
        y_energy = y_energy.mean()

        return (h1_energy, h2_energy, y_energy, h1, h2)

    def inference_cost(self, ph1, preact_h1, y, z1, z2, *params):
        if self.h1_bias:
            mu1 = eval(self.cond_to_h1.out_act)(preact_h1 + z1)
        else:
            mu1 = T.nnet.sigmoid(z1)
        h1 = self.cond_to_h1.sample(mu1)

        if self.h2_bias:
            if self.h2_sample_bias:
                h1_signal = h1
            else:
                h1_signal = mu1
            mu2 = eval(self.cond_to_h2.out_act)(
                self.preact_h1_to_h2(h1_signal, *params) + z2)
        else:
            mu2 = T.nnet.sigmoid(z2)

        ph2 = self.p_h2_given_h1(mu1, *params)
        py = self.p_y_given_h2(mu2, *params)

        cost = (self.cond_to_h1.neg_log_prob(mu1, ph1)
                + self.cond_to_h2.neg_log_prob(mu2, ph2)
                + self.cond_from_h2.neg_log_prob(y, py)
                - self.cond_to_h1.entropy(mu1)
                - self.cond_to_h2.entropy(mu2)
                ).sum(axis=0)

        grad1, grad2 = theano.grad(cost, wrt=[z1, z2],
                                   consider_constant=[ph1, preact_h1, y])
        return cost, (grad1, grad2)

    def _init_sgd(self, ph, y, z):
        return [self.inference_rate_1, self.inference_rate_2]

    def _step_sgd(self, ph1, y, z1, z2, l1, l2, *params):
        cost, (grad1, grad2) = self.inference_cost(ph1, y, z1, z2, *params)
        z1 = (z1 - l1 * grad1).astype(floatX)
        z2 = (z2 - l2 * grad2).astype(floatX)

        l1 *= self.inference_decay
        l2 *= self.inference_decay
        return z1, z2, l1, l2, cost

    def _unpack_sgd(self, outs):
        z1s, z2s, l1s, l2s, costs = outs
        return z1s, z2s, costs

    # Momentum
    def _step_momentum(self, ph1, preact_h1, y,
                       z1, z2, l1, l2, dz1_, dz2_,
                       m, *params):
        cost, (grad1, grad2) = self.inference_cost(ph1, preact_h1, y, z1, z2, *params)
        dz1 = (-l1 * grad1 + m * dz1_).astype(floatX)
        dz2 = (-l2 * grad2 + m * dz2_).astype(floatX)
        z1 = (z1 + dz1).astype(floatX)
        z2 = (z2 + dz2).astype(floatX)
        l1 *= self.inference_decay
        l2 *= self.inference_decay
        return z1, z2, l1, l2, dz1, dz2, cost

    def _init_momentum(self, ph, y, z):
        return [self.inference_rate_1, self.inference_rate_2,
                T.zeros_like(z), T.zeros_like(z)]

    def _unpack_momentum(self, outs):
        z1s, z2s, l1s, l2s, dz1s, dz2s, costs = outs
        return z1s, z2s, costs

    def _params_momentum(self):
        return [T.constant(self.momentum).astype('float32')]

    # ADAM
    def _step_adam(self, ph, preact_h1, y, z1, z2, m1_tm1, v1_tm1, m2_tm1, v2_tm1, cnt, b1, b2, lr, *params):

        b1 = b1 * (1 - 1e-7)**cnt
        cost, (grad1, grad2) = self.inference_cost(ph, preact_h1, y, z1, z2, *params)
        m1_t = b1 * m1_tm1 + (1 - b1) * grad1
        v1_t = b2 * v1_tm1 + (1 - b2) * grad1**2
        m2_t = b1 * m2_tm1 + (1 - b1) * grad2
        v2_t = b2 * v2_tm1 + (1 - b2) * grad2**2
        m1_t_hat = m1_t / (1. - b1**(cnt + 1))
        v1_t_hat = v1_t / (1. - b2**(cnt + 1))
        m2_t_hat = m2_t / (1. - b1**(cnt + 1))
        v2_t_hat = v2_t / (1. - b2**(cnt + 1))
        grad1_t = m1_t_hat / (T.sqrt(v1_t_hat) + 1e-7)
        grad2_t = m2_t_hat / (T.sqrt(v2_t_hat) + 1e-7)
        z1_t = (z1 - lr * grad1_t).astype(floatX)
        z2_t = (z2 - lr * grad2_t).astype(floatX)
        cnt += 1

        return z1_t, z2_t, m1_t, v1_t, m2_t, v2_t, cnt, cost

    def _init_adam(self, ph, y, z):
        return [T.zeros_like(z), T.zeros_like(z), T.zeros_like(z), T.zeros_like(z), 0]

    def _unpack_adam(self, outs):
        z1s, z2s, m1s, v1s, m2s, v2s, cnts, costs = outs
        return z1s, z2s, costs

    def _inference_cost_cg(self, ph1, preact_h1, y, z1, z2,  *params):
        mu1 = eval(self.cond_to_h1.out_act)(preact_h1 + z1)
        #mu1 = T.nnet.sigmoid(z1)
        ph2 = self.p_h2_given_h1(mu1, *params)

        h1 = self.cond_to_h1.sample(mu1)

        mu2 = eval(self.cond_to_h2.out_act)(
            self.preact_h1_to_h2(h1, *params) + z2)
        py = self.p_y_given_h2(mu2, *params)

        cost = (self.cond_to_h1.neg_log_prob(mu1, ph1)
                + self.cond_to_h2.neg_log_prob(mu2, ph2)
                + self.cond_from_h2.neg_log_prob(y, py)
                - self.cond_to_h1.entropy(mu1)
                - self.cond_to_h2.entropy(mu2)
                )
        return cost

     # Conjugate gradient with line search
    def _step_cg(self, ph1, preact_h1, y, z1, z2, s_, dz_sq_, alphas, *params):
        cost, (grad1, grad2) = self.inference_cost(ph1, preact_h1, y, z1, z2, *params)
        grad = T.concatenate([grad1, grad2], axis=1)
        dz = -grad
        dz_sq = (dz * dz).sum(axis=1)
        beta = dz_sq / (dz_sq_ + 1e-8)
        s = dz + beta[:, None] * s_

        z1_alpha = z1[None, :, :] + alphas[:, None, None] * s[None, :, :z1.shape[1]]
        z2_alpha = z2[None, :, :] + alphas[:, None, None] * s[None, :, z1.shape[1]:]
        costs = self._inference_cost_cg(
            ph1[None, :, :], preact_h1[None, :, :], y[None, :, :], z1_alpha, z2_alpha, *params)
        idx = costs.argmin(axis=0)
        z1 = z1 + alphas[idx][:, None] * s[:, :z1.shape[1]]
        z2 = z2 + alphas[idx][:, None] * s[:, z1.shape[1]:]
        return z1, z2, s, dz_sq, cost

    def _init_cg(self, ph, y, z):
        params = self.get_params()
        s0 = T.alloc(0., z.shape[0], 2 * z.shape[1]).astype(floatX)
        dz_sq0 = T.alloc(1., z.shape[0]).astype(floatX)
        return [s0, dz_sq0]

    def _unpack_cg(self, outs):
        z1s, z2s, ss, dz_sqs, costs = outs
        return z1s, z2s, costs

    # Separate CG for z1 and z2
    def _step_cg2(self, ph1, preact_h1, y, z1, z2, s1_, s2_, dz1_sq_, dz2_sq_, alphas, *params):
        cost, (grad1, grad2) = self.inference_cost(ph1, preact_h1, y, z1, z2, *params)
        dz1 = -grad1
        dz2 = -grad2
        dz1_sq = (dz1 * dz1).sum(axis=1)
        dz2_sq = (dz2 * dz2).sum(axis=1)
        beta1 = dz1_sq / (dz1_sq_ + 1e-8)
        beta2 = dz2_sq / (dz2_sq_ + 1e-8)
        s1 = dz1 + beta1[:, None] * s1_
        s2 = dz2 + beta2[:, None] * s2_

        z1_alpha = z1[None, :, :] + alphas[:, None, None] * s1[None, :, :]
        z2_alpha = z2[None, :, :] + alphas[:, None, None] * s2[None, :, :]

        z1_e = T.alloc(0., alphas.shape[0], z1.shape[0], z1.shape[1]).astype(floatX) + z1
        z2_e = T.alloc(0., alphas.shape[0], z2.shape[0], z2.shape[1]).astype(floatX) + z2

        costs2 = self._inference_cost_cg(
            ph1[None, :, :], preact_h1[None, :, :], y[None, :, :], z1_e, z2_alpha, *params)

        costs1 = self._inference_cost_cg(
            ph1[None, :, :], preact_h1[None, :, :], y[None, :, :], z1_alpha, z2_e, *params)

        idx2 = costs2.argmin(axis=0)
        z2 = z2 + alphas[idx2][:, None] * s2
        idx1 = costs1.argmin(axis=0)
        z1 = z1 + alphas[idx1][:, None] * s1
        return z1, z2, s1, s2, dz1_sq, dz2_sq, cost

    def _init_cg2(self, ph, y, z):
        params = self.get_params()
        s10 = T.zeros_like(z)
        s20 = T.zeros_like(z)
        dz1_sq0 = T.alloc(1., z.shape[0]).astype(floatX)
        dz2_sq0 = T.alloc(1., z.shape[0]).astype(floatX)
        return [s10, s20, dz1_sq0, dz2_sq0]

    def _unpack_cg2(self, outs):
        z1s, z2s, s1s, s2s, dz1_sqs, dz2_sqs, costs = outs
        return z1s, z2s, costs

    def _params_cg2(self):
        return [(self.inference_rate_1 * 2. ** T.arange(9)).astype(floatX)]

    def inference(self, x, y, z10=None, z20=None, n_samples=20):
        updates = theano.OrderedUpdates()

        xs, ys = self.init_inputs(x, y, steps=self.n_inference_steps)
        ph1 = self.cond_to_h1(xs)
        preact_h1 = self.cond_to_h1(xs, return_preact=True)

        if z10 is None:
            z10 = self.init_z(xs[0], ys[0])
        if z20 is None:
            z20 = self.init_z(xs[0], ys[0])

        if self.inference_step_scheduler is None:
            n_inference_steps = self.n_inference_steps
        else:
            n_inference_steps, updates_c = self.inference_step_scheduler(n_inference_steps)
            updates.update(updates_c)

        seqs = [ph1, preact_h1, ys]
        outputs_info = [z10, z20] + self.init_infer(ph1[0], ys[0], z10) + [None]
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

        z1s, z2s, i_costs = self.unpack_infer(outs)
        h1_energy, h2_energy, y_energy, h1s, h2s = self.sample_energy(
            ph1[0], preact_h1[0], ys[0], z1s[-1], z2s[-1], n_samples=n_samples)
        return (xs, ys, z1s, z2s, h1_energy, h2_energy, y_energy, i_costs[-1], h1s, h2s), updates

    def __call__(self, x, y, ph1=None, n_samples=100, from_z=False):
        x_n = self.trng.binomial(p=x, size=x.shape, n=1, dtype=x.dtype)
        ph1 = self.cond_to_h1(x_n)
        h1 = self.cond_to_h1.sample(ph1, size=(n_samples, ph1.shape[0], ph1.shape[1]))
        ph2 = self.cond_to_h2(h1)
        h2 = self.cond_to_h2.sample(ph2)
        py = self.cond_from_h2(h2)
        y_energy = self.cond_from_h2.neg_log_prob(y[None, :, :], py)
        y_energy = -log_mean_exp(-y_energy, axis=0).mean()
        return py, y_energy
