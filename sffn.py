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
                 weight_scale=0.1, weight_noise=False,
                 z_init=None, learn_z=False,
                 x_noise_mode=None, y_noise_mode=None, noise_amount=0.1,
                 momentum=0.9,
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
        self.set_params()

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

        if self.z_init == 'x':
            W0 = norm_weight(self.dim_in, self.dim_h, scale=self.weight_scale,
                             rng=self.rng)
            b0 = np.zeros((self.dim_h,)).astype('float32')
            self.params.update(W0=W0, b0=b0)
        elif self.learn_z:
            W0 = norm_weight(self.dim_in, self.dim_h, scale=self.weight_scale,
                             rng=self.rng)
            b0 = np.zeros((self.dim_h,)).astype('float32')
            W1 = norm_weight(self.dim_h, self.dim_h, scale=self.weight_scale,
                             rng=self.rng)
            b1 = np.zeros((self.dim_h,)).astype('float32')
            self.params.update(W0=W0, b0=b0, W1=W1, b1=b1)
        elif self.z_init == 'xy':
            W0 = norm_weight(self.dim_in, self.dim_h, scale=self.weight_scale,
                             rng=self.rng)
            U0 = norm_weight(self.dim_out, self.dim_h, scale=self.weight_scale,
                             rng=self.rng)
            b0 = np.zeros((self.dim_h,)).astype('float32')
            self.params.update(W0=W0, U0=U0, b0=b0)

    def set_tparams(self):
        tparams = super(SFFN, self).set_tparams()
        tparams.update(**self.cond_to_h.set_tparams())
        tparams.update(**self.cond_from_h.set_tparams())

        return tparams

    def initialize_z(self, Z):
        self.z_init = 'from_tensor'
        self.z0 = Z

    def init_z(self, x, y):
        # This needs to be factored out as it's a general intialization of
        # tensor problem.
        if self.z_init == 'average':
            z = T.alloc(0., x.shape[0], self.dim_h).astype(floatX) + self.z[None, :]
            z += self.trng.normal(avg=0, std=0.1, size=(x.shape[0], self.dim_h), dtype=x.dtype)
        elif self.z_init == 'xy':
            z = T.dot(x, self.W0) + T.dot(y, self.U0) + self.b0
        elif self.z_init == 'x':
            z = T.dot(x, self.W0) + self.b0
        elif self.z_init == 'y':
            z = T.dot(y, self.W0) + self.b0
        elif self.z_init == 'noise':
            z = self.trng.normal(avg=0, std=1, size=(x.shape[0], self.dim_h), dtype=x.dtype)
        elif self.z_init is None:
            z = T.alloc(0., x.shape[0], self.dim_h).astype(floatX)
        elif self.z_init == 'from_tensor':
            z = self.z0
        else:
            raise ValueError(self.z_init)
        return z

    def _sample(self, p, size):
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

    def sample_energy(self, x, y, z, n_samples=10):
        mu = T.nnet.sigmoid(z)

        if n_samples == 0:
            h = mu
        else:
            h = self.cond_to_h.sample(mu, size=(n_samples,
                                                mu.shape[0], mu.shape[1]))

        ph = self.cond_to_h(x)
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
                ).mean(axis=0)
        grad = theano.grad(cost, wrt=z, consider_constant=[ph, y])
        return cost, grad

    # SGD
    def _step_sgd(self, ph, y, z, l, *params):
        cost, grad = self.inference_cost(ph, y, z, *params)
        z = (z - ph.shape[0] * l * grad).astype(floatX)
        l *= self.inference_decay
        return z, l, cost

    def _init_sgd(self, z):
        return []

    def _unpack_sgd(self, outs):
        zs, ls, costs = outs
        return zs, costs

    def _params_sgd(self):
        return []

    # Momentum
    def _step_momentum(self, ph, y, z, l, dz_, m, *params):
        cost, grad = self.inference_cost(ph, y, z, *params)
        dz = (-ph.shape[0] * l * grad + m * dz_).astype(floatX)
        z = (z + dz).astype(floatX)
        l *= self.inference_decay
        return z, l, dz, cost

    def _init_momentum(self, z):
        return [T.zeros_like(z)]

    def _unpack_momentum(self, outs):
        zs, ls, dzs, costs = outs
        return zs, costs

    def _params_momentum(self):
        return [T.constant(self.momentum).astype('float32')]

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
        outputs_info = [z0, self.inference_rate] + self.init_infer(z0) + [None]
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
        h_energy, y_energy = self.sample_energy(xs[0], ys[0], zs[-1],
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


class SFFN_2Layer(SFFN):
    def __init__(self, dim_in, dim_h, dim_out, cond_to_h2=None,
                 name='sffn_2layer', **kwargs):

        self.cond_to_h2 = cond_to_h2

        super(SFFN_2Layer, self).__init__(
            dim_in, dim_h, dim_out, name=name, **kwargs)

    def set_params(self):
        super(SFFN_2Layer, self).set_params()
        self.cond_to_h1 = self.cond_to_h
        self.cond_from_h2 = self.cond_from_h
        del self.cond_to_h
        del self.cond_from_h

        self.cond_to_h1.name = 'cond_to_h1'
        self.cond_from_h2.name = 'cond_from_h2'

        if self.cond_to_h2 is None:
            self.cond_to_h2 = MLP(self.dim_h, self.dim_h, self.dim_h, 1,
                                  rng=self.rng, trng=self.trng,
                                  h_act='T.nnet.sigmoid',
                                  out_act='T.nnet.sigmoid')
        self.cond_to_h2.name = 'cond_to_h2'

    def set_tparams(self):
        tparams = super(SFFN, self).set_tparams()
        tparams.update(**self.cond_to_h1.set_tparams())
        tparams.update(**self.cond_to_h2.set_tparams())
        tparams.update(**self.cond_from_h2.set_tparams())

        return tparams

    def get_params(self):
        return (self.cond_to_h1.get_params()
                + self.cond_to_h2.get_params()
                + self.cond_from_h2.get_params())

    def p_h1_given_x(self, x, *params):
        ps = params[:len(self.cond_to_h1.get_params())]
        return self.cond_to_h1.step_call(x, *ps)

    def p_h2_given_h1(self, h1, *params):
        ps = params[len(self.cond_to_h1.get_params()):
            len(self.cond_to_h1.get_params())+
            len(self.cond_to_h2.get_params())]
        return self.cond_to_h1.step_call(h1, *ps)

    def preact_h1_to_h2(self, h1, *params):
        ps = params[len(self.cond_to_h1.get_params()):
            len(self.cond_to_h1.get_params())+
            len(self.cond_to_h2.get_params())]
        return self.cond_to_h1.preact(h1, *ps)

    def p_y_given_h2(self, h2, *params):
        ps = params[-len(self.cond_from_h2.get_params()):]
        return self.cond_from_h2.step_call(h2, *ps)

    def sample_energy(self, x, y, z1, z2, n_samples=10):
        mu1 = T.nnet.sigmoid(z1)
        h1 = self.cond_to_h1.sample(p=mu1, size=(n_samples, mu1.shape[0], mu1.shape[1]))
        mu2 = eval(self.cond_to_h2.out_act)(
            self.cond_to_h2(h1, return_preact=True) + z2[None, :, :])
        h2 = self.cond_to_h2.sample(p=mu2)

        ph1 = self.cond_to_h1(x)
        ph2 = self.cond_to_h2(h1)
        py = self.cond_from_h2(h2)

        h1_energy = self.cond_to_h1.neg_log_prob(h1, ph1[None, :, :])#.mean()
        h1_energy = -log_mean_exp(-h1_energy, axis=0).mean()
        h2_energy = self.cond_to_h2.neg_log_prob(h2, ph2)#.mean()
        h2_energy = -log_mean_exp(-h2_energy, axis=0).mean()
        y_energy = self.cond_from_h2.neg_log_prob(y[None, :, :], py)
        y_energy = -log_mean_exp(-y_energy, axis=0).mean()

        return (h1_energy, h2_energy, y_energy)

    def grad_step(self, z1, z2, x, y, l, *params):
        ph1 = self.p_h1_given_x(x, *params)
        mu1 = T.nnet.sigmoid(z1)
        #h1 = self.trng.binomial(p=mu1, size=mu1.shape, n=1, dtype=mu1.dtype)
        ph2 = self.p_h2_given_h1(mu1, *params)
        mu2 = eval(self.cond_to_h2.out_act)(
            self.preact_h1_to_h2(mu1, *params) + z2)
        #mu2 = T.nnet.sigmoid(z2)
        py = self.p_y_given_h2(mu2, *params)

        cost = (self.cond_to_h1.neg_log_prob(mu1, ph1)
                + self.cond_to_h2.neg_log_prob(mu2, ph2)
                + self.cond_from_h2.neg_log_prob(y, py)
                - self.cond_to_h1.entropy(mu1)
                - self.cond_to_h2.entropy(mu2)
                ).mean()

        grad1, grad2 = theano.grad(cost, wrt=[z1, z2], consider_constant=[x, y])
        z1 = (z1 - x.shape[0] * l * grad1).astype(floatX)
        z2 = (z2 - x.shape[0] * l * grad2).astype(floatX)
        return z1, z2, cost

    def step_infer(self, z1, z2, l, x, y, *params):
        x_n, y_n = self.noise_inputs(x, y, self.inference_noise)

        z1, z2, i_cost = self.grad_step(z1, z2, x_n, y_n, l, *params)
        mu1 = T.nnet.sigmoid(z1)
        mu2 = eval(self.cond_to_h2.out_act)(
            self.preact_h1_to_h2(mu1, *params) + z2)

        ph1 = self.p_h1_given_x(x_n, *params)
        ph2 = self.p_h2_given_h1(mu1, *params)
        py = self.p_y_given_h2(mu2, *params)
        y_hat = self.cond_from_h2.sample(py)

        pd = T.concatenate([x, py], axis=1)
        d_hat = T.concatenate([x, y_hat], axis=1)

        l *= self.inference_decay
        return z1, z2, l, y_hat, pd, d_hat, i_cost

    def inference(self, x, y, m=20):
        updates = theano.OrderedUpdates()

        x_n, y_n = self.noise_inputs(x, y, self.x_init)
        z1 = self.init_z(x, y)
        z2 = self.init_z(x, y)
        mu1 = T.nnet.sigmoid(z1)
        mu2 = eval(self.cond_to_h2.out_act)(
            self.cond_to_h2(mu1, return_preact=True) + z2)
        py = self.cond_from_h2(mu2)
        y_hat = self.cond_from_h2.sample(py)

        l = self.inference_rate
        seqs = []
        outputs_info = [z1, z2, l, None, None, None, None]
        non_seqs = [x, y] + self.get_params()

        (z1s, z2s, ls, y_hats, pds, d_hats, i_costs), updates_2 = theano.scan(
            self.step_infer,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, 'infer'),
            n_steps=self.n_inference_steps,
            profile=tools.profile,
            strict=True
        )
        updates.update(updates_2)

        z1s = T.concatenate([z1[None, :, :], z1s], axis=0)
        z2s = T.concatenate([z2[None, :, :], z2s], axis=0)

        y_hats = T.concatenate([y[None, :, :],
                                y_n[None, :, :],
                                y_hat[None, :, :],
                                y_hats], axis=0)

        d = T.concatenate([x, y], axis=1)
        d_hat = T.concatenate([x, py], axis=1)

        d_hats = T.concatenate([d[None, :, :],
                                d_hat[None, :, :],
                                d_hats], axis=0)

        pds = T.concatenate([d[None, :, :],
                             d_hat[None, :, :],
                             pds], axis=0)

        h1_energy, h2_energy, y_energy = self.sample_energy(x_n, y, z1s[-1],
                                                            z2s[-1],
                                                            n_samples=m)

        if self.z_init == 'average':
            z_mean = zs[-1].mean(axis=0)
            new_z = (1. - self.rate) * self.z + self.rate * z_mean
            updates += [(self.z, new_z)]

        return (z1s, z2s, y_hats, d_hats, pds, h1_energy, h2_energy, y_energy, i_costs[-1]), updates

    def __call__(self, x, y, ph1=None, n_samples=100, from_z=False):
        x_n = self.trng.binomial(p=x, size=x.shape, n=1, dtype=x.dtype)

        if ph1 is not None:
            pass
        elif from_z:
            assert self.learn_z
            zh = T.tanh(T.dot(x_n, self.W0) + self.b0)
            z1 = T.dot(zh, self.W1) + self.b1
            ph1 = T.nnet.sigmoid(z1)
        else:
            ph1 = self.cond_to_h1(x)

        h1 = self.cond_to_h1.sample(ph1, size=(n_samples, ph1.shape[0], ph1.shape[1]))

        ph2 = self.cond_to_h2(h1)
        h2 = self.cond_to_h1.sample(ph2)

        py = self.cond_from_h2(h2)
        x_e = T.zeros_like(py) + x[None, :, :]
        pd = T.concatenate([x_e, py], axis=2)

        y_energy = self.cond_from_h2.neg_log_prob(y[None, :, :], py)
        y_energy = -log_mean_exp(-y_energy, axis=0).mean()

        log_py = T.log(py).mean(axis=0)
        py = T.exp(log_py)

        y_hat = self.cond_from_h2.sample(py)
        d_hat = T.concatenate([x, y_hat], axis=1)

        pd = pd[:10]

        return y_hat, py, y_energy, pd, d_hat
