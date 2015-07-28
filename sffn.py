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


norm_weight = tools.norm_weight
ortho_weight = tools.ortho_weight
floatX = 'float32' # theano.config.floatX


class SFFN(Layer):
    def __init__(self, dim_in, dim_h, dim_out, rng=None, trng=None,
                 cond_to_h=None, cond_from_h=None,
                 weight_scale=0.1, weight_noise=False, noise=False, z_init=None,
                 learn_z=False, noise_mode=None, name='sffn'):
        if trng is None:
            self.trng = RandomStreams(6 * 10 * 2015)
        else:
            self.trng = trng

        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out

        self.weight_noise = weight_noise
        self.weight_scale = weight_scale
        self.noise = noise
        self.noise_mode = noise_mode
        self.z_init = z_init
        self.learn_z = learn_z
        self.cond_to_h = cond_to_h
        self.cond_from_h = cond_from_h

        if rng is None:
            rng = tools.rng_
        self.rng = rng

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

        self.cond_to_h.name = 'cond_to_h'
        self.cond_from_h.name = 'cond_from_h'

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

    def init_z(self, x, y):
        if self.z_init == 'average':
            z = T.alloc(0., x.shape[0], self.dim_h).astype(floatX) + self.z[None, :]
            z += self.trng.normal(avg=0, std=0.1, size=(x.shape[0], self.dim_h), dtype=x.dtype)
        elif self.z_init == 'xy':
            z = T.dot(x, self.W0) + T.dot(y, self.U0) + self.b0
            #z += self.trng.normal(avg=0, std=0.1, size=(x.shape[0], self.dim_h), dtype=x.dtype)
        elif self.z_init == 'x':
            z = T.dot(x, self.W0) + self.b0
            #z += self.trng.normal(avg=0, std=0.1, size=(x.shape[0], self.dim_h), dtype=x.dtype)
        elif self.z_init == 'y':
            z = T.dot(y, self.W0) + self.b0
        elif self.z_init == 'noise':
            z = self.trng.normal(avg=0, std=1, size=(x.shape[0], self.dim_h), dtype=x.dtype)
        elif self.z_init is None:
            z = T.alloc(0., x.shape[0], self.dim_h).astype(floatX)
        else:
            raise ValueError(self.z_init)

        return z

    def noise_inputs(self, x, y, noise_mode):
        if noise_mode == 'x':
            x_n = x * (1 - self.trng.binomial(p=self.noise, size=x.shape, n=1,
                                            dtype=x.dtype))
            y_n = T.zeros_like(y) + y
        if noise_mode == 'noise_all':
            x_n = x * (1 - self.trng.binomial(p=self.noise, size=x.shape, n=1,
                                              dtype=x.dtype))
            y_n = y * (1 - self.trng.binomial(p=self.noise, size=y.shape, n=1,
                                              dtype=y.dtype))
        elif noise_mode == 'sample':
            x_n = self.trng.binomial(p=x, size=x.shape, n=1, dtype=x.dtype)
            y_n = self.trng.binomial(p=y, size=y.shape, n=1, dtype=y.dtype)
        elif noise_mode is None:
            x_n = T.zeros_like(x) + x
            y_n = T.zeros_like(y) + y
        else:
            raise ValueError(noise_mode)

        return x_n, y_n

    def get_params(self):
        return self.cond_to_h.get_params() + self.cond_from_h.get_params()

    def p_h_given_x(self, x, *params):
        ps = params[:len(self.cond_to_h.get_params())]
        return self.cond_to_h.step_call(x, *ps)

    def p_y_given_h(self, h, *params):
        ps = params[len(self.cond_to_h.get_params()):]
        return self.cond_from_h.step_call(h, *ps)

    def energy(self, x, p):
        return -(x * T.log(p + 1e-8) +
                (1 - x) * T.log(1 - p + 1e-8)).sum(axis=1)

    def energy_step(self, x, y, z, *params):
        mu = T.nnet.sigmoid(z)
        h = self.trng.binomial(p=mu, size=mu.shape, n=1, dtype=mu.dtype)
        ph = self.p_h_given_x(x, *params)
        py = self.p_y_given_h(h, *params)

        h_energy = self.energy(h, ph)
        y_energy = self.energy(y, py)
        return h_energy, y_energy

    def sample_energy(self, x, y, z, n_samples=10):
        seqs = []
        outputs_info = [None, None]
        non_seqs = [x, y, z] + self.get_params()

        (h_energies, y_energies), updates = theano.scan(
            self.energy_step,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, 'energy'),
            n_steps=n_samples,
            profile=tools.profile,
            strict=True
        )

        h_energy = h_energies.mean()
        y_energy = y_energies.mean()
        return (h_energy, y_energy), updates

    def grad_step(self, z, x, y, l, *params):
        ph = self.p_h_given_x(x, *params)
        mu = T.nnet.sigmoid(z)
        py = self.p_y_given_h(mu, *params)

        cost = (self.energy(y, py)
                - (mu * T.log(ph)
                   + (1 - mu) * T.log(1 - ph)).sum(axis=1)
                + (mu * T.log(mu)
                   + (1 - mu) * T.log(1 - mu)).sum(axis=1)
                ).sum()

        grad = theano.grad(cost, wrt=z, consider_constant=[x, y])
        z = z - l * grad
        return z

    def step_infer(self, z, x, y, l, *params):
        x_n, y_n = self.noise_inputs(x, y, self.noise_mode)

        z = self.grad_step(z, x_n, y_n, l, *params)
        mu = T.nnet.sigmoid(z)

        ph = self.p_h_given_x(x_n, *params)
        py = self.p_y_given_h(mu, *params)
        y_hat = self.trng.binomial(p=py, size=py.shape, n=1, dtype=py.dtype)

        pd = T.concatenate([x, py], axis=1)
        d_hat = T.concatenate([x, y_hat], axis=1)

        return z, y_hat, pd, d_hat

    def inference(self, x, y, l, n_inference_steps=1, m=20):
        updates = theano.OrderedUpdates()

        x_n, y_n = self.noise_inputs(x, y, self.noise_mode)
        z = self.init_z(x, y)
        mu = T.nnet.sigmoid(z)
        py = self.cond_from_h(mu)
        y_hat = self.trng.binomial(p=py, size=py.shape, n=1, dtype=py.dtype)

        seqs = []
        outputs_info = [z, None, None, None]
        non_seqs = [x, y, l] + self.get_params()

        (zs, y_hats, pds, d_hats), updates_2 = theano.scan(
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

        zs = T.concatenate([z[None, :, :], zs], axis=0)

        y_hats = T.concatenate([y[None, :, :],
                                y_n[None, :, :],
                                y_hat[None, :, :],
                                y_hats], axis=0)

        d = T.concatenate([x, y], axis=1)
        d_hat = T.concatenate([x, y_hat], axis=1)

        d_hats = T.concatenate([d[None, :, :],
                                d_hat[None, :, :],
                                d_hats], axis=0)

        pds = T.concatenate([d[None, :, :],
                             d_hat[None, :, :],
                             pds], axis=0)

        (h_energy, y_energy), updates_3 = self.sample_energy(x_n, y, zs[-1], n_samples=m)
        updates.update(updates_3)

        if self.z_init == 'average':
            z_mean = zs[-1].mean(axis=0)
            new_z = (1. - self.rate) * self.z + self.rate * z_mean
            updates += [(self.z, new_z)]

        return (zs, y_hats, d_hats, pds, h_energy, y_energy), updates

    def step_sample(self, p, x, *params):
        h = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)
        py = self.p_y_given_h(h, *params)
        pd = T.concatenate([x, py], axis=1)

        return py, pd

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

        seqs = []
        outputs_info = [None, None]
        non_seqs = [ph, x] + self.get_params()

        (pys, pds), updates = theano.scan(
            self.step_sample,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, 'samples'),
            n_steps=n_samples,
            profile=tools.profile,
            strict=True
        )
        log_py = T.log(pys).mean(axis=0)
        py = T.exp(log_py)

        y_hat = self.trng.binomial(p=py, size=py.shape, n=1, dtype=py.dtype)
        d_hat = T.concatenate([x, y_hat], axis=1)

        y_energy = self.energy(y, py).mean()

        pds = pds[:10]

        return (y_hat, y_energy, pds, d_hat), updates


class DSFFN(Layer):
    def __init__(self, dim_in, dim_h, dim_out, rng=None, trng=None,
                 weight_scale=0.1, weight_noise=False, noise=False, z_init=None,
                 learn_z=False, noise_mode=None, name='sffn'):
        if trng is None:
            self.trng = RandomStreams(6 * 10 * 2015)
        else:
            self.trng = trng

        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out

        self.weight_noise = weight_noise
        self.weight_scale = weight_scale
        self.noise = noise
        self.noise_mode = noise_mode
        self.z_init = z_init
        self.learn_z = learn_z

        if rng is None:
            rng = tools.rng_
        self.rng = rng

        super(DSFFN, self).__init__(name=name)
        self.set_params()

    def set_params(self):
        XH1 = norm_weight(self.dim_in, self.dim_h)
        bh1 = np.zeros((self.dim_h,)).astype(floatX)
        XH2 = norm_weight(self.dim_h, self.dim_h)
        bh2 = np.zeros((self.dim_h,)).astype(floatX)
        HY = norm_weight(self.dim_h, self.dim_out)
        by = np.zeros((self.dim_out,)).astype(floatX)

        self.params = OrderedDict(XH1=XH1, bh1=bh1, XH2=XH2, bh2=bh2,
                                  HY=HY, by=by)

        if self.weight_noise:
            assert False
            XH_noise = (XH * 0).astype(floatX)
            HY_noise = (HY * 0).astype(floatX)
            self.params.update(XH_noise=XH_noise, HX_noise=HY_noise)

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

    def init_z(self, x, y):
        if self.z_init == 'average':
            z = T.alloc(0., x.shape[0], self.dim_h).astype(floatX) + self.z[None, :]
            z += self.trng.normal(avg=0, std=0.1, size=(x.shape[0], self.dim_h), dtype=x.dtype)
        elif self.z_init == 'xy':
            z = T.dot(x, self.W0) + T.dot(y, self.U0) + self.b0
            #z += self.trng.normal(avg=0, std=0.1, size=(x.shape[0], self.dim_h), dtype=x.dtype)
        elif self.z_init == 'x':
            z = T.dot(x, self.W0) + self.b0
            #z += self.trng.normal(avg=0, std=0.1, size=(x.shape[0], self.dim_h), dtype=x.dtype)
        elif self.z_init == 'y':
            z = T.dot(y, self.W0) + self.b0
        elif self.z_init == 'noise':
            z = self.trng.normal(avg=0, std=1, size=(x.shape[0], self.dim_h), dtype=x.dtype)
        elif self.z_init is None:
            z = T.alloc(0., x.shape[0], self.dim_h).astype(floatX)
        else:
            raise ValueError(self.z_init)

        return z

    def init_inputs(self, x, y, noise_mode):
        if noise_mode == 'x':
            x_n = x * (1 - self.trng.binomial(p=self.noise, size=x.shape, n=1,
                                            dtype=x.dtype))
            y_n = T.zeros_like(y) + y
        if noise_mode == 'noise_all':
            x_n = x * (1 - self.trng.binomial(p=self.noise, size=x.shape, n=1,
                                              dtype=x.dtype))
            y_n = y * (1 - self.trng.binomial(p=self.noise, size=y.shape, n=1,
                                              dtype=y.dtype))
        elif noise_mode == 'sample':
            x_n = self.trng.binomial(p=x, size=x.shape, n=1, dtype=x.dtype)
            y_n = self.trng.binomial(p=y, size=y.shape, n=1, dtype=y.dtype)
        elif noise_mode is None:
            x_n = T.zeros_like(x) + x
            y_n = T.zeros_like(y) + y
        else:
            raise ValueError(noise_mode)

        return x_n, y_n

    def energy(self, x, p):
        return -(x * T.log(p + 1e-7) +
                (1 - x) * T.log(1 - p + 1e-7)).sum(axis=1)

    def energy_step(self, x, y, z,
                    XH1, bh1, XH2, bh2,
                    HY, by):
        mu = T.nnet.sigmoid(z)
        h = self.trng.binomial(p=mu, size=mu.shape, n=1, dtype=mu.dtype)

        hx = T.tanh(T.dot(x, XH1) + bh1)
        ph = T.nnet.sigmoid(T.dot(hx, XH2) + bh2)
        py = T.nnet.sigmoid(T.dot(h, HY) + by)

        h_energy = self.energy(h, ph)
        y_energy = self.energy(y, py)
        return h_energy, y_energy

    def sample_energy(self, x, y, z, n_samples=10):
        seqs = []
        outputs_info = [None, None]
        non_seqs = [x, y, z,
                    self.XH1, self.bh1, self.XH2, self.bh2,
                    self.HY, self.by]

        (h_energies, y_energies), updates = theano.scan(
            self.energy_step,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, 'energy'),
            n_steps=n_samples,
            profile=tools.profile,
            strict=True
        )

        h_energy = h_energies.mean(axis=0).mean(axis=0)
        y_energy = y_energies.mean(axis=0).mean(axis=0)
        return (h_energy, y_energy), updates

    def grad_step(self, z, x, y, l,
                  XH1, bh1, XH2, bh2,
                  HY, by):
        hx = T.tanh(T.dot(x, XH1) + bh1)
        ph = T.nnet.sigmoid(T.dot(hx, XH2) + bh2)

        mu = T.nnet.sigmoid(z)
        mu_c = T.zeros_like(mu) + mu
        py = T.nnet.sigmoid(T.dot(mu, HY) + by)

        #energy = (self.energy(y, py) + self.energy(mu, ph)).sum()
        energy = (self.energy(y, py) - mu * T.log(ph) + (1 - mu) * T.log(1 - ph) + mu * T.log(mu) + (1 - mu) * T.log(1 - mu)).sum()
        #entropy = -(mu * T.log(mu)).sum()
        cost = energy# + entropy
        grad = theano.grad(cost, wrt=z, consider_constant=[x, y, mu_c])

        z = z - l * grad
        return z

    def step_infer(self, z, x, y, l,
                   XH1, bh1, XH2, bh2,
                   HY, by):
        if self.noise_mode in ['x', 'noise_all']:
            x_n = x * (1 - self.trng.binomial(p=self.noise, size=x.shape, n=1,
                                            dtype=x.dtype))
            y_n = y * (1 - self.trng.binomial(p=self.noise, size=y.shape, n=1,
                                            dtype=y.dtype))
        elif self.noise_mode == 'sample':
            x_n = self.trng.binomial(p=x, size=x.shape, n=1, dtype=x.dtype)
            y_n = self.trng.binomial(p=y, size=y.shape, n=1, dtype=y.dtype)
        else:
            x_n = T.zeros_like(x) + x
            y_n = T.zeros_like(y) + y

        z = self.grad_step(z, x_n, y_n, l,
                           XH1, bh1, XH2, bh2,
                           HY, by)
        mu = T.nnet.sigmoid(z)

        hx = T.tanh(T.dot(x, XH1) + bh1)
        ph = T.nnet.sigmoid(T.dot(hx, XH2) + bh2)
        py = T.nnet.sigmoid(T.dot(mu, HY) + by)

        y_hat = self.trng.binomial(p=py, size=py.shape, n=1, dtype=py.dtype)

        pd = T.concatenate([x, py], axis=1)
        d_hat = T.concatenate([x, y_hat], axis=1)

        return z, y_hat, pd, d_hat

    def inference(self, x, y, l, n_inference_steps=1, m=20):
        updates = theano.OrderedUpdates()

        x_n, y_n = self.init_inputs(x, y, self.noise_mode)
        z = self.init_z(x, y)
        mu = T.nnet.sigmoid(z)
        py = T.nnet.sigmoid(T.dot(mu, self.HY) + self.by)
        y_hat = self.trng.binomial(p=py, size=py.shape, n=1, dtype=py.dtype)

        seqs = []
        outputs_info = [z, None, None, None]
        non_seqs = [x, y, l,
                    self.XH1, self.bh1, self.XH2, self.bh2,
                    self.HY, self.by]

        (zs, y_hats, pds, d_hats), updates_2 = theano.scan(
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

        zs = T.concatenate([z[None, :, :], zs], axis=0)

        y_hats = T.concatenate([y[None, :, :],
                                y_n[None, :, :],
                                y_hat[None, :, :],
                                y_hats], axis=0)

        d = T.concatenate([x, y], axis=1)
        d_hat = T.concatenate([x, y_hat], axis=1)

        d_hats = T.concatenate([d[None, :, :],
                                d_hat[None, :, :],
                                d_hats], axis=0)

        pds = T.concatenate([d[None, :, :],
                             d_hat[None, :, :],
                             pds], axis=0)

        (h_energy, y_energy), updates_3 = self.sample_energy(x_n, y, zs[-1], n_samples=m)
        updates.update(updates_3)

        if self.z_init == 'average':
            z_mean = zs[-1].mean(axis=0)
            new_z = (1. - self.rate) * self.z + self.rate * z_mean
            updates += [(self.z, new_z)]

        return (zs, y_hats, d_hats, pds, h_energy, y_energy), updates

    def step_sample(self, p, x, HY, by):
        h = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)
        py = T.nnet.sigmoid(T.dot(h, HY) + by)
        pd = T.concatenate([x, py], axis=1)

        return py, pd

    def __call__(self, x, y, ph=None, n_samples=100, from_z=False):
        x_n = self.trng.binomial(p=x, size=x.shape, n=1, dtype=x.dtype)

        if ph is not None:
            pass
        if from_z:
            assert self.learn_z
            zh = T.tanh(T.dot(x_n, self.W0) + self.b0)
            z = T.dot(zh, self.W1) + self.b1
            ph = T.nnet.sigmoid(z)
        else:
            hx = T.tanh(T.dot(x, self.XH1) + self.bh1)
            ph = T.nnet.sigmoid(T.dot(hx, self.XH2) + self.bh2)

        seqs = []
        outputs_info = [None, None]
        non_seqs = [ph, x, self.HY, self.by]

        (pys, pds), updates = theano.scan(
            self.step_sample,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, 'samples'),
            n_steps=n_samples,
            profile=tools.profile,
            strict=True
        )
        log_py = T.log(pys).mean(axis=0)
        py = T.exp(log_py)

        y_hat = self.trng.binomial(p=py, size=py.shape, n=1, dtype=py.dtype)
        d_hat = T.concatenate([x, y_hat], axis=1)

        y_energy = self.energy(y, py).mean()

        pds = pds[:10]

        return (y_energy, pds, d_hat), updates


class SFFN_2Layer(SFFN):
    def __init__(self, dim_in, dim_h, dim_out, name='sffn_2layer', **kwargs):

        super(SFFN_2Layer, self).__init__(dim_in, dim_h, dim_out, name=name,
                                          **kwargs)

    def set_params(self):
        super(SFFN_2Layer, self).set_params()

        HH = norm_weight(self.dim_h, self.dim_h)
        bhh = np.zeros((self.dim_h,)).astype(floatX)

        self.params.update(HH=HH, bhh=bhh)

        if self.weight_noise:
            HH_noise = (HH * 0).astype(floatX)
            self.params.update(HH_noise=HH_noise)

    def energy_step(self, x, y, z1, z2, XH, bh, HH, bhh, HY, by):
        mu1 = T.nnet.sigmoid(z1)
        h1 = self.trng.binomial(p=mu1, size=mu1.shape, n=1, dtype=mu1.dtype)

        mu2 = T.nnet.sigmoid(z2)
        h2 = self.trng.binomial(p=mu2, size=mu2.shape, n=1, dtype=mu2.dtype)

        ph1 = T.nnet.sigmoid(T.dot(x, XH) + bh)
        ph2 = T.nnet.sigmoid(T.dot(h1, HH) + bhh)
        py = T.nnet.sigmoid(T.dot(h2, HY) + by)

        h1_energy = self.energy(h1, ph1)
        h2_energy = self.energy(h2, ph2)
        y_energy = self.energy(y, py)
        return h1_energy, h2_energy, y_energy

    def sample_energy(self, x, y, z1, z2, n_samples=10):
        seqs = []
        outputs_info = [None, None, None]
        non_seqs = [x, y, z1, z2, self.XH, self.bh, self.HH, self.bhh, self.HY, self.by]

        (h1_energies, h2_energies, y_energies), updates = theano.scan(
            self.energy_step,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, 'energy'),
            n_steps=n_samples,
            profile=tools.profile,
            strict=True
        )

        h1_energy = h1_energies.mean(axis=0).mean(axis=0)
        h2_energy = h2_energies.mean(axis=0).mean(axis=0)
        y_energy = y_energies.mean(axis=0).mean(axis=0)
        return (h1_energy, h2_energy, y_energy), updates

    def grad_step(self, z1, z2, x, y, l, XH, bh, HH, bhh, HY, by):
        ph1 = T.nnet.sigmoid(T.dot(x, XH) + bh)
        mu1 = T.nnet.sigmoid(z1)

        ph2 = T.nnet.sigmoid(T.dot(mu1, HH) + bhh)
        mu2 = T.nnet.sigmoid(z2)

        py = T.nnet.sigmoid(T.dot(mu2, HY) + by)

        energy = (self.energy(mu1, ph1) + self.energy(mu2, ph2) + self.energy(y, py)).sum()
        #entropy = -(mu * T.log(mu)).sum()
        cost = energy# + entropy
        grad = theano.grad(cost, wrt=[z1, z2], consider_constant=[x, y])

        z1 = z1 - l * grad[0]
        z2 = z2 - l * grad[1]
        return z1, z2

    def step_infer(self, z1, z2, x, y, l, XH, bh, HH, bhh, HY, by):
        if self.noise_mode in ['x', 'noise_all']:
            x_n = x * (1 - self.trng.binomial(p=self.noise, size=x.shape, n=1,
                                            dtype=x.dtype))
            y_n = y * (1 - self.trng.binomial(p=self.noise, size=y.shape, n=1,
                                            dtype=y.dtype))
        elif self.noise_mode == 'sample':
            x_n = self.trng.binomial(p=x, size=x.shape, n=1, dtype=x.dtype)
            y_n = self.trng.binomial(p=y, size=y.shape, n=1, dtype=y.dtype)
        else:
            x_n = T.zeros_like(x) + x
            y_n = T.zeros_like(y) + y

        z1, z2 = self.grad_step(z1, z2, x_n, y_n, l, XH, bh, HH, bhh, HY, by)
        mu1 = T.nnet.sigmoid(z1)
        mu2 = T.nnet.sigmoid(z2)

        py = T.nnet.sigmoid(T.dot(mu2, HY) + by)
        ph = T.nnet.sigmoid(T.dot(x, XH) + bh)
        y_hat = self.trng.binomial(p=py, size=py.shape, n=1, dtype=py.dtype)

        pd = T.concatenate([x, py], axis=1)
        d_hat = T.concatenate([x, y_hat], axis=1)

        return z1, z2, y_hat, pd, d_hat

    def inference(self, x, y, l, n_inference_steps=1, noise_mode='noise_all', m=20):
        updates = theano.OrderedUpdates()

        x_n, y_n = self.init_inputs(x, y, noise_mode)
        z1 = self.init_z(x, y)
        z2 = self.init_z(x, y)
        mu2 = T.nnet.sigmoid(z2)
        h2 = self.trng.binomial(p=mu2, size=mu2.shape, n=1, dtype=mu2.dtype)
        p = T.nnet.sigmoid(T.dot(h2, self.HY) + self.by)
        y_hat = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)

        seqs = []
        outputs_info = [z1, z2, None, None, None]
        non_seqs = [x, y, l, self.XH, self.bh, self.HH, self.bhh, self.HY, self.by,]

        (z1s, z2s, y_hats, pds, d_hats), updates_2 = theano.scan(
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

        z1s = T.concatenate([z1[None, :, :], z1s], axis=0)
        z2s = T.concatenate([z2[None, :, :], z2s], axis=0)

        y_hats = T.concatenate([y[None, :, :],
                                y_n[None, :, :],
                                y_hat[None, :, :],
                                y_hats], axis=0)

        d = T.concatenate([x, y], axis=1)
        d_hat = T.concatenate([x, y_hat], axis=1)

        d_hats = T.concatenate([d[None, :, :],
                                d_hat[None, :, :],
                                d_hats], axis=0)

        pds = T.concatenate([d[None, :, :],
                             d_hat[None, :, :],
                             pds], axis=0)

        (h1_energy, h2_energy, y_energy), updates_3 = self.sample_energy(x_n, y, z1s[-1], z2s[-1], n_samples=m)
        updates.update(updates_3)

        if self.z_init == 'average':
            z_mean = zs[-1].mean(axis=0)
            new_z = (1. - self.rate) * self.z + self.rate * z_mean
            updates += [(self.z, new_z)]

        return (z1s, z2s, y_hats, d_hats, pds, h1_energy, h2_energy, y_energy), updates

    def __call__(self, x, y, from_z=False):
        if from_z:
            assert False
            zh = T.tanh(T.dot(x, self.W0) + self.b0)
            z = T.dot(zh, self.W1) + self.b1
            ph = T.nnet.sigmoid(z)
        else:
            ph1 = T.nnet.sigmoid(T.dot(x, self.XH) + self.bh)

        h1 = self.trng.binomial(p=ph1, size=ph1.shape, n=1, dtype=ph1.dtype)

        ph2 = T.nnet.sigmoid(T.dot(h1, self.HH) + self.bhh)
        h2 = self.trng.binomial(p=ph2, size=ph2.shape, n=1, dtype=ph2.dtype)

        py = T.nnet.sigmoid(T.dot(h2, self.HY) + self.by)
        y_hat = self.trng.binomial(p=py, size=py.shape, n=1, dtype=py.dtype)
        d_hat = T.concatenate([x, y_hat], axis=1)
        pd = T.concatenate([x, py], axis=1)

        y_energy = self.energy(y, py).mean()

        return y_hat, y_energy, pd, d_hat


class SFFN_2Layer2(SFFN):
    def __init__(self, dim_in, dim_h, dim_out, name='sffn_2layer', **kwargs):

        super(SFFN_2Layer2, self).__init__(dim_in, dim_h, dim_out, name=name,
                                          **kwargs)

    def set_params(self):
        super(SFFN_2Layer2, self).set_params()

        HH = norm_weight(self.dim_h, self.dim_h)
        bhh = np.zeros((self.dim_h,)).astype(floatX)

        self.params.update(HH=HH, bhh=bhh)

        if self.weight_noise:
            HH_noise = (HH * 0).astype(floatX)
            self.params.update(HH_noise=HH_noise)

    def energy_step(self, x, y, z, XH, bh, HH, bhh, HY, by):
        mu = T.nnet.sigmoid(z)
        h1 = self.trng.binomial(p=mu, size=mu.shape, n=1, dtype=mu.dtype)

        ph2 = T.nnet.sigmoid(T.dot(h1, HH) + bhh)
        h2 = self.trng.binomial(p=ph2, size=ph2.shape, n=1, dtype=ph2.dtype)

        ph1 = T.nnet.sigmoid(T.dot(x, XH) + bh)
        py = T.nnet.sigmoid(T.dot(h2, HY) + by)

        h_energy = self.energy(h1, ph1)
        y_energy = self.energy(y, py)
        return h_energy, y_energy

    def sample_energy(self, x, y, z, n_samples=10):
        seqs = []
        outputs_info = [None, None]
        non_seqs = [x, y, z, self.XH, self.bh, self.HH, self.bhh, self.HY, self.by]

        (h_energies, y_energies), updates = theano.scan(
            self.energy_step,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, 'energy'),
            n_steps=n_samples,
            profile=tools.profile,
            strict=True
        )

        h_energy = h_energies.mean(axis=0).mean(axis=0)
        y_energy = y_energies.mean(axis=0).mean(axis=0)
        return (h_energy, y_energy), updates

    def grad_step(self, z, x, y, l, XH, bh, HH, bhh, HY, by):
        ph1 = T.nnet.sigmoid(T.dot(x, XH) + bh)
        mu1 = T.nnet.sigmoid(z)

        ph2 = T.nnet.sigmoid(T.dot(mu1, HH) + bhh)

        py = T.nnet.sigmoid(T.dot(ph2, HY) + by)

        energy = (self.energy(mu1, ph1) + self.energy(y, py)).sum()
        #entropy = -(mu * T.log(mu)).sum()
        cost = energy# + entropy
        grad = theano.grad(cost, wrt=z, consider_constant=[x, y])

        z = z - l * grad
        return z

    def step_infer(self, z, x, y, l, XH, bh, HH, bhh, HY, by):
        if self.noise_mode in ['x', 'noise_all']:
            x_n = x * (1 - self.trng.binomial(p=self.noise, size=x.shape, n=1,
                                            dtype=x.dtype))
            y_n = y * (1 - self.trng.binomial(p=self.noise, size=y.shape, n=1,
                                            dtype=y.dtype))
        elif self.noise_mode == 'sample':
            x_n = self.trng.binomial(p=x, size=x.shape, n=1, dtype=x.dtype)
            y_n = self.trng.binomial(p=y, size=y.shape, n=1, dtype=y.dtype)
        else:
            x_n = T.zeros_like(x) + x
            y_n = T.zeros_like(y) + y

        z = self.grad_step(z, x_n, y_n, l, XH, bh, HH, bhh, HY, by)
        mu1 = T.nnet.sigmoid(z)
        ph2 = T.nnet.sigmoid(T.dot(mu1, HH) + bhh)

        py = T.nnet.sigmoid(T.dot(ph2, HY) + by)
        ph = T.nnet.sigmoid(T.dot(x, XH) + bh)
        y_hat = self.trng.binomial(p=py, size=py.shape, n=1, dtype=py.dtype)

        pd = T.concatenate([x, py], axis=1)
        d_hat = T.concatenate([x, y_hat], axis=1)

        return z, y_hat, pd, d_hat

    def inference(self, x, y, l, n_inference_steps=1, noise_mode='noise_all', m=20):
        updates = theano.OrderedUpdates()

        x_n, y_n = self.init_inputs(x, y, noise_mode)
        z = self.init_z(x, y)
        mu1 = T.nnet.sigmoid(z)
        ph2 = T.nnet.sigmoid(T.dot(mu1, self.HH) + self.bhh)
        h2 = self.trng.binomial(p=ph2, size=ph2.shape, n=1, dtype=ph2.dtype)
        p = T.nnet.sigmoid(T.dot(h2, self.HY) + self.by)
        y_hat = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)

        seqs = []
        outputs_info = [z, None, None, None]
        non_seqs = [x, y, l, self.XH, self.bh, self.HH, self.bhh, self.HY, self.by,]

        (zs, y_hats, pds, d_hats), updates_2 = theano.scan(
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

        zs = T.concatenate([z[None, :, :], zs], axis=0)

        y_hats = T.concatenate([y[None, :, :],
                                y_n[None, :, :],
                                y_hat[None, :, :],
                                y_hats], axis=0)

        d = T.concatenate([x, y], axis=1)
        d_hat = T.concatenate([x, y_hat], axis=1)

        d_hats = T.concatenate([d[None, :, :],
                                d_hat[None, :, :],
                                d_hats], axis=0)

        pds = T.concatenate([d[None, :, :],
                             d_hat[None, :, :],
                             pds], axis=0)

        (h_energy, y_energy), updates_3 = self.sample_energy(x_n, y, zs[-1], n_samples=m)
        updates.update(updates_3)

        if self.z_init == 'average':
            z_mean = zs[-1].mean(axis=0)
            new_z = (1. - self.rate) * self.z + self.rate * z_mean
            updates += [(self.z, new_z)]

        return (zs, y_hats, d_hats, pds, h_energy, y_energy), updates

    def __call__(self, x, y, from_z=False):
        if from_z:
            assert self.learn_z
            zh = T.tanh(T.dot(x, self.W0) + self.b0)
            z = T.dot(zh, self.W1) + self.b1
            ph1 = T.nnet.sigmoid(z)
        else:
            ph1 = T.nnet.sigmoid(T.dot(x, self.XH) + self.bh)

        h1 = self.trng.binomial(p=ph1, size=ph1.shape, n=1, dtype=ph1.dtype)

        ph2 = T.nnet.sigmoid(T.dot(h1, self.HH) + self.bhh)
        h2 = self.trng.binomial(p=ph2, size=ph2.shape, n=1, dtype=ph2.dtype)

        py = T.nnet.sigmoid(T.dot(h2, self.HY) + self.by)
        y_hat = self.trng.binomial(p=py, size=py.shape, n=1, dtype=py.dtype)
        d_hat = T.concatenate([x, y_hat], axis=1)
        pd = T.concatenate([x, py], axis=1)

        y_energy = self.energy(y, py).mean()

        return y_hat, y_energy, pd, d_hat
