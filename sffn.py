'''
Module of Stochastic Feed Forward Networks
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from layers import Layer
import tools


norm_weight = tools.norm_weight
ortho_weight = tools.ortho_weight
floatX = 'float32' # theano.config.floatX


class SFFN(Layer):
    def __init__(self, dim_in, dim_h, dim_out, rng=None, trng=None,
                 weight_scale=0.1, weight_noise=False, noise=False, z_init=None,
                 noise_mode=None, name='sffn'):
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

        if rng is None:
            rng = tools.rng_
        self.rng = rng

        super(SFFN, self).__init__(name=name)
        self.set_params()

    def set_params(self):
        XH = norm_weight(self.dim_in, self.dim_h)
        bh = np.zeros((self.dim_h,)).astype(floatX)
        HY = norm_weight(self.dim_h, self.dim_out)
        by = np.zeros((self.dim_out,)).astype(floatX)

        self.params = OrderedDict(XH=XH, bh=bh, HY=HY, by=by)

        if self.weight_noise:
            XH_noise = (XH * 0).astype(floatX)
            HY_noise = (HY * 0).astype(floatX)
            self.params.update(XH_noise=XH_noise, HX_noise=HY_noise)

        if self.z_init == 'x':
            W0 = norm_weight(self.dim_in, self.dim_h, scale=self.weight_scale,
                             rng=self.rng)
            b0 = np.zeros((self.dim_h,)).astype('float32')
            self.params.update(W0=W0, b0=b0)
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

    def energy_step(self, x, y, z, XH, bh, HY, by):
        mu = T.nnet.sigmoid(z)
        h = self.trng.binomial(p=mu, size=mu.shape, n=1, dtype=mu.dtype)

        ph = T.nnet.sigmoid(T.dot(x, XH) + bh)
        py = T.nnet.sigmoid(T.dot(h, HY) + by)

        h_energy = self.energy(h, ph)
        y_energy = self.energy(y, py)
        return h_energy, y_energy

    def sample_energy(self, x, y, z, n_samples=10):
        seqs = []
        outputs_info = [None, None]
        non_seqs = [x, y, z, self.XH, self.bh, self.HY, self.by]

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

    def grad_step(self, z, x, y, l, XH, bh, HY, by):
        ph = T.nnet.sigmoid(T.dot(x, XH) + bh)

        mu = T.nnet.sigmoid(z)
        py = T.nnet.sigmoid(T.dot(mu, HY) + by)

        energy = (self.energy(y, py) + self.energy(mu, ph)).sum()
        entropy = -(mu * T.log(mu)).sum()
        cost = energy# + entropy
        grad = theano.grad(cost, wrt=z, consider_constant=[x, y])

        z = z - l * grad
        return z

    def step_infer(self, z, x, y, l, XH, bh, HY, by):
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

        z = self.grad_step(z, x_n, y_n, l, XH, bh, HY, by)
        mu = T.nnet.sigmoid(z)

        py = T.nnet.sigmoid(T.dot(mu, HY) + by)
        ph = T.nnet.sigmoid(T.dot(x, XH) + bh)
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
        non_seqs = [x_n, y_n, l, self.XH, self.bh, self.HY, self.by]

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

    def __call__(self, x, y):
        ph = T.nnet.sigmoid(T.dot(x, self.XH) + self.bh)
        h = self.trng.binomial(p=ph, size=ph.shape, n=1, dtype=ph.dtype)
        py = T.nnet.sigmoid(T.dot(h, self.HY) + self.by)
        y_hat = self.trng.binomial(p=py, size=py.shape, n=1, dtype=py.dtype)
        d_hat = T.concatenate([x, y_hat], axis=1)
        pd = T.concatenate([x, py], axis=1)

        y_energy = self.energy(y, py).mean()

        return y_hat, y_energy, pd, d_hat


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

    def energy_sample(self, x, y, z, XH, bh, HH, bhh, HY, by):

        ph = T.nnet.sigmoid(T.dot(x, XH) + bh)
        pz = T.nnet.sigmoid(z)
        h = self.trng.binomial(p=pz, size=pz.shape, n=1, dtype=pz.dtype)
        phh = T.nnet.sigmoid(T.dot(h, HH) + bhh)
        hh = self.trng.binomial(p=phh, size=phh.shape, n=1, dtype=phh.dtype)
        p = T.nnet.sigmoid(T.dot(hh, HY) + by)

        energy = (self.energy(h, ph) + self.energy(hh, phh) + self.energy(y, p)).mean()
        return energy

    def IS_energy(self, x, y, z, n_samples=10):
        seqs = []
        outputs_info = [None]
        non_seqs = [x, y, z, self.XH, self.bh, self.HH, self.bhh, self.HY, self.by]

        energies, updates = theano.scan(
            self.energy_sample,
            sequences=seqs,
            outputs_info=outputs_info,
            non_sequences=non_seqs,
            name=tools._p(self.name, 'energy'),
            n_steps=n_samples,
            profile=tools.profile,
            strict=True
        )

        energy = energies.mean()
        return energy, updates

    def grad_step(self, z, x, y, l, XH, bh, HH, bhh, HY, by):
        ph = T.nnet.sigmoid(T.dot(x, XH) + bh)

        h = T.nnet.sigmoid(z)
        hh = T.nnet.sigmoid(T.dot(h, HH) + bhh)
        p = T.nnet.sigmoid(T.dot(hh, HY) + by)

        energy = (self.energy(h, ph) + self.energy(y, p)).sum()
        grad = theano.grad(energy, wrt=z, consider_constant=[x, y])

        z = z - l * grad
        return z

    def step_infer(self, z, x, y, l, XH, bh, HH, bhh, HY, by):
        if self.noise:
            x_n = x * (1 - self.trng.binomial(p=self.noise, size=x.shape, n=1,
                                            dtype=x.dtype))
            y_n = y * (1 - self.trng.binomial(p=self.noise, size=y.shape, n=1,
                                            dtype=y.dtype))
        else:
            x_n = T.zeros_like(x) + x
            y_n = T.zeros_like(y) + y

        ph = T.nnet.sigmoid(T.dot(x, XH) + bh)
        z = self.grad_step(z, x_n, y_n, l, XH, bh, HH, bhh, HY, by)
        h = T.nnet.sigmoid(z)
        hh = T.nnet.sigmoid(T.dot(h, HH) + bhh)
        #h = self.trng.binomial(p=pz, size=z.shape, n=1, dtype=z.dtype)

        p = T.nnet.sigmoid(T.dot(hh, HY) + by)
        y_hat = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)
        d_hat = T.concatenate([x, y_hat], axis=1)
        dp = T.concatenate([x, p], axis=1)

        energy = self.energy(y, p) + self.energy(h, ph)

        return z, y_hat, d_hat, p, dp, ph, energy

    def inference(self, x, y, l, n_inference_steps=1, noise_mode='noise_all', m=20):
        updates = theano.OrderedUpdates()

        x_n, y_n = self.init_inputs(x, y, noise_mode)
        z = self.init_z(x, y)
        pz = T.nnet.sigmoid(z)
        h = self.trng.binomial(p=pz, size=z.shape, n=1, dtype=z.dtype)
        phh = T.nnet.sigmoid(T.dot(h, self.HH) + self.bhh)
        hh = self.trng.binomial(p=phh, size=phh.shape, n=1, dtype=phh.dtype)
        p = T.nnet.sigmoid(T.dot(hh, self.HY) + self.by)
        y_hat = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)

        seqs = []
        outputs_info = [z, None, None, None, None, None, None]
        non_seqs = [x_n, y_n, l, self.XH, self.bh, self.HH, self.bhh, self.HY, self.by]

        (zs, y_hats, d_hats, ps, dps, phs, energies), updates_2 = theano.scan(
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
        #energy = (self.energy(T.nnet.sigmoid(zs[-1]), phs[-1]) + self.energy(y, ps[-1])).mean()
        energy, updates_3 = self.IS_energy(x_n, y, zs[-1], n_samples=m)
        updates.update(updates_3)

        y_hats = T.concatenate([y[None, :, :],
                                y_n[None, :, :],
                                y_hat[None, :, :],
                                y_hats], axis=0)

        d = T.concatenate([x, y], axis=1)
        d_hat = T.concatenate([x, y_hat], axis=1)

        d_hats = T.concatenate([d[None, :, :],
                                d_hat[None, :, :],
                                d_hats], axis=0)

        ps = T.concatenate([y[None, :, :],
                            p[None, :, :],
                            ps], axis=0)

        dps = T.concatenate([d[None, :, :],
                             d_hat[None, :, :],
                             dps], axis=0)

        if self.h_mode == 'average':
            z_mean = zs[-1].mean(axis=0)
            new_z = (1. - self.rate) * self.h + self.rate * z_mean
            updates += [(self.h, new_z)]

        return (y_hats, d_hats, ps, dps, zs, energy), updates

    def __call__(self, x, y):
        ph = T.nnet.sigmoid(T.dot(x, self.XH) + self.bh)
        h = self.trng.binomial(p=ph, size=ph.shape, n=1, dtype=ph.dtype)
        phh = T.nnet.sigmoid(T.dot(h, self.HH) + self.bhh)
        hh = self.trng.binomial(p=phh, size=phh.shape, n=1, dtype=phh.dtype)
        p = T.nnet.sigmoid(T.dot(hh, self.HY) + self.by)
        y_hat = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)
        d_hat = T.concatenate([x, y_hat], axis=1)
        dp = T.concatenate([x, p], axis=1)

        y_energy = self.energy(y, p).mean()

        return y_hat, y_energy, dp, d_hat
