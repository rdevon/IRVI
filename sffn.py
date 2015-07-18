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
                 weight_scale=0.1, weight_noise=False, noise=0.1, h_mode=None,
                 name='sffn'):
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
        self.h_mode = h_mode

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

        if self.h_mode == 'x':
            W0 = norm_weight(self.dim_in, self.dim_h, scale=self.weight_scale,
                             rng=self.rng)
            b0 = np.zeros((self.dim_h,)).astype('float32')
            self.params.update(W0=W0, b0=b0)

    def init_h(self, x, y):
        if self.h_mode == 'average':
            h = T.alloc(0., x.shape[0], self.dim_h).astype(floatX) + self.h[None, :]
            h += self.trng.normal(avg=0, std=0.1, size=(x.shape[0], self.dim_h), dtype=x.dtype)
        elif self.h_mode == 'ffn':
            h = T.dot(x, self.W0) + T.dot(y, self.U0) + self.b0
            h += self.trng.normal(avg=0, std=0.1, size=(x.shape[0], self.dim_h), dtype=x.dtype)
        elif self.h_mode == 'x':
            h = T.dot(x, self.W0) + self.b0
            h += self.trng.normal(avg=0, std=0.1, size=(x.shape[0], self.dim_h), dtype=x.dtype)
        elif self.h_mode == 'y':
            h = T.dot(y, self.W0) + self.b0
        elif self.h_mode == 'noise':
            h = self.trng.normal(avg=0, std=1, size=(x.shape[0], self.dim_h), dtype=x.dtype)
        elif self.h_mode is None:
            h = T.alloc(0., x.shape[0], self.dim_h).astype(floatX)
        else:
            raise ValueError(self.h_mode)

        return h

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
        elif noise_mode is None:
            x_n = T.zeros_like(x) + x
            y_n = T.zeros_like(y) + y
        else:
            raise ValueError(noise_mode)

        return x_n, y_n

    def energy_sample(self, x, y, z, XH, bh, HY, by):
        pz = T.nnet.sigmoid(z)
        h = self.trng.binomial(p=pz, size=pz.shape, n=1, dtype=pz.dtype)

        p = T.nnet.sigmoid(T.dot(h, HY) + by)
        ph = T.nnet.sigmoid(T.dot(x, XH) + bh)

        energy = (self.energy(h, ph) + self.energy(y, p)).mean()
        return energy

    def IS_energy(self, x, y, z, n_samples=10):
        seqs = []
        outputs_info = [None]
        non_seqs = [x, y, z, self.XH, self.bh, self.HY, self.by]

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

    def energy(self, x, p):
        return -(x * T.log(p + 1e-7) +
                (1 - x) * T.log(1 - p + 1e-7)).sum(axis=1)

    def grad_step(self, z, x, y, l, XH, bh, HY, by):
        ph = T.nnet.sigmoid(T.dot(x, XH) + bh)

        h = T.nnet.sigmoid(z)
        p = T.nnet.sigmoid(T.dot(h, HY) + by)

        energy = (self.energy(y, p) + self.energy(h, ph)).sum()
        grad = theano.grad(energy, wrt=z, consider_constant=[x, y])

        z = z - l * grad
        return z

    def step_infer(self, z, x, y, l, XH, bh, HY, by):
        if self.noise:
            x_n = x * (1 - self.trng.binomial(p=self.noise, size=x.shape, n=1,
                                            dtype=x.dtype))
            y_n = y * (1 - self.trng.binomial(p=self.noise, size=y.shape, n=1,
                                            dtype=y.dtype))
        else:
            x_n = T.zeros_like(x) + x
            y_n = T.zeros_like(y) + y

        z = self.grad_step(z, x_n, y_n, l, XH, bh, HY, by)
        h = T.nnet.sigmoid(z)
        #h = self.trng.binomial(p=pz, size=z.shape, n=1, dtype=z.dtype)

        p = T.nnet.sigmoid(T.dot(h, HY) + by)
        ph = T.nnet.sigmoid(T.dot(x, XH) + bh)
        y_hat = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)
        d_hat = T.concatenate([x, y_hat], axis=1)
        dp = T.concatenate([x, p], axis=1)

        energy = self.energy(y, p) + self.energy(h, ph)

        return z, y_hat, d_hat, p, dp, ph, energy

    def inference(self, x, y, l, n_inference_steps=1, noise_mode='noise_all', m=20):
        updates = theano.OrderedUpdates()

        x_n, y_n = self.init_inputs(x, y, noise_mode)
        z = self.init_h(x, y)
        p = T.nnet.sigmoid(T.dot(T.nnet.sigmoid(z), self.HY) + self.by)
        y_hat = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)

        seqs = []
        outputs_info = [z, None, None, None, None, None, None]
        non_seqs = [x_n, y_n, l, self.XH, self.bh, self.HY, self.by]

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
        p = T.nnet.sigmoid(T.dot(h, self.HY) + self.by)
        y_hat = self.trng.binomial(p=p, size=p.shape, n=1, dtype=p.dtype)
        d_hat = T.concatenate([x, y_hat], axis=1)
        dp = T.concatenate([x, p], axis=1)

        y_energy = self.energy(y, p).mean()

        return y_hat, y_energy, dp, d_hat


class SFFN_2Layer(SFFN):
    def __init__(self, dim_in, dim_h, dim_out, rng=None, trng=None,
                 weight_scale=0.1, weight_noise=False, noise=0.1, h_mode=None,
                 name='sffn_2layer'):

        super(SFFN_2Layer, self).__init__(dim_in, dim_h, dim_out, rng=rng,
                                          trng=trng, weight_scale=weight_scale,
                                          weight_noise=weight_noise,
                                          noise=noise, h_mode=h_mode,
                                          name=name)

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

        energy = (self.energy(h, ph) + self.energy(y, p)).mean()
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
        z = self.init_h(x, y)
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
