'''
Module for sigmoid belief network
'''

from collections import OrderedDict
import numpy as np
import theano
from theano import tensor as T

from layers import MLP
from sffn import SFFN
import tools
from tools import log_mean_exp


norm_weight = tools.norm_weight
ortho_weight = tools.ortho_weight
floatX = 'float32'


class SBN(SFFN):
    def __init__(self, dim_h, dim_out, depth=2,
                 rng=None, trng=None,
                 weight_scale=1.0, weight_noise=False,
                 z_init=None,
                 h_conds=None,
                 x_noise_mode=None, noise_amount=0.1,
                 momentum=0.9, b1=0.9, b2=0.999,
                 inference_rates=[0.1], inference_decay=0.99, n_inference_steps=30,
                 inference_step_scheduler=None,
                 inference_method='sgd',
                 name='sffn'):

        self.dim_h = dim_h
        self.dim_out = dim_out
        self.depth = depth

        self.h_conds = h_conds

        self.weight_noise = weight_noise
        self.weight_scale = weight_scale

        self.momentum = momentum
        self.b1 = b1
        self.b2 = b2

        self.z_init = z_init

        self.x_mode = x_noise_mode
        self.noise_amount = noise_amount

        self.inference_rates = inference_rates
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
        if self.h_conds is None:
            self.h_conds = []
            for l in xrange(self.depth):
                if l == self.depth - 1:
                    dim_out = self.dim_out
                else:
                    dim_out = self.dim_h

                p = MLP(self.dim_h, self.dim_h, dim_out, 1,
                        rng=self.rng, trng=self.trng,
                        h_act='T.nnet.sigmoid',
                        out_act='T.nnet.sigmoid')
                self.h_conds.append(p)
        else:
            assert len(self.h_conds) == self.depth

    def set_tparams(self):
        tparams = OrderedDict()
        for p in self.h_conds:
            tparams.update(**p.set_tparams())

        return tparams

    def init_inputs(self, x, steps=1):
        x_size = (steps, x.shape[0], x.shape[1])
        x = self.set_input(x, self.x_mode, size=x_size)
        return x

    def get_params(self):
        params = []
        for p in self.h_conds:
            params += p.get_params()

        return params

    def ph_given_h(self, h, level, *params):
        start = sum([0] + [len(p.get_params()) for p in self.h_conds[:level]])
        length = len(self.h_conds[level].get_params())
        ps = params[start:start+length]
        print params, ps
        return self.h_conds[level].step_call(h, *ps)

    def preact_h_to_h(self, h, level, *params):
        start = sum([0] + [len(p.get_params()) for p in self.h_conds[:level]])
        length = len(self.h_conds[level].get_params())
        ps = params[start:start+length]
        return self.h_conds[level].preact(h, *ps)

    def sample_energy(self, x, zs, n_samples=10):
        if n_samples == 0:
            hs = [T.nnet.sigmoid(z) for z in zs]
        else:
            mu = T.nnet.sigmoid(zs[0])
            h = self.trng.binomial(
                p=mu, size=(n_samples, mu.shape[0], mu.shape[1]), n=1, dtype=mu.dtype)
            hs = [h]
            for z, p in zip(zs, self.h_conds[:-1]):
                mu = eval(p.out_act)(
                    p(h, return_preact=True) + z[None, :, :])

                h = self.trng.binomial(
                    p=mu, size=(
                        n_samples, mu.shape[0], mu.shape[1]), n=1, dtype=mu.dtype)

                hs.append(h)

        xs = hs[1:] + [x[None, :, :]]

        energies = []
        for l, (p, h, x) in enumerate(zip(self.h_conds, hs, xs)):
            ph = p(h)
            energy = p.neg_log_prob(x, ph)
            energy = -log_mean_exp(-energy, axis=0).mean()
            energies.append(energy)

        return energies, hs

    def inference_cost(self, x, zs, *params):
        mus = [T.nnet.sigmoid(z) for z in zs]

        h = self.trng.binomial(
            p=mus[-1], size=mus[-1].shape, n=1, dtype=mus[-1].dtype)
        hs = [h]
        for l, (z, p) in enumerate(zip(zs, self.h_conds[:-1])):
            mu = eval(p.out_act)(
                self.preact_h_to_h(h, l, *params) + z)

            h = self.trng.binomial(
                p=mu, size=mu.shape, n=1, dtype=mu.dtype)

            hs.append(h)

        xs = hs[1:] + [x]

        cost = T.constant(0.).astype(floatX)
        for l, (p, mu, x) in enumerate(zip(self.h_conds, mus, xs)):
            ph = self.ph_given_h(mu, l, *params)
            energy = p.neg_log_prob(x, ph)
            cost += energy
            if l != self.depth - 1:
                cost += -p.entropy(mu)

        cost = cost.sum(axis=0)
        grads = theano.grad(cost, wrt=zs, consider_constant=[x])

        return cost, grads

    def _step_momentum(self, x, *all_params):
        all_params = list(all_params)
        zs = all_params[:self.depth]
        ls = all_params[self.depth:2*self.depth]
        dzs_ = all_params[2*self.depth:3*self.depth]
        m = all_params[3*self.depth]
        params = all_params[3*self.depth+1:]

        cost, grads = self.inference_cost(x, zs, *params)
        dzs = [-l * grad + m * dz_ for l, grad, dz_ in zip(ls, grads, dzs_)]
        zs = [(z + dz).astype(floatX) for z, dz in zip(zs, dzs)]
        ls = [l * self.inference_decay for l in ls]

        return tuple(zs + ls + dzs + [cost])

    def _init_momentum(self, x, zs):
        return self.inference_rates + [T.zeros_like(z) for z in zs]

    def _params_momentum(self):
        return [T.constant(self.momentum).astype('float32')]

    def _unpack_momentum(self, outs):
        zs = outs[:self.depth]
        costs = outs[-1]
        return zs, costs

    def inference(self, x, n_samples=20):
        updates = theano.OrderedUpdates()

        xs = self.init_inputs(x, steps=self.n_inference_steps)
        z0s = [T.alloc(x.shape[0], self.dim_h).astype(floatX) for _ in xrange(self.depth)]

        if self.inference_step_scheduler is None:
            n_inference_steps = self.n_inference_steps
        else:
            n_inference_steps, updates_c = self.inference_step_scheduler(n_inference_steps)
            updates.update(updates_c)

        seqs = [xs]
        outputs_info = z0s + self.init_infer(xs[0], z0s) + [None]
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

        energies, hs = self.sample_energy(x[0], zs, n_samples=n_samples)
        return (xs, zs, hs, energies, i_costs[-1]), updates

    def __call__(self, x, n_samples=100):
        x_n = self.trng.binomial(p=x, size=x.shape, n=1, dtype=x.dtype)

        h = self.trng.binomial(p=0.5, size=(n_samples, self.dim_h), n=1,
                               dtype=x.dtype)
        for p in self.h_conds[:-1]:
            ph = p(h)
            h = p.sample(ph)

        px = self.h_conds[-1](h)
        x_energy = self.h_conds[-1].neg_log_prob(x[None, :, :], px)
        x_energy = -log_mean_exp(-x_energy, axis=0).mean()
        return px, x_energy