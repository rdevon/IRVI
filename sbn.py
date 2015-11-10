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
from tools import concatenate
from tools import init_rngs
from tools import init_weights
from tools import log_mean_exp
from tools import log_sum_exp
from tools import logit
from tools import _slice


norm_weight = tools.norm_weight
ortho_weight = tools.ortho_weight
floatX = 'float32' # theano.config.floatX
pi = theano.shared(np.pi).astype('float32')

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
                        n_inference_samples=20,
                        inference_scaling=None,
                        inference_method='momentum',
                        alpha=7,
                        center_latent=False,
                        extra_inference_args=dict(),
                        **kwargs):
    model.inference_rate = inference_rate
    model.inference_decay = inference_decay
    model.entropy_scale = entropy_scale
    model.importance_sampling = importance_sampling
    model.inference_scaling = inference_scaling
    model.n_inference_samples = n_inference_samples
    model.alpha = alpha
    model.center_latent = center_latent

    if inference_method == 'sgd':
        model.step_infer = model._step_sgd
        model.init_infer = model._init_sgd
        model.unpack_infer = model._unpack_sgd
        model.params_infer = model._params_sgd
        kwargs = init_sgd_args(model, **extra_inference_args)
    elif inference_method == 'momentum':
        model.step_infer = model._step_momentum
        model.init_infer = model._init_momentum
        model.unpack_infer = model._unpack_momentum
        model.params_infer = model._params_momentum
        kwargs = init_momentum_args(model, **extra_inference_args)
    elif inference_method == 'momentum_straight_through':
        model.step_infer = model._step_momentum_st
        model.init_infer = model._init_momentum
        model.unpack_infer = model._unpack_momentum
        model.params_infer = model._params_momentum
        kwargs = init_momentum_args(model, **extra_inference_args)
    elif inference_method == 'adaptive':
        model.step_infer = model._step_adapt
        model.init_infer = model._init_adapt
        model.unpack_infer = model._unpack_adapt
        model.params_infer = model._params_adapt
        model.strict = False
        model.init_variational_params = model._init_variational_params_adapt
    elif inference_method == 'momentum_then_adapt':
        model.step_infer = model._step_momentum_then_adapt
        model.init_infer = model._init_momentum
        model.unpack_infer = model._unpack_momentum_then_adapt
        model.params_infer = model._params_momentum
        model.init_variational_params = model._init_variational_params_adapt
        model.strict = False
        kwargs = init_momentum_args(model, **extra_inference_args)
    else:
        raise ValueError()

    return kwargs

def _sample(p, size=None, trng=None):
    if size is None:
        size = p.shape
    return trng.binomial(p=p, size=size, n=1, dtype=p.dtype)

def _noise(x, amount=0.1, size=None, trng=None):
    if size is None:
        size = x.shape
    return x * (1 - trng.binomial(p=amount, size=size, n=1, dtype=x.dtype))

def set_input(x, mode, trng=None):
    if mode == 'sample':
        x = _sample(x, trng=trng)
    elif mode == 'noise':
        x = _noise(x, trng=trng)
    elif mode is None:
        pass
    else:
        raise ValueError('% not supported' % mode)
    return x


class SigmoidBeliefNetwork(Layer):
    def __init__(self, dim_in, dim_h, dim_out,
                 posterior=None, conditional=None,
                 z_init=None,
                 name='sbn',
                 **kwargs):

        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out

        self.posterior = posterior
        self.conditional = conditional

        self.z_init = z_init

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

    def get_params(self):
        params = [self.z] + self.conditional.get_params() + self.posterior.get_params() + [self.inference_scale_factor]
        return params

    def p_y_given_h(self, h, *params):
        params = params[1:1+len(self.conditional.get_params())]
        return self.conditional.step_call(h, *params)

    def sample_from_prior(self, n_samples=100):
        p = T.nnet.sigmoid(self.z)
        h = self.posterior.sample(p=p, size=(n_samples, self.dim_h))
        if self.center_latent:
            py = self.conditional(h - p[None, :])
        else:
            py = self.conditional(h)
        return py

    def importance_weights(self, y, h, py, q, prior, normalize=True):
        y_energy = self.conditional.neg_log_prob(y, py)
        prior_energy = self.posterior.neg_log_prob(h, prior)
        entropy_term = self.posterior.neg_log_prob(h, q)

        log_p = -y_energy - prior_energy + entropy_term
        log_p_max = T.max(log_p, axis=0, keepdims=True)
        w = T.exp(log_p - log_p_max)

        if normalize:
            w = w / w.sum(axis=0, keepdims=True)

        return w

    def log_marginal(self, y, h, py, q, prior):
        y_energy = self.conditional.neg_log_prob(y, py)
        prior_energy = self.posterior.neg_log_prob(h, prior)
        entropy_term = self.posterior.neg_log_prob(h, q)

        log_p = -y_energy - prior_energy + entropy_term
        log_p_max = T.max(log_p, axis=0, keepdims=True)
        w = T.exp(log_p - log_p_max)

        return (T.log(w.sum(axis=0, keepdims=True)) + log_p_max).mean()

    def m_step(self, p_h_logit, y, z, n_samples=10):
        constants = []
        q = T.nnet.sigmoid(z)
        prior = T.nnet.sigmoid(self.z)
        p_h = T.nnet.sigmoid(p_h_logit)

        if n_samples == 0:
            h = q[None, :, :]
        else:
            h = self.posterior.sample(
                q, size=(n_samples, q.shape[0], q.shape[1]))

        if self.center_latent:
            print 'M step: Centering binary latent variables before passing to generation net'
            py = self.conditional(h - prior[None, None, :])
        else:
            py = self.conditional(h)

        entropy = self.posterior.entropy(q).mean()

        prior_energy = self.posterior.neg_log_prob(q, prior[None, :]).mean()
        y_energy = self.conditional.neg_log_prob(y[None, :, :], py).mean()
        h_energy = self.posterior.neg_log_prob(q, p_h).mean()

        return (prior_energy, h_energy, y_energy, entropy), constants

    def kl_divergence(self, p, q, entropy_scale=1.0):
        entropy_term = entropy_scale * self.posterior.entropy(p)
        prior_term = self.posterior.neg_log_prob(p, q)
        return -(entropy_term - prior_term)

    def e_step(self, y, z, *params):
        prior = T.nnet.sigmoid(params[0])
        q = T.nnet.sigmoid(z)

        if self.center_latent:
            print 'E step: Centering binary latent variables before passing to generation net'
            py  = self.p_y_given_h(q - prior[None, :], *params)
        else:
            py = self.p_y_given_h(q, *params)

        consider_constant = [y, prior]
        cond_term = self.conditional.neg_log_prob(y, py)

        kl_term = self.kl_divergence(q, prior[None, :], entropy_scale=self.entropy_scale)
        cost = (cond_term + kl_term).sum(axis=0)

        grad = theano.grad(cost, wrt=z, consider_constant=consider_constant)

        return cost, grad

    def step_infer(self, *params):
        raise NotImplementedError()

    def init_infer(self, z):
        raise NotImplementedError()

    def unpack_infer(self, outs):
        raise NotImplementedError()

    def params_infer(self):
        raise NotImplementedError()

    # Importance Sampling
    def _step_adapt(self, y, q, *params):
        prior = T.nnet.sigmoid(params[0])
        h = self.posterior.sample(
            q, size=(self.n_inference_samples, q.shape[0], q.shape[1]))

        if self.center_latent:
            py = self.p_y_given_h(h - prior[None, None, :], *params)
        else:
            py = self.p_y_given_h(h, *params)

        w = self.importance_weights(
            y[None, :, :], h, py, q[None, :, :], prior[None, None, :])

        cost = w.std()
        q = (w[:, :, None] * h).sum(axis=0)

        return q, cost

    def _init_adapt(self, q):
        return []

    def _init_variational_params_adapt(self, p_h_logit, z0=None):
        if z0 is None:
            if self.z_init == 'recognition_net':
                print 'Starting z0 at recognition net'
                q0 = T.nnet.sigmoid(p_h_logit)[0]
            else:
                q0 = T.alloc(0.5, p_h_logit.shape[0], self.dim_h).astype(floatX)

        return q0

    def _unpack_adapt(self, outs):
        qs, costs = outs
        return logit(qs), costs

    def _params_adapt(self):
        return []

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
        cost, grad = self.e_step(y, z, *params)
        dz = (-l * grad + m * dz_).astype(floatX)
        z = (z + dz).astype(floatX)

        l *= self.inference_decay
        return z, l, dz, cost

    def _step_momentum_st(self, y, z, l, dz_, m, *params):
        prior = T.nnet.sigmoid(params[0])
        consider_constant = [y, prior]

        q = T.nnet.sigmoid(z)

        h = self.posterior.sample(q, size=(self.n_inference_samples, q.shape[0], q.shape[1]))

        if self.center_latent:
            py = self.p_y_given_h(h - prior[None, None, :], *params)
        else:
            py = self.p_y_given_h(h, *params)

        kl_term = self.kl_divergence(q, prior[None, :])
        cond_term = self.conditional.neg_log_prob(y[None, :, :], py).mean(axis=0)

        grad_h = theano.grad(cond_term.sum(axis=0), wrt=h, consider_constant=consider_constant)
        #grad_q = (grad_h * q * (1 - q)).sum(axis=0)
        grad_q = grad_h.sum(axis=0)

        grad_k = theano.grad(kl_term.sum(axis=0), wrt=z, consider_constant=consider_constant)
        grad = grad_q + grad_k

        dz = (-l * grad + m * dz_).astype(floatX)
        z = (z + dz).astype(floatX)
        l *= self.inference_decay

        return z, l, dz, (grad).mean()

    def _step_momentum_then_adapt(self, y, q, l, dz_, m, *params):

        if False:
            z = logit(q)
            z, l, dz, cost = self._step_momentum(y, z, l, dz_, m, *params)
            q = T.nnet.sigmoid(z)
            q, cost = self._step_adapt(p_h_logit, y, q, *params)
        else:
            prior = T.nnet.sigmoid(params[0])
            consider_constant = [y, prior, p_h_logit]
            if self.center_latent:
                print 'E step: Centering binary latent variables before passing to generation net'
                py  = self.p_y_given_h(q - prior[None, :], *params)
            else:
                py = self.p_y_given_h(q, *params)

            cond_term = self.conditional.neg_log_prob(y, py)

            kl_term = self.kl_divergence(q, prior[None, :], entropy_scale=self.entropy_scale)
            cost = (cond_term + kl_term).sum(axis=0)

            grad = theano.grad(cost, wrt=q, consider_constant=consider_constant)

            dz = (-l * grad + m * dz_).astype(floatX)
            q = (q + dz).astype(floatX)

            q, cost = self._step_adapt(y, q, *params)

        return q, l, dz, cost

    def _init_momentum(self, z):
        return [self.inference_rate, T.zeros_like(z)]

    def _unpack_momentum(self, outs):
        zs, ls, dzs, costs = outs
        return zs, costs

    def _unpack_momentum_then_adapt(self, outs):
        qs, ls, dqs, costs = outs
        return logit(qs), costs

    def _params_momentum(self):
        return [T.constant(self.momentum).astype('float32')]

    def init_variational_params(self, p_h_logit, z0=None):
        if z0 is None:
            if self.z_init == 'recognition_net':
                print 'Starting z0 at recognition net'
                z0 = p_h_logit[0]
            else:
                z0 = T.alloc(0., p_h_logit.shape[0], self.dim_h).astype(floatX)

        return z0

    def infer_q(self, x, y, n_inference_steps, n_sampling_steps=0, z0=None):
        updates = theano.OrderedUpdates()

        xs = T.alloc(0., n_inference_steps + 1, x.shape[0], x.shape[1]) + x[None, :, :]
        ys = T.alloc(0., n_inference_steps + 1, y.shape[0], y.shape[1]) + y[None, :, :]

        p_h_logit = self.posterior(xs, return_preact=True)
        z0 = self.init_variational_params(p_h_logit, z0=z0)

        seqs = [ys]
        outputs_info = [z0] + self.init_infer(z0) + [None]
        non_seqs = self.params_infer() + self.get_params()

        if isinstance(n_inference_steps, T.TensorVariable) or n_inference_steps > 1:
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
            zs = T.concatenate([z0[None, :, :], zs], axis=0)

        elif n_inference_steps == 1:
            inps = [ys[0]] + outputs_info[:-1] + non_seqs
            outs = self.step_infer(*inps)
            z, i_cost = self.unpack_infer(outs)
            zs = concatenate([z0[None, :, :], z[None, :, :]], axis=0)
            i_costs = [i_cost]

        elif n_inference_steps == 0:
            zs = z0[None, :, :]
            i_costs = [T.constant(0.).astype(floatX)]

        return (p_h_logit, zs, i_costs[-1]), updates

    # Inference
    def inference(self, x, y, z0=None, n_inference_steps=20,
                  n_sampling_steps=0, n_samples=100):

        (p_h_logit, zs, _), updates = self.infer_q(
            x, y, n_inference_steps, n_sampling_steps=n_sampling_steps, z0=z0)

        (prior_energy, h_energy, y_energy, entropy), m_constants = self.m_step(
            p_h_logit[0], y, zs[-1], n_samples=n_samples)

        constants = [zs, entropy] + m_constants

        if self.inference_scaling == 'global':
            updates += [
                (self.inference_scale_factor, y_energy / y_energy_approx)]
            constants += [self.inference_scale_factor]

        return (zs, prior_energy, h_energy, y_energy, entropy), updates, constants

    def __call__(self, x, y, p_h=None,
                 n_samples=100, n_inference_steps=0, n_sampling_steps=0,
                 calculate_log_marginal=False):
        outs = OrderedDict()
        updates = theano.OrderedUpdates()
        prior = T.nnet.sigmoid(self.z)

        if p_h is None:
            z0 = None
        else:
            z0 = logit(p_h)

        (p_h_logit, zs, i_cost), updates_i = self.infer_q(
            x, y, n_inference_steps, n_sampling_steps=n_sampling_steps, z0=z0)
        updates.update(updates_i)
        q = T.nnet.sigmoid(zs[-1])
        outs.update(inference_cost=i_cost)

        if n_samples == 0:
            h = q[None, :, :]
        else:
            h = self.posterior.sample(
                q, size=(n_samples, q.shape[0], q.shape[1]))

        if self.center_latent:
            print 'Centering latents in call'
            py = self.conditional(h - prior[None, None, :])
        else:
            py = self.conditional(h)

        cond_term = self.conditional.neg_log_prob(y[None, :, :], py).mean()
        kl_term = self.kl_divergence(q, prior[None, :]).mean()

        outs.update(
            py=py,
            lower_bound=(cond_term+kl_term)
        )

        if calculate_log_marginal:
            nll = -self.log_marginal(y[None, :, :], h, py, q[None, :, :], prior[None, None, :])
            outs.update(nll=nll)

        return outs, updates


#Deep Sigmoid Belief Networks===================================================


class DeepSBN(Layer):
    def __init__(self, dim_in, dim_h, dim_out, n_layers=2,
                 posteriors=None, conditionals=None,
                 z_init=None,
                 name='sbn',
                 **kwargs):

        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out

        self.n_layers = n_layers

        self.posteriors = posteriors
        self.conditionals = conditionals

        self.z_init = z_init

        kwargs = init_inference_args(self, **kwargs)
        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        super(DeepSBN, self).__init__(name=name)

    def set_params(self):
        z = np.zeros((self.dim_h,)).astype(floatX)

        self.params = OrderedDict(z=z)

        if self.posteriors is None:
            self.posteriors = [None for _ in xrange(self.n_layers)]
        else:
            assert len(self.posteriors) == self.n_layers

        if self.conditionals is None:
            self.conditionals = [None for _ in xrange(self.n_layers)]
        else:
            assert len(self.conditionals) == self.n_layers

        for l in xrange(self.n_layers):
            if l == 0:
                dim_in = self.dim_in
            else:
                dim_in = self.dim_h

            if l == self.n_layers - 1:
                dim_out = self.dim_out
            else:
                dim_out = self.dim_h

            if self.posteriors[l] is None:
                self.posteriors[l] = MLP(
                    dim_in, dim_out, dim_out, 1,
                    rng=self.rng, trng=self.trng,
                    h_act='T.nnet.sigmoid',
                    out_act='T.nnet.sigmoid')

            if self.conditionals[l] is None:
                self.conditionals[l] = MLP(
                    dim_out, dim_out, dim_in, 1,
                    rng=self.rng, trng=self.trng,
                    h_act='T.nnet.sigmoid',
                    out_act='T.nnet.sigmoid')

            self.posteriors[l].name = self.name + '_posterior%d' % l
            self.conditionals[l].name = self.name + '_conditional%d' % l

    def set_tparams(self, excludes=[]):
        excludes = ['{name}_{key}'.format(name=self.name, key=key)
                    for key in excludes]
        tparams = super(DeepSBN, self).set_tparams()

        for l in xrange(self.n_layers):
            tparams.update(**self.posteriors[l].set_tparams())
            tparams.update(**self.conditionals[l].set_tparams())

        tparams = OrderedDict((k, v) for k, v in tparams.iteritems()
            if k not in excludes)

        return tparams

    def get_params(self):
        params = [self.z]
        for l in xrange(self.n_layers):
            params += self.conditionals[l].get_params()
        return params

    def p_y_given_h(self, h, level, *params):
        start = 1
        for l in xrange(level):
            start += len(self.conditionals[l].get_params())
        end = start + len(self.conditionals[level].get_params())

        params = params[start:end]
        return self.conditionals[level].step_call(h, *params)

    def sample_from_prior(self, n_samples=100):
        p = T.nnet.sigmoid(self.z)
        h = self.posteriors[-1].sample(p=p, size=(n_samples, self.dim_h))

        for conditional in self.conditionals[::-1]:
            p = conditional(h)
            h = conditional.sample(p)

        return p

    def importance_weights(self, y, h, py, q, prior, level, normalize=True):
        y_energy = self.conditionals[level].neg_log_prob(y, py)
        prior_energy = self.posteriors[level].neg_log_prob(h, prior)
        entropy_term = self.posteriors[level].neg_log_prob(h, q)

        log_p = -y_energy - prior_energy + entropy_term
        log_p_max = T.max(log_p, axis=0, keepdims=True)
        w = T.exp(log_p - log_p_max)

        if normalize:
            w = w / w.sum(axis=0, keepdims=True)

        return w

    def kl_divergence(self, p, q):
        p_c = T.clip(p, 1e-7, 1.0 - 1e-7)
        q = T.clip(q, 1e-7, 1.0 - 1e-7)

        entropy_term = T.nnet.binary_crossentropy(p_c, p)
        prior_term = T.nnet.binary_crossentropy(q, p)
        return -(entropy_term - prior_term).sum(axis=entropy_term.ndim-1)

    def m_step(self, p_h_logits, y, zs, n_samples=10):
        constants = []

        prior_energy = T.constant(0.).astype(floatX)
        y_energy = T.constant(0.).astype(floatX)
        h_energy = T.constant(0.).astype(floatX)

        p_y = T.nnet.sigmoid(self.z)[None, None, :]
        qs = [T.nnet.sigmoid(z) for z in zs]

        for l in xrange(self.n_layers - 1, -1, -1):
            q = qs[l]
            p_h = T.nnet.sigmoid(p_h_logits[l])
            prior_energy += self.posteriors[l].neg_log_prob(q[None, :, :], p_y).mean()
            h_energy += self.posteriors[l].neg_log_prob(q, p_h).mean()

            if n_samples == 0:
                h = q[None, :, :]
            else:
                h = self.posteriors[l].sample(
                    q, size=(n_samples, q.shape[0], q.shape[1]))

            p_y = self.conditionals[l](h)

            if l == 0:
                y_energy += self.conditionals[l].neg_log_prob(y[None, :, :], p_y).mean()
            else:
                y_energy += self.conditionals[l].neg_log_prob(qs[l-1][None, :, :], p_y).mean()

        return (prior_energy, h_energy, y_energy), constants

    def e_step(self, y, zs, *params):
        total_cost = T.constant(0.).astype(floatX)
        p_y = T.nnet.sigmoid(params[0])[None, :]
        grads = []
        for l in xrange(self.n_layers - 1, -1, -1):
            consider_constant = [p_y]

            z = zs[l]
            q = T.nnet.sigmoid(z)
            kl_term = self.kl_divergence(q, p_y)

            p_y = self.p_y_given_h(q, l, *params)

            if l == 0:
                cond_term = self.conditionals[l].neg_log_prob(y, p_y)
                consider_constant.append(y)
            else:
                cond_term = self.conditionals[l].neg_log_prob(
                    T.nnet.sigmoid(zs[l-1]), p_y)
                consider_constant.append(zs[l-1])

            cost = (cond_term + kl_term).sum(axis=0)

            grad = theano.grad(
                cost, wrt=z, consider_constant=consider_constant)
            grads.append(grad)

            total_cost += cost

        return total_cost, grads

    def step_infer(self, *params):
        raise NotImplementedError()

    def init_infer(self, z):
        raise NotImplementedError()

    def unpack_infer(self, outs):
        raise NotImplementedError()

    def params_infer(self):
        raise NotImplementedError()

    # Importance Sampling
    def _step_adapt(self, y, *params):
        params = list(params)
        qs = params[:self.n_layers]
        params = params[self.n_layers:]

        total_cost = T.constant(0.).astype(floatX)
        p_y = T.nnet.sigmoid(params[0])[None, None, :]
        new_qs = []

        for l in xrange(self.n_layers - 1, -1, -1):
            q = qs[l]
            h = self.posteriors[l].sample(
                q, size=(self.n_inference_samples, q.shape[0], q.shape[1]))

            prior_energy = self.posteriors[l].neg_log_prob(h, p_y)

            p_y = self.p_y_given_h(h, l, *params)

            if l == 0:
                y_energy = self.conditionals[l].neg_log_prob(
                    y[None, :, :], p_y)
            else:
                y_energy = self.conditionals[l].neg_log_prob(
                    qs[l-1][None, :, :], p_y)

            entropy_term = self.posteriors[l].neg_log_prob(h, q[None, :, :])

            log_p = -y_energy - prior_energy + entropy_term
            log_p_max = T.max(log_p, axis=0, keepdims=True)
            w = T.exp(log_p - log_p_max)
            w = w / w.sum(axis=0, keepdims=True)

            q = (w[:, :, None] * h).sum(axis=0)
            new_qs.append(q)

        cost = T.constant(0.).astype(floatX)

        return tuple(new_qs) + (cost,)

    def _init_adapt(self, qs):
        return []

    def _init_variational_params_adapt(self, state):
        z0s = []
        p_h_logits = []

        for l in xrange(self.n_layers):
            p_h_logit = self.posteriors[l](state, return_preact=True)
            p_h = T.nnet.sigmoid(p_h_logit)
            p_h_logits.append(p_h_logit)
            state = self.posteriors[l].sample(p_h)
            z0s.append(p_h[0])

        return z0s, p_h_logits

    def _unpack_adapt(self, outs):
        qss = outs[:self.n_layers]
        return [logit(qs) for qs in qss], outs[-1]

    def _params_adapt(self):
        return []

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
    def _step_momentum(self, y, *params):
        params = list(params)
        zs = params[:self.n_layers]
        l = params[self.n_layers]
        dzs_ = params[1+self.n_layers:1+2*self.n_layers]
        m = params[1+2*self.n_layers]
        params = params[2+2*self.n_layers:]

        cost, grads = self.e_step(y, zs, *params)

        dzs = [(-l * grad + m * dz_).astype(floatX) for dz_, grad in zip(dzs_, grads)]
        qz = [(z + dz).astype(floatX) for z, dz in zip(zs, dzs)]

        l *= self.inference_decay
        return tuple(zs) + (l,) + tuple(dzs) + (cost,)

    def _init_momentum(self, zs):
        return [self.inference_rate] + [T.zeros_like(z) for z in zs]

    def _unpack_momentum(self, outs):
        zss = outs[:self.n_layers]
        return zss, outs[-1]

    def _params_momentum(self):
        return [T.constant(self.momentum).astype('float32')]

    def init_variational_params(self, state):
        z0s = []
        p_h_logits = []

        for l in xrange(self.n_layers):
            p_h_logit = self.posteriors[l](state, return_preact=True)
            p_h = T.nnet.sigmoid(p_h_logit)
            p_h_logits.append(p_h_logit)
            state = self.posteriors[l].sample(p_h)

            z0s.append(p_h_logit[0])

        return z0s, p_h_logits

    def infer_q(self, x, y, n_inference_steps, n_sampling_steps=0):
        updates = theano.OrderedUpdates()

        xs = T.alloc(0., n_inference_steps + 1, x.shape[0], x.shape[1]) + x[None, :, :]
        ys = T.alloc(0., n_inference_steps + 1, y.shape[0], y.shape[1]) + y[None, :, :]

        z0s, p_h_logits = self.init_variational_params(xs)

        seqs = [ys]
        outputs_info = z0s + self.init_infer(z0s) + [None]
        non_seqs = self.params_infer() + self.get_params()

        if isinstance(n_inference_steps, T.TensorVariable) or n_inference_steps > 1:
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

            zss, i_costs = self.unpack_infer(outs)
            for l in xrange(self.n_layers):
                zss[l] = T.concatenate([z0s[l][None, :, :], zss[l]], axis=0)

        else:
            raise NotImplementedError()

        return (p_h_logits, zss, i_costs[-1]), updates

    # Inference
    def inference(self, x, y, n_inference_steps=20,
                  n_sampling_steps=0, n_samples=100):

        (p_h_logits, zss, _), updates = self.infer_q(
            x, y, n_inference_steps, n_sampling_steps=n_sampling_steps)

        zs = [z[-1] for z in zss]
        p_h_logits = [p_h_logit[-1] for p_h_logit in p_h_logits]

        (prior_energy, h_energy, y_energy), m_constants = self.m_step(
            p_h_logits, y, zs, n_samples=n_samples)

        constants = zss + m_constants

        return (zss, prior_energy, h_energy, y_energy), updates, constants

    def __call__(self, x, y,
                 n_samples=100, n_inference_steps=0, n_sampling_steps=0,
                 calculate_log_marginal=False):

        outs = OrderedDict()
        updates = theano.OrderedUpdates()

        (_, zss, i_cost), updates_i = self.infer_q(
            x, y, n_inference_steps, n_sampling_steps=n_sampling_steps)
        updates.update(updates_i)
        zs = [z[-1] for z in zss]
        qs = [T.nnet.sigmoid(z) for z in zs]

        outs.update(inference_cost=i_cost)
        lower_bound = T.constant(0.).astype(floatX)

        p_y = T.nnet.sigmoid(self.z)[None, None, :]

        for l in xrange(self.n_layers - 1, -1, -1):
            q = qs[l]

            kl_term = self.kl_divergence(q[None, :, :], p_y).mean(axis=0)

            outs['q%d' % l] = q
            outs['pya%d' % l] = p_y
            outs['kl%d' % l] = kl_term

            if n_samples == 0:
                h = q[None, :, :]
            else:
                h = self.posteriors[l].sample(
                    q, size=(n_samples, q.shape[0], q.shape[1]))
            p_y = self.conditionals[l](h)

            if l == 0:
                cond_term = self.conditionals[l].neg_log_prob(y[None, :, :], p_y).mean(axis=0)
            else:
                cond_term = self.conditionals[l].neg_log_prob(qs[l-1][None, :, :], p_y).mean(axis=0)

            outs['pyb%d' % l] = p_y
            outs['c%d' % l] = cond_term

            lower_bound += (kl_term + cond_term).mean(axis=0)

            if calculate_log_marginal:
                raise NotImplementedError()
                nll = -log_mean_exp(
                    -self.conditional.neg_log_prob(
                        y[None, :, :], pys[:, -1])
                    - self.posterior.neg_log_prob(
                        h[:, -1], prior[None, None, :]
                    )
                    + self.posterior.neg_log_prob(
                        h[:, -1], q[-1][None, :, :]
                    ),
                    axis=0).mean()
                outs.update(nll=nll)

        outs.update(
            py=p_y,
            lower_bound=lower_bound
        )

        return outs, updates