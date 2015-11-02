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
from tools import log_sum_exp
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
                        n_inference_samples=20,
                        inference_scaling=None,
                        inference_method='momentum',
                        alpha = 7,
                        **kwargs):
    model.inference_rate = inference_rate
    model.inference_decay = inference_decay
    model.entropy_scale = entropy_scale
    model.importance_sampling = importance_sampling
    model.inference_scaling = inference_scaling
    model.n_inference_samples = n_inference_samples
    model.alpha = alpha

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
    elif inference_method == 'momentum_2':
        model.step_infer = model._step_momentum_2
        model.init_infer = model._init_momentum
        model.unpack_infer = model._unpack_momentum
        model.params_infer = model._params_momentum
        kwargs = init_momentum_args(model, **kwargs)
    elif inference_method == 'momentum_3':
        model.step_infer = model._step_momentum_3
        model.init_infer = model._init_momentum
        model.unpack_infer = model._unpack_momentum
        model.params_infer = model._params_momentum
        kwargs = init_momentum_args(model, **kwargs)
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
                 input_mode=None,
                 name='sbn',
                 **kwargs):

        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out

        self.posterior = posterior
        self.conditional = conditional

        self.z_init = z_init
        self.input_mode = input_mode

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

    def init_inputs(self, x, samples=1):
        size = (samples, x.shape[0], x.shape[1])
        x = T.alloc(0., *size) + x[None, :, :]

        if isinstance(self.input_mode, list):
            for mode in self.input_mode:
                x = set_input(x, mode, trng=self.trng)
        else:
            x = set_input(x, self.input_mode, trng=self.trng)
        return x

    def p_y_given_h(self, h, *params):
        params = params[1:1+len(self.conditional.get_params())]
        return self.conditional.step_call(h, *params)

    def sample_from_prior(self, n_samples=100):
        p = T.nnet.sigmoid(self.z)
        h = self.posterior.sample(p=p, size=(n_samples, self.dim_h))
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
            w = T.clip(w, 1e-7, 1)
            w = w / w.sum(axis=0, keepdims=True)

        return w

    def m_step(self, ph, y, z, n_samples=10):
        constants = []
        q = T.nnet.sigmoid(z)
        prior = T.nnet.sigmoid(self.z)

        if n_samples == 0:
            h = q[None, :, :]
        else:
            h = self.posterior.sample(
                q, size=(n_samples, q.shape[0], q.shape[1]))

        py = self.conditional(h)


        entropy = self.posterior.entropy(q).mean()

        if self.importance_sampling:
            print 'Importance sampling in M'
            w = self.importance_weights(y[None, :, :], h, py, q[None, :, :], prior[None, None, :])
            y_energy = self.conditional.neg_log_prob(y[None, :, :], py)
            prior_energy = self.posterior.neg_log_prob(h, prior[None, None, :])
            h_energy = self.posterior.neg_log_prob(q, ph)

            y_energy = (w * y_energy).sum(axis=0).mean()
            prior_energy = (w * prior_energy).sum(axis=0).mean()
            h_energy = (w * h_energy).sum(axis=0).mean()
            constants += [w]
        else:
            prior_energy = self.posterior.neg_log_prob(q, prior[None, :]).mean()
            y_energy = self.conditional.neg_log_prob(y[None, :, :], py).mean()
            h_energy = self.posterior.neg_log_prob(q, ph).mean()

        return (prior_energy, h_energy, y_energy, entropy), constants

    def kl_divergence(self, p, q, entropy_scale=1.0):
        entropy_term = entropy_scale * self.posterior.entropy(p)
        prior_term = self.posterior.neg_log_prob(p, q)
        return -(entropy_term - prior_term)

    def e_step(self, ph, y, z, *params):
        prior = T.nnet.sigmoid(params[0])
        q = T.nnet.sigmoid(z)
        py = self.p_y_given_h(q, *params)

        consider_constant = [y, prior]
        cond_term = self.conditional.neg_log_prob(y, py)

        if isinstance(self.inference_scaling, float):
            cond_term = self.inference_scaling * cond_term

        elif self.inference_scaling == 'global':
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

        elif self.inference_scaling in ['marginal', 'stochastic', 'conditional_only', 'recognition_net']:
            pass

        elif self.inference_scaling == 'continuous':
            print 'Approximate continuous Bernoulli'
            u = self.trng.uniform(low=0, high=1, size=(self.n_inference_samples, q.shape[0], q.shape[1])).astype(floatX)
            p = (u + q[None, :, :] - 0.5)
            alpha = self.alpha
            h = p ** alpha / ((1 - p) ** alpha + p ** alpha)
            h = T.clip(h, .0, 1.)
            py = self.p_y_given_h(h, *params)
            cond_term = self.conditional.neg_log_prob(y[None, :, :], py).mean(axis=(0))

        elif self.inference_scaling == 'symmetrize':
            print 'Symmetrizing cond'
            py_neg = self.p_y_given_h(1.0 - q, *params)
            cond_term_neg = self.conditional.neg_log_prob(y, py_neg)
            cond_term = 0.5 * (cond_term + 1.0 - cond_term_neg)

        elif self.inference_scaling is not None:
            raise ValueError(self.inference_scaling)
        else:
            print 'No inference scaling'

        if self.inference_scaling == 'conditional_only':
            print 'Conditional-only inference'
            kl_term = 0. * cond_term
            cost = cond_term.sum(axis=0)
        elif self.inference_scaling == 'recognition_net':
            print 'Using recognition as posterior'
            kl_term = self.kl_divergence(
                q, ph, entropy_scale=self.entropy_scale)
            kl_term += self.kl_divergence(q, prior[None, :])
            cost = (cond_term + kl_term).sum(axis=0)
        else:
            kl_term = self.kl_divergence(q, prior[None, :], entropy_scale=self.entropy_scale)
            cost = (cond_term + kl_term).sum(axis=0)

        grad = theano.grad(cost, wrt=z, consider_constant=consider_constant)

        return cost, grad

    def sample_e(self, y, q, *params):
        prior =T.nnet.sigmoid(params[0])
        h = self.posterior.sample(q, size=(100, q.shape[0], q.shape[1]))
        py = self.p_y_given_h(h, *params)

        cond_term = self.conditional.neg_log_prob(y[None, :, :], py)
        prior_term = self.posterior.neg_log_prob(h, prior[None, None, :])
        posterior_term = self.posterior.neg_log_prob(h, q[None, :, :])
        cost = cond_term + prior_term - posterior_term

        w = T.exp(-cost)
        w = T.clip(w, 1e-7, 1.0)
        w_tilda = w / w.sum(axis=0)[None, :]
        q = (w_tilda[:, :, None] * h).sum(axis=0)
        return q, cost

    def step_infer(self, *params):
        raise NotImplementedError()

    def init_infer(self, z):
        raise NotImplementedError()

    def unpack_infer(self, outs):
        raise NotImplementedError()

    def params_infer(self):
        raise NotImplementedError()

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
        z_ = (z + dz).astype(floatX)

        if self.inference_scaling == 'reweight':
            q = T.nnet.sigmoid(z_)
            prior = T.nnet.sigmoid(params[0])
            h = self.posterior.sample(q, size=(100, q.shape[0], q.shape[1]))
            py_r = self.p_y_given_h(h, *params)
            cond_term = self.conditional.neg_log_prob(y[None, :, :], py_r)
            prior_term = self.posterior.neg_log_prob(h, prior[None, None, :])
            posterior_term = self.posterior.neg_log_prob(h, q[None, :, :])
            w = T.exp(-cond_term - prior_term + posterior_term)
            w = T.clip(w, 1e-7, 1.0)
            w_tilda = w / w.sum(axis=0)[None, :]
            mu = (w_tilda[:, :, None] * h).sum(axis=0)
            z = logit(q)
        else:
            z = z_
        l *= self.inference_decay
        return z, l, dz, cost

    def _step_momentum_2(self, ph, y, z, l, dz_, m, *params):

        def get_cond_terms(q):
            py = self.p_y_given_h(q, *params)
            cond_term = self.conditional.neg_log_prob(y, py)

            h = self.posterior.sample(q, size=(self.n_inference_samples, q.shape[0], q.shape[1]))
            py_mcmc = self.p_y_given_h(h, *params)
            cond_term_mcmc = self.conditional.neg_log_prob(y[None, :, :], py_mcmc).mean(axis=0)

            return cond_term, cond_term_mcmc

        prior = T.nnet.sigmoid(params[0])
        consider_constant = [y, prior, l, ph]

        q = T.nnet.sigmoid(z)
        cond_term, cond_term_mcmc = get_cond_terms(q)
        kl_term = self.kl_divergence(q, prior[None, :])

        grad_cond = theano.grad(cond_term.sum(axis=0), wrt=z, consider_constant=consider_constant)
        grad_kl = theano.grad(kl_term.sum(axis=0), wrt=z, consider_constant=consider_constant)

        q = T.nnet.sigmoid(z - l * (grad_cond + grad_kl) + m * dz_)
        cond_term_, cond_term_mcmc_ = get_cond_terms(q)

        r = (cond_term_mcmc - cond_term_mcmc_) / (cond_term - cond_term_ + 1e-7)
        r = T.clip(r, -100.0, 100.0)
        grad = r[:, None] * grad_cond + grad_kl

        dz = (-l * grad + m * dz_).astype(floatX)
        z = (z + dz).astype(floatX)

        l *= self.inference_decay

        cost = (cond_term + kl_term).sum(axis=0)

        return z, l, dz, cost

    def _step_momentum_3(self, ph, y, z, l, dz_, m, *params):
        prior = T.nnet.sigmoid(params[0])
        consider_constant = [y, prior, ph]

        q = T.nnet.sigmoid(z)

        h = self.posterior.sample(q, size=(self.n_inference_samples, q.shape[0], q.shape[1]))
        py = self.p_y_given_h(h, *params)

        if self.importance_sampling:
            print 'Importance sampling in E'
            w = self.importance_weights(y[None, :, :], h, py, q[None, :, :], prior[None, None, :])
            cond_term = (w * self.conditional.neg_log_prob(y[None, :, :], py)).sum(axis=0)
            prior_term = (w * self.posterior.neg_log_prob(h, prior[None, None, :])).sum(axis=0)
            entropy_term = -self.posterior.entropy(q)
            consider_constant += [w]

            grad_p = theano.grad((prior_term + cond_term).sum(axis=0), wrt=h, consider_constant=consider_constant)
            grad_q = theano.grad(entropy_term.sum(axis=0), wrt=z, consider_constant=consider_constant)

            grad = (grad_p * q * (1 - q)).sum(axis=0) + grad_q
        else:
            kl_term = self.kl_divergence(q, prior[None, :])
            cond_term = self.conditional.neg_log_prob(y[None, :, :], py).mean(axis=0)

            grad_h = theano.grad(cond_term.sum(axis=0), wrt=h, consider_constant=consider_constant)
            grad_q = (grad_h * q * (1 - q)).sum(axis=0)

            grad_k = theano.grad(kl_term.sum(axis=0), wrt=z, consider_constant=consider_constant)
            grad = grad_q + grad_k

        dz = (-l * grad + m * dz_).astype(floatX)
        z = (z + dz).astype(floatX)
        l *= self.inference_decay

        return z, l, dz, (grad).mean()

    def _init_momentum(self, ph, y, z):
        return [self.inference_rate, T.zeros_like(z)]

    def _unpack_momentum(self, outs):
        zs, ls, dzs, costs = outs
        return zs, costs

    def _params_momentum(self):
        return [T.constant(self.momentum).astype('float32')]

    def infer_q(self, x, n_inference_steps, n_sampling_steps=0, z0=None):
        updates = theano.OrderedUpdates()

        xs = self.init_inputs(x, samples=n_inference_steps+1)
        ph = self.posterior(xs)
        if z0 is None:
            if self.z_init == 'recognition_net':
                print 'Starting z0 at recognition net'
                z0 = logit(ph[0])
            else:
                z0 = T.alloc(0., x.shape[0], self.dim_h).astype(floatX)

        seqs = [ph, xs]
        outputs_info = [z0] + self.init_infer(ph[0], xs[0], z0) + [None]
        non_seqs = self.params_infer() + self.get_params()

        if isinstance(n_inference_steps, T.TensorVariable) or n_inference_steps > 0:
            if not isinstance(n_inference_steps, T.TensorVariable):
                print 'Gradient descent %d steps' % n_inference_steps
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
        else:
            zs = z0[None, :, :]
            i_costs = T.constant(0.)

        if n_sampling_steps > 0:
            print 'Importance sampling %d steps' % n_sampling_steps
            q0 = T.nnet.sigmoid(zs[-1])

            seqs = [xs]
            outputs_info = [q0, None]
            non_seqs = self.get_params()
            outs, updates_3 = theano.scan(
                self.sample_e,
                sequences=seqs,
                outputs_info=outputs_info,
                non_sequences=non_seqs,
                name=tools._p(self.name, 'infer_sample'),
                n_steps=n_sampling_steps,
                profile=tools.profile,
                strict=True
            )
            updates.update(updates_3)

            qs, s_costs = outs
            zs = T.concatenate([zs, logit(qs)])

        return (ph, xs, zs, i_costs[-1]), updates

    # Inference
    def inference(self, x, z0=None, n_inference_steps=20,
                  n_sampling_steps=0, n_samples=100):

        (ph, xs, zs, _), updates = self.infer_q(
            x, n_inference_steps, n_sampling_steps=n_sampling_steps, z0=z0)

        (prior_energy, h_energy, y_energy, entropy), m_constants = self.m_step(
            ph[0], xs[0], zs[-1], n_samples=n_samples)

        constants = [xs, zs, entropy] + m_constants

        if self.inference_scaling == 'global':
            updates += [
                (self.inference_scale_factor, y_energy / y_energy_approx)]
            constants += [self.inference_scale_factor]

        return (xs, zs, prior_energy, h_energy, y_energy, entropy), updates, constants

    def __call__(self, x, ph=None,
                 n_samples=100, n_inference_steps=0, n_sampling_steps=0,
                 calculate_log_marginal=False):
        outs = OrderedDict()
        updates = theano.OrderedUpdates()
        prior = T.nnet.sigmoid(self.z)

        if isinstance(n_inference_steps, T.TensorVariable) or n_inference_steps > 0:
            if ph is None:
                z0 = None
            else:
                z0 = logit(ph)
            (ph_x, xs, zs, i_cost), updates_i = self.infer_q(
                x, n_inference_steps, n_sampling_steps=n_sampling_steps, z0=z0)
            updates.update(updates_i)
            q = T.nnet.sigmoid(zs)
            outs.update(inference_cost=i_cost)
            x = xs[-1]
        elif ph is None:
            x = self.trng.binomial(p=x, size=x.shape, n=1, dtype=x.dtype)
            q = self.posterior(x)

        if isinstance(n_inference_steps, T.TensorVariable) or n_inference_steps > 0:
            if n_samples == 0:
                h = q[None, :, :, :]
            else:
                h = self.posterior.sample(
                    q, size=(n_samples, q.shape[0], q.shape[1], q.shape[2]))

            pys = self.conditional(h)
            py_approx = self.conditional(q)

            conds_app = self.conditional.neg_log_prob(x[None, :, :], py_approx).mean(axis=1)
            conds_mc = self.conditional.neg_log_prob(x[None, None, :, :], pys).mean(axis=(0, 2))
            kl_terms = self.kl_divergence(q, prior[None, None, :]).mean(axis=1)

            y_energy = conds_mc[-1]
            kl_term = kl_terms[-1]
            py = pys[:, -1]

            outs.update(
                c_a=conds_app,
                c_mc=conds_mc,
                kl=kl_terms
            )

            if calculate_log_marginal:
                nll = -log_mean_exp(
                    -self.conditional.neg_log_prob(
                        x[None, :, :], pys[:, -1])
                    - self.posterior.neg_log_prob(
                        h[:, -1], prior[None, None, :]
                    )
                    + self.posterior.neg_log_prob(
                        h[:, -1], q[-1][None, :, :]
                    ),
                    axis=0).mean()
                outs.update(nll=nll)
        else:
            if n_samples == 0:
                h = q[None, :, :]
            else:
                h = self.posterior.sample(
                    q, size=(n_samples, q.shape[0], q.shape[1]))

            py = self.conditional(h)
            y_energy = self.conditional.neg_log_prob(xs, py).mean(axis=(0, 1))
            kl_term = self.kl_divergence(q, prior[None, :]).mean(axis=0)

            if calculate_log_marginal:
                nll = -log_mean_exp(
                    -self.conditional.neg_log_prob(
                        x[None, :, :], py)
                    - self.posterior.neg_log_prob(
                        h, prior[None, None, :]
                    )
                    + self.posterior.neg_log_prob(
                        h, q[None, :, :]
                    ),
                    axis=0).mean()
                outs.update(nll=nll)

        outs.update(
            py=py,
            lower_bound=(y_energy+kl_term)
        )

        return outs, updates


class GaussianBeliefNet(Layer):
    def __init__(self, dim_in, dim_h, dim_out,
                 posterior=None, conditional=None,
                 z_init=None, input_mode=None,
                 name='gbn',
                 **kwargs):

        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out

        self.posterior = posterior
        self.conditional = conditional

        self.z_init = z_init
        self.input_mode = input_mode

        kwargs = init_inference_args(self, **kwargs)
        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

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
        excludes.append('inference_scale_factor')
        print 'Excluding log sigma from learned params'
        excludes.append('log_sigma')
        excludes = ['{name}_{key}'.format(name=self.name, key=key)
                    for key in excludes]
        tparams = super(GaussianBeliefNet, self).set_tparams()
        tparams.update(**self.posterior.set_tparams())
        tparams.update(**self.conditional.set_tparams())
        tparams = OrderedDict((k, v) for k, v in tparams.iteritems()
            if k not in excludes)

        return tparams

    def init_inputs(self, x, samples=1):
        size = (samples, x.shape[0], x.shape[1])
        x = T.alloc(0., *size) + x[None, :, :]

        if isinstance(self.input_mode, list):
            for mode in self.input_mode:
                x = set_input(x, mode, trng=self.trng)
        else:
            x = set_input(x, self.input_mode, trng=self.trng)
        return x

    def get_params(self):
        params = [self.mu, self.log_sigma] + self.conditional.get_params() + [self.inference_scale_factor]
        return params

    def p_y_given_h(self, h, *params):
        params = params[2:-1]
        return self.conditional.step_call(h, *params)

    def sample_from_prior(self, n_samples=100):
        p = T.concatenate([self.mu, self.log_sigma])
        h = self.posterior.sample(p=p, size=(n_samples, self.dim_h))
        py = self.conditional(h)
        return py

    def m_step(self, ph, y, q, n_samples=10):
        constants = []
        prior = T.concatenate([self.mu[None, :], self.log_sigma[None, :]], axis=1)

        if n_samples == 0:
            h = mu[None, :, :]
        else:
            h = self.posterior.sample(p=q, size=(n_samples, q.shape[0], q.shape[1] / 2))

        py = self.conditional(h)

        y_energy = self.conditional.neg_log_prob(y[None, :, :], py).mean()
        prior_energy = self.kl_divergence(q, prior).mean()
        h_energy = self.kl_divergence(q, ph).mean()

        entropy = self.posterior.entropy(q).mean()

        return (prior_energy, h_energy, y_energy, entropy), constants

    def kl_divergence(self, p, q,
                      entropy_scale=1.0):
        dim = self.dim_h
        mu_p = _slice(p, 0, dim)
        log_sigma_p = _slice(p, 1, dim)
        mu_q = _slice(q, 0, dim)
        log_sigma_q = _slice(q, 1, dim)

        kl = log_sigma_q - log_sigma_p + 0.5 * (
            (T.exp(2 * log_sigma_p) + (mu_p - mu_q) ** 2) /
            T.exp(2 * log_sigma_q)
            - 1)
        return kl.sum(axis=kl.ndim-1)

    def e_step(self, y, q, *params):
        prior = T.concatenate([params[0][None, :], params[1][None, :]], axis=1)

        mu_q = _slice(q, 0, self.dim_h)
        log_sigma_q = _slice(q, 1, self.dim_h)

        epsilon = self.trng.normal(
            avg=0, std=1.0,
            size=(self.n_inference_samples, mu_q.shape[0], mu_q.shape[1]))

        h = mu_q + epsilon * T.exp(log_sigma_q)
        py = self.p_y_given_h(h, *params)

        consider_constant = [y] + list(params[:1])
        cond_term = self.conditional.neg_log_prob(y[None, :, :], py).mean()

        kl_term = self.kl_divergence(q, prior)

        cost = (cond_term + kl_term).sum(axis=0)
        grad = theano.grad(cost, wrt=q, consider_constant=consider_constant)

        return cost, grad

    def step_infer(self, *params):
        raise NotImplementedError()

    def init_infer(self, q):
        raise NotImplementedError()

    def unpack_infer(self, outs):
        raise NotImplementedError()

    def params_infer(self):
        raise NotImplementedError()

    # Momentum
    def _step_momentum(self, y, q, l, dq_, m, *params):
        cost, grad = self.e_step(y, q, *params)
        dq = (-l * grad + m * dq_).astype(floatX)
        q = (q + dq).astype(floatX)
        l *= self.inference_decay
        return q, l, dq, cost

    def _init_momentum(self, ph, y, q):
        return [self.inference_rate, T.zeros_like(q)]

    def _unpack_momentum(self, outs):
        qs, ls, dqs, costs = outs
        return qs, costs

    def _params_momentum(self):
        return [T.constant(self.momentum).astype('float32')]

    def infer_q(self, x, n_inference_steps, q0=None):

        updates = theano.OrderedUpdates()

        xs = self.init_inputs(x, samples=n_inference_steps+1)
        ph = self.posterior(xs)
        if q0 is None:
            if self.z_init == 'recognition_net':
                print 'Starting q0 at recognition net'
                q0 = ph[0]
            else:
                q0 = T.alloc(0., x.shape[0], 2 * self.dim_h).astype(floatX)

        seqs = [xs]
        outputs_info = [q0] + self.init_infer(ph[0], xs[0], q0) + [None]
        non_seqs = self.params_infer() + self.get_params()

        if isinstance(n_inference_steps, T.TensorVariable) or n_inference_steps > 0:
            if not isinstance(n_inference_steps, T.TensorVariable):
                print '%d inference steps' % n_inference_steps

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

            qs, i_costs = self.unpack_infer(outs)
            qs = T.concatenate([q0[None, :, :], qs], axis=0)
        else:
            qs = q0[None, :, :]

        return (ph, xs, qs, i_costs[-1]), updates

    # Inference
    def inference(self, x, q0=None, n_inference_steps=20, n_sampling_steps=None, n_samples=100):
        (ph, xs, qs, _), updates = self.infer_q(x, n_inference_steps, q0=q0)

        (prior_energy, h_energy, y_energy, entropy), m_constants = self.m_step(
            ph[0], xs[0], qs[-1], n_samples=n_samples)

        constants = [xs, qs, entropy] + m_constants

        return (xs, qs,
                prior_energy, h_energy,
                y_energy, entropy), updates, constants

    def __call__(self, x, ph=None, n_samples=100, n_sampling_steps=None,
                 n_inference_steps=0, calculate_log_marginal=False):

        outs = OrderedDict()
        updates = theano.OrderedUpdates()
        prior = T.concatenate([self.mu[None, :], self.log_sigma[None, :]], axis=1)

        if isinstance(n_inference_steps, T.TensorVariable) or n_inference_steps > 0:
            if ph is None:
                q0 = None
            else:
                q0 = ph

            (ph_x, xs, qs, i_cost), updates_i = self.infer_q(
                x, n_inference_steps, q0=q0)
            updates.update(updates_i)
            q = qs[-1]
            outs.update(inference_cost=i_cost)
            x = xs[-1]
        elif ph is None:
            x = self.trng.binomial(p=x, size=x.shape, n=1, dtype=x.dtype)
            q = self.posterior(x)

        if n_samples == 0:
            h = q[None, :, :]
        else:
            h = self.posterior.sample(
                q, size=(n_samples, q.shape[0], q.shape[1] / 2))

        py = self.conditional(h)
        y_energy = self.conditional.neg_log_prob(x[None, :, :], py).mean(axis=(0, 1))
        kl_term = self.kl_divergence(q, prior).mean(axis=0)

        outs.update(
            py=py,
            lower_bound=(y_energy+kl_term)
        )

        if calculate_log_marginal:
            nll = -log_mean_exp(
                -self.conditional.neg_log_prob(
                    x[None, :, :], py)
                - self.posterior.neg_log_prob(
                    h, prior[None, :, :]
                )
                + self.posterior.neg_log_prob(
                    h, q[None, :, :]
                ),
                axis=0)
            outs.update(nll=nll)

        return outs, updates


class GaussianBeliefDeepNet(Layer):
    def __init__(self, dim_in, dim_h, dim_out, n_layers=2,
                 posteriors=None, conditionals=None,
                 z_init=None,
                 x_noise_mode=None, y_noise_mode=None, noise_amount=0.1,
                 name='gbn',
                 **kwargs):

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.dim_h = dim_h

        self.n_layers = n_layers

        self.posteriors = posteriors
        self.conditionals = conditionals

        self.z_init = z_init

        self.x_mode = x_noise_mode
        self.y_mode = y_noise_mode
        self.noise_amount = noise_amount

        kwargs = init_inference_args(self, **kwargs)
        kwargs = init_weights(self, **kwargs)
        kwargs = init_rngs(self, **kwargs)

        super(GaussianBeliefDeepNet, self).__init__(name=name)

    def set_params(self):
        self.params = OrderedDict()

        if self.posteriors is None:
            self.posteriors = [None for _ in xrange(self.n_layers)]
        else:
            assert len(posteriors) == self.n_layers

        if self.conditionals is None:
            self.conditionals = [None for _ in xrange(self.n_layers)]
        else:
            assert len(conditionals) == self.n_layers

        for l in xrange(self.n_layers):

            if l == 0:
                dim_in = self.dim_in
            else:
                dim_in = self.dim_h

            if l == self.n_layers - 1:
                dim_out = self.dim_out
            else:
                dim_out = self.dim_h

            mu = np.zeros((dim_out,)).astype(floatX)
            log_sigma = np.zeros((dim_out,)).astype(floatX)

            self.params['mu%d' % l] = mu
            self.params['log_sigma%d' % l] = log_sigma

            if self.posteriors[l] is None:
                self.posteriors[l] = MLP(
                    dim_in, dim_out, dim_out, 1,
                    rng=self.rng, trng=self.trng,
                    h_act='T.nnet.sigmoid',
                    out_act='lambda x: x')

            if l == 0:
                out_act = 'T.nnet.sigmoid'
            else:
                out_act = 'lambda x: x'

            if self.conditionals[l] is None:
                self.conditionals[l] = MLP(
                    dim_out, dim_out, dim_in, 1,
                    rng=self.rng, trng=self.trng,
                    h_act='T.nnet.sigmoid',
                    out_act=out_act)

            self.posteriors[l].name = self.name + '_posterior%d' % l
            self.conditionals[l].name = self.name + '_conditional%d' % l

    def set_tparams(self, excludes=[]):
        excludes = ['{name}_{key}'.format(name=self.name, key=key)
                    for key in excludes]
        tparams = super(GaussianBeliefDeepNet, self).set_tparams()

        for l in xrange(self.n_layers):
            tparams.update(**self.posteriors[l].set_tparams())
            tparams.update(**self.conditionals[l].set_tparams())

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
        params = []
        for l in xrange(self.n_layers):
            params += self.conditionals[l].get_params()
            mu = self.__dict__['mu%d' % l]
            log_sigma = self.__dict__['mu%d' % l]
            params += [mu, log_sigma]

        return params

    def get_prior_params(self, level, **params):
        start = 0
        for n in xrange(level):
            start += len(self.conditionals[n].get_params()) + 2
        start -= 2
        params = params[start:start+2]
        return params

    def p_y_given_h(self, h, level, *params):
        for n in xrange(level):
            start += len(self.conditionals[n].get_params()) + 2
        end = start + len(self.conditionals[level].get_params())

        params = params[start:end]
        return self.conditionals[level].step_call(h, *params)

    def sample_from_prior(self, level, n_samples=100):
        mu = self.__dict__['mu%d' % level]
        log_sigma = self.__dict__['mu%d' % level]
        p = T.concatenate([mu, log_sigma])
        h = self.posteriors[l].sample(p=p, size=(n_samples, self.dim_h))

        for l in xrange(level - 1, -1, -1):
            p = self.conditionals[l](h)
            h = self.conditionals[l].sample(p)

        return p

    def kl_divergence(self, p, q,
                      entropy_scale=1.0):
        dim = self.dim_h
        mu_p = _slice(p, 0, dim)
        log_sigma_p = _slice(p, 1, dim)
        mu_q = _slice(q, 0, dim)
        log_sigma_q = _slice(q, 1, dim)

        kl = log_sigma_q - log_sigma_p + 0.5 * (
            (T.exp(2 * log_sigma_p) + (mu_p - mu_q) ** 2) /
            T.exp(2 * log_sigma_q)
            - 1)
        return kl.sum(axis=kl.ndim-1)

    def m_step(self, phs, y, qs, n_samples=10):
        constants = []

        prior_energy = T.constant(0.).astype(floatX)
        y_energy = T.constant(0.).astype(floatX)
        h_energy = T.constant(0.).astype(floatX)

        for l in xrange(self.n_layers):
            q = qs[l]
            ph = phs[l]

            h_energy += self.kl_divergence(q, ph).mean()

            mu = self.__dict__['mu%d' % l]
            log_sigma = self.__dict__['mu%d' % l]
            prior = T.concatenate([mu[None, :], log_sigma[None, :]], axis=1)

            prior_energy += self.kl_divergence(q, prior).mean()

            if n_samples == 0:
                h = mu[None, :, :]
            else:
                h = self.posteriors[l].sample(p=q, size=(n_samples, q.shape[0], q.shape[1] / 2))

            py = self.conditional(h)

            if l == 0:
                y_energy += self.conditionals[l].neg_log_prob(y[None, :, :], py).mean()
            else:
                y_energy += self.kl_divergence(q[l - 1], py).mean()

        return (prior_energy, h_energy, y_energy), constants

    def e_step(self, y, qs, *params):
        consider_constant = [y]

        cost = T.constant(0.).astype(floatX)

        for l in xrange(self.n_layers):
            q = qs[l]
            mu_q = _slice(q, 0, self.dim_h)
            log_sigma_q = _slice(q, 1, self.dim_h)

            prior = T.concatenate(get_prior_params(l), axis=1)
            kl_term = self.kl_divergence(q, prior).mean(axis=0)

            epsilon = self.trng.normal(
                avg=0, std=1.0,
                size=(self.n_inference_samples, mu_q.shape[0], mu_q.shape[1]))

            h = mu_q + epsilon * T.exp(log_sigma_q)
            p = self.p_y_given_h(h, *params)

            if l == 0:
                cond_term = self.conditional.neg_log_prob(y[None, :, :], p).mean(axis=0)
            else:
                cond_term = self.kl_divergence(q[l-1][None, :, :], p)

            cost += (kl_term + cond_term).sum(axis=0)

        grads = theano.grad(cost, wrt=qs, consider_constant=consider_constant)

        return cost, grads

    def step_infer(self, *params):
        raise NotImplementedError()

    def init_infer(self, q):
        raise NotImplementedError()

    def unpack_infer(self, outs):
        raise NotImplementedError()

    def params_infer(self):
        raise NotImplementedError()

    # Momentum
    def _step_momentum(self, y, *params):
        params = list(params)
        qs = params[:self.n_layers]
        l = params[self.n_layers]
        dqs_ = params[1+self.n_layers:1+2*self.n_layers]
        m = params[1+2*self.n_layers]

        params = params[2+2*self.n_layers:]

        cost, grads = self.e_step(y, qs, *params)

        dqs = [(-l * grad + m * dq_).astype(floatX) for dq_, grad in zip(dqs_, grads)]
        qs = [(q + dq).astype(floatX) for q, dq in zip(qs, dqs)]

        l *= self.inference_decay
        return qs + (l,) + dqs + (cost,)

    def _init_momentum(self, qs):
        return [self.inference_rate] + [T.zeros_like(q) for q in qs]

    def _unpack_momentum(self, outs):
        qss = outs[:self.n_layers]
        costs = outs[-1]
        return qss, costs

    def _params_momentum(self):
        return [T.constant(self.momentum).astype('float32')]

    def infer_q(self, x, y, n_inference_steps, q0s=None):
        updates = theano.OrderedUpdates()

        xs, ys = self.init_inputs(x, y, steps=n_inference_steps+1)

        state = xs
        q0s = []

        for l in xrange(self.n_layers):
            ph = self.posteriors[l](state)
            phs.append(ph)

            if q0 is None:
                if self.z_init == 'recognition_net':
                    print 'Starting q0 at recognition net'
                    q0 = ph[0]
                else:
                    q0 = T.alloc(0., x.shape[0], 2 * self.dim_h).astype(floatX)
            q0s.append(q0)

        seqs = [ys]
        outputs_info = q0s + self.init_infer(q0s) + [None]
        non_seqs = self.params_infer() + self.get_params()

        if n_inference_steps > 0:
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

            qss, i_costs = self.unpack_infer(outs)

            for l in xrange(self.n_layers):
                qss[l] = T.concatenate([q0s[l][None, :, :], qss[l]], axis=0)
        else:
            qss = [q0[None, :, :] for q0 in q0s]

        return (phs, xs, ys, qss), updates

    # Inference
    def inference(self, x, y, q0s=None, n_inference_steps=20, n_sampling_steps=None, n_samples=100):
        (phs, xs, ys, qss), updates = self.infer_q(
            x, y, n_inference_steps, q0s=q0s)

        qs = [qs[-1] for qs in qss]
        phs = [ph[0] for ph in phs]

        (prior_energy, h_energy, y_energy), m_constants = self.m_step(
            phs, ys[0], qs, n_samples=n_samples)

        constants = [xs, ys, qs, entropy] + m_constants

        return (xs, ys, qs,
                prior_energy, h_energy, y_energy,
                y_energy_approx, entropy), updates, constants

    def __call__(self, x, y, n_samples=100, n_sampling_steps=None,
                 n_inference_steps=0, end_with_inference=True):

        outs = OrderedDict()
        updates = theano.OrderedUpdates()

        q0s = []

        prior = T.concatenate([self.mu[None, :], self.log_sigma[None, :]], axis=1)

        if end_with_inference:
            (_, xs, ys, qss), updates_i = self.infer_q(
                x, y, n_inference_steps,)
            updates.update(updates_i)

            q = [qs[-1] for qs in qss]
        else:
            ys = x.copy()

        if n_samples == 0:
            h = q[None, :, :]
        else:
            h = self.posterior.sample(
                q, size=(n_samples, q.shape[0], q.shape[1] / 2))

        py = self.conditional(h)
        y_energy = self.conditionals[0].neg_log_prob(ys, py).mean(axis=(0, 1))
        kl_term = self.kl_divergence(q, prior).mean(axis=0)

        outs.update(
            py=py,
            lower_bound=(y_energy+kl_term)
        )

        return outs, updates