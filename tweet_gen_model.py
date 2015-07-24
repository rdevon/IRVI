'''
Basic model for tweet RNN.
'''

from collections import OrderedDict
import itertools
import logging
import math
import numpy as np
import pprint
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano

import costs
from gru import GRUWithOutput
from layers import FFN
from layers import Logistic
from layers import Softmax
import logger
from twitter_api import TwitterFeed

logger = logger.setup_custom_logger('nmt', logging.DEBUG)

default_hyperparams = OrderedDict(
    epochs=5000,
    display_interval=10,
    learning_rate=0.01,
    optimizer='rmsprop',
    saveto='outs',
    disp_freq=100,
    valid_freq=1000,
    save_freq=1000,
    sample_freq=100,
    weight_noise=False
)


class DataIter():
    def __init__(self, dataset):
        if dataset is None:
            return None
        self.dataset = dataset
        self.count = self.dataset.count

    def next(self):
        x, y, m = self.dataset.next()
        self.count = self.dataset.count
        return x, m


def get_model(**kwargs):
    dim_h = 512
    dim_emb = 512
    sampling_steps = 20

    train = TwitterFeed(mode='microsoft', batch_size=32, n_tweets=1, limit_unks=0.2)
    valid = TwitterFeed(mode='feed', batch_size=8, n_tweets=1)
    test = None

    trng = RandomStreams(7 * 2 * 2015)

    X = T.tensor3('X')
    M = T.matrix('M')

    inps = OrderedDict()
    inps['x'] = X
    inps['m'] = M

    embedding = FFN(train.dim, dim_emb, ortho=False, name='embedding')
    tparams = embedding.set_tparams()
    rnn = GRUWithOutput(dim_emb, dim_h, dim_emb)
    tparams.update(rnn.set_tparams())
    exclude_params = rnn.get_excludes()

    emb_logit = FFN(dim_emb, dim_emb, ortho=False, name='emb_logit')
    tparams.update(emb_logit.set_tparams())
    logit_ffn = FFN(dim_emb, train.dim, ortho=False, name='logit_ffn')
    tparams.update(logit_ffn.set_tparams())

    softmax = Softmax()

    outs = OrderedDict()
    logger.info('Enbedding words')
    outs_emb, _ = embedding(X)
    outs[embedding.name] = outs_emb

    emb_s = T.zeros_like(outs_emb['z'])
    emb_s = T.set_subtensor(emb_s[1:], outs_emb['z'][:-1])

    logger.info('Pushing through heirarchal RNN')
    outs_rnn, updates = rnn(emb_s, M)
    outs[rnn.name] = outs_rnn

    o = outs_rnn['o']
    outs_emb_logit, _ = emb_logit(emb_s)
    outs[emb_logit.name] = outs_emb_logit
    logit = T.tanh(o + outs_emb_logit['z'])

    outs_logit, _ = logit_ffn(logit)
    outs[logit_ffn.name] = outs_logit
    outs['logit'] = OrderedDict(y=logit)
    outs['logit'].update(outs_logit)

    outs_softmax, _ = softmax(outs_logit['z'])
    outs[softmax.name] = outs_softmax

    logger.info('Done setting up model')
    logger.info('Adding validation graph')
    vouts = OrderedDict()
    vouts_rnn, vupdates = rnn(emb_s, M, suppress_noise=True)
    vouts[rnn.name] = vouts_rnn

    vo = vouts_rnn['o']
    vouts_emb_logit, _ = emb_logit(emb_s)
    vouts[emb_logit.name] = vouts_emb_logit
    vlogit = T.tanh(vo + vouts_emb_logit['z'])

    vouts_logit, _ = logit_ffn(vlogit)
    vouts[logit_ffn.name] = vouts_logit

    vouts_softmax, _ = softmax(vouts_logit['z'])
    vouts[softmax.name] = vouts_softmax

    def step_sample(x_, h_,
                    U, W, b, Ux, Wx, bx, Wo, bo,
                    E, be, EL, bel, L, bl):

        preact = T.dot(h_, U) + T.dot(x_, W) + b
        r, u = rnn.get_gates(preact)
        preactx = T.dot(h_, Ux) * r + T.dot(x_, Wx) + bx
        h = T.tanh(preactx)
        h = u * h_ + (1. - u) * h
        o = T.dot(h, Wo) + bo

        x_l = T.dot(x_, EL) + bel
        l = T.dot(T.tanh(x_l + o), L) + bl
        p = T.nnet.softmax(l)
        x = trng.multinomial(pvals=p).argmax(axis=1)
        y = T.zeros_like(p)
        y = T.set_subtensor(y[:, x], 1.)
        x_e = (T.dot(y, E) + be).astype('float32')

        return x_e, h, o, x

    seqs = []
    outputs_info = [T.alloc(0., 2, rnn.dim_h).astype('float32'),
        T.alloc(0., 2, rnn.dim_in).astype('float32'), None, None]
    non_seqs = [rnn.U, rnn.W, rnn.b, rnn.Ux, rnn.Wx, rnn.bx, rnn.Wo, rnn.bo,
                embedding.W, embedding.b, emb_logit.W, emb_logit.b,
                logit_ffn.W, logit_ffn.b]

    (x_e, h, o, x), updates_sampler = theano.scan(step_sample,
                                        sequences=seqs,
                                        outputs_info=outputs_info,
                                        non_sequences=non_seqs,
                                        name='sampling',
                                        n_steps=sampling_steps,
                                        profile=False,
                                        strict=True)
    vupdates.update(updates_sampler)

    vouts.update(
        sampler=OrderedDict(
            x_e=x_e,
            h=h,
            o=o,
            x=x
        )
    )

    errs = OrderedDict(
        perc_unks = T.eq(X[:, :, 1], 1).mean()
    )

    consider_constant = []

    return OrderedDict(
        name='tweet_gen',
        inps=inps,
        outs=outs,
        vouts=vouts,
        errs=errs,
        updates=updates,
        vupdates=vupdates,
        exclude_params=exclude_params,
        consider_constant=consider_constant,
        tparams=tparams,
        data=dict(train=DataIter(train), valid=DataIter(valid), test=None)
    )

def get_samplers(inps=None, outs=None, dataset=None):
    trng = RandomStreams(7 * 2 * 2015)
    probs = outs['softmax']['y_hat'][:, 0]
    gt = inps['x'][:, 0]
    gt_prob = (probs * gt).max(axis=1)
    gt_sample = trng.multinomial(pvals=probs).argmax(axis=1)

    sampler = outs['sampler']

    return OrderedDict(
        es=gt_sample,
        gt=gt.argmax(axis=1),
        pr=gt_prob,
        xs=sampler['x'][:, 0]
    )

def get_costs(inps=None, outs=None, **kwargs):
    mask = inps['m']
    cost = costs.categorical_cross_entropy(outs['softmax']['y_hat'],
                                           inps['x'], mask)
    return OrderedDict(
        cost=cost,
        known_grads=OrderedDict()
    )
