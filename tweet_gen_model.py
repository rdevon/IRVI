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
    saveto='model.npz',
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

    train = TwitterFeed(batch_size=17, n_tweets=1)
    valid = TwitterFeed(mode='feed', batch_size=17)
    test = None

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

    errs = OrderedDict(
        perc_unks = T.eq(X[:, :, 1], 1).mean()
    )

    consider_constant = []

    return OrderedDict(
        inps=inps,
        outs=outs,
        vouts=vouts,
        errs=errs,
        updates=updates,
        vupdates=vupdates,
        exclude_params=exclude_params,
        consider_constant=consider_constant,
        tparams=tparams,
        data=dict(train=DataIter(train), valid=None, test=None)
    )

def get_samplers(inps=None, outs=None):
    return OrderedDict()

def get_costs(inps=None, outs=None, **kwargs):
    mask = inps['m']
    cost = costs.categorical_cross_entropy(outs['softmax']['y_hat'],
                                           inps['x'])

    return OrderedDict(
        cost=cost,
        known_grads=OrderedDict()
    )
