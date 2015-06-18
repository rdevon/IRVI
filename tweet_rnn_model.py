'''
Basic model for tweet RNN.
'''

from collections import OrderedDict
import logging
import pprint
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

from gru import HeirarchalGRU
from layers import Logistic
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
    weight_noise=True
)

def get_model(**kwargs):
    dim_h = 500
    dim_s = 200

    train = TwitterFeed()
    valid = None
    test = None

    X = T.tensor3('X')
    R = T.matrix('R')
    inps = OrderedDict()
    inps['x'] = X
    inps['r'] = R

    rnn = HeirarchalGRU(train.dim, dim_h, dim_s)
    tparams = rnn.set_tparams()
    exclude_params = rnn.get_excludes()
    logistic = Logistic()
    exclude_params += logistic.get_excludes()

    outs = OrderedDict()

    logger.info('Pushing through heirarchal RNN')
    outs_rnn, updates = rnn(X)
    outs[rnn.name] = outs_rnn

    outs_l, updates_l = logistic(outs_rnn['o'])
    outs[logistic.name] = outs_l
    updates.update(updates_l)

    logger.info('Done setting up model')

    consider_constant = []

    return OrderedDict(
        inps=inps,
        outs=outs,
        updates=updates,
        exclude_params=exclude_params,
        consider_constant=consider_constant,
        tparams=tparams,
        data=dict(train=train, valid=valid, test=test)
    )

def get_costs(inps=None, outs=None, **kwargs):
    r_hat = outs['logistic']['y_hat']
    r = inps['r']
    # Fix here
    mask = outs['hiero_gru']['mask']

    cost = ((r - r_hat * (1 - mask))**2).mean()

    return OrderedDict(
        cost=cost,
        known_grads=OrderedDict()
    )


