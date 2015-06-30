'''
Basic model for tweet RNN.
'''

from collections import OrderedDict
import logging
import pprint
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import theano
from gru import HeirarchalGRU
from layers import Logistic
import logger
from twitter_api import TwitterFeed
import itertools
import math
logger = logger.setup_custom_logger('nmt', logging.DEBUG)
import numpy as np

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

def get_model(**kwargs):
    dim_h = 500
    dim_s = 200

    train = TwitterFeed()
    valid = TwitterFeed(mode='feed')
    test = None

    X = T.tensor3('X')
    R = T.matrix('R')
    inps = OrderedDict()
    inps['x'] = X
    inps['r'] = R

    rnn = HeirarchalGRU(train.dim, dim_h, dim_s, dropout=0.5, top_fb=True)
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
    logger.info('Adding validation graph')
    vouts = OrderedDict()
    vouts_rnn, vupdates = rnn(X, suppress_noise=True)
    vouts[rnn.name] = vouts_rnn

    vouts_l, vupdates_l = logistic(vouts_rnn['o'])
    vouts[logistic.name] = vouts_l

    top_30 = compute_top(vouts['logistic']['y_hat'][:,:,0], R,
                         vouts['hiero_gru']['mask'],0.3)
    top_50 = compute_top(vouts['logistic']['y_hat'][:,:,0], R,
                         vouts['hiero_gru']['mask'],0.5)
    errs = OrderedDict(
        top_30_acc = top_30,
        top_50_acc = top_50
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
        data=dict(train=train, valid=valid, test=test)
    )

def compute_top(r_hat,R,mask,threshold):
    mask = 1-mask[1:]
    r_hat = r_hat[1:]*mask
    R = R[1:]*mask

    GT_ids = T.argsort(R,axis=0)
    RH_ids = T.argsort(r_hat,axis=0)
    n_total = T.floor(np.float32(threshold) * mask.sum().astype('float32')).astype('int64')


    GT_ids = GT_ids[-n_total:]
    mask_gt = T.zeros_like(mask)
    mask_gt = T.set_subtensor(mask_gt[GT_ids], 1).astype('float32')

    RH_ids = RH_ids[-n_total:]
    mask_r = T.zeros_like(mask)
    mask_r = T.set_subtensor(mask_r[RH_ids], 1).astype('float32')

    mask_final = mask_gt*mask_r
    return mask_final.sum().astype('float32')/n_total.astype('float32')

def get_costs(inps=None, outs=None, **kwargs):
    r_hat = outs['logistic']['y_hat']
    r = inps['r']
    mask = outs['hiero_gru']['mask']

    cost = (((r - r_hat[:, :, 0]) * (1 - mask))**2).sum() / (1 - mask).sum().astype('float32')

    return OrderedDict(
        cost=cost,
        known_grads=OrderedDict()
    )
