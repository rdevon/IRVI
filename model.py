'''
Simple model example.
'''

from collections import OrderedDict
import logging
import pprint
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import costs
import gru
import layers
from mnist import mnist_iterator
from rbm import RBM
import tools
from tools import f_clip
import yaml

import logger
logger = logger.setup_custom_logger('nmt', logging.DEBUG)


default_hyperparams = OrderedDict(
    epochs=5000,
    display_interval=10,
    learning_rate=0.1,
    optimizer='rmsprop',
    saveto='model.npz',
    disp_freq=100,
    valid_freq=1000,
    save_freq=1000,
    sample_freq=100,
    weight_noise=True
)


class DataIter():
    def __init__(self, dataset):
        self.dataset = dataset
        if dataset is None:
            return
        self.count = self.dataset.pos

    def next(self):
        if self.dataset is None:
            return
        x, y = self.dataset.next()
        self.count = self.dataset.pos
        return (x,)


def get_model():
    dim_r = 500
    dim_g = 200
    batch_size = 10
    n_steps = 10

    train = mnist_iterator(batch_size=2 * batch_size, mode='train')
    valid = mnist_iterator(batch_size=2 * batch_size, mode='valid')
    test = None

    x = T.matrix('x', dtype='float32')
    x0 = x[:batch_size]
    xT = x[batch_size:]

    trng = RandomStreams(6 * 10 * 2015)

    dim_in = train.dim

    rnn = gru.CondGenGRU(dim_in, dim_r, trng=trng)
    rbm = RBM(dim_in, dim_g, trng=trng)

    tparams = rnn.set_tparams()
    tparams.update(rbm.set_tparams())

    inps = OrderedDict(x=x)

    logger.info('Pushing through GRU')
    outs = OrderedDict()
    outs_rnn, updates = rnn(x0, xT, reversed=True, n_steps=n_steps)
    outs[rnn.name] = outs_rnn

    logger.info('Pushing through RBM')
    outs_rbm, updates_rbm = rbm.energy(outs[rnn.name]['x'])
    outs[rbm.name] = outs_rbm
    updates.update(updates_rbm)

    logger.info('Done setting up model')
    # If you want to keep something constant
    exclude_params = []

    return OrderedDict(
        inps=inps,
        outs=outs,
        updates=updates,
        exclude_params=exclude_params,
        tparams=tparams,
        data=dict(train=DataIter(train), valid=DataIter(valid), test=DataIter(test))
    )

def get_costs(inps=None, outs=None, **kwargs):
    consider_constant = []
    q = outs['cond_gen_gru']['p']
    samples = outs['cond_gen_gru']['x']
    energy_q = samples * T.log(q + 1e-7) + (1. - samples) * T.log(1. - q + 1e-7)
    energy_p = outs['rbm']['log_p']

    reward = (energy_p - energy_q)
    consider_constant += [reward]

    cost = -(energy_p + reward * energy_q).mean()

    return OrderedDict(
        energy_q=energy_q.mean(),
        energy_p=energy_p.mean(),
        reward=reward.mean(),
        cost=cost,
        known_grads=OrderedDict()
    )