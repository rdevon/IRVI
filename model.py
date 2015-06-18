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
    learning_rate=0.00001,
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


def get_model(**kwargs):
    dim_r = 500
    dim_g = 200
    batch_size = 17
    n_steps = 11

    train = mnist_iterator(batch_size=2 * batch_size, mode='train')
    valid = None
    test = None

    x = T.matrix('x', dtype='float32')
    x0 = x[:batch_size]
    xT = x[batch_size:]

    trng = RandomStreams(6 * 10 * 2015)

    dim_in = train.dim

    rnn = gru.CondGenGRU(dim_in, dim_r, trng=trng)
    rbm = RBM(dim_in, dim_g, trng=trng)
    baseline = layers.BaselineWithInput((dim_in, dim_in), n_steps + 1,
        name='reward_baseline')

    tparams = rnn.set_tparams()
    exclude_params = rnn.get_excludes()
    tparams.update(rbm.set_tparams())
    exclude_params += rbm.get_excludes()
    tparams.update(baseline.set_tparams())
    exclude_params += baseline.get_excludes()

    inps = OrderedDict(x=x)

    logger.info('Pushing through GRU')
    outs = OrderedDict()
    outs_rnn, updates = rnn(x0, xT, reversed=True, n_steps=n_steps)
    outs[rnn.name] = outs_rnn

    logger.info('Pushing through RBM')
    outs_rbm, updates_rbm = rbm.energy(outs[rnn.name]['x'])
    outs[rbm.name] = outs_rbm
    updates.update(updates_rbm)

    logger.info('Computing reward')
    q = outs[rnn.name]['p']
    samples = outs[rnn.name]['x']
    energy_q = (samples * T.log(q + 1e-7) + (1. - samples) * T.log(1. - q + 1e-7)).sum(axis=2)
    outs[rnn.name]['log_p'] = energy_q
    energy_p = outs[rbm.name]['log_p']
    reward = (energy_p - energy_q)
    reward.name = 'reward'

    logger.info('Pushing reward through baseline')
    outs_baseline, updates_baseline = baseline(reward, x0, xT)
    outs[baseline.name] = outs_baseline
    updates.update(updates_baseline)

    consider_constant = [
        outs[baseline.name]['x'],
        outs[baseline.name]['x_centered']
    ]

    logger.info('Done setting up model')

    errs = OrderedDict()

    return OrderedDict(
        inps=inps,
        outs=outs,
        updates=updates,
        exclude_params=exclude_params,
        consider_constant=consider_constant,
        tparams=tparams,
        data=dict(train=DataIter(train), valid=DataIter(valid), test=DataIter(test))
    )

def get_costs(inps=None, outs=None, **kwargs):
    print outs
    q = outs['cond_gen_gru']['p']
    samples = outs['cond_gen_gru']['x']
    energy_q = outs['cond_gen_gru']['log_p']
    energy_p = outs['rbm']['log_p']

    reward0 = outs['reward_baseline']['x']
    centered_reward = outs['reward_baseline']['x_centered']

    base_cost = -(energy_p + centered_reward * energy_q).mean()
    idb = outs['reward_baseline']['idb']
    c = outs['reward_baseline']['c']
    idb_cost = ((reward0 - idb - c)**2).mean()
    cost = base_cost + idb_cost

    return OrderedDict(
        energy_q=energy_q.mean(),
        energy_p=energy_p.mean(),
        centered_reward=centered_reward.mean(),
        idb_cost=idb_cost,
        cost=cost,
        base_cost=base_cost,
        known_grads=OrderedDict()
    )