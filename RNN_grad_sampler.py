'''
Sampling and inference with LSTM models
'''

from collections import OrderedDict
import numpy as np
import os
from scipy.cluster.vq import vq
from scipy.cluster.vq import kmeans
from scipy.cluster.vq import whiten
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time

from gru import GRU
from gru import GenStochasticGRU
from gru import SimpleInferGRU
from layers import BaselineWithInput
from mnist import mnist_iterator
import op
from rnn import SimpleInferRNN
from rnn import OneStepInferRNN
from tools import itemlist

floatX = theano.config.floatX


def analyze(model_file):
    train = mnist_iterator(batch_size=1, mode='train', inf=False,
                           out_mode='pairs')

    trng = RandomStreams(6 * 23 * 2015)
    dim_in = train.dim
    dim_h=200
    l=.01
    n_inference_steps=30
    rnn = SimpleInferGRU(dim_in, dim_h, trng=trng, h0_mode='ffn')

    pretrained_model = np.load(model_file)

    for k, v in rnn.params.iteritems():
        try:
            pretrained_v = pretrained_model[
                '{name}_{key}'.format(name=rnn.name, key=k)]
            rnn.params[k] = pretrained_v
        except KeyError:
            print '{} not found'.format(k)

    tparams = rnn.set_tparams()

    X = T.tensor3('x', dtype=floatX)

    mask = T.alloc(1., 2).astype('float32')

    (x_hats, h0s, energy), updates = rnn.inference(
        X, mask, l, n_inference_steps=n_inference_steps)

    f_h0 = theano.function([X], h0s[-1], updates=updates)

    hs = []

    try:
        while True:
            x, _ = train.next()
            hs.append(f_h0(x))
    except:
        StopIteration()

    return hs

def test(batch_size=100, dim_h=256, l=.01, n_inference_steps=30,
         save_graphs=False):
    train = mnist_iterator(batch_size=batch_size, mode='train', inf=True,
                           out_mode='pairs')

    dim_in = train.dim
    X = T.tensor3('x', dtype=floatX)

    trng = RandomStreams(6 * 23 * 2015)
    rnn = OneStepInferRNN(dim_in, dim_h, trng=trng, h0_mode='ffn', x_init='run')
    tparams = rnn.set_tparams()

    (x_hats, ps, h0s, energy, entropy), updates = rnn.inference(
        X, l, n_inference_steps=n_inference_steps, noise_mode=None)

    consider_constant = [h0s, x_hats]

    cost = energy.mean()
    cost -= (h0s[-1] * T.log(h0s[-1])).sum(1).mean()

    if rnn.h0_mode == 'ffn':
        print 'Using a ffn h0 with inputs x0 x1'
        h0 = h0s[0]
        ht = h0s[-1]
        ht_c = T.zeros_like(ht) + ht
        h0_cost = ((ht_c - h0)**2).sum(axis=1).mean()
        cost += h0_cost
        consider_constant.append(ht_c)

    if rnn.x_init == 'ffn':
        print 'Using a ffn x1.5 with inputs x0 x1'
        x10 = x
        ht = h0s[-1]
        ht_c = T.zeros_like(ht) + ht
        h0_cost = ((ht_c - h0)**2).sum(axis=1).mean()
        cost += h0_cost
        consider_constant.append(ht_c)

    E = T.scalar('E', dtype='int64')
    new_k = T.min([10, T.floor(E / 200) * 0]).astype('int64')
    k_update = theano.OrderedUpdates([(rnn.k, new_k)])
    updates.update(k_update)

    tparams = OrderedDict((k, v)
        for k, v in tparams.iteritems()
        if (v not in updates.keys()))

    grads = T.grad(cost, wrt=itemlist(tparams),
                   consider_constant=consider_constant)

    (chain, p_chain), updates_s = rnn.sample(X[0], X[1])
    updates.update(updates_s)

    if save_graphs:
        print 'Saving graphs'
        theano.printing.pydotprint(
            cost,
            outfile='/Users/devon/tmp/rnn_grad_cost_graph.png',
            var_with_name_simple=True)
        theano.printing.pydotprint(
            grads,
            outfile='/Users/devon/tmp/rnn_grad_grad_graph.png',
            var_with_name_simple=True)

    print 'Building optimizer'
    lr = T.scalar(name='lr')
    optimizer = 'adam'
    f_grad_shared, f_grad_updates = eval('op.' + optimizer)(
        lr, tparams, grads, [X, E], cost,
        extra_ups=updates,
        extra_outs=[ps, p_chain, energy.mean(), h0_cost, rnn.k])

    print 'Actually running'
    learning_rate = 0.01

    try:
        for e in xrange(100000):
            x, _ = train.next()
            rval = f_grad_shared(x, e)
            r = False
            for k, out in zip(['cost', 'x_hats', 'chain', 'energy', 'h0_cost'], rval):
                if np.any(np.isnan(out)):
                    print k, 'nan'
                    r = True
                elif np.any(np.isinf(out)):
                    print k, 'inf'
                    r = True
            if r:
                return
            if e % 10 == 0:
                print ('%d: cost: %.5f | prob: %.5f | energy: %.5f | '
                       'h0_cost: %.5f | k: %d'
                       % (e, rval[0], np.exp(-rval[0]), rval[3], rval[4], rval[5]))
            if e % 10 == 0:
                idx = np.random.randint(rval[1].shape[2])
                sample = rval[1][:, :, idx]
                train.save_images(sample, '/Users/devon/tmp/grad_sampler2.png')
                sample_chain = rval[2][:, :batch_size/10]
                train.save_images(sample_chain, '/Users/devon/tmp/grad_chain2.png')
                #prob_chain = rval[3]
                #train.save_images(prob_chain, '/Users/devon/tmp/grad_probs.png')

            f_grad_updates(learning_rate)
    except KeyboardInterrupt:
        print 'Training interrupted'

    outfile = os.path.join('/Users/devon/tmp/',
                           'rnn_grad_model_{}.npz'.format(int(time.time())))

    print 'Saving'
    np.savez(outfile, **dict((k, v.get_value()) for k, v in tparams.items()))
    print 'Done saving. Bye bye.'

if __name__ == '__main__':
    test()