'''
Sampling and inference with LSTM models
'''

from collections import OrderedDict
import numpy as np
import os
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
from tools import itemlist

floatX = theano.config.floatX


def test(batch_size=10, dim_h=500, l=.01, n_inference_steps=30,
         save_graphs=False):
    train = mnist_iterator(batch_size=batch_size, mode='train', inf=True,
                           restrict_digits=[3, 8, 9])

    dim_in = train.dim
    X = T.tensor3('x', dtype=floatX)

    trng = RandomStreams(6 * 23 * 2015)
    rnn = SimpleInferGRU(dim_in, dim_h, trng=trng, h0_mode='ffn')
    tparams = rnn.set_tparams()

    mask = T.alloc(1., 2).astype('float32')

    (x_hats, h0s, energy), updates = rnn.inference(
        X, mask, l, n_inference_steps=n_inference_steps)

    consider_constant = [h0s]

    def reg_step(e_, e_accum, e):
        e_accum = e_accum + ((e - e_)**2).sum()
        return e_accum

    reg_terms, _ = theano.scan(
        reg_step,
        sequences=[energy],
        outputs_info=[T.constant(0.).astype(floatX)],
        non_sequences=[energy],
        name='reg_term',
        strict=True
    )

    reg_term = reg_terms[-1]
    cost = energy.mean()

    if rnn.h0_mode == 'ffn':
        print 'Using a ffn h0 with inputs x1 x2'
        h0 = h0s[0]
        ht = h0s[-1]
        ht_c = T.zeros_like(ht) + ht
        h0_cost = ((ht_c - h0)**2).sum(axis=1).mean()
        cost += h0_cost
        consider_constant.append(ht_c)

    E = T.scalar('E', dtype='int64')
    new_k = T.min([10, T.floor(E / 100)]).astype('int64')
    k_update = theano.OrderedUpdates([(rnn.k, new_k)])
    updates.update(k_update)

    tparams = OrderedDict((k, v)
        for k, v in tparams.iteritems()
        if (v not in updates.keys()))

    grads = T.grad(cost, wrt=itemlist(tparams),
                   consider_constant=consider_constant)

    (chain, p_chain), updates_s = rnn.sample(X[0])
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
    optimizer = 'rmsprop'
    f_grad_shared, f_grad_updates = eval('op.' + optimizer)(
        lr, tparams, grads, [X, E], cost,
        extra_ups=updates,
        extra_outs=[x_hats, chain, energy.mean(), reg_term, h0_cost, rnn.k])

    print 'Actually running'
    learning_rate = 0.001

    try:
        for e in xrange(10000):
            data, _ = train.next()
            x = np.zeros((2, batch_size * batch_size, data.shape[1])).astype('float32')
            for i in xrange(batch_size):
                for j in xrange(batch_size):
                    batch = np.concatenate([data[i][None, :], data[j][None, :]])
                    x[:, batch_size * j + i] = batch
            rval = f_grad_shared(x, e)
            r = False
            for k, out in zip(['cost', 'x_hats', 'chain', 'energy', 'reg_term', 'h0_cost'], rval):
                if np.any(np.isnan(out)):
                    print k, 'nan'
                    r = True
                elif np.any(np.isinf(out)):
                    print k, 'inf'
                    r = True
            if r:
                return
            if e % 10 == 0:
                print ('%d: cost: %.5f | prob: %.5f | energy: %.5f | reg_term: %.5f | h0_cost: %.5f | k: %d'
                       % (e, rval[0], np.exp(-rval[0]), rval[3], rval[4], rval[5], rval[6]))
            if e % 10 == 0:
                idx = np.random.randint(rval[1].shape[2])
                sample = rval[1][:, :, idx]
                train.save_images(sample, '/Users/devon/tmp/grad_sampler2.png')
                sample_chain = rval[2][:, :batch_size]
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