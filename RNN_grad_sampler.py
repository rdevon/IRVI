'''
Sampling and inference with LSTM models
'''

from collections import OrderedDict
import numpy as np
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


def test(batch_size=20, dim_h=200, l=.01, n_inference_steps=30):
    train = mnist_iterator(batch_size=2*batch_size, mode='train', inf=True,
                           restrict_digits=[3, 8, 9])
    dim_in = train.dim

    X = T.tensor3('x', dtype=floatX)

    trng = RandomStreams(6 * 23 * 2015)
    rnn = SimpleInferGRU(dim_in, dim_h, trng=trng)
    exclude_params = rnn.get_excludes()
    tparams = rnn.set_tparams()
    baseline = BaselineWithInput((dim_in, dim_in), 1, name='energy_baseline')
    tparams.update(baseline.set_tparams())
    exclude_params += baseline.get_excludes()

    mask = T.alloc(1., 2).astype('float32')

    (x_hats, energies), updates = rnn.inference(
        X, mask, l, n_inference_steps=n_inference_steps)

    energy = energies[-1]

    outs_baseline, updates_baseline = baseline(energy, True, X[0], X[1])
    updates.update(updates_baseline)
    centered_energy = outs_baseline['x_centered']
    energy_c = outs_baseline['x_c']
    idb = outs_baseline['idb']
    idb_c = outs_baseline['idb_c']
    m = outs_baseline['m']
    var = outs_baseline['var']

    consider_constant = [
        energy_c,
        idb_c,
        m,
        var
    ]

    #cost = -(log_p + centered_reward * log_q).mean()
    idb_cost = (((energy_c - idb - m))**2).mean()
    cost = centered_energy.mean() + idb_cost

#    e_idx = T.argsort(energy)
#    threshold = energy[e_idx[batch_size / 20]]
#    energy = T.switch(T.lt(energy, threshold), energy, 0.).mean()
#    thresholded_energy = (e_sorted[:(batch_size / 20)]).mean()
#    consider_constant = e_sorted[(batch_size / 20):]

    exclude_keys = exclude_params
    exclude_params = [tparams[ep] for ep in exclude_params]
    tparams = OrderedDict((k, v) for k, v in tparams.iteritems() if k not in exclude_keys)
    grads = T.grad(cost, wrt=itemlist(tparams),
                   consider_constant=consider_constant)

    (chain, p_chain), updates_s = rnn.sample(X[0])
    updates.update(updates_s)

    lr = T.scalar(name='lr')
    optimizer = 'rmsprop'
    f_grad_shared, f_grad_updates = eval('op.' + optimizer)(
        lr, tparams, grads, [X], cost,
        extra_ups=updates,
        extra_outs=[x_hats, chain, centered_energy.mean(), idb_cost.mean()])

    print 'Actually running'
    learning_rate = 0.001
    for e in xrange(10000):
        x, _ = train.next()
        x = x.reshape(2, batch_size, x.shape[1]).astype(floatX)
        rval = f_grad_shared(x)
        r = False
        for k, out in zip(['energy'], rval):
            if np.any(np.isnan(out)):
                print k, 'nan'
                r = True
            elif np.any(np.isinf(out)):
                print k, 'inf'
                r = True
        if r:
            return
        if e % 10 == 0:
            print ('%d: cost: %.5f | prob: %.5f | c_energy: %.5f | idb_cost: %.5f' %
                   (e, rval[0], np.exp(-rval[0]), rval[3], rval[4]))
        if e % 100 == 0:
            sample = rval[1][:, :, 0]
            sample[0] = x[:, 0, :]
            train.save_images(sample, '/Users/devon/tmp/grad_sampler.png')
            sample_chain = rval[2]
            train.save_images(sample_chain, '/Users/devon/tmp/grad_chain.png')
            #prob_chain = rval[3]
            #train.save_images(prob_chain, '/Users/devon/tmp/grad_probs.png')

        f_grad_updates(learning_rate)

if __name__ == '__main__':
    test()