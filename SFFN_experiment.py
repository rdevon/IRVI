'''
SFFN experiment
'''

from collections import OrderedDict
from monitor import SimpleMonitor
import numpy as np
import os
from os import path
import sys
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import time

from mnist import mnist_iterator
import op
from sffn import SFFN
from sffn import SFFN_2Layer
from sffn import SFFN_2Layer2
from tools import itemlist


floatX = theano.config.floatX

def test(batch_size=16, dim_h=256, l=0.1, n_inference_steps=30, out_path=''):

    train = mnist_iterator(batch_size=batch_size, mode='train', inf=True, repeat=1)
    valid = mnist_iterator(batch_size=batch_size, mode='valid', inf=True, repeat=1)

    dim_in = train.dim / 2
    dim_out = train.dim / 2
    D = T.matrix('x', dtype=floatX)
    X = D[:, :dim_in]
    Y = D[:, dim_in:]

    trng = RandomStreams(6 * 23 * 2015)
    sffn = SFFN(dim_in, dim_h, dim_out, trng=trng, z_init=None,
                        noise=0.2, learn_z=True, noise_mode='sample')
    tparams = sffn.set_tparams()

    (zs, y_hats, d_hats, pds, h_energy, y_energy), updates = sffn.inference(
        X, Y, l, n_inference_steps=n_inference_steps, m=100)

    (y_energy_s, pd_s, d_hat_s), updates_s1 = sffn(X, Y, from_z=True)
    updates.update(updates_s1)
    f_d_hat = theano.function([X, Y], [y_energy_s, pd_s, d_hat_s])

    (y_energy_t, pd_t, d_hat_t), updates_s2 = sffn(X, Y, from_z=False)
    updates.update(updates_s2)
    f_d_hat2 = theano.function([X, Y], [y_energy_t, pd_t, d_hat_t])

    consider_constant = [X, Y, zs, zs[-1], y_hats]
    cost = h_energy + y_energy

    if sffn.z_init == 'xy':
        print 'Using a ffn h with inputs x y'
        z0 = T.dot(X, sffn.W0) + T.dot(Y, sffn.U0) + sffn.b0
        zt = zs[-1]
        z_cost = ((zt - z0)**2).sum(axis=1).mean()
        cost += z_cost
    elif sffn.z_init == 'x':
        z0 = T.dot(X, sffn.W0) + sffn.b0
        zt = zs[-1]
        z_cost = ((zt - z0)**2).sum(axis=1).mean()
        cost += z_cost
    elif sffn.learn_z:
        print 'Learning z with a ffn'
        zh = T.tanh(T.dot(X, sffn.W0) + sffn.b0)
        z0 = T.dot(zh, sffn.W1) + sffn.b1
        zt = zs[-1]
        z_cost = ((zt - z0)**2).sum(axis=1).mean()
        cost += z_cost
    else:
        z_cost = T.constant(0.)

    tparams = OrderedDict((k, v)
        for k, v in tparams.iteritems()
        if (v not in updates.keys()))

    grads = T.grad(cost, wrt=itemlist(tparams),
                   consider_constant=consider_constant)

    print 'Building optimizer'
    lr = T.scalar(name='lr')
    optimizer = 'adam'
    f_grad_shared, f_grad_updates = eval('op.' + optimizer)(
        lr, tparams, grads, [D], cost,
        extra_ups=updates,
        extra_outs=[h_energy, y_energy, z_cost, pds, d_hats])

    monitor = SimpleMonitor()

    print 'Actually running'
    learning_rate = 0.0001
    min_lr = 0.00001
    lr_decay = 0.99

    try:
        for e in xrange(100000):
            x, _ = train.next()
            rval = f_grad_shared(x)
            r = False
            for k, out in zip(['cost', 'h_energy', 'y_energy', 'z_cost', 'pds',
                               'd_hats'], rval):
                if np.any(np.isnan(out)):
                    print k, 'nan'
                    r = True
                elif np.any(np.isinf(out)):
                    print k, 'inf'
                    r = True
            if r:
                return

            if e % 10 == 0:
                d_v, _ = valid.next()
                x_v = d_v[:, :dim_in]
                y_v = d_v[:, dim_in:]
                ye_v, pd_v, d_hat_v = f_d_hat(x_v, y_v)
                ye_u, pd_u, d_hat_u = f_d_hat2(x_v, y_v)

                monitor.update(
                    **OrderedDict({
                        'cost': rval[0],
                        'h energy': rval[1],
                        'train y energy': rval[2],
                        'train y prob': np.exp(-rval[2]),
                        'valid y energy': ye_v,
                        'valid y prob': np.exp(-ye_v),
                        'valid y energy 2': ye_u,
                        'valid y prob 2': np.exp(-ye_u),
                        'z cost': rval[3],
                        'learning rate': learning_rate
                    })
                )
                monitor.display(e * batch_size)
                monitor.save(path.join(out_path, 'sffn_monitor.png'))

                idx = np.random.randint(rval[4].shape[1])
                inference = rval[4][:, idx]
                i_samples = rval[5][:, idx]
                inference = np.concatenate([inference[:, None, :],
                                            i_samples[:, None, :]], axis=1)
                train.save_images(inference, path.join(out_path, 'sffn_inference.png'))

                samples = np.concatenate([d_v[None, :, :],
                                          pd_v,
                                          d_hat_v[None, :, :]], axis=0)
                samples = samples[:, :min(10, samples.shape[1] - 1)]
                train.save_images(samples, path.join(out_path, 'sffn_samples.png'))

            f_grad_updates(learning_rate)

            learning_rate = max(learning_rate * lr_decay, min_lr)

    except KeyboardInterrupt:
        print 'Training interrupted'

    outfile = path.join(out_path,
                        'rnn_grad_model_{}.npz'.format(int(time.time())))

    print 'Saving'
    np.savez(outfile, **dict((k, v.get_value()) for k, v in tparams.items()))
    print 'Done saving. Bye bye.'

if __name__ == '__main__':
    out_path = sys.argv[1]
    test(out_path=out_path)