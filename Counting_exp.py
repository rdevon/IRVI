'''
Module to train RNN to add
'''

import argparse
from collections import OrderedDict
import numpy as np
import os
from os import path
import theano
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano import tensor as T

from gru import GenGRU
from monitor import SimpleMonitor
from rnn import GenRNN
import op
from tools import itemlist


floatX = theano.config.floatX

class BitCounter(object):
    def __init__(self, n_bits, n_steps):
        self.n_bits = n_bits
        self.count = 0
        self.n_steps = n_steps

    def next(self):
        x = np.zeros((1, self.n_steps, self.n_bits)).astype('float32')

        for i in xrange(self.n_steps):
            c = self.count + i % 2**self.n_bits
            try:
                bits = [int(y) for y in bin(c)[2:]]
                rem = self.n_bits - len(bits)
                bits = [0 for _ in range(rem)] + bits
                x[:, i] = np.array(bits)
            except ValueError as e:
                print c, bits
                raise e

        return x

    def translate(self, x):
        y = 0
        for i, b in enumerate(x):
            y += b * 2**i
        return y

def train_model(mode='rnn', n_bits=4, dim_h=8, n_steps=10, learning_rate=0.001,
                out_path='', load_last=False, lr_decay=False,
                optimizer = 'rmsprop'):

    train = BitCounter(n_bits, n_steps)

    trng = RandomStreams(8 * 3 * 2015)
    dim_in = n_bits

    if mode == 'gru':
        C = GenGRU
    elif mode == 'rnn':
        C = GenRNN
    else:
        raise ValueError()

    X = T.tensor3('x', dtype=floatX)

    rnn = C(dim_in, dim_h, trng=trng, h0_mode=None, condition_on_x=False)
    tparams = rnn.set_tparams()

    X_s = T.zeros_like(X)
    X_s = T.set_subtensor(X_s[1:], X[:-1])
    outs, updates = rnn(X_s)
    h = outs['h']
    p = outs['p']
    x = outs['x']

    consider_constant = []

    energy = -(X * T.log(p + 1e-7) + (1 - X) * T.log(1 - p + 1e-7)).sum(axis=(0, 2))
    cost = energy.mean()

    tparams = OrderedDict((k, v)
        for k, v in tparams.iteritems()
        if (v not in updates.keys()))

    grads = T.grad(cost, wrt=itemlist(tparams),
                   consider_constant=consider_constant)

    zero = T.alloc(0., 2, n_bits).astype(floatX)
    out_s, updates_s = rnn.sample(zero, n_samples=2, n_steps=2 * 2**n_bits)
    updates.update(updates_s)

    print 'Building optimizer'
    lr = T.scalar(name='lr')
    f_grad_shared, f_grad_updates = eval('op.' + optimizer)(
        lr, tparams, grads, [X], cost,
        extra_ups=updates,
        extra_outs=[outs['x'], out_s['x']])

    monitor = SimpleMonitor()

    try:
        for e in xrange(1000000):
            x = train.next()
            rval = f_grad_shared(x)
            r = False
            for k, out in zip(['cost'], rval):
                if np.any(np.isnan(out)):
                    print k, 'nan'
                    r = True
                elif np.any(np.isinf(out)):
                    print k, 'inf'
                    r = True
            if r:
                return

            if e % 1000 == 0:
                monitor.update(
                    **OrderedDict({
                        'cost': rval[0]
                    })
                )

                monitor.display(e)

                samples = rval[2]
                print [train.translate(s) for s in samples[:, 0]]

            f_grad_updates(learning_rate)

            if lr_decay:
                learning_rate = max(learning_rate * lr_decay, min_lr)

    except KeyboardInterrupt:
        print 'Training interrupted'

    print 'Saving'
    np.savez(outfile, **dict((k, v.get_value()) for k, v in tparams.items()))
    print 'Done saving. Bye bye.'

def make_argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--out_path', default=None)
    parser.add_argument('-l', '--load_last', action='store_true')
    return parser

if __name__ == '__main__':
    parser = make_argument_parser()
    args = parser.parse_args()

    out_path = args.out_path
    if path.isfile(out_path):
        raise ValueError()
    elif not path.isdir(out_path):
        os.mkdir(path.abspath(out_path))

    train_model(out_path=args.out_path, load_last=args.load_last)
