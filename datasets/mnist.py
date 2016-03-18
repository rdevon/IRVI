'''
Module for MNIST dataset.
'''

from collections import OrderedDict
import cPickle
import gzip
import multiprocessing as mp
import numpy as np
from os import path
import PIL
import random
import sys
from sys import stdout
import theano
from theano import tensor as T
import time
import traceback

from . import Dataset
from utils.tools import (
    concatenate,
    init_rngs,
    resolve_path,
    rng_,
    scan
)
from utils.vis_utils import tile_raster_images


class MNIST(Dataset):
    '''MNIST Dataset class.

    Loads MNIST data from a .pkl file. Train, valid, test must be organized
    as per the `get_data` method.

    Attributes:
        mode: str. 'train', 'test', or 'valid'
        image_shape: list of int. Actual image dimensions for visualization.
        out_path: str. Path for saving images.
        dims: dict of int. Dimensions of each mode.
        dimensions: dict of str. Distribution of each mode.
            See `distributions.py`.
        mean_image: floatX np.array. mean image of data for centering.

    '''
    def __init__(self, source=None, restrict_digits=None, mode='train',
                 binarize=False, name='mnist',
                 out_path=None, **kwargs):
        '''Init function for MNIST dataset.

        Args:
            source: str. location of source file.
            restrict_digits: optional list of ints. If not None, only digits in
                this list will be loaded.
            mode: str. mode of iterator: 'train', 'test', or 'valid'.
            binarize: bool. If True, then sample from data.
            name: str.
            out_path: str. Out path for image saving.
            **kwargs: optional args send to Dataset parent class __init__.
        '''
        super(MNIST, self).__init__(name=name, **kwargs)
        source = resolve_path(source)

        if source is None:
            raise ValueError('No source file provided')
        print 'Loading {name} ({mode}) from {source}'.format(
            name=name, mode=mode, source=source)

        X, Y = self.get_data(source, mode)
        self.mode = mode

        self.image_shape = (28, 28)
        self.out_path = out_path

        uniques = np.unique(Y).tolist()
        if restrict_digits is None:
            n_classes = len(uniques)
        else:
            n_classes = len(restrict_digits)

        O = np.zeros((X.shape[0], n_classes), dtype='float32')

        if restrict_digits is None:
            for idx in xrange(X.shape[0]):
                i = uniques.index(Y[idx])
                O[idx, i] = 1.;
        else:
            print 'Restricting to digits %s' % restrict_digits
            new_X = []
            i = 0
            for j in xrange(X.shape[0]):
                if Y[j] in restrict_digits:
                    new_X.append(X[j])
                    c_idx = restrict_digits.index(Y[j])
                    O[i, c_idx] = 1.;
                    i += 1
            X = new_X.astype(floatX)

        if self.stop is not None:
            X = X[:self.stop]
        self.n = X.shape[0]

        self.dims = dict(label=len(np.unique(Y)))
        self.dims[name] = X.shape[1]
        self.distributions = dict(label='binomial')
        self.distributions[name] = 'binomial'

        if binarize:
            X = rng_.binomial(p=X, size=X.shape, n=1).astype('float32')

        self.X = X
        self.O = O

        self.mean_image = self.X.mean(axis=0)

        if self.shuffle:
            self.randomize()

    def get_data(self, source, mode):
        '''Fetches data from source file.

        Source file must be .pkl.

        Args:
            source: str. Path to source file.
            mode: str. Iterator mode.
        '''
        with gzip.open(source, 'rb') as f:
            x = cPickle.load(f)

        if mode == 'train':
            X = np.float32(x[0][0])
            Y = np.float32(x[0][1])
        elif mode == 'valid':
            X = np.float32(x[1][0])
            Y = np.float32(x[1][1])
        elif mode == 'test':
            X = np.float32(x[2][0])
            Y = np.float32(x[2][1])
        else:
            raise ValueError()

        return X, Y

    def randomize(self):
        '''Randomize dataset function.'''
        rnd_idx = np.random.permutation(np.arange(0, self.n, 1))
        self.X = self.X[rnd_idx, :]
        self.O = self.O[rnd_idx, :]

    def next(self, batch_size=None):
        '''Pull next batch.

        Args:
            batch_size: int (Optional).

        Returns:
            rval: OrderedDict of data.

        '''
        if batch_size is None:
            batch_size = self.batch_size

        if self.pos == -1:
            self.reset()

            if not self.inf:
                raise StopIteration

        x = self.X[self.pos:self.pos+batch_size]
        y = self.O[self.pos:self.pos+batch_size]

        self.pos += batch_size
        if self.pos + batch_size > self.n:
            self.pos = -1

        rval = OrderedDict()
        rval[self.name] = x
        rval['label'] = y

        return rval

    def save_images(self, x, imgfile, transpose=False, x_limit=None):
        '''Save images from flattened array.

        Unflattens array using self.image_shape. Will save as a montage of
        multiple images.

        Args:
            x: floatX np.array. array of flattened images.
            imgfile: str. Output file path.
            transpose: bool. Transpose the images.
            x_limit: int (optional). If set, the number of images in the x
                direction for the montage will be limited.
        '''
        if len(x.shape) == 2:
            x = x.reshape((x.shape[0], 1, x.shape[1]))

        if x_limit is not None and x.shape[0] > x_limit:
            x = np.concatenate([x, np.zeros((x_limit - x.shape[0] % x_limit,
                                             x.shape[1],
                                             x.shape[2])).astype('float32')],
                axis=0)
            x = x.reshape((x_limit, x.shape[0] * x.shape[1] // x_limit, x.shape[2]))

        if transpose:
            x = x.reshape((x.shape[0], x.shape[1], self.image_shape[0], self.image_shape[1]))
            x = x.transpose(0, 1, 3, 2)
            x = x.reshape((x.shape[0], x.shape[1], self.image_shape[0] * self.image_shape[1]))

        tshape = x.shape[0], x.shape[1]
        x = x.reshape((x.shape[0] * x.shape[1], x.shape[2]))
        image = self.show(x.T, tshape)
        image.save(imgfile)

    def show(self, image, tshape):
        '''Arrange montage of images.'''
        fshape = self.image_shape
        X = image.T

        return PIL.Image.fromarray(tile_raster_images(
            X=X, img_shape=fshape, tile_shape=tshape,
            tile_spacing=(1, 1)))

    def translate(self, x):
        return x
