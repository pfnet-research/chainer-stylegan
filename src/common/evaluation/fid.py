#Modified from https://github.com/pfnet-research/chainer-gan-lib/blob/master/common/evaluation.py

import os
import sys
import math

import numpy as np
import scipy.linalg
import pickle

import chainer
import chainer.cuda
from chainer import Variable
from chainer import serializers
import chainer.functions as F
import tqdm
from .utils import get_classifer


def calc_FID(m0, c0, m1, c1):
    ret = 0
    ret += np.sum((m0-m1)**2)
    ret += np.trace(c0 + c1 - 2.0 * scipy.linalg.sqrtm(np.dot(c0, c1)))
    return np.real(ret)


def fid_extension(fidapi,
                  generate_func,
                  seed=None,
                  report_key='FID',
                  verbose=True):
    @chainer.training.make_extension()
    def calc(trainer):
        if verbose:
            print('Running FID...')
        fidapi.calc_fake(generate_func, seed)
        fid = fidapi.calc_FID()
        chainer.report({report_key: fid})
        if verbose:
            print(report_key + ' Value: ', fid)
    return calc


class API:
    def __init__(self,
                clsf_type,
                clsf_path,
                gpu,
                load_real_stat=None,
                n_batches=4000,
                batch_size=5):

        self.cnn, self.input_args = get_classifer(clsf_type, clsf_path)
        if gpu >= 0:
            self.cnn.to_gpu(gpu)
            print("Send to GPU ", gpu)
        self.n_batches  = n_batches
        self.batch_size = batch_size
        self.features = {}
        if load_real_stat is not None:
            self.load_real_statistics(load_real_stat)

    def get_mean_cov(self, seed, get_image_func=None, n_batches=None):
        xp = self.cnn.xp

        batch_size = self.batch_size
        n_batches = self.n_batches if n_batches is None else n_batches

        if seed is not None:
            np.random.seed(seed)

        result = []
        print("Calculating FID Features...")
        for i in tqdm.tqdm(range(n_batches)):
            # should return numpy array on CPU
            imgs = get_image_func(batch_size)
            imgs = xp.asarray(imgs)
            imgs = Variable(imgs)

            # Feed images to the inception module to get the features
            with chainer.using_config('train', False):
                with chainer.using_config('enable_backprop', False):
                    y = self.cnn(imgs, **self.input_args)
            result.append(y.data.get())

        if seed is not None:
            np.random.seed()

        result = np.asarray(result)
        result = result.reshape(batch_size*n_batches, result.shape[2])
        mean = np.mean(result, axis=0)
        cov = np.cov(result.T)
        return mean, cov

    def init_real(self, generate_func, seed=None, n_batches=None):
        mean, cov = self.get_mean_cov(seed, generate_func, n_batches=n_batches)
        real_feature = { 
            'mean': mean,
            'cov': cov
        }
        self.features['real'] = real_feature

    def calc_fake(self, generate_func, seed=None, n_batches=None):
        mean, cov = self.get_mean_cov(seed, generate_func, n_batches=n_batches)
        fake_feature = { 
                'mean': mean,
                'cov': cov
            }
        self.features['fake'] = fake_feature

    def calc_FID(self):
        assert 'fake' in self.features and 'real' in self.features
        return calc_FID(self.features['fake']['mean'], self.features['fake']['cov'], self.features['real']['mean'], self.features['real']['cov'])

    def save_real_statistics(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.features['real'], f)

    def load_real_statistics(self, path):
        with open(path, 'rb') as f:
            self.features['real'] = pickle.load(f)
