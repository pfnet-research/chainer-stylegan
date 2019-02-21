import sys
import os
import chainer
import chainer.functions as F
import chainer.links as L
from chainer import cuda
import numpy as np

class LinkRelu(chainer.Chain):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return F.relu(x)

class LinkTanh(chainer.Chain):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return F.tanh(x)

class LinkLeakyRelu(chainer.Chain):
    def __init__(self):
        super().__init__()

    def __call__(self, x):
        return F.leaky_relu(x)

class LinkReshape(chainer.Chain):
    def __init__(self, shape):
        self.shape = shape
        super().__init__()

    def __call__(self, x):
        b = x.shape[0]
        return F.reshape(x, (b,) + self.shape)

class LinkAct(chainer.Chain):
    def __init__(self, act):
        self.act = act
        super().__init__()
    
    def __call__(self, x):
        return self.act(x)

class LinkSum(chainer.Chain):
    def __init__(self, axis):
        self.axis = axis
        super().__init__()

    def __call__(self, x):
        return F.sum(x, self.axis)

class LinkAveragePooling2D(chainer.Chain):
    def __init__(self, ksize, stride=None, pad=0):
        super().__init__()
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
    
    def __call__(self, x):
        return F.average_pooling_2d(x, self.ksize, self.stride, self.pad)

class LinkMixPooling2D(chainer.Chain):
    def __init__(self, ksize, stride=None, pad=0):
        super().__init__()
        self.ksize = ksize
        self.stride = stride
        self.pad = pad
    
    def __call__(self, x):
        return F.max_pooling_2d(x, self.ksize, self.stride, self.pad)

