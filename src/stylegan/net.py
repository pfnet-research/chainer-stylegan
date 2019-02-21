import sys
import os
import math
import chainer
import chainer.functions as F 
import chainer.links as L 
import numpy as np

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.path.sep + os.path.pardir)

from common.networks.component.pggan import EqualizedConv2d, EqualizedLinear, feature_vector_normalization
from common.networks.component.auxiliary_links import LinkLeakyRelu
from common.networks.component.normalization.adain import AdaIN
from common.networks.component.scale import Scale
from common.networks.component.rescale import upscale2x, downscale2x, blur

class MappingNetwork(chainer.Chain):
    def __init__(self, ch=512):
        super().__init__()
        self.ch = ch
        with self.init_scope():
            self.l = chainer.ChainList(
                EqualizedLinear(ch, ch),
                LinkLeakyRelu(),
                EqualizedLinear(ch, ch),
                LinkLeakyRelu(),
                EqualizedLinear(ch, ch),
                LinkLeakyRelu(),
                EqualizedLinear(ch, ch),
                LinkLeakyRelu(),
                EqualizedLinear(ch, ch),
                LinkLeakyRelu(),
                EqualizedLinear(ch, ch),
                LinkLeakyRelu(),
                EqualizedLinear(ch, ch),
                LinkLeakyRelu(),
                EqualizedLinear(ch, ch),
                LinkLeakyRelu(),
            )
            self.ln = len(self.l)

    def make_hidden(self, batch_size):
        xp = self.xp
        if xp != np:
            z = xp.random.normal(size=(batch_size, self.ch, 1, 1), dtype='f')
        else:
            # no "dtype" in kwargs for numpy.random.normal
            z = xp.random.normal(size=(batch_size, self.ch, 1, 1)).astype('f')
        # z /= xp.sqrt(xp.sum(z * z, axis=1, keepdims=True) / self.ch + 1e-8)
        return z

    def __call__(self, x):
        h = feature_vector_normalization(x)
        for i in range(self.ln):
            h = self.l[i](h)
        return h

class NoiseBlock(chainer.Chain):
    def __init__(self, ch):
        super().__init__()
        with self.init_scope():
            self.b = Scale(axis=1, W_shape=ch, initialW=0)
        self.ch = ch

    def get_noise(self, batch_size, ch, shape):
        xp = self.xp
        if xp != np:
            z = xp.random.normal(size=(batch_size,) + shape, dtype='f')
        else:
            # no "dtype" in kwargs for numpy.random.normal
            z = xp.random.normal(size=(batch_size,) + shape).astype('f') 
        z = xp.broadcast_to(z, (ch, batch_size,) + shape)
        z = z.transpose((1, 0, 2, 3))
        return z

    def __call__(self, h):
        batch_size = h.shape[0]
        noise = self.get_noise(batch_size, self.ch, h.shape[2:])
        h = h + self.b(noise)
        return h


class StyleBlock(chainer.Chain):
    def __init__(self, w_in, ch):
        super().__init__()
        self.w_in = w_in
        self.ch = ch
        with self.init_scope(): 
            self.s = EqualizedLinear(w_in, ch, initial_bias=chainer.initializers.One(), gain=1)
            self.b = EqualizedLinear(w_in, ch, initial_bias=chainer.initializers.Zero(), gain=1)
    
    def __call__(self, w, h):
        ws = self.s(w)
        wb = self.b(w)
        return AdaIN(h, ws, wb)


class SynthesisBlock(chainer.Chain):
    def __init__(self, ch=512, ch_in=512, w_ch= 512, upsample=True, enable_blur=False):
        super().__init__()
        self.upsample = upsample
        self.ch = ch
        self.ch_in = ch_in
        self.w_ch = w_ch
        with self.init_scope(): 
            if not upsample:
                self.W = chainer.Parameter(shape=(ch_in, 4, 4))
                # w_data_tmp = np.random.normal(size=(ch_in, 4, 4))
                # w_data_tmp /= np.sqrt(np.sum(w_data_tmp * w_data_tmp, axis=0, keepdims=True) + 1e-8)
                self.W.data[:] = 1 # w_data_tmp

            #self.b0 = Scale(axis=1, W_shape=ch, initialW=0)
            #self.b1 = Scale(axis=1, W_shape=ch, initialW=0)

            self.b0 = L.Bias(axis=1, shape=(ch,))
            self.b1 = L.Bias(axis=1, shape=(ch,))
            self.n0 = NoiseBlock(ch)
            self.n1 = NoiseBlock(ch)

            self.s0 = StyleBlock(w_ch, ch)
            self.s1 = StyleBlock(w_ch, ch)

            self.c0 = EqualizedConv2d(ch_in, ch, 3, 1, 1, nobias=True)
            self.c1 = EqualizedConv2d(ch, ch, 3, 1, 1, nobias=True)

        self.blur_k = None
        self.enable_blur = enable_blur

    def __call__(self, w, x=None, add_noise=False):
        h = x
        batch_size, _ = w.shape
        if self.upsample:
            assert h is not None
            # h = F.unpooling_2d(h, 2, 2, 0, outsize=(2 * h.shape[2], 2 * h.shape[3]))
            # h = F.resize_images(h, (2*h.shape[2], 2*h.shape[3]))
            if self.blur_k is None:
                k = np.asarray([1, 2, 1]).astype('f')
                k = k[:, None] * k[None, :]
                k = k / np.sum(k)
                self.blur_k = self.xp.asarray(k)[None, None, :]
            #    self.blur_k = self.xp.asarray(np.broadcast_to(k, (h.shape[1], 3, 3)))[None, :]
            # h = blur(upscale2x(h), self.blur_k)
            if self.enable_blur:
                h = blur(upscale2x(h), self.blur_k)
            else:
                h = upscale2x(h)
            h = self.c0(h)
        else:
            h = F.broadcast_to(self.W, (batch_size, self.ch_in, 4, 4))
        
        # h should be (batch, ch, size, size)
        if add_noise:
            h = self.n0(h)

        h = F.leaky_relu(self.b0(h))
        h = self.s0(w, h)

        h = self.c1(h)
        if add_noise:
            h = self.n1(h)

        h = F.leaky_relu(self.b1(h))
        h = self.s1(w, h)
        return h


class StyleGenerator(chainer.Chain):
    def __init__(self, ch=512, enable_blur=False):
        super(StyleGenerator, self).__init__()
        self.max_stage = 17
        with self.init_scope():
            self.blocks = chainer.ChainList(
                SynthesisBlock(ch, ch, upsample=False), #4
                SynthesisBlock(ch, ch, upsample=True, enable_blur=enable_blur), #8 
                SynthesisBlock(ch, ch, upsample=True, enable_blur=enable_blur), #16
                SynthesisBlock(ch, ch, upsample=True, enable_blur=enable_blur), # 32
                SynthesisBlock(ch // 2, ch, upsample=True, enable_blur=enable_blur), #64
                SynthesisBlock(ch // 4, ch // 2, upsample=True, enable_blur=enable_blur), #128
                SynthesisBlock(ch // 8, ch // 4, upsample=True, enable_blur=enable_blur), #256
                SynthesisBlock(ch // 16, ch // 8, upsample=True, enable_blur=enable_blur), #512
                SynthesisBlock(ch // 32, ch // 16, upsample=True, enable_blur=enable_blur) #1024
            )
            self.outs = chainer.ChainList(
                EqualizedConv2d(ch, 3, 1, 1, 0, gain=1),
                EqualizedConv2d(ch, 3, 1, 1, 0, gain=1),
                EqualizedConv2d(ch, 3, 1, 1, 0, gain=1),
                EqualizedConv2d(ch, 3, 1, 1, 0, gain=1),
                EqualizedConv2d(ch // 2, 3, 1, 1, 0, gain=1),
                EqualizedConv2d(ch // 4, 3, 1, 1, 0, gain=1),
                EqualizedConv2d(ch // 8, 3, 1, 1, 0, gain=1),
                EqualizedConv2d(ch // 16, 3, 1, 1, 0, gain=1),
                EqualizedConv2d(ch // 32, 3, 1, 1, 0, gain=1)
            )

        self.n_blocks = len(self.blocks)
        self.image_size = 1024
        self.enable_blur = enable_blur

    def __call__(self, w, stage, add_noise=True, w2=None):
        '''
            for alpha in [0, 1), and 2*k+2 + alpha < self.max_stage (-1 <= k <= ...):
            stage 0 + alpha       : z ->        block[0] -> out[0] * 1
            stage 2*k+1 + alpha   : z -> ... -> block[k] -> (up -> out[k]) * (1 - alpha)
                                    .................... -> (block[k+1] -> out[k+1]) * (alpha)
            stage 2*k+2 + alpha   : z -> ............... -> (block[k+1] -> out[k+1]) * 1
            over flow stages continues.
        '''

        stage = min(stage, self.max_stage - 1e-8)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)

        h = None
        if stage % 2 == 0:
            k = (stage - 2) // 2
            
            # Enable Style Mixing:
            if w2 is not None and k >= 0:
                lim = np.random.randint(1, k+2)
            else:
                lim = k+2

            for i in range(0, (k + 1) + 1):  # 0 .. k+1
                if i == lim:
                    w = w2
                h = self.blocks[i](w, x=h, add_noise=add_noise)

            h = self.outs[k + 1](h)

        else:
            k = (stage - 1) // 2

            if w2 is not None and k >= 1:
                lim = np.random.randint(1, k+1)
            else:
                lim = k+1

            for i in range(0, k + 1):  # 0 .. k
                if i == lim:
                    w = w2
                h = self.blocks[i](w, x=h, add_noise=add_noise)

            h_0 = self.outs[k](upscale2x(h)) #F.resize_images(h, (2*h.shape[2], 2*h.shape[3])))
            h_1 = self.outs[k + 1](self.blocks[k + 1](w, x=h, add_noise=add_noise))
            assert 0. <= alpha < 1.
            h = (1.0 - alpha) * h_0 + alpha * h_1

        if chainer.configuration.config.train:
            return h
        else:
            min_sample_image_size = 64
            if h.data.shape[2] < min_sample_image_size:  # too small
                scale = int(min_sample_image_size // h.data.shape[2])
                return F.unpooling_2d(h, scale, scale, 0, outsize=(min_sample_image_size, min_sample_image_size))
            else:
                return h

class Generator(chainer.Chain):
    def __init__(self, ch):
        super(Generator, self).__init__()
        self.ch = ch
        with self.init_scope():
            self.mapping = MappingNetwork()
            self.gen = StyleGenerator(ch)

    def make_hidden(self, batch_size):
        xp = self.xp
        if xp != np:
            z = xp.random.normal(size=(batch_size, self.ch, 1, 1), dtype='f')
        else:
            # no "dtype" in kwargs for numpy.random.normal
            z = xp.random.normal(size=(batch_size, self.ch, 1, 1)).astype('f')
        z /= xp.sqrt(xp.sum(z * z, axis=1, keepdims=True) / self.ch + 1e-8)
        return z

    def __call__(self, z, stage):
        w = self.mapping(z)
        return self.gen(w, stage)


class DiscriminatorBlockBase(chainer.Chain):

    def __init__(self, ch):
        super(DiscriminatorBlockBase, self).__init__()
        with self.init_scope():
            self.c0 = EqualizedConv2d(ch, ch, 3, 1, 1)
            self.c1 = EqualizedConv2d(ch, ch, 4, 1, 0)
            self.l2 = EqualizedLinear(ch, 1, gain=1)

    def __call__(self, x):
        h = x
        h = F.leaky_relu((self.c0(h)))
        h = F.leaky_relu((self.c1(h)))
        h = self.l2(h)
        return h


class DiscriminatorBlock(chainer.Chain):

    def __init__(self, in_ch, out_ch, enable_blur=False):
        super(DiscriminatorBlock, self).__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        with self.init_scope():
            self.c0 = EqualizedConv2d(in_ch, out_ch, 3, 1, 1)
            self.c1 = EqualizedConv2d(out_ch, out_ch, 3, 1, 1)
        self.blur_k = None
        self.enable_blur = enable_blur

    def __call__(self, x):
        h = x
        h = F.leaky_relu((self.c0(h)))
        h = F.leaky_relu((self.c1(h)))
        # h = F.average_pooling_2d(h, 2, 2, 0)
        # h = F.resize_images(h, (h.shape[2]//2, h.shape[3]//2))
        if self.blur_k is None:
            k = np.asarray([1, 2, 1]).astype('f')
            k = k[:, None] * k[None, :]
            k = k / np.sum(k)
            self.blur_k = self.xp.asarray(k)[None, None, :]
        if self.enable_blur:
            h = blur(downscale2x(h), self.blur_k)
        else:
            h = downscale2x(h)
        return h


class Discriminator(chainer.Chain):

    def __init__(self, ch=512, enable_blur=False):
        super(Discriminator, self).__init__()
        self.max_stage = 17

        with self.init_scope():
            # NOTE: called in reversed order.
            self.blocks = chainer.ChainList(
                DiscriminatorBlockBase(ch),
                DiscriminatorBlock(ch, ch, enable_blur=enable_blur),
                DiscriminatorBlock(ch, ch, enable_blur=enable_blur),
                DiscriminatorBlock(ch, ch, enable_blur=enable_blur),
                DiscriminatorBlock(ch // 2, ch, enable_blur=enable_blur),
                DiscriminatorBlock(ch // 4, ch // 2, enable_blur=enable_blur),
                DiscriminatorBlock(ch // 8, ch // 4, enable_blur=enable_blur),
                DiscriminatorBlock(ch // 16, ch // 8, enable_blur=enable_blur),
                DiscriminatorBlock(ch // 32, ch // 16, enable_blur=enable_blur),)
            self.ins = chainer.ChainList(
                EqualizedConv2d(3, ch, 1, 1, 0),
                EqualizedConv2d(3, ch, 1, 1, 0),
                EqualizedConv2d(3, ch, 1, 1, 0),
                EqualizedConv2d(3, ch, 1, 1, 0),
                EqualizedConv2d(3, ch // 2, 1, 1, 0),
                EqualizedConv2d(3, ch // 4, 1, 1, 0),
                EqualizedConv2d(3, ch // 8, 1, 1, 0),
                EqualizedConv2d(3, ch // 16, 1, 1, 0),
                EqualizedConv2d(3, ch // 32, 1, 1, 0),)
            self.enable_blur = enable_blur
            
    def __call__(self, x, stage):
        '''
            for alpha in [0, 1), and 2*k+2 + alpha < self.max_stage (-1 <= k <= ...):
            stage 0 + alpha       : p <-        block[0] <- in[0] * 1
            stage 2*k+1 + alpha   : p <- ... <- block[k] <- (up <- in[k]) * (1 - alpha)
                                    .................... <- (block[k+1] <- in[k+1]) * (alpha)
            stage 2*k+2 + alpha   : p <- ............... <- (block[k+1] <- in[k+1]) * 1
            over flow stages continues.
        '''
        stage = min(stage, self.max_stage - 1e-8)
        alpha = stage - math.floor(stage)
        stage = math.floor(stage)

        h = x
        if stage % 2 == 0:
            k = (stage - 2) // 2
            h = F.leaky_relu(self.ins[k + 1](h))
            for i in reversed(range(0, (k + 1) + 1)):  # k+1 .. 0
                h = self.blocks[i](h)
        else:
            k = (stage - 1) // 2

            h_0 = F.leaky_relu(self.ins[k](downscale2x(h))) #  F.resize_images(h, (h.shape[2]//2, h.shape[3]//2))))
            h_1 = self.blocks[k + 1](F.leaky_relu(self.ins[k + 1](x)))
            assert 0. <= alpha < 1.
            h = (1.0 - alpha) * h_0 + alpha * h_1

            for i in reversed(range(0, k + 1)):  # k .. 0
                h = self.blocks[i](h)

        return h
