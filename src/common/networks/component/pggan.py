import numpy as np
import chainer
import chainer.functions as F
import chainer.links as L


def feature_vector_normalization(x, eps=1e-8):
    # x: (B, C, H, W)
    alpha = 1.0 / F.sqrt(F.mean(x * x, axis=1, keepdims=True) + eps)
    return F.broadcast_to(alpha, x.data.shape) * x


class EqualizedConv2d(chainer.Chain):

    def __init__(self, in_ch, out_ch, ksize, stride, pad, nobias=False, gain=np.sqrt(2), lrmul=1):
        w = chainer.initializers.Normal(1.0/lrmul)  # equalized learning rate
        self.inv_c = gain * np.sqrt(1.0 / (in_ch * ksize ** 2))
        self.inv_c = self.inv_c * lrmul
        super(EqualizedConv2d, self).__init__()
        with self.init_scope():
            self.c = L.Convolution2D(in_ch, out_ch, ksize, stride, pad, initialW=w, nobias=nobias)

    def __call__(self, x):
        return self.c(self.inv_c * x)


class EqualizedDeconv2d(chainer.Chain):

    def __init__(self, in_ch, out_ch, ksize, stride, pad, nobias=False, gain=np.sqrt(2), lrmul=1):
        w = chainer.initializers.Normal(1.0/lrmul)  # equalized learning rate
        self.inv_c = gain * np.sqrt(1.0 / (in_ch))
        self.inv_c = self.inv_c * lrmul
        super(EqualizedDeconv2d, self).__init__()
        with self.init_scope():
            self.c = L.Deconvolution2D(in_ch, out_ch, ksize, stride, pad, initialW=w, nobias=nobias)

    def __call__(self, x):
        return self.c(self.inv_c * x)


class EqualizedLinear(chainer.Chain):
    def __init__(self, in_ch, out_ch, initial_bias=None, nobias=False, gain=np.sqrt(2), lrmul=1):
        w = chainer.initializers.Normal(1.0/lrmul) # equalized learning rate
        self.inv_c = gain * np.sqrt(1.0 / in_ch)
        self.inv_c = self.inv_c * lrmul
        super(EqualizedLinear, self).__init__()
        with self.init_scope():
            self.c = L.Linear(in_ch, out_ch, initialW=w, initial_bias=initial_bias, nobias=nobias)
            
    def __call__(self, x):
        return self.c(self.inv_c * x)
        

def minibatch_std(x):
    m = F.mean(x, axis=0, keepdims=True)
    v = F.mean((x - F.broadcast_to(m, x.shape)) * (x - F.broadcast_to(m, x.shape)), axis=0, keepdims=True)
    std = F.mean(F.sqrt(v + 1e-8), keepdims=True)
    std = F.broadcast_to(std, (x.shape[0], 1, x.shape[2], x.shape[3]))
    return F.concat([x, std], axis=1)
    