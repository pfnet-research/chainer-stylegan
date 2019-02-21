import numpy as np
import chainer.functions as F

def upscale2x(h):
    return F.unpooling_2d(h, 2, 2, 0, outsize=(h.shape[2] * 2, h.shape[3] * 2))

def downscale2x(h):
    return F.average_pooling_2d(h, 2, 2, 0)

def blur(h, w_k):
    b, ch, w_s, h_s = h.shape
    h = F.reshape(h, (b*ch, 1, w_s, h_s))
    h = F.convolution_2d(h, w_k, stride=1, pad=1)
    h = F.reshape(h, (b, ch, w_s, h_s))
    return h
    