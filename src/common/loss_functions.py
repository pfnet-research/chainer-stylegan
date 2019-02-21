import numpy as np
import chainer.functions as F
from chainer import cuda


def loss_l1(h, t):
    return F.sum(F.absolute(h - t)) / np.prod(h.data.shape)


def loss_l1_no_avg(h, t):
    return F.sum(F.absolute(h - t)) / np.prod(h.data.shape[1:])


def loss_l2(h, t):
    return F.sum((h - t)**2) / np.prod(h.data.shape)


def loss_l2_no_avg(h, t):
    return F.sum((h - t)**2) / np.prod(h.data.shape[1:])


def loss_l2_norm(h, t, axis=(1)):
    return F.sum(F.sqrt(F.sum((h - t)**2, axis=axis))) / h.data.shape[0]


def loss_func_dcgan_gen(y_fake):
    return F.sum(F.softplus(-y_fake)) / np.prod(y_fake.data.shape)


def loss_func_dcgan_dis(y_fake, y_real):
    loss = F.sum(F.softplus(y_fake)) / np.prod(y_fake.data.shape)
    loss += F.sum(F.softplus(-y_real)) / np.prod(y_real.data.shape)
    return loss


def loss_func_hinge_gen(y_fake):
    loss = - F.mean(y_fake)
    return loss


def loss_func_hinge_dis(y_fake, y_real):
    loss = F.mean(F.relu(1. - y_real))
    loss += F.mean(F.relu(1. + y_fake))
    return loss


def loss_func_batch_hinge_gen(y_fake, cond_fake):
    loss = - F.sum(y_fake * cond_fake) / F.sum(cond_fake)
    # loss = - F.mean(y_fake)
    return loss


def loss_func_batch_hinge_dis(y_fake, cond_fake, y_real, cond_real):
    loss = F.sum(F.relu(1. - y_real) * cond_real) / F.sum(cond_real)
    loss += F.sum(F.relu(1. + y_fake) * cond_fake) / F.sum(cond_fake)
    # loss = F.mean(F.relu(1. - y_real))
    # loss += F.mean(F.relu(1. + y_fake))
    return loss


def loss_binary_cross_entropy(x, y, EPS=1e-10):
    return F.averae(-(F.log(x + EPS) * y + F.log(1. - x + EPS) * (1. - y)))


def loss_binary_cross_entropy_no_avg(x, y, EPS=1e-10):
    return F.sum(-(F.log(x + EPS) * y + F.log(1. - x + EPS) * (1. - y)))  # / np.prod(x.data.shape[1:])


def loss_sigmoid_cross_entropy_with_logits(x, t):
    return F.average(x - x * t + F.softplus(-x))


def loss_sigmoid_cross_entropy_with_logits_no_avg(x, t):
    return F.sum(x - x * t + F.softplus(-x))


def loss_func_tv_l1(x_out):
    xp = cuda.get_array_module(x_out.data)
    b, ch, h, w = x_out.data.shape
    Wx = xp.zeros((ch, ch, 2, 2), dtype="f")
    Wy = xp.zeros((ch, ch, 2, 2), dtype="f")
    for i in range(ch):
        Wx[i, i, 0, 0] = -1
        Wx[i, i, 0, 1] = 1
        Wy[i, i, 0, 0] = -1
        Wy[i, i, 1, 0] = 1
    return F.sum(F.absolute(F.convolution_2d(x_out, W=Wx))) + F.sum(F.absolute(F.convolution_2d(x_out, W=Wy)))


def loss_func_tv_l2(x_out):
    xp = cuda.get_array_module(x_out.data)
    b, ch, h, w = x_out.data.shape
    Wx = xp.zeros((ch, ch, 2, 2), dtype="f")
    Wy = xp.zeros((ch, ch, 2, 2), dtype="f")
    for i in range(ch):
        Wx[i, i, 0, 0] = -1
        Wx[i, i, 0, 1] = 1
        Wy[i, i, 0, 0] = -1
        Wy[i, i, 1, 0] = 1
    return F.sum(F.convolution_2d(x_out, W=Wx) ** 2) + F.sum(F.convolution_2d(x_out, W=Wy) ** 2)
