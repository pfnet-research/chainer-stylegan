import numpy as np
import chainer.functions as F

def loss_l1(h, t):
    return F.sum(F.absolute(h - t)) / np.prod(h.data.shape)


def loss_l1_no_avg(h, t):
    return F.sum(F.absolute(h - t)) / np.prod(h.data.shape[1:])


def loss_l2(h, t):
    return F.sum((h - t)**2) / np.prod(h.data.shape)


def loss_l2_no_avg(h, t):
    return F.sum((h - t)**2) / np.prod(h.data.shape[1:])


def loss_func_dcgan_gen(y_fake):
    return F.sum(F.softplus(-y_fake)) / np.prod(y_fake.data.shape)


def loss_func_dcgan_dis(y_fake, y_real):
    loss = F.sum(F.softplus(y_fake)) / np.prod(y_fake.data.shape)
    loss += F.sum(F.softplus(-y_real)) / np.prod(y_real.data.shape)
    return loss
