import chainer
from .optimizers.optimAdam import OptimAdam
try:
    import chainermn
except:
    pass

def make_adam(model, lr=0.0002, beta1=0.9, beta2=0.999):
    optimizer = chainer.optimizers.Adam(alpha=lr, beta1=beta1, beta2=beta2)

    if chainer.config.using_chainermn:
        optimizer = chainermn.create_multi_node_optimizer(optimizer, chainer.config.communicator)

    optimizer.setup(model)
    return optimizer


def make_optim_adam(model, lr=0.0002, beta1=0.9, beta2=0.999):
    optimizer = OptimAdam(alpha=lr, beta1=beta1, beta2=beta2)
    
    if chainer.config.using_chainermn:
        optimizer = chainermn.create_multi_node_optimizer(optimizer, chainer.config.communicator)

    optimizer.setup(model)
    return optimizer

