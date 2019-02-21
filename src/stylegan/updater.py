#!/usr/bin/env python3

from itertools import accumulate
import os
import sys
import six

import chainer
from chainer.datasets import SubDataset
import chainer.functions as F
from chainer import Variable
import numpy as np
from config import get_lr_scale_factor

import chainer.computational_graph as c

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.abspath(os.path.dirname(__file__)) + os.path.sep + os.path.pardir)

from common.loss_functions import loss_func_dcgan_dis, loss_func_dcgan_gen, loss_l2
from common.utils.copy_param import soft_copy_param
from common.utils.pggan import downsize_real
from common.utils.save_images import convert_batch_images

class StageManager(object):

    def __init__(self, stage_interval, 
                        dynamic_batch_size, 
                        make_dataset_func, 
                        make_iterator_func, 
                        **kwargs):
        self.stage_interval = stage_interval
        self.dynamic_batch_size = dynamic_batch_size
        self._make_dataset_func = make_dataset_func
        self._make_iterator_func = make_iterator_func

        # Padding to the length + 1 so all calculation (such as self.ratio_in_stage)
        # remains valid when stage = max_stage
        self.dynamic_batch_count = list(map(lambda bs: (stage_interval + bs - 1) // bs, self.dynamic_batch_size)) + [1]
        self.dynamic_batch_count_previous = list(accumulate([0] + self.dynamic_batch_count))

        self.stage_int = 0
        self.counter_batch = 0

        self._iterators = {}
        self._datasets = {}

        self.total_batch = sum(self.dynamic_batch_count)

        if 'debug_start_instance' in kwargs:
            debug_start_instance = kwargs['debug_start_instance']
            while self.total_instance < debug_start_instance:
                self.tick_counter()

    @property
    def counter_batch_in_stage(self):
        return self.counter_batch - self.dynamic_batch_count_previous[self.stage_int]

    @property
    def total_batch_in_stage(self):
        return self.dynamic_batch_count[self.stage_int]

    @property
    def ratio_in_stage(self):
        return self.counter_batch_in_stage * 1.0 / self.total_batch_in_stage

    @property
    def stage(self):
        ratio_in_stage = self.ratio_in_stage
        assert 0. <= ratio_in_stage < 1.0
        return float(self.stage_int) + ratio_in_stage

    @property
    def should_stop(self):
        return self.stage_int >= len(self.dynamic_batch_size)

    @property
    def total_instance(self):
        count = 0
        for stage in range(self.stage_int):
            count += self.dynamic_batch_count[stage] * self.dynamic_batch_size[stage]
        count += self.counter_batch_in_stage * self.dynamic_batch_size[self.stage_int]
        return count

    def tick_counter(self):
        self.counter_batch += 1
        if self.counter_batch_in_stage == self.dynamic_batch_count[self.stage_int]:
            self.stage_int += 1

    def tick_a_batch(self):
        stage_int = self.stage_int

        key = 'stage%d' % stage_int
        # update dataset and iterator if needed.
        if key not in self._iterators:
            for old_key in list(self._iterators.keys()):
                self._iterators[old_key].finalize()
                current_dataset = self._datasets[old_key]
                if isinstance(current_dataset, SubDataset):
                    current_dataset = current_dataset._dataset  #pylint:disable=protected-access
                current_dataset.close()
                del self._iterators[old_key]
                del self._datasets[old_key]

            new_dataset = self._make_dataset_func(stage_int)
            batch_size = self.dynamic_batch_size[self.stage_int]
            new_iterator = self._make_iterator_func(new_dataset, batch_size)
            self._datasets[key] = new_dataset
            self._iterators[key] = new_iterator

        batch = self._iterators[key].next()

        # tick counters
        self.tick_counter()

        return batch

    def serialize(self, serializer):
        for name, iterator in six.iteritems(self._iterators):
            iterator.serialize(serializer['iterator:' + name])

        self.stage_int = int(serializer('stage_int', self.stage_int))
        self.counter_batch = int(serializer('counter_batch', self.counter_batch))


class Updater(chainer.training.Updater):

    def __init__(self, models, optimizer, stage_manager, device=None, **kwargs):
        if len(models) == 3:
            models = models + [None, None]
        self.map, self.gen, self.dis, self.smoothed_gen, self.smoothed_map = models

        assert isinstance(optimizer, dict)
        self._optimizers = optimizer

        if device is not None and device >= 0:
            for _optimizer in six.itervalues(self._optimizers):
                _optimizer.target.to_gpu(device)
        self.device = device

        # Stage manager
        self.stage_manager = stage_manager

        # Parse kwargs for updater
        self.use_cleargrads = kwargs.pop('use_cleargrads')
        self.smoothing = kwargs.pop('smoothing')
        self.lambda_gp = kwargs.pop('lambda_gp')

        self.total_gpu = kwargs.pop('total_gpu')

        self.style_mixing_rate = kwargs.pop('style_mixing_rate')

    def finalize(self):
        pass

    def get_optimizer(self, name):
        return self._optimizers[name]

    def get_all_optimizers(self):
        return dict(self._optimizers)

    @property
    def iteration(self):
        return self.stage_manager.counter_batch

    @property
    def epoch(self):
        return self.stage_manager.stage_int

    @property
    def epoch_detail(self):
        return self.stage_manager.ratio_in_stage

    @property
    def total_iteration(self):
        return self.stage_manager.total_batch

    @property
    def stage(self):
        return self.stage_manager.stage

    def get_x_real_data(self, batch, batch_size):
        xp = self.gen.xp
        x_real_data = []
        for i in range(batch_size):
            this_instance = batch[i]
            if isinstance(this_instance, tuple):
                this_instance = this_instance[0]  # It's (data, data_id), so take the first one.
            x_real_data.append(np.asarray(this_instance).astype("f"))
        x_real_data = xp.asarray(x_real_data)
        return x_real_data

    def get_z_fake_data(self, batch_size):
        xp = self.map.xp
        return xp.asarray(self.map.make_hidden(batch_size))

    def gen_cleargrads(self):
        if self.use_cleargrads:
            self.map.cleargrads()
            self.gen.cleargrads()
        else:
            self.map.zerograds()
            self.gen.zerograds()

    def dis_cleargrads(self):
        if self.use_cleargrads:
            self.dis.cleargrads()
        else:
            self.dis.zerograds()

    def update(self):
        xp = self.gen.xp
        
        self.gen_cleargrads()
        self.dis_cleargrads()
        
        opt_g_m = self.get_optimizer('map')
        opt_g_g = self.get_optimizer('gen')
        opt_d = self.get_optimizer('dis')

        # z: latent | x: data | y: dis output
        # *_real/*_fake/*_pertubed: Variable
        # *_data: just data (xp array)

        stage = self.stage  # Need to retrive the value since next statement may change state (at the stage boundary)
        batch = self.stage_manager.tick_a_batch()
        batch_size = len(batch)

        lr_scale = get_lr_scale_factor(self.total_gpu, stage)

        x_real_data = self.get_x_real_data(batch, batch_size)
        z_fake_data = self.get_z_fake_data(batch_size)

        x_real = Variable(x_real_data)
        # Image.fromarray(convert_batch_images(x_real.data.get(), 4, 4)).save('no_downsized.png')
        x_real = downsize_real(x_real, stage)
        x_real = Variable(x_real.data)
        # Image.fromarray(convert_batch_images(x_real.data.get(), 4, 4)).save('downsized.png')
        image_size = x_real.shape[2]
        z_fake = Variable(z_fake_data)
        w_fake = self.map(z_fake)

        if self.style_mixing_rate > 0 and np.random.rand() < self.style_mixing_rate:
            z_fake2 = Variable(self.get_z_fake_data(batch_size))
            w_fake2 = self.map(z_fake2)
            x_fake = self.gen(w_fake, stage=stage, w2=w_fake2)
        else:
            x_fake = self.gen(w_fake, stage=stage)
        y_fake = self.dis(x_fake, stage=stage)
        loss_gen = loss_func_dcgan_gen(y_fake) * lr_scale
        if chainer.global_config.debug:
            g = c.build_computational_graph(loss_gen)
            with open('out_loss_gen', 'w') as o:
                o.write(g.dump())
        assert not xp.isnan(loss_gen.data)
        chainer.report({'loss_adv': loss_gen}, self.gen)
        loss_gen.backward()
        opt_g_m.update()
        opt_g_g.update()

        # keep smoothed generator if instructed to do so.
        if self.smoothed_gen is not None:
            # layers_in_use = self.gen.get_layers_in_use(stage=stage)
            soft_copy_param(self.smoothed_gen, self.gen, 1.0 - self.smoothing)
            soft_copy_param(self.smoothed_map, self.map, 1.0 - self.smoothing)

        z_fake_data = self.get_z_fake_data(batch_size)
        z_fake = Variable(z_fake_data)

        with chainer.using_config('enable_backprop', False):
            w_fake = self.map(z_fake)
            if self.style_mixing_rate > 0 and np.random.rand() < self.style_mixing_rate:
                z_fake2 = Variable(self.get_z_fake_data(batch_size))
                w_fake2 = self.map(z_fake2)
                x_fake = self.gen(w_fake, stage=stage, w2=w_fake2)
            else:
                x_fake = self.gen(w_fake, stage=stage)

        x_fake.unchain_backward()
        y_fake = self.dis(x_fake, stage=stage)
        y_real = self.dis(x_real, stage=stage)
        loss_adv = loss_func_dcgan_dis(y_fake, y_real)

        if self.lambda_gp > 0:
            x_perturbed = x_real
            y_perturbed = y_real 
            # y_perturbed = self.dis(x_perturbed, stage=stage)
            grad_x_perturbed, = chainer.grad([y_perturbed], [x_perturbed], enable_double_backprop=True)
            grad_l2 = F.sqrt(F.sum(grad_x_perturbed ** 2, axis=(1, 2, 3)))
            loss_gp = self.lambda_gp * loss_l2(grad_l2, 0.0)
            chainer.report({'loss_gp': loss_gp}, self.dis)
        else:
            loss_gp = 0.

        loss_dis = ( loss_adv + loss_gp ) * lr_scale
        assert not xp.isnan(loss_dis.data)

        chainer.report({'loss_adv': loss_adv}, self.dis)

        self.dis_cleargrads()
        loss_dis.backward()
        opt_d.update()

        chainer.reporter.report({'stage': stage})
        chainer.reporter.report({'batch_size': batch_size})
        chainer.reporter.report({'image_size': image_size})

    def serialize(self, serializer):
        self.stage_manager.serialize(serializer['stage_manager:'])

        for name, optimizer in six.iteritems(self._optimizers):
            optimizer.serialize(serializer['optimizer:' + name])
            optimizer.target.serialize(serializer['model:' + name])
