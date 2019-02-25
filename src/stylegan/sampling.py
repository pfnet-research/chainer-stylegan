import os
import sys
import gflags
import tqdm
import numpy as np
import chainer
from chainer import Variable
from PIL import Image
from net import StyleGenerator, MappingNetwork 

FLAGS = gflags.FLAGS
gflags.DEFINE_integer('seed', '19260817', 'Random seed')
gflags.DEFINE_string('m_style', '', 'Style Generator model file')
gflags.DEFINE_string('m_mapping', '', 'Mapping Network model file')
gflags.DEFINE_float('stage', 17, 'Input Stage')
gflags.DEFINE_integer('n', 100, '# of output images')
gflags.DEFINE_integer('ch', 512, '# of channels')
gflags.DEFINE_string('out', 'results', 'Output folder')
gflags.DEFINE_string('img_name', 'images_', 'Output prefix')
gflags.DEFINE_integer('n_avg_w', 20000, '# of inputs for esitmating average W')
gflags.DEFINE_float('trc_psi', 0.7, 'Style strength multiplier for the truncation trick')
gflags.DEFINE_integer('gpu', 0, 'GPU ID')
gflags.DEFINE_boolean('enable_blur', False, 'Enable blur function after upscaling/downscaling')

def convert_batch_images(x, rows, cols):
    x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
    _, _, H, W = x.shape
    x = x.reshape((rows, cols, 3, H, W))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((rows * H, cols * W, 3)) 
    return x

def main():
    FLAGS(sys.argv)
    mapping = MappingNetwork(FLAGS.ch)
    gen = StyleGenerator(FLAGS.ch, FLAGS.enable_blur)
    chainer.serializers.load_npz(FLAGS.m_mapping, mapping)
    chainer.serializers.load_npz(FLAGS.m_style, gen)
    if FLAGS.gpu >= 0:
        chainer.cuda.get_device_from_id(FLAGS.gpu).use()
        mapping.to_gpu()
        gen.to_gpu()
    xp = gen.xp

    np.random.seed(FLAGS.seed)
    xp.random.seed(FLAGS.seed)

    enable_trunction_trick = FLAGS.trc_psi != 1.0
    
    if enable_trunction_trick:
        print("Calculate average W...")
        w_batch_size = 100
        n_batches = FLAGS.n_avg_w // w_batch_size
        w_avg = xp.zeros(FLAGS.ch).astype('f')
        with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
            for i in tqdm.tqdm(range(n_batches)):
                z = Variable(xp.asarray(mapping.make_hidden(w_batch_size)))
                w_cur = mapping(z)
                w_avg = w_avg + xp.average(w_cur.data, axis=0)
        w_avg = w_avg / n_batches

    np.random.seed(FLAGS.seed)
    xp.random.seed(FLAGS.seed)

    print("Generating...")
    os.makedirs(FLAGS.out, exist_ok=True)
    with chainer.using_config('train', False), chainer.using_config('enable_backprop', False):
        for i in tqdm.tqdm(range(FLAGS.n)):
            z = mapping.make_hidden(1)
            w = mapping(z).data
            if enable_trunction_trick:
                delta = w - w_avg
                w = delta * FLAGS.trc_psi + w_avg

            x = gen(w, FLAGS.stage)
            x = chainer.cuda.to_cpu(x.data)
            x = convert_batch_images(x, 1, 1)
            preview_path = FLAGS.out + '/' + FLAGS.img_name + str(i) + '.jpg'
            Image.fromarray(x).save(preview_path)

import pdb, traceback, sys, code 

if __name__ == '__main__':
    try:
        main()
    except: 
        type, value, tb = sys.exc_info()
        traceback.print_exc()
        pdb.post_mortem(tb)
        pdb.post_mortem(tb)