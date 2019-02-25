#!/usr/bin/env python3
'''Converts folder dataset to hdf5 dataset.'''

import math
import os
import sys
import io
import json
import glob
import pickle
import argparse
import threading
import queue
import traceback

import numpy as np
import scipy.ndimage
from PIL import Image
import h5py
import numpy as np

#----------------------------------------------------------------------------
'''Modified from https://github.com/tkarras/progressive_growing_of_gans/blob/master/h5tool.py'''

class HDF5Exporter:
    def __init__(self, h5_filename, resolution, channels=3, buffer_size_mb=512):
        rlog2 = int(np.floor(np.log2(resolution)))
        assert resolution == 2 ** rlog2
        self.resolution = resolution
        self.channels = channels
        self.h5_file = h5py.File(h5_filename, 'w', libver='latest')
        self.h5_lods = []
        self.lods = []
        self.buffers = []
        self.buffer_sizes = []
        self.metadata = {}
        for lod in range(rlog2, -1, -1):
            r = 2 ** lod; c = channels
            bytes_per_item = c * (r ** 2)
            chunk_size = int(np.ceil(128.0 / bytes_per_item))
            buffer_size = int(np.ceil(float(buffer_size_mb) * np.exp2(20) / bytes_per_item))
            lod = self.h5_file.create_dataset('%dx%d' % (r,r), shape=(0,c,r,r), dtype=np.uint8,
                maxshape=(None,c,r,r), chunks=(chunk_size,c,r,r), compression='gzip', compression_opts=4)
            self.metadata['%dx%d' % (r, r)] = []
            self.h5_lods.append(lod)
            self.lods.append('%dx%d' % (r, r))
            self.buffers.append(np.zeros((buffer_size,c,r,r), dtype=np.uint8))
            self.buffer_sizes.append(0)
        print('HDF5 Exporter will use following LODs', self.lods)
 

    def close(self):
        for lod in range(len(self.h5_lods)):
            self.flush_lod(lod)
        self.h5_file.close()

    def add_images(self, img, fn):
        assert img.ndim == 4 and img.shape[1] == self.channels and img.shape[2] == img.shape[3]
        # assert img.shape[2] >= self.resolution and img.shape[2] == 2 ** int(np.floor(np.log2(img.shape[2])))
        if img.shape[2] == 512:
            start = 1
        elif img.shape[2] == 256:
            start = 2
        else:
            start = 0
        
        for lod in range(start, len(self.h5_lods)):
            while img.shape[2] > self.resolution / (2 ** lod):
                img = img.astype(np.float32)
                img = (img[:, :, 0::2, 0::2] + img[:, :, 0::2, 1::2] + img[:, :, 1::2, 0::2] + img[:, :, 1::2, 1::2]) * 0.25
            quant = np.uint8(np.clip(np.round(img), 0, 255))
            ofs = 0
            self.metadata[self.lods[lod]].append(fn)
            while ofs < quant.shape[0]:
                num = min(quant.shape[0] - ofs, self.buffers[lod].shape[0] - self.buffer_sizes[lod])
                self.buffers[lod][self.buffer_sizes[lod] : self.buffer_sizes[lod] + num] = quant[ofs : ofs + num]
                self.buffer_sizes[lod] += num
                if self.buffer_sizes[lod] == self.buffers[lod].shape[0]:
                    self.flush_lod(lod)
                ofs += num
        del img

    def num_images(self):
        return self.h5_lods[0].shape[0] + self.buffer_sizes[0]

    def flush_lod(self, lod):
        num = self.buffer_sizes[lod]
        if num > 0:
            self.h5_lods[lod].resize(self.h5_lods[lod].shape[0] + num, axis=0)
            self.h5_lods[lod][-num:] = self.buffers[lod][:num]
            self.buffer_sizes[lod] = 0

#----------------------------------------------------------------------------

class ExceptionInfo(object):
    def __init__(self):
        self.type, self.value = sys.exc_info()[:2]
        self.traceback = traceback.format_exc()

#----------------------------------------------------------------------------

class WorkerThread(threading.Thread):
    def __init__(self, task_queue):
        threading.Thread.__init__(self)
        self.task_queue = task_queue

    def run(self):
        while True:
            func, args, result_queue = self.task_queue.get()
            if func is None:
                break
            try:
                result = func(*args)
            except:
                result = ExceptionInfo()
            result_queue.put((result, args))

#----------------------------------------------------------------------------

class ThreadPool(object):
    def __init__(self, num_threads):
        assert num_threads >= 1
        self.task_queue = queue.Queue()
        self.result_queues = dict()
        self.num_threads = num_threads
        for idx in range(self.num_threads):
            thread = WorkerThread(self.task_queue)
            thread.daemon = True
            thread.start()

    def add_task(self, func, args=()):
        assert hasattr(func, '__call__') # must be a function
        if func not in self.result_queues:
            self.result_queues[func] = queue.Queue()
        self.task_queue.put((func, args, self.result_queues[func]))

    def get_result(self, func, verbose_exceptions=True): # returns (result, args)
        result, args = self.result_queues[func].get()
        if isinstance(result, ExceptionInfo):
            if verbose_exceptions:
                print('\n\nWorker thread caught an exception:\n' + result.traceback + '\n', end=' ')
            raise result.type(result.value)
        return result, args

    def finish(self):
        for idx in range(self.num_threads):
            self.task_queue.put((None, (), None))

    def __enter__(self): # for 'with' statement
        return self

    def __exit__(self, *excinfo):
        self.finish()

    def process_items_concurrently(self, item_iterator,
        process_func=lambda x: x, pre_func=lambda x: x, post_func=lambda x: x, max_items_in_flight=None):

        if max_items_in_flight is None: max_items_in_flight = self.num_threads * 4
        assert max_items_in_flight >= 1
        results = []
        retire_idx = [0]

        def task_func(prepared, idx):
            return process_func(prepared)

        def retire_result():
            processed, (prepared, idx) = self.get_result(task_func)
            results[idx] = processed
            while retire_idx[0] < len(results) and results[retire_idx[0]] is not None:
                yield post_func(results[retire_idx[0]])
                results[retire_idx[0]] = None
                retire_idx[0] += 1

        for idx, item in enumerate(item_iterator):
            prepared = pre_func(item)
            results.append(None)
            self.add_task(func=task_func, args=(prepared, idx))
            while retire_idx[0] < idx - max_items_in_flight + 2:
                for res in retire_result(): yield res
        while retire_idx[0] < len(results):
            for res in retire_result(): yield res

#----------------------------------------------------------------------------
# Helpers
def is_power_of_2(v):
    rlog2 = int(np.floor(np.log2(v)))
    return 2 ** rlog2 == v

#----------------------------------------------------------------------------


import gflags
FLAGS = gflags.FLAGS

# hps: I/O
gflags.DEFINE_string('folder_path', '', '')
gflags.DEFINE_string('h5_filename', '', '')

# hps: config
gflags.DEFINE_integer('image_size', 1024, 'image sizes')
gflags.DEFINE_integer('min_input_image_size', -1, 'minimal input image sizes')
gflags.DEFINE_integer('num_threads', 4, 'Number of concurrent threads')
gflags.DEFINE_integer('num_tasks', 100 , 'Number of concurrent processing tasks')
gflags.DEFINE_integer('buffer_size_mb', 1024 , 'Size of buffer in MB.')
gflags.DEFINE_string(
    'resample', 'ANTIALIAS',
    'Method for resampling. see https://pillow.readthedocs.io/en/4.3.x/handbook/concepts.html#concept-filters'
)

def main():  #pylint: disable=missing-docstring
    FLAGS(sys.argv)

    glob_pattern = os.path.join(FLAGS.folder_path, '**/*')
    files = glob.glob(glob_pattern, recursive=True)
    files = [_ for _ in files if _.endswith('.jpg') or _.endswith('.png')]
    files = list(sorted(files))
    files = np.random.permutation(files)
    print(('Found %d images in %s' % (len(files), FLAGS.folder_path)))
    if len(files) == 0:
        print("No image found!")
        return

    if FLAGS.min_input_image_size > 0:
        _thres = FLAGS.min_input_image_size
        _pred = lambda img: img.size[0] >= _thres and img.size[1] > _thres
        files = list([f for f in files if _pred(Image.open(f))])

        print('Filtered images with less than %d size. %d images remain.' % (FLAGS.min_input_image_size, len(files)))
        if len(files) == 0:
            print("No image remains!")
            return

    if not is_power_of_2(FLAGS.image_size):
        print(("Image size specified as %d is invalid. It must be a power of 2." % FLAGS.image_size))
        return

    relative_files = [os.path.basename(_) for _ in files]


    def process_func(filepath):
        img = Image.open(filepath)
        if img.size[0] >= 800:
            img = img.resize((1024, 1024), resample=getattr(Image, FLAGS.resample))
        elif img.size[0] >= 400:
            img = img.resize((512, 512), resample=getattr(Image, FLAGS.resample))
        else:
            img = img.resize((256, 256), resample=getattr(Image, FLAGS.resample))

       #  img = img.resize((FLAGS.image_size, FLAGS.image_size), resample=getattr(Image, FLAGS.resample))
        imgx = img.convert('RGB')
        img = np.asarray(imgx, dtype=np.uint8)
        img = img.transpose((2, 0, 1))
        if img.shape[0] == 1:
            img = np.broadcast_to(img, (3, img.shape[1], img.shape[2]))
        imgx.close()
        return img

    h5_filename = FLAGS.h5_filename
    print(('Creating %s' % h5_filename))
    h5 = HDF5Exporter(h5_filename, FLAGS.image_size, 3, buffer_size_mb=FLAGS.buffer_size_mb)

    with ThreadPool(FLAGS.num_threads) as pool:
        print('%d / %d\r' % (0, len(files)), end='')
        for idx, img in enumerate(pool.process_items_concurrently(files, process_func=process_func, max_items_in_flight=FLAGS.num_tasks)):
            h5.add_images(img[np.newaxis], relative_files[idx])
            print('%d / %d\r' % (idx + 1, len(files)), end='')

    print('%-40s\r' % 'Flushing data...', end='')
    h5.close()
    print('%-40s\r' % '', end='')
    print('Added %d images.' % len(files))
    metadata_filename = FLAGS.h5_filename + '.metadata.json'
    json.dump(h5.metadata, open(metadata_filename, 'w'))


import pdb, traceback, sys, code  #pylint: disable=multiple-imports,wrong-import-position,wrong-import-order,unused-import,ungrouped-imports,reimported

if __name__ == '__main__':
    try:
        main()
    except:  #pylint:disable=bare-except
        type, value, tb = sys.exc_info()  #pylint:disable=invalid-name,redefined-builtin
        traceback.print_exc()
        pdb.post_mortem(tb)
