import gflags
import math
FLAGS = gflags.FLAGS

# hps (training dynamics)
gflags.DEFINE_integer('seed', 19260817, '')
gflags.DEFINE_float('adam_alpha_g', 0.001, 'alpha in Adam optimizer')
gflags.DEFINE_float('adam_alpha_d', 0.001, 'alpha in Adam optimizer')
gflags.DEFINE_float('adam_beta1', 0.0, 'beta1 in Adam optimizer')
gflags.DEFINE_float('adam_beta2', 0.999, 'beta2 in Adam optimizer')
gflags.DEFINE_float('lambda_gp', 1.0, 'Lambda GP')
gflags.DEFINE_float('smoothing', 0.999, '')
gflags.DEFINE_boolean('keep_smoothed_gen', False, 'Whether to keep a smoothed version of generator.')
gflags.DEFINE_string('dynamic_batch_size', '16,16,16,16,16,16,16,16,16,8,8,4,4,2,2,1,1',
                     'comma-split list of dynamic batch size w.p.t. stage')
gflags.DEFINE_integer('stage_interval', 600000, '')
gflags.DEFINE_integer('max_stage', 13, 'Size of image.')


gflags.DEFINE_boolean('auto_resume', False, 'Whether to automatically resume')

# algorithm & architecture
gflags.DEFINE_integer('ch', 512, '#Channels')
gflags.DEFINE_integer('debug_start_instance', 0, 'Change starting iteration for debugging.')


# hps (device)
gflags.DEFINE_integer('gpu', 0, 'GPU ID (negative value indicates CPU)')
gflags.DEFINE_boolean('use_mpi', False, 'Whether to use MPI for multi-GPU training.')
gflags.DEFINE_string('comm_name', 'pure_nccl', 'ChainerMN communicator name')
gflags.DEFINE_boolean('enable_cuda_profiling', False, 'Whether to enable CUDA profiling.')


# hps (I/O)
gflags.DEFINE_string('out', 'result', 'Directory to output the result')
gflags.DEFINE_string('auto_resume_dir', '', 'Directory for loading the saved models')
gflags.DEFINE_string('dataset_config', '', 'Dataset config json')
gflags.DEFINE_integer(
    'dataset_worker_num', 12,
    'Number of threads in dataset loader'
)

gflags.DEFINE_integer('snapshot_interval', 5000, 'Interval of snapshot')
gflags.DEFINE_integer('evaluation_sample_interval', 500, 'Interval of evaluation sampling')
gflags.DEFINE_integer('display_interval', 100, 'Interval of displaying log to console')
gflags.DEFINE_string('get_model_from_interation', '', 'Load this iteration (it is a string)')

# hps FID
gflags.DEFINE_integer('fid_interval', 0, 'Enable FID when > 0')
gflags.DEFINE_string('fid_real_stat', '', 'Save NPZ of real images')
gflags.DEFINE_string('fid_clfs_type', '', 'i2v_v5/inception')
gflags.DEFINE_string('fid_clfs_path', '', 'classifier path')
gflags.DEFINE_boolean('fid_skip_first', False, 'Whether to skip FID calculation when iter = 0')

# Style GAN
gflags.DEFINE_float('style_mixing_rate', 0.9, ' Style Mixing Prob')
gflags.DEFINE_boolean('enable_blur', False, 'Enable blur function after upscaling/downscaling')


stage2reso = {
    0: 4,
    1: 8,
    2: 8,
    3: 16,
    4: 16,
    5: 32,
    6: 32,
    7: 64,
    8: 64,
    9: 128,
    10: 128,
    11: 256,
    12: 256,
    13: 512,
    14: 512,
    15: 1024,
    16: 1024,
    17: 1024
}

gpu_lr = {
    1: {15: 1.5, 16: 1.5, 17: 1.5},
    2: {13: 1.5, 14: 1.5, 15: 2, 16: 2, 17: 2},
    3: {11: 1.5, 12: 1.5, 13: 2, 14: 2, 15: 2.5, 16: 2.5, 17: 2.5},
    4: {11: 1.5, 12: 1.5, 13: 2, 14: 2, 15: 3, 16: 3, 17: 3},
    8: {9: 1.5, 10: 1.5, 11: 2, 12: 2, 13: 3, 14: 3, 15: 3, 16: 3, 17: 3},
}

def get_lr_scale_factor(total_gpu, stage):
    gpu_lr_d = gpu_lr.get(total_gpu, gpu_lr[1])
    stage = math.floor(stage)
    if stage >= 18:
        return gpu_lr_d[17]
    return gpu_lr_d.get(stage, 1)
