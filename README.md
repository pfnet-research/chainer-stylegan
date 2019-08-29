# Chainer implementation of Style-based Generator
A Style-Based Generator Architecture for Generative Adversarial Networks

[https://arxiv.org/abs/1812.04948](https://arxiv.org/abs/1812.04948)  

## Requirements
```
opencv-python
python-gflags
Augmentor
h5py
Pillow
scipy
mpi4py
chainer >= 5.0.0
cupy >= 5.0.0

```
* python >= 3.6.0
* NVIDIA driver 391.35 or newer, CUDA toolkit 9.0 or newer, cuDNN 7.3.1 or newer.
* NCCL2 
* A graphic card with at least 11GB memory to train the 1024x1024 model.
* Tested on 8 Tesla P100.

## Datasets

1. Please follow [ffhq-dataset](https://github.com/NVlabs/ffhq-dataset) to obtain the ffhq dataset.

```
python download_ffhq.py -h -i
```

2. Convert raw ffhq images to a HDF5 file. (Around 198GB)

```
cd src/hdf5_tools
bash folder_to_multisize_hdf5_cmds.sh 1 YOUR_PATH_TO_RAW_FFHQ_IMAGES
```

## Run

* 8 GPUs setting
```
cd src/stylegan
bash run_ffhq.sh 2 
```

* 1 GPU setting (up to 256x256)
```
cd src/stylegan
bash run_ffhq.sh 1
```

## Pre-trained Model on FFHQ
[GDrive](https://drive.google.com/open?id=1Yde3i7knsJ3JK9nZ_oHngvweQODFVLD6)

Sampling images on CPU
`
python sampling.py --m_style SmoothedGenerator_405000.npz --m_mapping SmoothedMapping_405000.npz --gpu -1
`
## Results

![samples](https://i.imgur.com/OUDgy5y.jpg)


