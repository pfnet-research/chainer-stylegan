from PIL import Image
import numpy as np
from chainer import cuda
import chainer
try:
    import cupy
except:
    pass
import os

from .image_processing import copy_to_cpu, postprocessing_tanh, postprocessing_sigmoid


def save_single_image(img, path, using_tanh=True):
    img = copy_to_cpu(img)
    if using_tanh:
        img = postprocessing_tanh(img)
    else:
        img = postprocessing_sigmoid(img)
    #ch, w, h = img.shape
    img = img.transpose((1, 2, 0))
    pilImg = Image.fromarray(img)
    pilImg.save(path, "JPEG")
    # cv2.imwrite(path, img)

def create_image_grid(imgs, grid_w=4, grid_h=4,
            using_tanh=True, transposed=False, bgr2rgb=False):

    imgs = copy_to_cpu(imgs)

    if using_tanh:
        imgs = postprocessing_tanh(imgs)
    else:
        imgs = postprocessing_sigmoid(imgs)

    b, ch, w, h = imgs.shape
    assert b == grid_w*grid_h

    if bgr2rgb:
        imgs2 = imgs.copy()
        imgs[:,0,:,:], imgs[:,2,:,:] = imgs2[:,2,:,:], imgs2[:,0,:,:]

    imgs = imgs.reshape((grid_w, grid_h, ch, w, h))
    imgs = imgs.transpose(0, 1, 3, 4, 2)

    if transposed:
        imgs = imgs.reshape((grid_w, grid_h, w, h, ch)).transpose(1, 2, 0, 3, 4).reshape((grid_h*w, grid_w*h, ch))
    else:
        imgs = imgs.reshape((grid_w, grid_h, w, h, ch)).transpose(0, 2, 1, 3, 4).reshape((grid_w*w, grid_h*h, ch))

    # Automaticly compress grayscale
    if ch==1:
        imgs = imgs.reshape((grid_w*w, grid_h*h))

    return imgs

def save_images_grid(imgs, path, grid_w=4, grid_h=4,
            using_tanh=True, transposed=False, bgr2rgb=False):

    imgs = copy_to_cpu(imgs)

    if using_tanh:
        imgs = postprocessing_tanh(imgs)
    else:
        imgs = postprocessing_sigmoid(imgs)


    b, ch, w, h = imgs.shape
    assert b == grid_w*grid_h

    imgs = imgs.reshape((grid_w, grid_h, ch, w, h))
    imgs = imgs.transpose(0, 1, 3, 4, 2)

    if transposed:
        imgs = imgs.reshape((grid_w, grid_h, w, h, ch)).transpose(1, 2, 0, 3, 4).reshape((grid_h*w, grid_w*h, ch))
    else:
        imgs = imgs.reshape((grid_w, grid_h, w, h, ch)).transpose(0, 2, 1, 3, 4).reshape((grid_w*w, grid_h*h, ch))

    if ch==1:
        imgs = imgs.reshape((grid_w*w, grid_h*h))


    pilImg = Image.fromarray(imgs)
    pilImg.save(path, "JPEG")
    # if bgr2rgb:
    #    imgs = cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)
    # cv2.imwrite(path, imgs)

def convert_batch_images(x, rows, cols):
    x = np.asarray(np.clip(x * 127.5 + 127.5, 0.0, 255.0), dtype=np.uint8)
    _, _, H, W = x.shape
    x = x.reshape((rows, cols, 3, H, W))
    x = x.transpose(0, 3, 1, 4, 2)
    x = x.reshape((rows * H, cols * W, 3)) 
    return x
