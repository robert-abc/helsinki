import numpy as np
from PIL import Image
import torch

def pil_to_np(img_PIL):
    '''Converts image in PIL format to np.array.

    From W x H x C [0...255] to C x W x H [0..1]
    '''
    ar = np.array(img_PIL)

    if len(ar.shape) == 3:
        ar = ar.transpose(2,0,1)
    else:
        ar = ar[None, ...]

    return ar.astype(np.float32) / np.max(ar)

def np_to_pil(img_np):
    '''Converts image in np.array format to PIL image.

    From C x W x H [0..1] to  W x H x C [0...255]
    '''
    ar = np.clip(img_np*255,0,255).astype(np.uint8)

    if img_np.shape[0] == 1:
        ar = ar[0]
    else:
        ar = ar.transpose(1, 2, 0)

    return Image.fromarray(ar)

def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.
    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.
    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]

def load(path):
    """Load PIL image."""
    img = Image.open(path)
    return img

def get_image(path, imsize=-1):
    """Load an image and resize to a cpecific size.
    Args:
        path: path to image
        imsize: tuple or scalar with dimensions; -1 for `no resize`
    """
    img = load(path)

    if isinstance(imsize, int):
        imsize = (imsize, imsize)

    if imsize[0]!= -1 and img.size != imsize:
        if imsize[0] > img.size[0]:
            img = img.resize(imsize, Image.BICUBIC)
        else:
            img = img.resize(imsize, Image.ANTIALIAS)

    img_np = pil_to_np(img)

    return img, img_np

def load_img(fname, width, enforse_div32=None):
    '''Loads an image, resizes it, center crops and downscales.
    Args:
        fname: path to the image
        width: desired width size
        enforse_div32: if 'EXTEND' pads bottom and right side of image,
                       so that its dimensions are divisible by 32.
    '''
    _,img_np = get_image(fname, -1)
    img_pil = np_to_pil(img_np)

    factor = width / img_pil.size[0]

    orig_dim=[img_pil.size[0],img_pil.size[1]]

    LR_size = [
               width,
               np.round(img_pil.size[1]*factor).astype(int)
    ]

    img_LR_pil = img_pil.resize(LR_size, Image.BICUBIC)
    img_LR_np = pil_to_np(img_LR_pil)

    new_size=[0,0]

    if enforse_div32 == 'EXTEND':
        new_size = [32 - img_LR_np.shape[1] % 32,
                    32 - img_LR_np.shape[2] % 32]

        new_size = (np.array(new_size)!=32)*new_size

        bbox = (
                (0,0),
                (0,new_size[0]),
                (0,new_size[1])
                )

        img_LR_np = np.pad(img_LR_np,bbox,mode='reflect')

    return img_LR_np, orig_dim, new_size

def save_img(img_arr, fname, orig_dim, extend_dim):
    img_arr=img_arr[0:img_arr.shape[1]-extend_dim[0],
                    0:img_arr.shape[2]-extend_dim[1]]

    img_pil=np_to_pil(img_arr)

    img_pil=img_pil.resize(orig_dim, Image.ANTIALIAS)
    img_pil.save(fname)
