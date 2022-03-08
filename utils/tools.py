from utils.process import *
import torch
import torch.optim
import torch.nn as nn
import cv2
from math import sqrt, exp
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation

class Blur(nn.Module):
    def __init__(self, n_planes, kernel_type, im_shape, kernel_parameter=None, kernel_width=None):
        super(Blur, self).__init__()

        if kernel_type in ['gauss', 'circle']:
            kernel_type_ = kernel_type
        else:
            assert False, 'wrong kernel type'
        
        assert kernel_parameter, 'radius or sigma is not specified'

        if kernel_width is None:
          kernel_width = int(2*kernel_parameter + 8) #3)

        if(kernel_width%2 == 0):
          kernel_width += 1

        self.kernel = get_kernel(kernel_type_, kernel_width, kernel_parameter)

        blur = nn.Conv2d(n_planes, n_planes, kernel_size=self.kernel.shape,
                                padding='same', padding_mode='replicate')
        blur.weight.data[:] = 0
        blur.bias.data[:] = 0

        kernel_torch = torch.from_numpy(self.kernel)

        for i in range(n_planes):
            blur.weight.data[i, i] = kernel_torch

        self.blur_ = blur
        
        self.a = nn.Parameter(torch.ones([1,1,im_shape[1],im_shape[2]])*0.35)
        self.b = nn.Parameter(torch.ones([1,1,im_shape[1],im_shape[2]])*0.6)

    def forward(self, input):
        res_conv = self.blur_(input)
        output = self.a*res_conv + self.b

        return output

def get_kernel(kernel_type, kernel_width, kernel_parameter, norm=True):
    kernel = np.zeros([kernel_width, kernel_width])

    if kernel_type == 'gauss':
        sigma = kernel_parameter
        center = (kernel_width + 1.)/2.
        sigma_sq =  sigma**2

        for i in range(1, kernel.shape[0] + 1):
            for j in range(1, kernel.shape[1] + 1):
                di = (i - center)/2.
                dj = (j - center)/2.
                kernel[i - 1][j - 1] = np.exp(-(di * di + dj * dj)/(2 * sigma_sq))
                kernel[i - 1][j - 1] = kernel[i - 1][j - 1]/(2. * np.pi * sigma_sq)

    elif kernel_type == 'circle':
      r = kernel_parameter
      n=kernel_width
      m=kernel_width

      kernel=np.ones((m,n))/(np.pi*r**2)

      x=np.arange(-np.fix(n/2),np.ceil(n/2))
      y=np.arange(-np.fix(m/2),np.ceil(m/2))

      X,Y=np.meshgrid(x,y)

      mask = (X)**2 + (Y)**2 < r**2

      kernel = mask*kernel
    else:
        assert False, 'wrong kernel type'

    if(norm):
      kernel /= kernel.sum()
    else:
      kernel = 1*(kernel>0)

    return kernel

class Rings(nn.Module):
  def __init__(self, ini_kernel, n_down, n_up, dtype):
    super().__init__()

    rings = []

    mask = 1*(ini_kernel>0)

    for i in range(n_up):
      over = 1*binary_dilation(mask)
      kernel = (over-mask)
      rings.insert(0, kernel)

      mask = over.copy()
    
    mask = 1*(ini_kernel>0)

    for i in range(n_down):
      sub = 1*binary_erosion(mask)
      kernel = (mask-sub)
      rings.append(kernel)

      if(np.sum(sub)==0):
        break
      
      mask = sub.copy()

    self.fix = torch.from_numpy(sub*ini_kernel).type(dtype)
    self.rings = torch.from_numpy(np.array(np.expand_dims(rings,0))).type(dtype)

  def forward(self, X):
    f = nn.ReLU()
    sep = torch.multiply(self.rings,f(X))
    val = (torch.sum(sep,dim=[0,2,3])/torch.sum(self.rings,dim=[0,2,3])).view(1,-1,1,1)
    output = torch.multiply(self.rings,val)
    output = torch.sum(output,dim=1,keepdim=True)
    output = output + self.fix
    output = output / output.sum()

    return output

def fill_noise(x, noise_type):
    """Fills tensor `x` with noise of type `noise_type`."""
    if noise_type == 'u':
        x.uniform_()
    elif noise_type == 'n':
        x.normal_()
    else:
        assert False

def get_noise(input_depth, method, spatial_size, noise_type='u', var=1./10):
    """Returns a pytorch.Tensor of size (1 x `input_depth` x `spatial_size[0]` x `spatial_size[1]`)
    initialized in a specific way.
    Args:
        input_depth: number of channels in the tensor
        method: `noise` for fillting tensor with noise; `meshgrid` for np.meshgrid
        spatial_size: spatial size of the tensor to initialize
        noise_type: 'u' for uniform; 'n' for normal
        var: a factor, a noise will be multiplicated by. Basically it is standard deviation scaler.
    """
    if isinstance(spatial_size, int):
        spatial_size = (spatial_size, spatial_size)
    if method == 'noise':
        shape = [1, input_depth, spatial_size[0], spatial_size[1]]
        net_input = torch.zeros(shape)

        fill_noise(net_input, noise_type)
        net_input *= var
    elif method == 'meshgrid':
        assert input_depth == 2
        X, Y = np.meshgrid(np.arange(0, spatial_size[1])/float(spatial_size[1]-1), np.arange(0, spatial_size[0])/float(spatial_size[0]-1))
        meshgrid = np.concatenate([X[None,:], Y[None,:]])
        net_input=  np_to_torch(meshgrid)
    else:
        assert False

    return net_input

def get_params(opt_over, net, net_input, blur=None):
    '''Returns parameters that we want to optimize over.
    Args:
        opt_over: comma separated list, e.g. "net,input" or "net"
        net: network
        net_input: torch.Tensor that stores input `z`
    '''
    opt_over_list = opt_over.split(',')
    params = []
    params_reduc = []

    for opt in opt_over_list:
        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='blur':
            assert blur is not None
            for x in blur.parameters():
              if(len(x.size())==4 and x.requires_grad):
                params_reduc += [x]
              else:
                params += [x]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params, params_reduc

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size,channel,dtype):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous().type(dtype)
    return window

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(nn.Module):
    def __init__(self, dtype=torch.cuda.FloatTensor, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.dtype = dtype
        self.window = create_window(window_size, self.channel, dtype)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel, self.dtype)

            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)

            self.window = window
            self.channel = channel

        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, dtype, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel,dtype)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)