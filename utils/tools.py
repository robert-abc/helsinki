from utils.process import *
import torch
import torch.optim
import torch.nn as nn
import cv2
from math import sqrt, exp
import torch.nn.functional as F
from utils.autoencoder_tools import get_dl_estim

class Blur(nn.Module):
    def __init__(self, n_planes, kernel_type, kernel_parameter=None, kernel_width=None):
        super(Blur, self).__init__()

        if kernel_type in ['gauss', 'circle']:
            kernel_type_ = kernel_type
        else:
            assert False, 'wrong kernel type'
        
        assert kernel_parameter, 'radius or sigma is not specified'

        if kernel_width is None:
          kernel_width = int(2*kernel_parameter + 3)

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

    def forward(self, input):
        return self.blur_(input)

def get_kernel(kernel_type, kernel_width, kernel_parameter):
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

    kernel /= kernel.sum()

    return kernel

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

    for opt in opt_over_list:

        if opt == 'net':
            params += [x for x in net.parameters() ]
        elif  opt=='blur':
            assert blur is not None
            params = [x for x in blur.parameters()]
        elif opt == 'input':
            net_input.requires_grad = True
            params += [net_input]
        else:
            assert False, 'what is it?'

    return params

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
    def __init__(self, window_size=11, size_average=True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)

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

def deblur_image(deblur_net, deblur_input, blur, img_np,
        OPT_OVER, num_iter, reg_noise_std, LR, iter_lr, iter_mean,
        dtype, autoencoder, iter_dl, dl_param):

    img_torch = torch.from_numpy(img_np).type(dtype)
    out_mean_deblur = np.zeros(img_np.shape)
    torch_dl = torch.zeros(img_torch[0].size())

    ind_dl=-1

    net_input_saved = deblur_input.detach().clone()
    noise = deblur_input.detach().clone()

    p = get_params(OPT_OVER,deblur_net,deblur_input,blur)

    optimizer = torch.optim.Adam(p, lr=LR)

    for i in range(num_iter):
        optimizer.zero_grad()

        if reg_noise_std > 0:
            deblur_input = net_input_saved + (noise.normal_() * reg_noise_std)
        else:
            deblur_input = net_input_saved

        out_sharp = deblur_net(deblur_input)

        if autoencoder is not None:
            if i in iter_dl:
                if i == iter_dl[0]:
                    out_sharp_np = torch_to_np(out_sharp)
                else:
                    out_sharp_np = torch_to_np((1-dl_param[ind_dl])*out_sharp+dl_param[ind_dl]*torch_dl)

                img_dl = get_dl_estim(out_sharp_np[0], autoencoder, dtype)
                img_dl = np.expand_dims(img_dl, axis=0)
                torch_dl = np_to_torch(img_dl).type(dtype)
                ind_dl += 1

            if i >= iter_dl[0]:
              out_sharp = ((1-dl_param[ind_dl])*out_sharp+dl_param[ind_dl]*torch_dl)

        out_blur = blur(out_sharp)

        total_loss = 1 - ssim(out_blur, img_torch, dtype)

        if i >= iter_mean:
          out_sharp_np = torch_to_np(out_sharp)
          out_mean_deblur += out_sharp_np

        total_loss.backward()

        optimizer.step()
    
    return out_mean_deblur