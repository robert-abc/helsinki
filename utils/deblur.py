from utils.process import *
from utils.tools import *
from utils.dip import *
import numpy as np

def deblur(img_np, blur, autoencoder, dtype, num_iter=1500, dl_param=[1e-2,1e-2,5e-3,5e-3]):
    #  Image Parameters
    width = 512 # Desired image width
    enforse_div32 = 'EXTEND' # Force image to have dims multiple of 32

    # Input Parameters
    input_depth = 32
    input_type = 'noise'
    OPT_OVER = 'net'

    # Optimization Parameters
    OPTIMIZER = 'adam'
    pad = 'reflection'
    NET_TYPE = 'skip'
    LR = 0.01
    LR_kernel = 1e-6
    reg_noise_std = 0.03
    iter_dl = num_iter - np.arange(len(dl_param)+1,1,-1)*100
    iter_mean = num_iter-50

    deblur_input = get_noise(input_depth,input_type,
                    (img_np.shape[1],img_np.shape[2])).type(dtype).detach()

    deblur_net = get_net(input_depth, NET_TYPE, pad,
                  skip_n33d=128,
                  skip_n33u=128,
                  skip_n11=4,
                  n_channels=1,
                  num_scales=5,
                  upsample_mode='bilinear').type(dtype)
    
    out_mean_deblur = deblur_image(deblur_net, deblur_input, blur, img_np,
      OPT_OVER, num_iter, reg_noise_std, LR, LR_kernel, iter_mean, dtype, autoencoder,
      iter_dl, dl_param)

    img_mean=((out_mean_deblur[0]-np.min(out_mean_deblur[0]))/(np.max(out_mean_deblur[0])-np.min(out_mean_deblur[0])))
    img_mean=np.expand_dims(img_mean,axis=0)

    return img_mean