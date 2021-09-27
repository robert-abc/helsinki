from process import *
from tools import *
from dip import *
from autoencoder_tools import get_dl_estim

def deblur(img_np,blur,autoencoder):
    #  Image Parameters
    width = 512 # Desired image width
    enforse_div32 = 'EXTEND' # Force image to have dims multiple of 32

    # Input Parameters
    input_depth = 32
    input_type = 'noise'
    OPT_OVER = 'net'

    # Optimization Parameters
    LR = 0.01
    OPTIMIZER = 'adam'
    pad = 'reflection'
    reg_noise_std= 0.03
    NET_TYPE = 'skip'
    num_iter= 1500
    iter_lr=[200,400,600]
    iter_dl=[1000,1100,1200,1300]
    iter_mean=1400
    dl_param=[1e-2,1e-2,5e-3,5e-3]

    img_torch=get_torch_imgs(img_np)
    out_mean_deblur = np.zeros(img_np.shape)

    deblur_input = get_noise(input_depth,input_type,
                    (img_np.shape[1],img_np.shape[2])).type(dtype).detach()

    deblur_net = get_net(input_depth, NET_TYPE, pad,
                  skip_n33d=128,
                  skip_n33u=128,
                  skip_n11=4,
                  n_channels=1,
                  num_scales=5,
                  upsample_mode='bilinear').type(dtype)

    ind_lr=0
    ind_dl=-1

    net_input_saved = deblur_input.detach().clone()
    noise = deblur_input.detach().clone()

    i = 0
    p = get_params(OPT_OVER,deblur_net,deblur_input,blur)
    optimize(OPTIMIZER, p, deblur_closure, LR, num_iter)

    img_mean=((out_mean_deblur[0]-np.min(out_mean_deblur[0]))/(np.max(out_mean_deblur[0])-np.min(out_mean_deblur[0])))

    return img_mean

def deblur_closure():
  global i, deblur_input, out_mean_deblur, ind_lr, torch_dl, ind_dl

  if reg_noise_std > 0:
      deblur_input = net_input_saved + (noise.normal_() * reg_noise_std)
  else:
      deblur_input = net_input_saved

  out_sharp = deblur_net(deblur_input)
  out_blur = blur(out_sharp)

  if i in iter_lr:
    ind_lr+=1

  total_loss = 1 - ssim(out_blur, img_torch[ind_lr])

  if i>=iter_mean:
    out_sharp_np = torch_to_np(out_sharp)
    out_mean_deblur += out_sharp_np

  if i in iter_dl:
    out_sharp_np = torch_to_np(out_sharp)
    img_dl = get_dl_estim(out_sharp_np[0],autoencoder)
    img_dl=np.expand_dims(img_dl,axis=0)
    torch_dl=np_to_torch(img_dl).type(dtype)
    ind_dl+=1

  if i >= iter_dl[0]:
    total_loss += dl_param[ind_dl]*(1 - ssim(torch_dl, out_sharp))

  total_loss.backward()

  i += 1

  return total_loss
