from utils.process import *
from utils.tools import *
from utils.dip import *
from utils.autoencoder_tools import get_dl_estim

def deblur(img_np,blur,autoencoder,dtype,num_iter=1500):
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
    iter_lr=[200,400,600]
    LR = 0.01
    reg_noise_std= 0.03
    iter_dl=[num_iter-500,num_iter-400,num_iter-300,num_iter-200]
    iter_mean=num_iter-100
    dl_param=[1e-2,1e-2,5e-3,5e-3]

    img_torch=get_torch_imgs(img_np,dtype=dtype)
    out_mean_deblur = np.zeros(img_np.shape)
    torch_dl=torch.zeros(img_torch[0].size())

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

    def deblur_closure():
        nonlocal i, deblur_input, out_mean_deblur, ind_lr, torch_dl, ind_dl

        if reg_noise_std > 0:
            deblur_input = net_input_saved + (noise.normal_() * reg_noise_std)
        else:
            deblur_input = net_input_saved

        out_sharp = deblur_net(deblur_input)
        out_blur = blur(out_sharp)

        if i in iter_lr:
          ind_lr+=1

        total_loss = 1 - ssim(out_blur, img_torch[ind_lr],dtype)

        if i>=iter_mean:
          out_sharp_np = torch_to_np(out_sharp)
          out_mean_deblur += out_sharp_np

        if autoencoder is not None:
            if i in iter_dl:
              out_sharp_np = torch_to_np(out_sharp)
              img_dl = get_dl_estim(out_sharp_np[0],autoencoder)
              img_dl=np.expand_dims(img_dl,axis=0)
              torch_dl=np_to_torch(img_dl).type(dtype)
              ind_dl+=1

            if i >= iter_dl[0]:
              total_loss += dl_param[ind_dl]*(1 - ssim(torch_dl, out_sharp,dtype))

        total_loss.backward()
        i += 1

        return total_loss

    net_input_saved = deblur_input.detach().clone()
    noise = deblur_input.detach().clone()

    i = 0
    p = get_params(OPT_OVER,deblur_net,deblur_input,blur)
    optimize(OPTIMIZER, p, deblur_closure, LR, num_iter)

    img_mean=((out_mean_deblur[0]-np.min(out_mean_deblur[0]))/(np.max(out_mean_deblur[0])-np.min(out_mean_deblur[0])))
    img_mean=np.expand_dims(img_mean,axis=0)

    return img_mean
