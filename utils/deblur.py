from utils.process import *
from utils.tools import *
from utils.dip import *
import numpy as np

def deblur(img_np, blur, autoencoder, dtype, config):
    deblur_input = get_noise(config['input_depth'], 'noise',
                    (img_np.shape[1],img_np.shape[2])).type(dtype).detach()

    deblur_net = get_net(config['input_depth'], 'skip', 'reflection',
                  skip_n33d=128,
                  skip_n33u=128,
                  skip_n11=config['skip_n11'],
                  n_channels=1,
                  num_scales=config['num_scales'],
                  upsample_mode='bilinear').type(dtype)

    img_torch = torch.from_numpy(np.expand_dims(img_np,0)).type(dtype)
    torch_dl = torch.zeros(img_torch[0].size())

    out_mean_deblur = np.zeros(img_np.shape)
    mean_ae = True

    ind_dl=-1

    net_input_saved = deblur_input.detach().clone()
    noise = deblur_input.detach().clone()

    p, p_kernel = get_params('net,blur',deblur_net,deblur_input,blur)
    
    optimizer = torch.optim.Adam([{'params':p},
     {'params': p_kernel, 'lr': config['LR_kernel']}], lr=config['LR'])

    for i in range(config['num_iter']):
        optimizer.zero_grad()

        if config['reg_noise_std'] > 0:
            deblur_input = net_input_saved + (noise.normal_() * config['reg_noise_std'])
        else:
            deblur_input = net_input_saved

        out_sharp = deblur_net(deblur_input)
        out_blur = blur(out_sharp)

        if (i >= (config['num_iter']-50)):
            out_sharp_np = torch_to_np(out_sharp)
            out_mean_deblur += out_sharp_np

        total_loss = 1 - ssim(out_blur, img_torch, dtype)

        if autoencoder is not None:
            if ((i >= (config['num_iter']-100)) and mean_ae):
                out_sharp_np = torch_to_np(out_sharp)
                out_mean_deblur += out_sharp_np

            if i in config['iter_dl']:
                img_mean = ((out_mean_deblur[0]-np.min(out_mean_deblur[0]))/(np.max(out_mean_deblur[0])-np.min(out_mean_deblur[0])))
                img_mean = np.expand_dims(img_mean,axis=0)
                out_mean_deblur = np.zeros(img_np.shape)
                mean_ae = False

                img_dl = get_dl_estim(img_mean[0],autoencoder,dtype,tam=96,out=16)
                img_dl = np.expand_dims(img_dl,axis=0)
                torch_dl = np_to_torch(img_dl).type(dtype)
                ind_dl += 1

            if i >= config['iter_dl'][0]:
                total_loss += config['dl_param'][ind_dl]*(1 - ssim(torch_dl, out_sharp, dtype))

        total_loss.backward()

        optimizer.step()

    out_deblur = out_mean_deblur

    img_res=((out_deblur[0]-np.min(out_deblur[0]))/(np.max(out_deblur[0])-np.min(out_deblur[0])))
    img_res=np.expand_dims(img_res,axis=0)

    return img_res