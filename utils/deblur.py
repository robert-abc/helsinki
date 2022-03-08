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

        total_loss = 1 - ssim(out_blur, img_torch, dtype)

        if autoencoder is not None:
            if i in config['iter_dl']:
                out_sharp_np = torch_to_np(out_sharp)
                img_dl = get_dl_estim(out_sharp_np[0],autoencoder,dtype,tam=96,out=16)
                img_dl = np.expand_dims(img_dl,axis=0)
                torch_dl = np_to_torch(img_dl).type(dtype)
                ind_dl += 1

            if i >= config['iter_dl'][0]:
                total_loss += config['dl_param'][ind_dl]*(1 - ssim(torch_dl, out_sharp, dtype))

        total_loss.backward()

        blur.blur_.weight.grad *= blur.grad_mask

        optimizer.step()

    out_deblur = deblur_net(deblur_input)
    out_deblur = torch_to_np(out_deblur)

    img_res=((out_deblur[0]-np.min(out_deblur[0]))/(np.max(out_deblur[0])-np.min(out_deblur[0])))
    img_res=np.expand_dims(img_res,axis=0)

    return img_res