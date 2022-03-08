import argparse
import re
import os
from utils import process
from utils import deblur
from utils import tools
from utils import autoencoder_tools
from torch.nn.utils import parametrize
import torch

# Get input arguments
parser = argparse.ArgumentParser(description=
            'Deblur images with different levels of blur.')

parser.add_argument('input_path', type=str,
                    help='Path with images to be deblurred')
parser.add_argument('output_path', type=str,
                    help='Path to save deblurred images')
parser.add_argument('deblur_level', type=int,
                    choices=range(0,20), metavar='[0-19]',
                    help='Level of blur')
parser.add_argument('--extension', dest='extension',
                    type=str, default='tif', required=False,
                    help='Image extension (default: tif)')
parser.add_argument('--num_iter', dest='num_iter',
                    type=int, default=None, required=False,
                    help='Number of iterations (default: 225 for config 1, 750 for config 2)')
parser.add_argument('--congif_option', dest='congif_option',
                    type=int, default=1, required=False,
                    help='Parameters of DIP architecture (default: 1)')
parser.add_argument('--dl_param', dest='dl_param', nargs="+",
                    type=float, default=None, required=False,
                    help='Regularization weight of autoencoder (default: [0.01])')

args = parser.parse_args()

# Get image names
img_names=sorted(os.listdir(args.input_path))
r=re.compile(".*"+args.extension)
img_names=list(filter(r.match,img_names))
print(f"{len(img_names)} images were found.")

# Use of GPU
if torch.cuda.is_available():
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True
  dtype = torch.cuda.FloatTensor
  map_location = None
else:
  torch.backends.cudnn.enabled = False
  torch.backends.cudnn.benchmark = False
  dtype = torch.FloatTensor
  map_location = 'cpu'

# Radius of PSF with respect to deblur levels
r_list=[1,2,3,4,6,8,9,11,13,15,17,18,20,21,22,26,31,35.5,41,44]
radius=r_list[args.deblur_level]

# Model of blur
blur = tools.Blur(n_planes=1,kernel_type='circle', kernel_parameter=radius).type(dtype)
parametrize.register_parametrization(blur.blur_,"weight", tools.Rings(blur.kernel,6,4,dtype))

# Find weights for the blur level
n_weight=args.deblur_level
weights_names=os.listdir('weights')
weight_lvl="weights_"+str(n_weight)+".pth"
aux=1

while(weight_lvl not in weights_names):
    n_weight+= (-1)**(aux+1) * aux
    weight_lvl="weights_"+str(n_weight)+".pth"
    aux+=1

    if(n_weight>60):
        break

# Autoencoder
autoencoder=autoencoder_tools.Autoencoder().type(dtype) 
autoencoder.load_state_dict(torch.load(os.path.join('weights',weight_lvl), map_location=map_location))
autoencoder.eval()

# DIP Parameters
if(args.config_option==1):
  if(args.num_iter is None):
    args.num_iter=225
  if(args.dl_param is None):
    args.dl_param=0.01
    iter_dl = [175]

  config = {
      'num_iter': args.num_iter, 
      'dl_param': args.dl_param,
      'input_depth': 16,
      'skip_n11': 32,
      'num_scales': 8,
      'LR': 0.001,
      'LR_kernel': 5e-6,
      'reg_noise_std': 0.003,
      'iter_dl': iter_dl
  }
elif(args.config_option==2):
  if(args.num_iter is None):
    args.num_iter=750
  if(args.dl_param is None):
    args.dl_param=0.01
    iter_dl = [700]

  config = {
      'num_iter': args.num_iter, 
      'dl_param': args.dl_param,
      'input_depth': 32,
      'skip_n11': 4,
      'num_scales': 5,
      'LR': 0.01,
      'LR_kernel': 5e-6,
      'reg_noise_std': 0.03,
      'iter_dl': iter_dl
  }

for img in img_names:
    path_in=os.path.join(args.input_path,img)
    path_out=os.path.join(args.output_path,img)
    path_out=path_out[0:-3]+'png'

    img_arr,orig_dim,extend_dim=process.load_img(path_in,width=512,enforse_div32='EXTEND')
    img_out=deblur.deblur(img_arr, blur, autoencoder, dtype, config)

    process.save_img(img_out,path_out,orig_dim,extend_dim)
