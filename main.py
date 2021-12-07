import argparse
import re
import os
from utils import process
from utils import deblur
from utils import tools
from utils import autoencoder_tools
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
                    type=int, default=1500, required=False,
                    help='Number of iterations (default: 1500)')
parser.add_argument('--dl_param', dest='dl_param', nargs="+",
                    type=float, default=[1e-2,1e-2,5e-3,5e-3], required=False,
                    help='Regularization weight of autoencoder (default: [1e-2,1e-2,5e-3,5e-3])')

args = parser.parse_args()

# Get image names
img_names=sorted(os.listdir(args.input_path))
r=re.compile(".*"+args.extension)
img_names=list(filter(r.match,img_names))
print(f"{len(img_names)} images were found.")

# Use of GPU
torch.backends.cudnn.enabled = True #False #
torch.backends.cudnn.benchmark = True #False #
dtype = torch.cuda.FloatTensor #torch.FloatTensor #

# Radius of PSF with respect to deblur levels
r_list=[1,2,3,4,6,8,9,11,13,15,17,18,20,21,22,26,31,35.5,41,44]
radius=r_list[args.deblur_level]

# Model of blur
blur = tools.Blur(n_planes=1,kernel_type='circle',sigma=radius).type(dtype)

# Find weights for the blur level
n_weight=args.deblur_level
weights_names=os.listdir('weights')
weight_lvl="weights_"+str(n_weight)+".h5"
aux=1

while(weight_lvl not in weights_names):
    n_weight+= (-1)**(aux+1) * aux
    weight_lvl="weights_"+str(n_weight)+".h5"
    aux+=1

    if(n_weight>60):
        break

# Autoencoder
autoencoder=autoencoder_tools.get_nn(os.path.join('weights',weight_lvl))

for img in img_names:
    path_in=os.path.join(args.input_path,img)
    path_out=os.path.join(args.output_path,img)
    path_out=path_out[0:-3]+'png'

    img_arr,orig_dim,extend_dim=process.load_img(path_in,width=512,enforse_div32='EXTEND')
    img_out=deblur.deblur(img_arr,blur,autoencoder,dtype,num_iter=args.num_iter,dl_param=args.dl_param)

    process.save_img(img_out,path_out,orig_dim,extend_dim)
