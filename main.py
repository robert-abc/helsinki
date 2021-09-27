#python main.py  C:\Users\guys_\Documents\UFABC\Doutorado\Deblur\dados_helsinki\input C:\Users\guys_\Documents\UFABC\Doutorado\Deblur\dados_helsinki\output 0

import argparse
import re
import os
from utils import process
from utils import deblur
from utils import tools
from utils import autoencoder_tools
from utils import deblur
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

args = parser.parse_args()

# Get image names
img_names=os.listdir(args.input_path)
r=re.compile(".*"+args.extension)
img_names=list(filter(r.match,img_names))
print(f"{len(img_names)} images were found.")

# Use of GPU
torch.backends.cudnn.enabled = False #True #
torch.backends.cudnn.benchmark = False #True #
dtype = torch.FloatTensor #torch.cuda.FloatTensor #

# Radius of PSF with respect to deblur levels
r_list=[0,0,0,0,0,0,0,0,0,15,0,0,0,0,0,26,0,0,0,0]
radius=r_list[args.deblur_level]

# Model of blur
blur = tools.Blur(n_planes=1,kernel_type='circle',sigma=radius).type(dtype)

# Autoencoder
autoencoder=autoencoder_tools.get_nn(os.path.join('weights','binary_64_mse.h5'))

for img in img_names:
    path_in=os.path.join(args.input_path,img)
    path_out=os.path.join(args.output_path,img)
    path_out=path_out[0:-3]+'png'

    img_arr,orig_dim,extend_dim=process.load_img(path_in,width=512,enforse_div32='EXTEND')
    img_out=deblur.deblur(img_arr,blur,autoencoder,dtype)

    process.save_img(img_out,path_out,orig_dim,extend_dim)
