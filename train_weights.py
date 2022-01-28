import argparse
import re
import os
import random
import numpy as np
from utils import process
from utils import deblur
from utils import tools
from utils import autoencoder_tools
from utils import dip
from torch.utils.data import Dataset, DataLoader
from sklearn import feature_extraction
import torch

# Get input arguments
parser = argparse.ArgumentParser(description=
            'Get weights to be used in the autoencoder for deblurring.')

parser.add_argument('blur_path', type=str,
                    help='Path with images to be deblurred')
parser.add_argument('sharp_path', type=str,
                    help='Path with sharp images')
parser.add_argument('weight_path', type=str,
                    help='Path to save weight')
parser.add_argument('blur_level', type=int,
                    choices=range(0,20), metavar='[0-19]',
                    help='Level of blur')
parser.add_argument('--extension', dest='extension',
                    type=str, default='tif', required=False,
                    help='Image extension (default: tif)')
parser.add_argument('--save_intermediary', dest='out_dip_path',
                    type=str, default=None, required=False,
                    help='Path to save output of DIP phase (default: None)')
parser.add_argument('--have_intermediary', dest='in_dip_path',
                    type=str, default=None, required=False,
                    help='Path with output of DIP phase (default: None)')
parser.add_argument('--num_iter', dest='num_iter',
                    type=int, default=500, required=False,
                    help='Number of iterations of the DIP (default: 500)')
parser.add_argument('--train_iter', dest='train_iter',
                    type=int, default=30, required=False,
                    help='Number of iterations of the autoencoder training (default: 30)')

args = parser.parse_args()

# Get image names
img_names=sorted(os.listdir(args.blur_path))
r=re.compile(".*"+args.extension)
img_names=list(filter(r.match,img_names))
print(f"{len(img_names)} images were found.")

# Use of GPU
if torch.cuda.is_available():
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark = True
  dtype = torch.cuda.FloatTensor
else:
  torch.backends.cudnn.enabled = False
  torch.backends.cudnn.benchmark = False
  dtype = torch.FloatTensor

if(args.in_dip_path is None):
  # Radius of PSF with respect to deblur levels
  r_list=[1,2,3,4,6,8,9,11,13,15,17,18,20,21,22,26,31,35.5,41,44]
  radius=r_list[args.blur_level]

  # Model of blur
  blur = tools.Blur(n_planes=1,kernel_type='circle', kernel_parameter=radius).type(dtype)

  for i,img in enumerate(img_names):
      path_blur=os.path.join(args.blur_path,img)

      img_arr,_,_=process.load_img(path_blur,width=512,enforse_div32='EXTEND')
      img_out=deblur.deblur(img_arr,blur,None,dtype,num_iter=args.num_iter)

      path_out=os.path.join(args.out_dip_path,img)
      path_out=path_out[0:-3]+'npy'
      np.save(path_out,img_out)
else:
  #  Remove part of background
  crop_x = None
  crop_y = None

  # Structures to store results
  arr_x=np.zeros(len(img_names),dtype=np.object)
  arr_y_orig=np.zeros(len(img_names),dtype=np.object)
  arr_y=np.zeros(len(img_names),dtype=np.object)

  for i,img in enumerate(img_names):
    path_sharp=os.path.join(args.sharp_path,img)

    img_arr,_,_=process.load_img(path_sharp,width=512,enforse_div32='EXTEND')
    arr_y_orig[i]=autoencoder_tools.preprocess_array(img_arr,crop_x,crop_y,binary_threshold=0.5)

  out_dip_names=sorted(os.listdir(args.in_dip_path))
  r=re.compile(".*npy")
  out_dip_names=list(filter(r.match,out_dip_names))

  for i,img in enumerate(out_dip_names):
    path_in=os.path.join(args.in_dip_path,img)
    arr_x[i]=autoencoder_tools.preprocess_array(np.load(path_in),crop_x,crop_y)

  # Spatial normalization of images
  for i in range(len(img_names)):
    warp_matrix=autoencoder_tools.get_transform(arr_x[i],arr_y_orig[i])
    arr_y[i]=autoencoder_tools.apply_transform(arr_y_orig[i],warp_matrix)

  # Patch Extraction
  patch_p_img=1000
  patch_size=96

  train_x=[]
  train_y=[]
  valid_x=[]
  valid_y=[]

  for i in range(0,arr_x.shape[0]-5):
    i_rand=random.randint(0,1e4)

    train_x.append(feature_extraction.image.extract_patches_2d(arr_x[i],patch_size=(patch_size, patch_size),max_patches=patch_p_img,random_state=i_rand))
    train_y.append(feature_extraction.image.extract_patches_2d(arr_y[i],patch_size=(patch_size, patch_size),max_patches=patch_p_img,random_state=i_rand))

  for i in range(arr_x.shape[0]-5,arr_x.shape[0]):
    i_rand=random.randint(0,1e4)

    valid_x.append(feature_extraction.image.extract_patches_2d(arr_x[i],patch_size=(patch_size, patch_size),max_patches=patch_p_img,random_state=i_rand))
    valid_y.append(feature_extraction.image.extract_patches_2d(arr_y[i],patch_size=(patch_size, patch_size),max_patches=patch_p_img,random_state=i_rand))

  train_x=np.expand_dims(np.concatenate(train_x,axis=0),1)
  train_y=np.expand_dims(np.concatenate(train_y,axis=0),1)
  valid_x=np.expand_dims(np.concatenate(valid_x,axis=0),1)
  valid_y=np.expand_dims(np.concatenate(valid_y,axis=0),1)

  del arr_x
  del arr_y_orig
  del arr_y

  class ImageDataset(Dataset):
    def __init__(self,x,y):
      self.n_samples = x.shape[0]

      self.x_data = torch.from_numpy(x).type(dtype) 
      self.y_data = torch.from_numpy(y).type(dtype)

    def __getitem__(self, index):
      return self.x_data[index], self.y_data[index]

    def __len__(self):
      return self.n_samples

  train_dataset = ImageDataset(train_x,train_y)
  train_loader = DataLoader(dataset=train_dataset,
                            batch_size=32,
                            shuffle=True)
  val_dataset = ImageDataset(valid_x,valid_y)
  val_loader = DataLoader(dataset=val_dataset,
                            batch_size=32,
                            shuffle=False)

  model = autoencoder_tools.Autoencoder().type(dtype)
  best_params = autoencoder_tools.train_model(model, train_loader, val_loader,
    n_epoch=args.train_iter, loss_tol=0.0001, epoch_tol=3, l1_reg=10e-10)

  torch.save(best_params, os.path.join(args.weight_path,'weights_'+str(args.blur_level)+'.pth'))
