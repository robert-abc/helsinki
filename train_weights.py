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
from torch.nn.utils import parametrize
import shutil

# Get input arguments
parser = argparse.ArgumentParser(description=
            'Get weights to be used in the autoencoder for deblurring.')

parser.add_argument('--blur_path', type=str, nargs="+",
                    help='Path with images to be deblurred')
parser.add_argument('--sharp_path', type=str, nargs="+",
                    help='Path with sharp images')
parser.add_argument('--weight_path', type=str,
                    help='Path to save weight')
parser.add_argument('--blur_level', type=int, nargs="+", required = False,
                    choices=range(0,20), metavar='[0-19]',
                    help='Level of blur')
parser.add_argument('--extension', dest='extension',
                    type=str, default='tif', required=False,
                    help='Image extension (default: tif)')
parser.add_argument('--save_intermediary', dest='out_dip_path', nargs="+",
                    type=str, default=None, required=False,
                    help='Path to save output of DIP phase (default: None)')
parser.add_argument('--have_intermediary', dest='in_dip_path',  nargs="+",
                    type=str, default=None, required=False,
                    help='Path with output of DIP phase (default: None)')
parser.add_argument('--config_option', dest='config_option',
                    type=int, default=1, required=False,
                    help='Parameters of DIP architecture (default: 1)')
parser.add_argument('--num_iter', dest='num_iter',
                    type=int, default=None, required=False,
                    help='Number of iterations (default: 175 for config 1, 700 for config 2)')
parser.add_argument('--train_iter', dest='train_iter',
                    type=int, default=30, required=False,
                    help='Number of iterations of the autoencoder training (default: 30)')

args = parser.parse_args()

# Get image names
ind_path = []
img_names = []

for i, base_path in enumerate(args.blur_path):
  img_names_i=sorted(os.listdir(args.blur_path[i]))
  r=re.compile(".*"+args.extension)
  img_names_i = list(filter(r.match,img_names_i))
  ind_path_i = np.ones(len(img_names_i),dtype=int)*i
  img_names = [*img_names, *img_names_i]
  ind_path = [*ind_path, *ind_path_i]

ind_path = np.array(ind_path)

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
  # DIP Parameters
  if(args.config_option==1):
    if(args.num_iter is None):
      args.num_iter=175

    config = {
        'num_iter': args.num_iter, 
        'input_depth': 16,
        'skip_n11': 32,
        'num_scales': 8,
        'LR': 0.01,
        'LR_kernel': 5e-6,
        'reg_noise_std': 0.003,
    }
  elif(args.config_option==2):
    if(args.num_iter is None):
      args.num_iter=700

    config = {
        'num_iter': args.num_iter, 
        'input_depth': 32,
        'skip_n11': 4,
        'num_scales': 5,
        'LR': 0.01,
        'LR_kernel': 5e-6,
        'reg_noise_std': 0.03,
    }

  # Radius of PSF with respect to deblur levels
  r_list=[1,2,3,4,6,8,9,11,13,15,17,18,20,21,22,26,31,35.5,41,44]
  radius=r_list[args.blur_level[ind_path[i]]]

  for i,img in enumerate(img_names):
      path_blur=os.path.join(args.blur_path[ind_path[i]],img)

      img_arr,_,_=process.load_img(path_blur,width=512,enforse_div32='EXTEND')

      # Model of blur
      blur = tools.Blur(n_planes=1,kernel_type='circle', kernel_parameter=radius, im_shape=img_arr.shape).type(dtype)
      parametrize.register_parametrization(blur.blur_,"weight", tools.Rings(blur.kernel,6,4,dtype))

      img_out=deblur.deblur(img_arr,blur,None,dtype,config)

      path_out=os.path.join(args.out_dip_path[ind_path[i]],img)
      path_out=path_out[0:-3]+'npy'
      np.save(path_out,img_out)
else:
  #  Remove part of background
  crop_x = None
  crop_y = None

  # Structures to store results
  arr_x=np.zeros(len(img_names),dtype=object)
  arr_y_orig=np.zeros(len(img_names),dtype=object)
  arr_y=np.zeros(len(img_names),dtype=object)

  for i,img in enumerate(img_names):
    path_sharp=os.path.join(args.sharp_path[ind_path[i]],img)

    img_arr,_,_=process.load_img(path_sharp,width=512,enforse_div32='EXTEND')
    arr_y_orig[i]=autoencoder_tools.preprocess_array(img_arr,crop_x,crop_y,binary_threshold=0.5)

  for i,img in enumerate(img_names):
    img = img[0:-3]+'npy'
    path_in=os.path.join(args.in_dip_path[ind_path[i]],img)
    arr_x[i]=autoencoder_tools.preprocess_array(np.load(path_in),crop_x,crop_y)

  # Spatial normalization of images
  for i in range(len(img_names)):
    warp_matrix=autoencoder_tools.get_transform(arr_x[i],arr_y_orig[i])
    arr_y[i]=autoencoder_tools.apply_transform(arr_y_orig[i],warp_matrix)

  # Patch Extraction
  patch_p_img=500
  patch_size=96

  num_types = np.max(ind_path)+1
  num_ant = 0
  ind_train = []
  ind_valid = []

  for i in range(num_types):
    inds_type = np.arange(num_ant,num_ant+np.sum(ind_path==i))
    num_ant += len(inds_type)
    ind_train = [*ind_train, *inds_type[0:-4]]
    ind_valid = [*ind_valid, *inds_type[-4:]]

  ind_train = np.array(ind_train)
  ind_valid = np.array(ind_valid)
  
  train_x_path = os.path.join('temp','train_x')
  train_y_path = os.path.join('temp','train_y')
  valid_x_path = os.path.join('temp','valid_x')
  valid_y_path = os.path.join('temp','valid_y')

  os.mkdir('temp')
  os.mkdir(train_x_path)
  os.mkdir(train_y_path)
  os.mkdir(valid_x_path)
  os.mkdir(valid_y_path)

  for i in ind_train:
    i_rand=random.randint(0,1e4)

    train_x = feature_extraction.image.extract_patches_2d(arr_x[i],patch_size=(patch_size, patch_size),max_patches=patch_p_img,random_state=i_rand)
    train_y = feature_extraction.image.extract_patches_2d(arr_y[i],patch_size=(patch_size, patch_size),max_patches=patch_p_img,random_state=i_rand)

    for j in range(len(train_x)):
      k = np.where(ind_train==i)[0][0]
      np.save(os.path.join(train_x_path,str(k*patch_p_img+j)+'.npy'), train_x[j])
      np.save(os.path.join(train_y_path,str(k*patch_p_img+j)+'.npy'), train_y[j])

  for i in ind_valid:
    i_rand=random.randint(0,1e4)

    valid_x = feature_extraction.image.extract_patches_2d(arr_x[i],patch_size=(patch_size, patch_size),max_patches=patch_p_img,random_state=i_rand)
    valid_y = feature_extraction.image.extract_patches_2d(arr_y[i],patch_size=(patch_size, patch_size),max_patches=patch_p_img,random_state=i_rand)

    for j in range(len(valid_x)):
      k = np.where(ind_valid==i)[0][0]
      np.save(os.path.join(valid_x_path,str(k*patch_p_img+j)+'.npy'), valid_x[j])
      np.save(os.path.join(valid_y_path,str(k*patch_p_img+j)+'.npy'), valid_y[j])

  del arr_x
  del arr_y_orig
  del arr_y

  class ImageDataset(Dataset):
    def __init__(self,x_path,y_path):
      self.x_path = x_path
      self.y_path = y_path

    def __getitem__(self, index):
      x = np.load(os.path.join(self.x_path,str(index)+'.npy'))
      x = (x-x.min())/(x.max()-x.min())
      y = np.load(os.path.join(self.y_path,str(index)+'.npy'))

      x = np.expand_dims(x,0)
      y = np.expand_dims(y,0)

      x = torch.from_numpy(x).type(dtype) 
      y = torch.from_numpy(y).type(dtype)

      return x, y

    def __len__(self):
      return len(os.listdir(self.x_path))


  train_dataset = ImageDataset(train_x_path,train_y_path)
  train_loader = DataLoader(dataset=train_dataset,
                            batch_size=64,
                            shuffle=True)

  val_dataset = ImageDataset(valid_x_path,valid_y_path)
  val_loader = DataLoader(dataset=val_dataset,
                            batch_size=64,
                            shuffle=False)

  model = autoencoder_tools.Autoencoder().type(dtype)
  best_params = autoencoder_tools.train_model(model, train_loader, val_loader,
    n_epoch=args.train_iter, loss_tol=0.0001, epoch_tol=3, l1_reg=10e-10)

  torch.save(best_params, os.path.join(args.weight_path,'weights_'+str(args.blur_level)+'.pth'))
  shutil.rmtree('temp')
