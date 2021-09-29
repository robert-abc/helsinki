import argparse
import re
import os
import random
import numpy as np
from utils import process
from utils import deblur
from utils import tools
from utils import autoencoder_tools
from utils import deblur
from sklearn import feature_extraction
import torch
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import tensorflow as tf

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

args = parser.parse_args()

# Get image names
img_names=sorted(os.listdir(args.blur_path))
r=re.compile(".*"+args.extension)
img_names=list(filter(r.match,img_names))
print(f"{len(img_names)} images were found.")

# Use of GPU
torch.backends.cudnn.enabled = True #False #
torch.backends.cudnn.benchmark = True #False #
dtype = torch.cuda.FloatTensor #torch.FloatTensor #

# Radius of PSF with respect to deblur levels
r_list=[0,0,0,0,0,0,0,0,0,15,0,0,0,0,0,26,31,0,0,0]
radius=r_list[args.blur_level]

# Model of blur
blur = tools.Blur(n_planes=1,kernel_type='circle',sigma=radius).type(dtype)

#  Remove part of background
crop_x=[35,265]
crop_y=[50,460]

# Structures to store results
arr_x=np.zeros(len(img_names),dtype=np.object)
arr_y_orig=np.zeros(len(img_names),dtype=np.object)
arr_y=np.zeros(len(img_names),dtype=np.object)

for i,img in enumerate(img_names):
    path_sharp=os.path.join(args.sharp_path,img)

    img_arr,_,_=process.load_img(path_sharp,width=512,enforse_div32='EXTEND')
    arr_y_orig[i]=autoencoder_tools.preprocess_array(img_arr,crop_x,crop_y,binary_threshold=0.5)

if(args.in_dip_path is None):
    for i,img in enumerate(img_names):
        path_blur=os.path.join(args.blur_path,img)

        img_arr,_,_=process.load_img(path_blur,width=512,enforse_div32='EXTEND')
        img_out=deblur.deblur(img_arr,blur,None,dtype)
        arr_x[i]=autoencoder_tools.preprocess_array(img_out,crop_x,crop_y)

        if(args.out_dip_path is not None):
            path_out=os.path.join(args.out_dip_path,img)
            path_out=path_out[0:-3]+'npy'
            np.save(path_out,img_out)
else:
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
patch_size=64

train_x=[]
train_y=[]
valid_x=[]
valid_y=[]

for i in range(0,arr_x.shape[0]-5):
  i_rand=random.randint(0,1e4)

  train_x.append(feature_extraction.image.extract_patches_2d(arr_x[i],patch_size=(patch_size, patch_size),max_patches=patch_p_img,random_state=i_rand))
  train_y.append(feature_extraction.image.extract_patches_2d(arr_y[i],patch_size=(patch_size, patch_size),max_patches=patch_p_img,random_state=i_rand))

for i in range(arr_x.shape[0]-5,arr_x.shape[0]):
  i_rand=random.randint(0,1000)

  valid_x.append(feature_extraction.image.extract_patches_2d(arr_x[i],patch_size=(patch_size, patch_size),max_patches=patch_p_img,random_state=i_rand))
  valid_y.append(feature_extraction.image.extract_patches_2d(arr_y[i],patch_size=(patch_size, patch_size),max_patches=patch_p_img,random_state=i_rand))

train_x=np.concatenate(train_x,axis=0)
train_y=np.concatenate(train_y,axis=0)
valid_x=np.concatenate(valid_x,axis=0)
valid_y=np.concatenate(valid_y,axis=0)

# Autoencoder
mse = tf.keras.losses.MeanSquaredError()
autoencoder=autoencoder_tools.get_nn()
autoencoder.compile(optimizer='adam', loss=mse)
early_stopper = EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=3, verbose=1, mode='min')
model_checkpoint = ModelCheckpoint(join(args.weight_path,'weights_'+str(args.blur_level)+'.h5'),save_best_only=True)

history = autoencoder.fit(train_x,train_y,
            epochs=100,
            batch_size=64,
            steps_per_epoch=300,
            validation_data=(valid_x, valid_y),
            callbacks=[early_stopper, model_checkpoint])
