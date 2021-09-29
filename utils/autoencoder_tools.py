import tensorflow as tf
from tensorflow.keras import Model, Input, regularizers
from tensorflow.keras.layers import Conv2D, MaxPool2D, UpSampling2D, Add
import numpy as np
import cv2

def get_nn(weight_path=None):
    Input_img = Input(shape=(64,64,1))

    #encoding architecture
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(Input_img)
    x2 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x1)
    x3 = MaxPool2D(padding='same')(x2)

    x4 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x3)
    x5 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x4)
    x6 = MaxPool2D(padding='same')(x5)

    encoded = Conv2D(256, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x6)

    # decoding architecture
    x7 = UpSampling2D()(encoded)
    x8 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x7)
    x9 = Conv2D(128, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x8)
    x10 = Add()([x5, x9])

    x11 = UpSampling2D()(x10)
    x12 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x11)
    x13 = Conv2D(64, (3, 3), activation='relu', padding='same', kernel_regularizer=regularizers.l1(10e-10))(x12)
    x14 = Add()([x2, x13])

    decoded = Conv2D(1, (3, 3), padding='same',activation='relu', kernel_regularizer=regularizers.l1(10e-10))(x14)

    autoencoder = Model(Input_img, decoded)

    if(weight_path is not None):
        autoencoder.load_weights(weight_path)

    return autoencoder

def get_dl_estim(img_x,autoencoder):
  ind_x=np.arange(0,img_x.shape[0]+1,64)
  ind_y=np.arange(0,img_x.shape[1]+1,64)

  if(ind_x[-1]!=img_x.shape[0]):
    ind_x=np.concatenate((ind_x,[img_x.shape[0]]))

  if(ind_y[-1]!=img_x.shape[1]):
    ind_y=np.concatenate((ind_y,[img_x.shape[1]]))

  Px=len(ind_x)-1
  Py=len(ind_y)-1

  patch_test=np.zeros((Px*Py,64,64))

  for i in range(Px):
    for j in range(Py):
      patch_extract=img_x[ind_x[i]:ind_x[i+1],
                          ind_y[j]:ind_y[j+1]]

      dif_x=64-patch_extract.shape[0]
      dif_y=64-patch_extract.shape[1]

      if(dif_x>0):
        patch_extract=np.pad(patch_extract,((0,dif_x),(0,0)),mode='edge')

      if(dif_y>0):
        patch_extract=np.pad(patch_extract,((0,0),(0,dif_y)),mode='edge')

      patch_test[i*Py+j,:,:]=patch_extract

  preds=autoencoder.predict(patch_test)
  preds=np.squeeze(preds)

  img_rec=np.zeros((64*Px,64*Py))

  for i in range(Px):
    for j in range(Py):
      img_rec[i*64:i*64+64,j*64:j*64+64]=preds[i*Py+j]

  img_rec=img_rec[0:img_x.shape[0],0:img_x.shape[1]]
  img_rec=(img_rec-np.min(img_rec))/(np.max(img_rec)-np.min(img_rec))

  return img_rec

def preprocess_array(img,crop_x,crop_y,binary_threshold=None):
    img=img[0,crop_x[0]:crop_x[1],crop_y[0]:crop_y[1]]
    img=(img-np.min(img))/(np.max(img)-np.min(img))

    if(binary_threshold is not None):
        img=img>binary_threshold

    return img.astype(np.float32)

def get_transform(img_fix,img_mov,n_iter=500,end_eps=1e-10):
  warp_mode = cv2.MOTION_AFFINE
  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,n_iter,end_eps)
  warp_matrix = np.eye(2, 3, dtype=np.float32)

  (_, warp_matrix) = cv2.findTransformECC (img_fix,img_mov,warp_matrix,
                                            warp_mode,criteria,None,1)

  return warp_matrix

def apply_transform(img,warp_matrix):
  img_align = cv2.warpAffine(img, warp_matrix, (img.shape[1],img.shape[0]),
                             flags=cv2.INTER_LINEAR+cv2.WARP_INVERSE_MAP,
                             borderMode=cv2.BORDER_REPLICATE);

  img_align=(img_align-np.min(img_align))/(np.max(img_align)-np.min(img_align))

  return img_align
