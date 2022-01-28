import numpy as np
import cv2

import torch.nn as nn
import torch.nn.functional as F
import torch

class Autoencoder(nn.Module):
  def __init__(self, input_channel=1, pad_m='zeros'):
    super(Autoencoder, self).__init__()
    self.c1 = nn.Conv2d(input_channel, 64, 3, padding='same', padding_mode=pad_m)
    self.c2 = nn.Conv2d(64, 64, 3, padding='same', padding_mode=pad_m)
    
    self.c3 = nn.Conv2d(64, 128, 3, padding='same', padding_mode=pad_m)
    self.c4 = nn.Conv2d(128, 128, 3, padding='same', padding_mode=pad_m)

    self.encoded = nn.Conv2d(128,256,3, padding='same', padding_mode=pad_m)

    self.c5 = nn.Conv2d(256, 128, 3, padding='same', padding_mode=pad_m)
    self.c6 = nn.Conv2d(128, 128, 3, padding='same', padding_mode=pad_m)
    
    self.c7 = nn.Conv2d(128, 64, 3, padding='same', padding_mode=pad_m)
    self.c8 = nn.Conv2d(64, 64, 3, padding='same', padding_mode=pad_m)

    self.decoded = nn.Conv2d(64, input_channel, 3, padding='same', padding_mode=pad_m)

    self.pool = nn.MaxPool2d(2)
    self.up = nn.Upsample(scale_factor=2)
    
  def forward(self,x):
    x1 = F.relu(self.c1(x))
    x2 = F.relu(self.c2(x1))
    x3 = self.pool(x2)

    x4 = F.relu(self.c3(x3))
    x5 = F.relu(self.c4(x4))
    x6 = self.pool(x5)

    enc = F.relu(self.encoded(x6))

    x7 = self.up(enc)
    x8 = F.relu(self.c5(x7))
    x9 = F.relu(self.c6(x8))
    x10 = x5+x9

    x11 = self.up(x10)
    x12 = F.relu(self.c7(x11))
    x13 = F.relu(self.c8(x12))
    x14 = x13 + x2

    dec = F.relu(self.decoded(x14))

    return dec

def train_model(model, train_loader, valid_loader,
  loss=torch.nn.MSELoss(), optim='Adam', n_epoch=20,
  val_loss_best=np.inf, loss_tol = 0, epoch_tol = -1,
  l1_reg=0):

  if(optim=='Adam'):
    optimizer = torch.optim.Adam(model.parameters())

  i_tol=0

  for epoch in range(n_epoch):
    model.train()
    for i, (x, y) in enumerate(train_loader):
      y_predicted = model(x)

      # loss
      l = loss(y_predicted, y)

      if(l1_reg > 0):
        param_layers = [lp.view(-1) for lp in model.parameters()]
        param_w = torch.cat([param_layers[wi] for wi in range(0,len(param_layers),2)])
        l1_loss = l1_reg * torch.norm(param_w, 1)
        l = l + l1_loss

      # calculate gradients = backward pass
      l.backward()

      # update weights
      optimizer.step()

      # zero the gradients after updating
      optimizer.zero_grad()

      if i % 50 == 0:
        #[w, b] = deblur_net.parameters() # unpack parameters
        print('epoch ', epoch,' batch ', i, ' loss = ', l.item())
      
      if i == 500:
        break
      
    model.eval()
    val_loss_batchs = []
    with torch.no_grad():
      for i, (x, y) in enumerate(valid_loader):
        y_predicted = model(x)

        # loss
        l = loss(y, y_predicted)
        val_loss_batchs.append(l.item())

      val_loss_mean = np.array(val_loss_batchs).mean()
      print('epoch ', epoch, ' val_loss = ', val_loss_mean)

      if ((val_loss_mean+loss_tol) < (val_loss_best)):
        val_loss_best = val_loss_mean
        best_params = model.state_dict()
        i_tol = 0
      else:
        i_tol+=1
    
    if(i_tol == epoch_tol):
      print("Early Stop!")
      break
  
  return best_params

def get_dl_estim(img_x_orig, model, dtype, tam=64, out=0):
  dim_orig = img_x_orig.shape
  val_size = tam-2*out
  rest=dim_orig[0]%tam
  if(rest!=0):
    img_x = np.pad(img_x_orig,((0,96-rest),(0,96-rest)),mode='edge')
  else:
    img_x = img_x_orig

  dim_expand=img_x.shape
  img_x = (img_x-np.min(img_x))/(np.max(img_x)-np.min(img_x))
  img_x = np.pad(img_x,((out,out),(out,out)),mode='edge')

  ind_x=np.arange(0,img_x.shape[0]-tam+1,val_size)
  ind_y=np.arange(0,img_x.shape[1]-tam+1,val_size)

  Px=len(ind_x)
  Py=len(ind_y)

  patch_test=np.zeros((Px*Py,tam,tam))

  for i in range(Px):
    for j in range(Py):
      patch_extract=img_x[ind_x[i]:ind_x[i]+tam,
                          ind_y[j]:ind_y[j]+tam]

      patch_test[i*Py+j,:,:]=patch_extract

  patch_test_torch = torch.from_numpy(np.expand_dims(patch_test,1)).type(dtype)
  preds_torch = model(patch_test_torch)
  preds = preds_torch.detach().cpu().numpy()
  preds = np.squeeze(preds)

  img_rec=np.zeros(dim_expand)

  for i in range(Px):
    for j in range(Py):
      img_rec[i*val_size:i*val_size+val_size,j*val_size:j*val_size+val_size]=preds[i*Py+j,out:tam-out,out:tam-out]

  img_rec=img_rec[0:dim_orig[0],0:dim_orig[1]]
  img_rec=(img_rec-np.min(img_rec))/(np.max(img_rec)-np.min(img_rec))

  return img_rec

def preprocess_array(img,crop_x=None,crop_y=None,binary_threshold=None):
  if((crop_x is not None) and (crop_y is not None)):
    img=img[0,crop_x[0]:crop_x[1],crop_y[0]:crop_y[1]]
  else:
    img = img[0]
    
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
