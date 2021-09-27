# Helsinki deblur challenge 2021

## Authors, institution, location

* André Kazuo Takahata¹ - andre.t@ufabc.edu.br
* Leonardo Ferreira Alves¹ - leonardo.alves@ufabc.edu.br
* Ricardo Suyama¹ - ricardo.suyama@ufabc.edu.br
* Roberto Gutierrez Beraldo¹ - roberto.gutierrez@ufabc.edu.br

¹Federal University of ABC (Santo André, São Paulo, Brazil) - https://www.ufabc.edu.br/

## Brief description of the algorithm
This deblurring work is to join the Helsinki Deblur Challenge 2021 (HDC2021) [[1]](#1)
URL: https://www.fips.fi/HDC2021.php)  
It will be evaluated the results of out-of-focus deblurring text images, although it is expected to be a general purpose deblurring algorithm.  

### Database from the HDC2021 (https://zenodo.org/record/4916176)
There are 20 steps of blur (from 0 to 19), each one including 100 sharp-blurred image pairs for each font (times and verdana), resulting in 4000 images, as well as the point, the  horizontal and the vertical spread functions of each blur.  
The images are separated in folders:  
1. step
   1. Font
     - CAM01: sharp images
      - CAM02: blurred images
The images are .TIFF files. 
Image size: 2360 x 1460 pixels
For a single step, the training set includes 70 images (70%) and the test set the 30 remaining images (30%). 

### Forward problem
We consider the forward problem, i.e., to blur the image, as the convolution of an image x with a Point Spread Function (PSF) k  
<img src="https://render.githubusercontent.com/render/math?math=y = k*x %2B e,">  
where y is the resulting blurred image and e is an additive noise.

To simulate the out of focus blur the PSF is considered as a disc, where the only parameter is its radius. Inside the disc, the value is 1 and outside the disc the value is 0 [[2]](#2).. For each blur step (from 0 to 19), the PSF radius was visually estimated from the sharp-blurred image pairs.

It should be noted that we used no blurring matrix, because t would be computational costly. All the blurring is computed directly with the PSF.

The Blur category number is one of the three input arguments of the function. It is also important to select the correct image folder. 

### Inverse problem 
There are three parts to reconstruct the sharp images.

#### Reconstruction part one: Deep image prior (DIP)
* Input: blurred images from the dataset (training set)
* Output: resulting images from the DIP network (only)


The first step is to fit a generator network (defined by the user) to a single degraded image, repeating for all images of the training set.
The deep generator network is a parametric function <img src="https://render.githubusercontent.com/render/math?math=f_{\theta}(z)"> 
where the weights θ are  randomly initialized. Then, the weights are adjusted to map the random vector z to the image x [[3]](#3).

<img src="https://render.githubusercontent.com/render/math?math=\theta_1^* = \arg\underset{\theta_1}{\min} E (f_{\theta_1}(z) * k, y) "> 

After this, the partial reconstructed image is obtained by  
<img src="https://render.githubusercontent.com/render/math?math=x_1^* = f_{\theta_1^*}(z) ">   
(in this sense, DIP is a learning-free method, as it depend soolely on the degraded image).    
This results in a (third) folder of images, named 'res', with partial reconstructions of the blurred images. 


#### Reconstruction part two: "Autoencoder" network with bottleneck architeture
* Input: resulting images from the DIP network and sharp images from the dataset (traning set)
* Output: "autoencoder" network weights

The second part of the reconstruction task is to train an bottleneck deep neural network to map the (first) DIP output to the database sharp images. 
It resembles an autoencoder (this is the reason for the quotation marks on "autoencoder"), but this is not about self-supervised learning. In fact, this part two is an image-to-image translation task in a supervised fashion.

Both the part one and part two could be repeated for each blur step, saving the autoencoder weights for each ot them. 

<img src="https://render.githubusercontent.com/render/math?math=\Theta^* = \arg\underset{\Theta}{\min} E (h_{\Theta}(x_1^*), y) "> 

#### Reconstruction part three: regularized DIP

* Input: blurred images from the dataset (test set)
* Output: resulting images from the regularized DIP network 

The architeture of the deep generative network from the DIP method used here is the same as in the part one, with some different hyperparameters.  
The main difference is that after 1000 iterations (DIP only), the loss function now includes the sum of both the DIP and the autoencoder outputs.   
The idea is to use the autoencoder as an regularizer controlled by a regularization parameter.   

<img src="https://render.githubusercontent.com/render/math?math=\theta_2^* = \arg\underset{\theta_2}{\min} E [(f_{\theta_2}(z) * k, y) %2B \lambda h_{\Theta^*}(x_1^*)">   
where <img src="https://render.githubusercontent.com/render/math?math=\lambda"> is the regularization parameter.

After this, the final reconstructed image <img src="https://render.githubusercontent.com/render/math?math=x_2^*"> is obtained by   
<img src="https://render.githubusercontent.com/render/math?math=x_2^* = f_{\theta_2^*}(z) ">  

# Installation instructions, including any requirements.

We first mention we adapted functions from the following two papers:
* From the original "Deep Image prior" paper[[3]](#3)
Available at https://github.com/DmitryUlyanov/deep-image-prior/, under Apache License 2.0
The particular requisites are shown here: https://github.com/DmitryUlyanov/deep-image-prior/blob/master/README.md
* From a derivative work: Neural Blind Deconvolution Using Deep Priors [[4]](#4)
https://github.com/csdwren/SelfDeblur (no copyright disclaimer was found)
The particular requisites are shown here: https://github.com/csdwren/SelfDeblur/blob/master/README.md

Although these toolboxes have their prerequisites, all the prerequisites needed for our code are shown below:

## Prerequisites
* Python
* PyTorch (torch)
* torch.nn
* TensorFlow
* Keras
* numpy
* matplotlib
* tqdm - progress bar
* print_function
* argparse
* os
* sys
* re
* PIL
* math
* cv2
* torchvision
* sklearn
* skimage
* pytesseract
* scipy.ndimage



We assume the input images are .TIFF files in our code, but the user can define the file extension.



## Executing the code via Google Colab
The URLs to the Google Colab Notebooks are:

The images can be uploaded to the Google Drive and 

## Executing the files via Anaconda

In this repository there are 3 files available, one for each step.





# Usage instructions.
*Python: The main function must be a callable function from the command line. To achieve this you can use sys.argv or argparse module.
*Example calling the function:
*$ python3 main.py path/to/input/files path/to/output/files 3


# Examples.
Some results with the corresponding text obtained by the OCR pytesseract are:


Step 15:
<img src="focusStep_3_timesR_size_30_sample_0001.jpg" width="70">
OCR text:
Target:





## References
<a id="1">[1]</a> 
Juvonen, Markus, et al. “Helsinki Deblur Challenge 2021: Description of Photographic Data.” ArXiv:2105.10233 [Cs, Eess], May 2021. arXiv.org, http://arxiv.org/abs/2105.10233.

<a id="2">[2]</a> 
C.   P.   Hansen,   G.   Nagy,   and   D.   P.   O’Leary.,Deblurring   images:   matrices,   spectra,   and   filtering. Philadelphia:   SIAM,   Societyfor  Industrial  and  Applied  Mathematics,  2006.  [Online].  Available:http://www.imm.dtu.dk/∼pcha/HNO

<a id="3">[3]</a> 
D. Ulyanov, A. Vedaldi, and V. Lempitsky.
“Deep image prior” International Journal of Computer Vision, vol. 128, no. 7, pp.1867–1888, Mar. 2020. [Online]. Available: https://doi.org/10.1007/s11263-020-01303-4

<a id="4">[4]</a> 
D. Ren, K. Zhang, Q. Wang, Q. Hu and W. Zuo, "Neural Blind Deconvolution Using Deep Priors," 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 3338-3347, doi: 10.1109/CVPR42600.2020.00340.



