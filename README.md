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
The results will be evaluated in out-of-focus text deblurring images, although it is expected to be a general-purpose deblurring algorithm.  

### Dataset from the HDC2021 (https://zenodo.org/record/4916176)
There are 20 steps of blur (from 0 to 19), each one including 100 sharp-blurred image pairs.  
There are 2 different text fonts (Times New Roman and Verdana), resulting in 4000 images.
There is also the point, the horizontal, and the vertical spread functions of each blur.  

The images are separated into folders:  
1. step (20 folders, each one for a blur step)
   1. Font (2 folders - Times and Verdana)
     - CAM01: sharp images
      - CAM02: blurred images
The images are .TIFF files. 
Image size: 2360 x 1460 pixels
For a single step, the training set includes 70 images (70%) and the test set the 30 remaining images (30%). 

### Forward problem
We consider the forward problem, i.e., to blur the image, as 
<img src="https://render.githubusercontent.com/render/math?math=y = k*x,">  
where x is the sharp image, k is the point spread function (PSF), and y is the resulting blurred image.  
Although there is visible noise in both sharp and blurred images from the HDC dataset, no explicit noise model (e.g. gaussian additive noise) was considered.    

To simulate the out-of-focus blur the PSF is considered as a disk, where the only parameter is the disk radius.  
Inside the disk, the corresponding value is 1 and, outside the disk, the value is 0 [[2]](#2).  
For each blur step (from 0 to 19), the PSF radius was visually estimated from the sharp-blurred image pairs.  
The Blur category number is one of the three input arguments of the function. It is important to select the correct image folder and the PSF radius. 

It should be noted that we used no blurring matrix because it would be computationally expensive. All the blurring are computed directly with the PSF.
It is a non-blind deblurring algorithm, the PSF is not updated while iterating, assuming the PSF is known.
 
### Inverse problem 
There are three parts to reconstruct the sharp images.

#### Reconstruction part one: Deep image prior (DIP)
* Input: blurred images from the dataset (training set)
* Output: resulting images from the DIP network (only)

The part one is to fit a generator network (defined by the user) to a single degraded image, repeating for all images of the training set.  
In this sense, DIP is a learning-free method, as it depends solely on the degraded image.  
Also, no sharp image from the HDC is used in this part one.  
The deep generator network is a parametric function <img src="https://render.githubusercontent.com/render/math?math=f_{\theta}(z)"> 
where the generator weights θ are randomly initialized and z is a random vector.  

During the traning phase, the weights are adjusted to map <img src="https://render.githubusercontent.com/render/math?math=f_{\theta}(z)"> to the image x [[3]](#3), as the equation below includes the convolution with the PSF: 

<img src="https://render.githubusercontent.com/render/math?math=\hat{\theta}_1 = \arg\underset{\theta_1}{\min} E (f_{\theta_1}(z) * k, y) ">   
where <img src="https://render.githubusercontent.com/render/math?math=\hat{\theta}_1"> are the weights of the generator network f after fitting to the degraded image (the subscript refers to the part one) and E is the loss function.  

After this, the partial reconstructed image <img src="https://render.githubusercontent.com/render/math?math=x_1^* "> from part one is obtained by  
<img src="https://render.githubusercontent.com/render/math?math=x_1^* = f_{\theta_1^*}(z) ">   

This results in a (third) folder of images, named 'res', with partial reconstructions of the blurred images, with the same number of images as the traning set. 

#### Reconstruction part two: "Autoencoder" network with bottleneck architecture
* Input: resulting images from the DIP network and sharp images from the dataset (training set)
* Output: "autoencoder" network weights

The second part of the reconstruction task is to train a second deep neural network with a bottleneck architecture to map the (first) DIP output to the sharp images from the HDC2021 dataset. That is, ideally, <img src="https://render.githubusercontent.com/render/math?math=h_{\Theta}(x_1^*) = y) ">   
where <img src="https://render.githubusercontent.com/render/math?math=\Theta "> are the weights of the autoencoder h.

It resembles an autoencoder (this is the reason for the quotation marks on "autoencoder"), but this is not about self-supervised learning. In fact, this part two is an image-to-image translation task in a supervised fashion.  
The training in part two can be described by
<img src="https://render.githubusercontent.com/render/math?math=\hat{\Theta} = \arg\underset{\Theta}{\min} E (h_{\Theta}(x_1^*), y) ">  
where <img src="https://render.githubusercontent.com/render/math?math=\hat{\Theta} "> are the estimated autoencoder weights and E is a loss function (not necessarily the same as in part one.

Both part one and part two could be repeated for each blur step, saving the autoencoder weights for each of them. 

#### Reconstruction part three: regularized DIP

* Input: blurred images from the dataset (test set)
* Output: resulting images from the regularized DIP network 

The architecture of the deep generative network from the DIP method used here is the same as in part one, with some different hyperparameters.  
The main difference is that after 1000 iterations (DIP only), the loss function now includes the sum of both the DIP and the autoencoder outputs.   
The idea is to use the autoencoder as a regularizer, controlled by a regularization parameter.   

<img src="https://render.githubusercontent.com/render/math?math=\theta_2^* = \arg\underset{\theta_2}{\min} E [(f_{\theta_2}(z) * k, y) %2B \lambda h_{\Theta^*}(x_1^*)">   
where <img src="https://render.githubusercontent.com/render/math?math=\lambda">is the regularization parameter.

After this, the final reconstructed image <img src="https://render.githubusercontent.com/render/math?math=x_2^*"> is obtained by   
<img src="https://render.githubusercontent.com/render/math?math=x_2^* = f_{\theta_2^*}(z) ">  

# Installation, usage instructions and examples.

All the codes are available in this repository. There is a jupyter notebook called 'notebook_example.ipynb' explaining how to clone the repository, how to generate the results and how to visualize them. It includes an example from the blur step 15. 
It is also possible to execute the code via Google Colab. The HDC dataset can be uploaded to a google drive account, linking it to the Google Colab via 

directly into the Colab (not recommended) or  and linking to github via 
<img src="drive-to-colab.png" width="220">  

We need to mention that we adapted functions from the following two papers:
* From the original "Deep Image prior" paper[[3]](#3)
Available at https://github.com/DmitryUlyanov/deep-image-prior/, under Apache License 2.0
The particular requisites are shown here: https://github.com/DmitryUlyanov/deep-image-prior/blob/master/README.md
* From a derivative work: Neural Blind Deconvolution Using Deep Priors [[4]](#4)
https://github.com/csdwren/SelfDeblur (no copyright disclaimer was found)
The particular requisites are shown here: https://github.com/csdwren/SelfDeblur/blob/master/README.md

Although these toolboxes have their prerequisites, all the prerequisites needed for our code are shown below:

## Prerequisites 
* Python: 3.7.12

Main Packages:
* argparse==1.1
* cv2==4.1.2
* joblib==1.0.1
* Keras==2.6.0
* matplotlib==3.2.2
* numpy==1.19.5
* Pillow==7.1.2
* pytesseract==0.3.8
* re==2.2.1
* scipy==1.4.1
* skimage==0.16.2
* sklearn==0.22.2
* tensorflow==2.6.0
* tqdm==4.62.2
* torch==1.9.0
* torchvision==0.10.0

Observations:
* We assume the input images are .TIFF files in our code, but the user can define the file extension.
* The complete list of packages in the Google Colab (obtained by pip freeze > requirements.txt) can be found in the main repository folder. Some of the versions were removed to prevent conflicts, but not all of them are necessary running the code outside Google Colab.  

# Usage instructions.



## 
The URLs to the Google Colab Notebooks are:
 

## Executing the files via Anaconda

There are 3 files available in this repository, one for each step.

# Examples.
Some results with the corresponding text obtained by the OCR pytesseract are:


Step 15: Resulting image  

OCR text:  
Target:  

## References
<a id="1">[1]</a> 
Juvonen, Markus, et al. “Helsinki Deblur Challenge 2021: Description of Photographic Data.” ArXiv:2105.10233 [Cs, Eess], May 2021. arXiv.org, http://arxiv.org/abs/2105.10233.

<a id="2">[2]</a> 
C. P. Hansen, G. Nagy, and D. P. O’Leary. Deblurring images: matrices, spectra, and filtering. Philadelphia: SIAM, Society for Industrial and Applied Mathematics, 2006. 

<a id="3">[3]</a> 
D. Ulyanov, A. Vedaldi, and V. Lempitsky.
“Deep image prior” International Journal of Computer Vision, vol. 128, no. 7, pp.1867–1888, Mar. 2020. [Online]. Available: https://doi.org/10.1007/s11263-020-01303-4

<a id="4">[4]</a> 
D. Ren, K. Zhang, Q. Wang, Q. Hu and W. Zuo, "Neural Blind Deconvolution Using Deep Priors," 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 3338-3347, doi: 10.1109/CVPR42600.2020.00340.
