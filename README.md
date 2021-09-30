# Helsinki deblur challenge 2021: Brief description of the algorithm

## Authors
* André Kazuo Takahata¹ - andre.t@ufabc.edu.br
* Leonardo Ferreira Alves¹ - leonardo.alves@ufabc.edu.br
* Ricardo Suyama¹ - ricardo.suyama@ufabc.edu.br
* Roberto Gutierrez Beraldo¹ - roberto.gutierrez@ufabc.edu.br

¹Federal University of ABC (Santo André, São Paulo, Brazil) - https://www.ufabc.edu.br/

## 
This deblurring work is to join the Helsinki Deblur Challenge 2021 (HDC2021) [[1]](#1)
URL: https://www.fips.fi/HDC2021.php)  
The results will be evaluated in out-of-focus text deblurring images, although it is expected to be a general-purpose deblurring algorithm.  

The main idea here is based on the Deep Image Prior (DIP) reconstruction, but it uses only the degraded image.  
So, instead of using the DIP alone, a second DNN with bottleneck architecture (as an autoencoder) is used to help the deblurring task, as it includes (prior) information from the sharp images too.

## 1. Dataset from the HDC2021 (https://zenodo.org/record/4916176)
There are 20 steps of blur (from 0 to 19), each one including 100 sharp-blurred image pairs.  
There are 2 different text fonts (Times New Roman and Verdana), resulting in 4000 images.
The images are separated into folders:  
1. step (20 folders, each one for a blur step)
   1. Font (2 folders - Times and Verdana)
     - CAM01: sharp images
      - CAM02: blurred images  

### Notes:
*  Each image has its ground-truth text.
* There are also the point, the horizontal, and the vertical spread functions of each blur.  
* All the images from the dataset are .TIFF files which we assume is the case in our code, but the user can define another file extension.  
* The input image size: 2360 x 1460 pixels.
* Expected output image size: 2360 x 1460 pixels.

## 2. Forward problem 
We consider the forward problem, i.e., to blur the image, as 
<img src="https://render.githubusercontent.com/render/math?math=y = k*x,">  
where x is the sharp image, k is the point spread function (PSF), and y is the resulting blurred image [[2]](#2).  
Although there is visible noise in both sharp and blurred images from the HDC dataset, no explicit noise model (e.g. Gaussian additive noise) was considered.    

The PSF is considered as a disk to simulate the out-of-focus blur, where the only parameter is the disk radius.  
Inside the disk, the corresponding value is 1 and, outside the disk, the value is 0 [[3]](#3).  

### Notes:
* The Blur category number is one of the three input arguments of the function. It is important to select the correct image folder and the PSF radius. 
* All the blurring is computed by convolution with the PSF (like the conv2 function). We used no blurring matrix because it would be computationally expensive (like the Ax = b linear system). 
* We assume we know the PSF, although there could be better PSF estimations. In this sense, it is a non-blind deblurring algorithm.
* The PSF is not updated while iterating (and that could be accomplished in the future).
 
## 3. Inverse problem part one: partial reconstruction via Deep image prior (DIP)
* Input: blurred images from the dataset (training set)
* Output: resulting images from the DIP network (only)

Part one is to fit a generator network (defined by the user) to a single degraded image, repeating for all images of the training set.
In this sense, DIP is a learning-free method, as it depends solely on the degraded image. No sharp image from the HDC is used in this part one.  

### 3.1 DIP overview
The deep generator network is a parametric function <img src="https://render.githubusercontent.com/render/math?math=f_{\theta}(z)"> 
where the generator weights θ are randomly initialized and z is a random vector.  

During the traning phase, the weights are adjusted to map <img src="https://render.githubusercontent.com/render/math?math=f_{\theta}(z)"> to the image x [[4]](#4), as the equation below includes the convolution with the PSF:  
<img src="https://render.githubusercontent.com/render/math?math=\hat{\theta}_1 = \arg\underset{\theta_1}{\min} E (f_{\theta_1}(z) * k, y) ">  
where <img src="https://render.githubusercontent.com/render/math?math=\hat{\theta_1} ">  are the weights of the generator network f after fitting to the degraded image, the subscript 1 refers to the part one (and so on), the superscript ^ denotes an estimation, the operator * denotes convolution and E is the loss function.  

After this, the partial reconstructed image <img src="https://render.githubusercontent.com/render/math?math=\hat{x_1} "> from part one is generated by the network by  
<img src="https://render.githubusercontent.com/render/math?math=\hat{x_1} = f_{\hat{\theta_1}}(z) ">   

### Notes:
* For a single blur step, our training set included 70 blurred images (70% of the total).
* This results in a (third) folder of images, named 'res', with 70 partial reconstructions of the blurred images (the same number of images as the training set). 


### 3.2 Estimating the PSF radius in part one
For each blur step (from 0 to 19), the PSF radius was visually estimated from the sharp-blurred image pairs:
* We ran part one with a single degraded image (from each step), varying the PSF radius, comparing the output to the corresponding sharp image, and choosing the "best" radius.
* We limited our radius to integer numbers, but it was possible to choose non-integer numbers too.

One example can be seen in the notebook "Find_Radius-s5r8.ipynb" of this repository, where s5 denotes step 05 and r8 denotes radius = 8.   
The result is shown in cells #16 and #17.  
This notebook also illustrates the reconstruction part one: given the radius, reconstruct all the blurred images in the training set.

## 4. Inverse problem part two: "Autoencoder" network with bottleneck architecture
* Input: resulting images from the DIP network and sharp images from the dataset (training set)
* Output: "autoencoder" network weights

The second part of the reconstruction task is to train a second deep neural network with a bottleneck architecture to map the (first) DIP output to the sharp images from the HDC2021 dataset. 

### 4.1 Autoencoder overview
Ideally, we want the learning machine to be able to convert the DIP output <img src="https://render.githubusercontent.com/render/math?math=\hat{x_1} "> to the sharp image x (of the training set.  
That is, <img src="https://render.githubusercontent.com/render/math?math=h_{\Theta}(\hat{x_1}) = x ">,  where <img src="https://render.githubusercontent.com/render/math?math=\Theta "> are the weights of the autoencoder h.

The training in part two can be described by
<img src="https://render.githubusercontent.com/render/math?math=\hat{\Theta} = \arg\underset{\Theta}{\min} E (h_{\Theta}(\hat{x_1}), x) ">  
where <img src="https://render.githubusercontent.com/render/math?math=\hat{\Theta} ">  are the estimated autoencoder weights and E is a loss function (not necessarily the same as in part one.

### Notes: 
* The architecture resembles an autoencoder (this is the reason for the quotation marks on "autoencoder"), but this is not about self-supervised learning. 
* In fact, this part two is an image-to-image translation task in a supervised fashion.  
* Both part one and part two should be repeated for each blur step, saving the autoencoder weights for each of them. 
* The two networks architectures and the training itself do not change over the different blur steps, only the PSF radius and the input images (to respect the HDC2021 rules).

## 5. Inverse problem part three: Regularized DIP

* Input: blurred images from the dataset (test set), autoencoder weights Θ
* Output: resulting images from the regularized DIP network 

The idea is to use the autoencoder as prior information (as a regularizer), controlled by a regularization parameter.  
The main difference from part one is that after 1000 iterations (DIP only), the loss function will include the sum of both the DIP and the autoencoder outputs. 
  
### 5.1 Regularized DIP overview

During the initial 1000 iterations, the training phase can be described as in part one. 
After 1000 iterations, a autoencoder output term is included in the loss function, that is,  
<img src="https://render.githubusercontent.com/render/math?math=\hat{\theta_3} = \arg\underset{\theta_3}{\min} E [(f_{\theta_3}(z) * k, y) %2B \lambda h_{\hat{\Theta}}(\hat{x_1})]">   
where λ is the regularization parameter.

After the user-defined number of iterations, the final reconstructed image <img src="https://render.githubusercontent.com/render/math?math=\hat{x}_{\text{final}}"> is obtained by   
<img src="https://render.githubusercontent.com/render/math?math=\hat{x}_{\text{final}} = f_{\hat{\theta_3}}(z) ">  

### Notes:
* The architecture of the deep generative network from the DIP method used in part three is the same as in part one, with some different parameters.  
* This architecture from part three remains the same in all blur steps.
* In part three, a single blurred image from the test set (30 remaining images from that blur step) is reconstructed at one time, so we will not necessarily use all the test set images.
 
## 6. Installation, usage instructions, and examples

All the codes we used are available in this repository.   
There is also Jupyter Notebooks to run the codes (as seen in the usage instructions section)

The main point is that we didn't "install" anything, because we did everything on Google Colab, for the following reasons:
* Because of the COVI19 pandemic, our university is closed, so it allows us to work together online.
* In this context, we are using our personal notebooks, but they have basic specifications. Even the free Colab account is more powerful (although there are usage limits). 
* It is necessary to have a compatible Python CUDA for GPU support and Google Colab allows us to access them. 

In the following table, there is a small list of the main packages used (with "import" functions, for example) and it is important to note that:
* The complete list of packages in the Google Colab (obtained by pip freeze > requirements.txt) can be found in the main repository folder: <a href="requirements.txt">requirements</a>. 
* We understand that not all of them are necessary to run the code outside Google Colab. In fact, we did not run the code locally (e.g. with Anaconda). 
* It should be possible to create, for example, an Anaconda environment, with this requirements.txt, but we didn't do it.  
* It would be like creating Google Colab in an environment, but, with more than 400 packages, conflicts between them could happen.

| Package  | Version | Package  | Version |
| ------------- | ------------- | ------------- | ------------- |
| Python  | 3.7.12  |  re* | 2.2.1  | 
| argparse  | 1.1  |  scipy | 1.4.1  | 
| cv2  | 4.1.2  | skimage | 0.16.2  | 
| joblib | 1.0.1 | sklearn | 0.22.2  | 
| Keras | 2.6.0 | tensorflow | 2.6.0  |
| matplotlib | 3.2.2  | tqdm | 4.62.2  | 
| numpy | 1.19.5  | torch | 1.9.0  | 
| Pillow | 7.1.2  | torchvision | 0.10.0  | 

*or regex

It is not part of our code, but we tested reading the resulting images with the OCR pytesseract==0.3.8. 

### 6.1 External codes
We need to mention that we adapted functions from the following two papers:
1. From the original "Deep Image prior" paper[[4]](#4)
Available at https://github.com/DmitryUlyanov/deep-image-prior/, under Apache License 2.0
The particular requisites are shown here: https://github.com/DmitryUlyanov/deep-image-prior/blob/master/README.md
1. From a derivative work: Neural Blind Deconvolution Using Deep Priors [[5]](#5)
https://github.com/csdwren/SelfDeblur (no copyright disclaimer was found)
The particular requisites are shown here: https://github.com/csdwren/SelfDeblur/blob/master/README.md

Although these toolboxes have their own prerequisites, the requirements.txt includes the ones we need. 

### 6.2. Usage instructions

We created three notebooks to run with Google Colab, so here is a little setup to them both. 

First we clone the private git repository.   
it's not recommended to upload the images directly into the Colab with a free account because of running time limitations. 

So, next, the HDC dataset can be uploaded to a google drive account, linking it to the Google Colab via "mount drive"  
<img src="drive-to-colab.png" width="300">  
Google Colab will ask for a verification code and then it is possible to access Google Drive directly from the Google Colab.

After this, it is possible to execute the rest of the code.

* Part one and part two refer to the <a href="training_example.ipynb">training_example</a> notebook. It results in the autoencoder weights that are in the "weights" folder in this repository.
* Part three, the reconstruction step, can be seen in the Jupyter Notebook called <a href="notebook_example.ipynb">notebook_example</a> explaining how to clone the (private) repository, how to generate the part three results, and how to visualize them. It also includes an example from the blur step 15. 
* Part three can also be seen for blur step 19 in the <a href="notebook_example_19.ipynb">notebook_example_19</a>. We used the same network, but some parameters were different from the blur step 15 notebook. 

## 7. Final comments 

Regarding the autoencoder 
* We didn't "optimize" the autoencoder architecture and traning phase too much, which means that maybe it is possible to get better results with lower loss values. 
* In lower blur steps, the DIP-only deblurring (part one) may be sufficient, but is not preferable, as it uses no information of the sharp images from the dataset. 

Regarding the obtained autoencoder weights:
* We would like to have the autoencoder weights for all the blur steps, but unfortunately it was not possible in time (as Google Colab limits GPU usage).
* What we actually have in this release (folder "weights") are the autoencoder weights of the blur steps &#x1F536;&#x1F536;&#x1F536; 15, 19  &#x1F536;&#x1F536;&#x1F536;.
* We are not sure if HDC2021 will train the autoencoder for the blur steps we didn't (parts one and two) or if it is expected just the reconstruction itself (part three). 
* If we had all the autoencoder weights in the folder "weights", the algorithm would automatically select the weights corresponding to the blur step. 
* We also tested using the autoencoder trained in another blur step level. It worked in some cases, but we didn't have exhaustive testing of this option and it is harder to justify how to choose the (another) blur step selected. 
* In this context, the code verifies if the autoencoder weights are available (fot the right blur step). If it is not, it chooses the nearest.  
* Even after the end date (September 30) and the submitted release, we will keep training the autoencoder in the other blur steps. That is, not changing the code itself, just generating more results with it.

## References
<a id="1">[1]</a> 
Juvonen, Markus, et al. “Helsinki Deblur Challenge 2021: Description of Photographic Data.” ArXiv:2105.10233 [Cs, Eess], May 2021. arXiv.org, http://arxiv.org/abs/2105.10233.

<a id="2">[2]</a>
Mueller, Jennifer, and Samuli Siltanen. Linear and nonlinear inverse problems with practical applications. Philadelphia: Society for Industrial and Applied Mathematics, 2012. 

<a id="3">[3]</a> 
C. P. Hansen, G. Nagy, and D. P. O’Leary. Deblurring images: matrices, spectra, and filtering. Philadelphia: SIAM, Society for Industrial and Applied Mathematics, 2006. 

<a id="4">[4]</a> 
D. Ulyanov, A. Vedaldi, and V. Lempitsky.
“Deep image prior” International Journal of Computer Vision, vol. 128, no. 7, pp.1867–1888, Mar. 2020. [Online]. Available: https://doi.org/10.1007/s11263-020-01303-4

<a id="5">[5]</a> 
D. Ren, K. Zhang, Q. Wang, Q. Hu and W. Zuo, "Neural Blind Deconvolution Using Deep Priors," 2020 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2020, pp. 3338-3347, doi: 10.1109/CVPR42600.2020.00340.
