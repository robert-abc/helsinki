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

The main idea here is based on the deep image prior reconstruction, but, instead of using the deep image prior alone, a DNN with a bottleneck architecture (as an autoencoder) is used to help the deblurring task.

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
* There is also the point, the horizontal, and the vertical spread functions of each blur.  
* All the images are .TIFF files. We assume the input images are .TIFF files in our code, but the user can define another file extension.  
* The input image size: 2360 x 1460 pixels.
* Expected output image size: 2360 x 1460 pixels.

## 2. Hypothesis of the forward problem 
We consider the forward problem, i.e., to blur the image, as 
<img src="https://render.githubusercontent.com/render/math?math=y = k*x,">  
where x is the sharp image, k is the point spread function (PSF), and y is the resulting blurred image.  
Although there is visible noise in both sharp and blurred images from the HDC dataset, no explicit noise model (e.g. gaussian additive noise) was considered.    

The PSF is considered as a disk to simulate the out-of-focus blur , where the only parameter is the disk radius.  
Inside the disk, the corresponding value is 1 and, outside the disk, the value is 0 [[2]](#2).  

### Notes:
* The Blur category number is one of the three input arguments of the function. It is important to select the correct image folder and the PSF radius. 
* All the blurring are computed by convolution with the PSF (like the conv2 function). 
* We used no blurring matrix because it would be computationally expensive (like the Ax = b linear system). 
* We assume we know the PSF, altough there could be better PSF estmations. In this sense, it is a non-blind deblurring algorithm.
* The PSF is not updated while iterating (and that could be accomplished in the future).
 
## 3. Inverse problem part one: partial reconstruction via Deep image prior (DIP)
* Input: blurred images from the dataset (training set)
* Output: resulting images from the DIP network (only)

The part one is to fit a generator network (defined by the user) to a single degraded image, repeating for all images of the training set.
In this sense, DIP is a learning-free method, as it depends solely on the degraded image. No sharp image from the HDC is used in this part one.  

### 3.1 DIP overview
The deep generator network is a parametric function <img src="https://render.githubusercontent.com/render/math?math=f_{\theta}(z)"> 
where the generator weights θ are randomly initialized and z is a random vector.  

During the traning phase, the weights are adjusted to map <img src="https://render.githubusercontent.com/render/math?math=f_{\theta}(z)"> to the image x [[3]](#3), as the equation below includes the convolution with the PSF:  
<img src="https://render.githubusercontent.com/render/math?math=\hat{\theta}_1 = \arg\underset{\theta_1}{\min} E (f_{\theta_1}(z) * k, y) ">  
where <img src="https://render.githubusercontent.com/render/math?math=\hat{\theta_1} ">  are the are the weights of the generator network f after fitting to the degraded image (the subscript refers to the part one) and E is the loss function.  

After this, the partial reconstructed image <img src="https://render.githubusercontent.com/render/math?math=x_1^* "> from part one is obtained by  
<img src="https://render.githubusercontent.com/render/math?math=x_1^* = f_{\theta_1^*}(z) ">   

### Notes:
* For a single blur step, our training set included 70 blurred images (70% of the total).
* This results in a (third) folder of images, named 'res', with 70 partial reconstructions of the blurred images (the same number of images as the traning set). 


### 3.2 Estimating the PSF radius in part one
For each blur step (from 0 to 19), the PSF radius was visually estimated from the sharp-blurred image pairs:
* We ran part one with a single degraded image (from each step), varying the PSF radius, comparing the output to the corresponding sharp image and choosing the "best" radius.
* We limited our radius to integer numbers, but it was possible to choose non-integer numbers too.

One example can be seen in the notebook "Find_Radius-s5r8.ipynb" of this repository, where s5 denotes step 05 and r8 denotes radius = 8.   
The result is shown in the cells #16 and #17.  
This notebook also illustrates the reconstruction part one: given the radius, reconstruct all the blurred images in the training set.

## 4. Inverse problem part two: "Autoencoder" network with bottleneck architecture
* Input: resulting images from the DIP network and sharp images from the dataset (training set)
* Output: "autoencoder" network weights

The second part of the reconstruction task is to train a second deep neural network with a bottleneck architecture to map the (first) DIP output to the sharp images from the HDC2021 dataset. 

### 4.1 Autoencoder overview
Ideally, wee wwant the leanring machine to be able to convert the DIP output <img src="https://render.githubusercontent.com/render/math?math=x_1^* "> to the sharp image x, that is, <img src="https://render.githubusercontent.com/render/math?math=h_{\Theta}(x_1^*) = x ">,  where <img src="https://render.githubusercontent.com/render/math?math=\Theta "> are the weights of the autoencoder h.

The training in part two can be described by
<img src="https://render.githubusercontent.com/render/math?math=\hat{\Theta} = \arg\underset{\Theta}{\min} E (h_{\Theta}(x_1^*), x) ">  
where <img src="https://render.githubusercontent.com/render/math?math=\hat{\Theta} ">  are the estimated autoencoder weights and E is a loss function (not necessarily the same as in part one.

### Notes: 
* The architecture resembles an autoencoder (this is the reason for the quotation marks on "autoencoder"), but this is not about self-supervised learning. 
* In fact, this part two is an image-to-image translation task in a supervised fashion.  
* Both part one and part two should be repeated for each blur step, saving the autoencoder weights for each of them.  
* The two networks architectures and the training itself do not change over the different blur steps, only the PSF radius and the input images (as in the HDC2021 rules).   

## 5. Inverse problem part three: Regularized DIP

* Input: blurred images from the dataset (test set), autoencoder weights Θ
* Output: resulting images from the regularized DIP network 

The architecture of the deep generative network from the DIP method used in part three is the same as in part one, with some different parameters.  
The main difference is that after 1000 iterations (DIP only), the loss function will include the sum of both the DIP and the autoencoder outputs.   
The idea is to use the autoencoder as prior information (as a regularizer), controlled by a regularization parameter.  

### 5.1 Regularized DIP overview

During the initial 1000 iterations, the training phase can be described as in part one. 
After 1000 iterations, a autoencoder output term is included in the loss function, that is,  
<img src="https://render.githubusercontent.com/render/math?math=\theta_2^* = \arg\underset{\theta_2}{\min} E [(f_{\theta_2}(z) * k, y) %2B \lambda h_{\Theta^*}(x_1^*)">   
where λ is the regularization parameter.

After the user-defined number of iterations, the final reconstructed image <img src="https://render.githubusercontent.com/render/math?math=x_2^*"> is obtained by   
<img src="https://render.githubusercontent.com/render/math?math=x_2^* = f_{\theta_2^*}(z) ">  

### Notes:
* In part three, a single blurred image from the test set (30 remaining images from that blur step) is reconstructed at one time, so we will not necessarily use all the test set images.
 
## 6. Installation 
All the codes we used are available in this repository.   
There is the main.py, the "utils" folder with several functions and the "weights" folder with the autoencoder weights.  
There is also the notebooks to run the codes (as seen in the usage instructions section)

The main point is that we didn't "install" anything, because we did everything on Google Colab, for the following reasons:
* Because of the COVI19 pandemic, out university is closed, so it allow us to work together online.
* In this context, we are using our personal notebooks, but they have basic specifications. Even the free Colab account is more powerful (although there are usage limits). 
* It is necessary to have a compatible Python CUDA for GPU support and Google Colab allow us to access them. 

In the following table, there is a small list of the main packages used (with "import" functions, for example) and it is important to note that:
* The complete list of packages in the Google Colab (obtained by pip freeze > requirements.txt) can be found in the main repository folder.  
* We understand that not all of them are necessary running the code outside Google Colab, but in fact we did not run the code locally. 
* It should be possible to create, for example, an Anaconda enviroment, with this requirements.txt, but we didn't do it.  
* It would be like creating google colab in a environment, but, with more than 400 packages, conflicts between them could happen.

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
| pytesseract | 0.3.8  | | |

*or regex

That being said, we can share the Google Colab URLs to execute the codes:  (INSERIR URL)

### 6.1 External codes
We need to mention that we adapted functions from the following two papers:
1. From the original "Deep Image prior" paper[[3]](#3)
Available at https://github.com/DmitryUlyanov/deep-image-prior/, under Apache License 2.0
The particular requisites are shown here: https://github.com/DmitryUlyanov/deep-image-prior/blob/master/README.md
1. From a derivative work: Neural Blind Deconvolution Using Deep Priors [[4]](#4)
https://github.com/csdwren/SelfDeblur (no copyright disclaimer was found)
The particular requisites are shown here: https://github.com/csdwren/SelfDeblur/blob/master/README.md

Although these toolboxes have their own prerequisites, the requirements.txt includes the ones we need. 

## 7. Usage instructions and examples.

### 7.1 Part one and part two to get the autoencoder weights

Part one and Part two refers to the training_example.ipynb notebook. It results in the autoencoder weights. 
We would like to have the autoencoder weights for all the blur steps, but unfortunately it was not possible in time.

### 7.2 Part three

Part three, the reconstruction step (and the only part you are actually requiring), can be seen in the jupyter notebook called 'notebook_example.ipynb' explaining how to clone the (private) repository, how to generate the part three results and how to visualize them. 
It also includes an example from the blur step 15. 


The HDC dataset can be uploaded to a google drive account, linking it to the Google Colab via  
<img src="drive-to-colab.png" width="300">  
(it's not recommended to upload the images directly into the Colab with a free account because of running time limitations) 




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
