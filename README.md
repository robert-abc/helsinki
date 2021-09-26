# Helsinki deblur challenge 2021

## Authors, institution, location

* André Kazuo Takahata¹ - andre.t@ufabc.edu.br
* Leonardo Ferreira Alves¹ - leonardo.alves@ufabc.edu.br
* Ricardo Suyama¹ - ricardo.suyama@ufabc.edu.br
* Roberto Gutierrez Beraldo¹ - roberto.gutierrez@ufabc.edu.br

¹Federal University of ABC (Santo André, São Paulo, Brazil) - https://www.ufabc.edu.br/

## Brief description of your algorithm and a mention of the competition.
For images with blur caused by lack of focus, it is necessary to restore it to a sharper image, the so-called deblurring task. 
This work is to join the Helsinki Deblur Challenge 2021 (HDC2021, https://www.fips.fi/HDC2021.php), where it will be evaluated in deblurring text images, although it is expected to be a general purpose deblurring algorithm.   

<img src="focusStep_3_timesR_size_30_sample_0001.jpg" width="70">

### Database from the HDC2021 (https://zenodo.org/record/4916176)
There are 20 steps of blur (from 0 to 19), each step including 100 sharp-blurred image pairs for each font (times and verdana), resulting in 4000 images, as well as the point, the  horizontal and the vertical spread functions of each blur.
The images are separated in folders:
- step
-- Font
--- CAM01: sharp images
--- CAM02: blurred images

For a single step, the training set includes 70 images (70%) and the test set the 30 remaining images (30%). 

### Forward problem
We consider the forward problem, i.e., to blur the image, as the convolution of an image X with a Point Spread Function (PSF) K
<img src="https://render.githubusercontent.com/render/math?math=y = k*x,">
where Y is the resulting blurred image.

To simulate the out of focus blur the PSF is considered as a disc, where the only parameter is its radius. Inside the disc, the value is 1 and outside the disc the value is 0 [[2]](#1).. For each blur step (from 0 to 19), the PSF radius was visually estimated from the sharp-blurred image pairs.

It should be noted that we used no blurring matrix, because t would be computational costly. All the blurring is computed directly with the PSF.

The Blur category number is one of the three input arguments of the function. It is also important to select the correct image folder. 

### Inverse problem 
There are three steps to reconstruct the sharp images.

#### First step: Deep image prior (DIP)
* Input: blurred images from the dataset (training set)
* Output: resulting images from the DIP network (only)


The first step is to fit a generator network (defined by the user) to a single degraded image, repeating for all images of the training set.
This results in a (third) folder of images, named 'res', with partial reconstructions of the blurred images. 

The deep generator network is a parametric function <img src="https://render.githubusercontent.com/render/math?math=x = f_{\theta}}(z)"> 
where the weights θ are  randomly initialized. Then, the weights are adjusted to map the random vector z to the image x [[1]](#1)..

<img src="https://render.githubusercontent.com/render/math?math=\theta^* = \arg\underset{\theta}{\min} E (f_{\theta}(z), x_0) "> 


#### Second step: Autoencoder
* Input: resulting images from the DIP network and sharp images from the dataset (traning set)
* Output: autoencoder weights

<img src="https://render.githubusercontent.com/render/math?math=(f_{\theta}(z), x_0) "> 


A

#### Third step: regularized DIP

* Input: blurred images from the dataset 
* Output: resulting images from the DIP network (only)


## Installation instructions, including any requirements.
In this repository there are 3 files available, one for each step.
At the same time, there are Google Colab notebooks 


## Usage instructions.
*Python: The main function must be a callable function from the command line. To achieve this you can use sys.argv or argparse module.
*Example calling the function:
*$ python3 main.py path/to/input/files path/to/output/files 3


## Show a few examples.





## References
<a id="1">[1]</a> 
D. Ulyanov, A. Vedaldi, and V. Lempitsky.
“Deep image prior” International Journal of Computer Vision, vol. 128, no. 7, pp.1867–1888, Mar. 2020. [Online]. Available: https://doi.org/10.1007/s11263-020-01303-4

<a id="2">[2]</a> 
C.   P.   Hansen,   G.   Nagy,   and   D.   P.   O’Leary.,Deblurring   images:   matrices,   spectra,   and   filtering. Philadelphia:   SIAM,   Societyfor  Industrial  and  Applied  Mathematics,  2006.  [Online].  Available:http://www.imm.dtu.dk/∼pcha/HNO
