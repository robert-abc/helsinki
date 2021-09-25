# Helsinki deblur challenge

## Authors, institution, location

Leonardo Ferreira Alves, Roberto Gutierrez Beraldo, André Kazuo Takahata and Ricardo Suyama - Federal University of ABC (Brazil)

## Brief description of your algorithm and a mention of the competition.
The identification of alphanumeric characters depends on the quality of the image obtained. For images with blur caused by lack of focus, it is necessary to restore it to a sharper image, the so-called deblurring task. In this work, we used convolutional neural networks to deblur images from the Helsinki Deblur Challenge 2021 (HDC2021, https://www.fips.fi/HDC2021.php).

![Image](focusStep_3_timesR_size_30_sample_0001.jpg)


### Forward problem

Deblurring na forma matricial:
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{Y}=\mathbf{K}*\mathbf{X}+\mathbf{N}">
onde Y representa a imagem borrada, X representa a imagem nítida, K é o kernel de borramento e N um ruído externo. 

Para simular o efeito de uma imagem fora de foco, a PSF pode ser modelada como um disco[[2]](#1).. 
 
Para cada nível de dificuldade, o raio da PSF foi estimada visualmente a partir do par nítido-borrado.  

### Inverse problem

#### First step: Deep image prior (DIP)

"randomly-initialized neural network can be used as a handcrafted prior" [[1]](#1).
we fit a generator network to a single degraded image. In this scheme, the network weights serve as a parametrization of the restored image. The weights are randomly initialized and fitted to a specific degraded image under a task-dependent observation model. In this manner, the only information used to perform reconstruction is contained in the single degraded input image and the handcrafted structure of the network used for reconstruction [[1]](#1)..


Adeep generator network is a parametric function <img src="https://render.githubusercontent.com/render/math?math=x = f_{\theta_{DIP}}(z)"> 
that maps a code vector z to an image x [[1]](#1)..


<img src="https://render.githubusercontent.com/render/math?math=\theta^* = \arg\underset{\theta}{\min} E (f_{\theta}(z), x_0) "> 

#### Second step: autoencoder

f_{\theta_{autoencoder}}(z)




#### Third step: regularized DIP




## Installation instructions, including any requirements.



## Usage instructions.



## Show a few examples.





## References
<a id="1">[1]</a> 
D. Ulyanov, A. Vedaldi, and V. Lempitsky.
“Deep image prior” International Journal of Computer Vision, vol. 128, no. 7, pp.1867–1888, Mar. 2020. [Online]. Available: https://doi.org/10.1007/s11263-020-01303-4

<a id="2">[2]</a> 
C.   P.   Hansen,   G.   Nagy,   and   D.   P.   O’Leary.,Deblurring   images:   matrices,   spectra,   and   filtering. Philadelphia:   SIAM,   Societyfor  Industrial  and  Applied  Mathematics,  2006.  [Online].  Available:http://www.imm.dtu.dk/∼pcha/HNO
