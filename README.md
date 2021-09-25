# Helsinki deblur challenge


A identificação de caracteres alfanuméricos depende da qualidade da imagem obtida. Para os casos em que há borramento, seja ele por falta de foco ou por movimento, é necessária a sua restauração para uma imagem mais nítida, o chamado \textit{deblurring}. Nos últimos anos, redes neurais convolucionais tem sido utilizadas em diversos problemas de imagens, incluindo o \textit{deblurring}. Neste trabalho, uma rede convolucional do tipo "gargalo" foi utilizada para realização do \textit{deblurring} a partir da sua formulação como  um algoritmo de aprendizagem supervisionado, onde a imagem de entrada era a imagem borrada e a imagem de saída era a imagem nítida. 


Deep image prior and deep inverse priors




"randomly-initialized neural network can be used as a handcrafted prior" [[1]](#1).


Deblurring na forma matricial:
<img src="https://render.githubusercontent.com/render/math?math=\mathbf{Y}=\mathbf{K}*\mathbf{X}+\mathbf{N}">
onde Y representa a imagem borrada, X representa a imagem nítida, K é o kernel de borramento e N um ruído externo. 



## References
<a id="1">[1]</a> 
D. Ulyanov, A. Vedaldi, and V. Lempitsky.
“Deep image prior”
International Journal of Computer Vision, vol. 128, no. 7, pp.1867–1888, Mar. 2020.   
[Online].   Available:   https://doi.org/10.1007/s11263-020-01303-4
