# Neural-Style-Transfer

## Intro
In this repo, I implemented Neural Style Transfer Algorithm. This algorithm was created by Gatys et al. (2015) (http://arxiv.org/abs/1508.06576). This Deep Learning technique takes two images, namely, the content image(C) and a style image(S) and generates a new image(G) which combines the content of image C with style of image S.

I used a pre-trained VGG-19 model which is a very deep convolutional neural network because these deep pre-trained models can detect low-level features such as edges and vertex at earlier layers and high-level features at deep layers, pretty accurately.

### Pre-trained model license
----------------------------------------------------------------------------------
Here, we are using a pretrained model that can be downloaded at the following link: http://www.vlfeat.org/matconvnet/pretrained/. The pretrained model's parameters are due to the MatConvNet Team. Their software comes with the license replicated below.
Copyright (c) 2014-16 The MatConvNet Team.
All rights reserved. Redistribution and use in source and binary forms are permittedprovided that the above copyright notice and this paragraph are duplicated in all such forms and that any documentation, advertising materials, and other materials related to such distribution and use acknowledge that the software was developed by the MatConvNet Team. The name of the MatConvNet Team may not be used to endorse or promote products derived from this software without specific prior written permission.  THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.---------------------------------------------------------------------
