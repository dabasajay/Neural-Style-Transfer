## Neural Style Transfer

[![Issues](https://img.shields.io/github/issues/dabasajay/Neural-Style-Transfer.svg?color=%231155cc)](https://github.com/dabasajay/Image-Caption-Generator/issues)
[![Forks](https://img.shields.io/github/forks/dabasajay/Neural-Style-Transfer.svg?color=%231155cc)](https://github.com/dabasajay/Image-Caption-Generator/network)
[![Stars](https://img.shields.io/github/stars/dabasajay/Neural-Style-Transfer.svg?color=%231155cc)](https://github.com/dabasajay/Image-Caption-Generator/stargazers)
[![Ajay Dabas](https://img.shields.io/badge/Ajay-Dabas-ff0000.svg)](https://dabasajay.github.io/)

<h4 align="center">Example Image</h4>
<p align="center">
  <img src="example image.png?raw=true" width="85%" title="Example Image" alt="Example Image">
</p>

## Intro
In this repo, I implemented Neural Style Transfer Algorithm. This algorithm was created by Gatys et al. (2015) (<a href='http://arxiv.org/abs/1508.06576'>Link to paper</a>). This Deep Learning technique takes two images, namely, the content image(C) and a style image(S) and generates a new image(G) which combines the content of image C with style of image S.

I used a pre-trained VGG-19 model which is a very deep convolutional neural network because these deep pre-trained models can detect low-level features such as edges and vertex at earlier layers and high-level features at deep layers, pretty accurately.

## Try Yourself

1. Download and put VGG-19 weights (<a href="http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat">Link ~ 510MB</a>) in vgg_model folder so path looks like `vgg_model/VGG-19.mat`
2. Put style and content image in images folder. <br>
Example: `images/mystyleimage.jpeg` & `images/mycontentimage.jpeg` <br>
**NOTE**: Images must be of dimensions 400x300
3. Review `config.py` for paths and other configurations (explained below)
4. Run `nst_script.py`

## Requirements

Recommended System Requirements to run model.

<ul type="square">
	<li>A good CPU and a GPU with atleast 8GB memory</li>
	<li>Atleast 8GB of RAM</li>
</ul>

Required Libraries for Python along with their version numbers used while making & testing of this project

<ul type="square">
	<li>Python - 3.6.7</li>
	<li>Numpy - 1.16.2</li>
	<li>Tensorflow - 1.13.1</li>
	<li>imageio - 2.4.1</li>
	<li>scipy - 1.2.1</li>
	<li>Matplotlib - 3.0.3</li>
</ul>

## Pre-trained model license

Here, we are using a pretrained model that can be downloaded at the following link: http://www.vlfeat.org/matconvnet/pretrained/. The pretrained model's parameters are due to the MatConvNet Team. Their software comes with the license replicated below.
Copyright (c) 2014-16 The MatConvNet Team.
All rights reserved. Redistribution and use in source and binary forms are permittedprovided that the above copyright notice and this paragraph are duplicated in all such forms and that any documentation, advertising materials, and other materials related to such distribution and use acknowledge that the software was developed by the MatConvNet Team. The name of the MatConvNet Team may not be used to endorse or promote products derived from this software without specific prior written permission.  THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
