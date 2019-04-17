<h1>Neural Style Transfer</h1>
<h4 align="center">Example Image</h4>
<p align="center">
  <img src="example image.png?raw=true" width="85%" title="Example Image" alt="Example Image">
</p>
<h2>Intro</h2>
In this repo, I implemented Neural Style Transfer Algorithm. This algorithm was created by Gatys et al. (2015) (<a href='http://arxiv.org/abs/1508.06576'>Link to paper</a>). This Deep Learning technique takes two images, namely, the content image(C) and a style image(S) and generates a new image(G) which combines the content of image C with style of image S.

I used a pre-trained VGG-19 model which is a very deep convolutional neural network because these deep pre-trained models can detect low-level features such as edges and vertex at earlier layers and high-level features at deep layers, pretty accurately.

<h2>Try Yourself</h2>
<ol>
	<li>Download and put VGG-19 weights (<a href="http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat">Link ~ 510MB</a>) in vgg_model folder so path looks like 'vgg_model/VGG-19.mat'</li>
	<li>Put style and content image in images folder.
	Ex: 'images/mystyleimage.jpeg' & 'images/mycontentimage.jpeg'</li>
	<strong>NOTE: Images must be of dimensions 400x300</strong>
	<li>Review config.py for paths</li>
	<li>Run nst_script.py</li>
</ol>

<h2>Pre-trained model license</h2>
Here, we are using a pretrained model that can be downloaded at the following link: http://www.vlfeat.org/matconvnet/pretrained/. The pretrained model's parameters are due to the MatConvNet Team. Their software comes with the license replicated below.
Copyright (c) 2014-16 The MatConvNet Team.
All rights reserved. Redistribution and use in source and binary forms are permittedprovided that the above copyright notice and this paragraph are duplicated in all such forms and that any documentation, advertising materials, and other materials related to such distribution and use acknowledge that the software was developed by the MatConvNet Team. The name of the MatConvNet Team may not be used to endorse or promote products derived from this software without specific prior written permission.  THIS SOFTWARE IS PROVIDED ``AS IS'' AND WITHOUT ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, WITHOUT LIMITATION, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
