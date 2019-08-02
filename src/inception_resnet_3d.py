# -*- coding: utf-8 -*-
"""Inception-ResNet V2 model for Keras.

Model naming and structure follows TF-slim implementation (which has some additional
layers and different number of filters from the original arXiv paper):
https://github.com/tensorflow/models/blob/master/research/slim/nets/inception_resnet_v2.py

Pre-trained ImageNet weights are also converted from TF-slim, which can be found in:
https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models

# Reference
- [Inception-v4, Inception-ResNet and the Impact of
   Residual Connections on Learning](https://arxiv.org/abs/1602.07261)

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras import backend as K, Input, Model
from keras.applications import InceptionResNetV2
from keras.layers import BatchNormalization, Activation, Conv3D, Concatenate, Lambda, MaxPooling3D, AveragePooling3D, \
	Dropout, GlobalAveragePooling3D, Dense
import numpy as np

BASE_WEIGHT_URL = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.7/'




def conv2d_bn(x,
			  filters,
			  kernel_size,
			  strides=1,
			  padding='same',
			  activation='relu',
			  use_bias=False,
			  name=None):
	"""Utility function to apply conv + BN.

	# Arguments
		x: input tensor.
		filters: filters in `Conv2D`.
		kernel_size: kernel size as in `Conv2D`.
		strides: strides in `Conv2D`.
		padding: padding mode in `Conv2D`.
		activation: activation in `Conv2D`.
		use_bias: whether to use a bias in `Conv2D`.
		name: name of the ops; will become `name + '_ac'` for the activation
			and `name + '_bn'` for the batch norm layer.

	# Returns
		Output tensor after applying `Conv2D` and `BatchNormalization`.
	"""
	x = Conv3D(filters,
			   kernel_size,
			   strides=strides,
			   padding=padding,
			   use_bias=use_bias,
			   name=name)(x)
	if not use_bias:
		bn_axis = 1 if K.image_data_format() == 'channels_first' else 4
		bn_name = None if name is None else name + '_bn'
		x = BatchNormalization(axis=bn_axis, scale=False, name=bn_name)(x)
	if activation is not None:
		ac_name = None if name is None else name + '_ac'
		x = Activation(activation, name=ac_name)(x)
	return x


def inception_resnet_block(x, scale, block_type, block_idx, activation='relu', pretrained=False):
	"""Adds a Inception-ResNet block.

	This function builds 3 types of Inception-ResNet blocks mentioned
	in the paper, controlled by the `block_type` argument (which is the
	block name used in the official TF-slim implementation):
		- Inception-ResNet-A: `block_type='block35'`
		- Inception-ResNet-B: `block_type='block17'`
		- Inception-ResNet-C: `block_type='block8'`

	# Arguments
		x: input tensor.
		scale: scaling factor to scale the residuals (i.e., the output of
			passing `x` through an inception module) before adding them
			to the shortcut branch. Let `r` be the output from the residual branch,
			the output of this block will be `x + scale * r`.
		block_type: `'block35'`, `'block17'` or `'block8'`, determines
			the network structure in the residual branch.
		block_idx: an `int` used for generating layer names. The Inception-ResNet blocks
			are repeated many times in this network. We use `block_idx` to identify
			each of the repetitions. For example, the first Inception-ResNet-A block
			will have `block_type='block35', block_idx=0`, ane the layer names will have
			a common prefix `'block35_0'`.
		activation: activation function to use at the end of the block
			(see [activations](../activations.md)).
			When `activation=None`, no activation is applied
			(i.e., "linear" activation: `a(x) = x`).

		pretrained:
			use extra asymmetric strides only in a not pretrained scenario

	# Returns
		Output tensor for the block.

	# Raises
		ValueError: if `block_type` is not one of `'block35'`,
			`'block17'` or `'block8'`.
	"""
	if block_type == 'block35':
		branch_0 = conv2d_bn(x, 32, 1)
		branch_1 = conv2d_bn(x, 32, 1)
		branch_1 = conv2d_bn(branch_1, 32, 3)
		branch_2 = conv2d_bn(x, 32, 1)
		branch_2 = conv2d_bn(branch_2, 48, 3)
		branch_2 = conv2d_bn(branch_2, 64, 3)
		branches = [branch_0, branch_1, branch_2]
	elif block_type == 'block17':
		branch_0 = conv2d_bn(x, 192, 1)
		branch_1 = conv2d_bn(x, 128, 1)
		if not pretrained:
			branch_1 = conv2d_bn(branch_1, 160, [1, 1, 7])
		branch_1 = conv2d_bn(branch_1, 160, [1, 7, 1])
		branch_1 = conv2d_bn(branch_1, 192, [7, 1, 1])
		branches = [branch_0, branch_1]
	elif block_type == 'block8':
		branch_0 = conv2d_bn(x, 192, 1)
		branch_1 = conv2d_bn(x, 192, 1)
		if not pretrained:
			branch_1 = conv2d_bn(branch_1, 224, [1, 1, 3])
		branch_1 = conv2d_bn(branch_1, 224, [1, 3, 1])
		branch_1 = conv2d_bn(branch_1, 256, [3, 1, 1])
		branches = [branch_0, branch_1]
	else:
		raise ValueError('Unknown Inception-ResNet block type. '
						 'Expects "block35", "block17" or "block8", '
						 'but got: ' + str(block_type))

	block_name = block_type + '_' + str(block_idx)
	channel_axis = 1 if K.image_data_format() == 'channels_first' else 4
	mixed = Concatenate(axis=channel_axis, name=block_name + '_mixed')(branches)
	up = conv2d_bn(mixed,
				   K.int_shape(x)[channel_axis],
				   1,
				   activation=None,
				   use_bias=True,
				   name=block_name + '_conv')

	x = Lambda(lambda inputs, scale: inputs[0] + inputs[1] * scale,
			   output_shape=K.int_shape(x)[1:],
			   arguments={'scale': scale},
			   name=block_name)([x, up])
	if activation is not None:
		x = Activation(activation, name=block_name + '_ac')(x)
	return x


def InceptionResNet3D(input_shape=None, classes=3, include_top=True, dropout=True, pretrained=False):
	"""Instantiates the Inception-ResNet v2 architecture.

	Optionally loads weights pre-trained on ImageNet.
	Note that when using TensorFlow, for best performance you should
	set `"image_data_format": "channels_last"` in your Keras config
	at `~/.keras/keras.json`.

	The model and the weights are compatible with TensorFlow, Theano and
	CNTK backends. The data format convention used by the model is
	the one specified in your Keras config file.

	Note that the default input image size for this model is 299x299, instead
	of 224x224 as in the VGG16 and ResNet models. Also, the input preprocessing
	function is different (i.e., do not use `imagenet_utils.preprocess_input()`
	with this model. Use `preprocess_input()` defined in this module instead).

	# Arguments
		include_top: whether to include the fully-connected
			layer at the top of the network.
		weights: one of `None` (random initialization),
			  'imagenet' (pre-training on ImageNet),
			  or the path to the weights file to be loaded.
		input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
			to use as image input for the model.
		input_shape: optional shape tuple, only to be specified
			if `include_top` is `False` (otherwise the input shape
			has to be `(299, 299, 3)` (with `'channels_last'` data format)
			or `(3, 299, 299)` (with `'channels_first'` data format).
			It should have exactly 3 inputs channels,
			and width and height should be no smaller than 139.
			E.g. `(150, 150, 3)` would be one valid value.
		pooling: Optional pooling mode for feature extraction
			when `include_top` is `False`.
			- `None` means that the output of the model will be
				the 4D tensor output of the last convolutional layer.
			- `'avg'` means that global average pooling
				will be applied to the output of the
				last convolutional layer, and thus
				the output of the model will be a 2D tensor.
			- `'max'` means that global max pooling will be applied.
		classes: optional number of classes to classify images
			into, only to be specified if `include_top` is `True`, and
			if no `weights` argument is specified.

	# Returns
		A Keras `Model` instance.

	# Raises
		ValueError: in case of invalid argument for `weights`,
			or invalid input shape.
	"""

	img_input = Input(shape=input_shape)

	# Stem block: 35 x 35 x 192
	x = conv2d_bn(img_input, 32, 3, strides=(2,2,1), padding='valid')
	x = conv2d_bn(x, 32, 3, padding='valid')
	x = conv2d_bn(x, 64, 3)
	x = MaxPooling3D(3, strides=(2,2,1))(x) # make last dimension decrease less up to factor 4
	x = conv2d_bn(x, 80, 1, padding='valid')
	x = conv2d_bn(x, 192, 3, padding='valid')
	x = MaxPooling3D(3, strides=2)(x)

	# Mixed 5b (Inception-A block): 35 x 35 x 320
	branch_0 = conv2d_bn(x, 96, 1)
	branch_1 = conv2d_bn(x, 48, 1)
	branch_1 = conv2d_bn(branch_1, 64, 5)
	branch_2 = conv2d_bn(x, 64, 1)
	branch_2 = conv2d_bn(branch_2, 96, 3)
	branch_2 = conv2d_bn(branch_2, 96, 3)
	branch_pool = AveragePooling3D(3, strides=1, padding='same')(x)
	branch_pool = conv2d_bn(branch_pool, 64, 1)
	branches = [branch_0, branch_1, branch_2, branch_pool]
	channel_axis = 1 if K.image_data_format() == 'channels_first' else 4
	x = Concatenate(axis=channel_axis, name='mixed_5b')(branches)

	# 10x block35 (Inception-ResNet-A block): 35 x 35 x 320
	for block_idx in range(1, 11):
		x = inception_resnet_block(x,
								   scale=0.17,
								   block_type='block35',
								   block_idx=block_idx,
								   pretrained=pretrained)

	# Mixed 6a (Reduction-A block): 17 x 17 x 1088
	branch_0 = conv2d_bn(x, 384, 3, strides=2, padding='valid')
	branch_1 = conv2d_bn(x, 256, 1)
	branch_1 = conv2d_bn(branch_1, 256, 3)
	branch_1 = conv2d_bn(branch_1, 384, 3, strides=2, padding='valid')
	branch_pool = MaxPooling3D(3, strides=2, padding='valid')(x)
	branches = [branch_0, branch_1, branch_pool]
	x = Concatenate(axis=channel_axis, name='mixed_6a')(branches)

	# 20x block17 (Inception-ResNet-B block): 17 x 17 x 1088
	for block_idx in range(1, 21):
		x = inception_resnet_block(x,
								   scale=0.1,
								   block_type='block17',
								   block_idx=block_idx,
								   pretrained=pretrained)

	# Mixed 7a (Reduction-B block): 8 x 8 x 2080
	branch_0 = conv2d_bn(x, 256, 1)
	branch_0 = conv2d_bn(branch_0, 384, 3, strides=2, padding='valid')
	branch_1 = conv2d_bn(x, 256, 1)
	branch_1 = conv2d_bn(branch_1, 288, 3, strides=2, padding='valid')
	branch_2 = conv2d_bn(x, 256, 1)
	branch_2 = conv2d_bn(branch_2, 288, 3)
	branch_2 = conv2d_bn(branch_2, 320, 3, strides=2, padding='valid')
	branch_pool = MaxPooling3D(3, strides=2, padding='valid')(x)
	branches = [branch_0, branch_1, branch_2, branch_pool]
	x = Concatenate(axis=channel_axis, name='mixed_7a')(branches)

	# 10x block8 (Inception-ResNet-C block): 8 x 8 x 2080
	for block_idx in range(1, 10):
		x = inception_resnet_block(x,
								   scale=0.2,
								   block_type='block8',
								   block_idx=block_idx,
								   pretrained=pretrained)
	x = inception_resnet_block(x,
							   scale=1.,
							   activation=None,
							   block_type='block8',
							   block_idx=10,
							   pretrained=pretrained)

	# Final convolution block: 8 x 8 x 1536
	x = conv2d_bn(x, 1536, 1, name='conv_7b')


	if include_top:
		x = GlobalAveragePooling3D()(x)
		if dropout:
			x = Dropout(0.6)(x)
		out = Dense(classes, activation='softmax')(x)
	else:
		out = GlobalAveragePooling3D()(x)
	model = Model(inputs=img_input, outputs=out)

	if pretrained:

		# this is the pretraining is based on the averaging of weights over new layers
		model2D = InceptionResNetV2(include_top=False,weights='imagenet') # pretrained back bone
		model3D = model

		weights2DList = model2D.get_weights()
		weights3DList = model3D.get_weights()

		list = []
		for i, weight2D in enumerate(weights2DList):
			weight3D = weights3DList[i]
			weight2D3D = np.zeros(weight3D.shape)
			if weight2D.ndim == 4:
				# expand to 5 dimensions
				expand_dim = weight3D.shape[2]
				for j in range(expand_dim):
					weight2D3D[:, :, j, :, :] = (weight2D / expand_dim) # average over number of inserts to get equal results as in 2d
					# mind that 1-5 layers are than averaged into one -> might not work
			else:
				weight2D3D = weight2D

			list.append(weight2D3D)

		model3D.set_weights(list)

		# this is the pretraining where weights are just inserted, we didn't check if this code still works
		# just for your knowledge

		# model3D = model
		#
		# weights2DList = model2D.get_weights()
		# weights3DList = model3D.get_weights()
		#
		# list = []
		# # not identical to 2d prediction due to batch normalization
		# for i, weight3D in enumerate(weights3DList):
		# 	if i < len(weights2DList):
		# 		weight2D = weights2DList[i]
		#
		# 		weight2D3D = np.zeros(weight3D.shape)
		# 		if weight2D.ndim == 4:
		# 			# expand to 5 dimensions
		# 			expand_dim = weight3D.shape[2] // 2
		# 			# print(weight3D.shape[2])
		# 			weight2D3D[:, :, expand_dim, :, :] = weight2D
		# 		else:
		# 			weight2D3D = weight2D
		#
		#
		# 	else:
		# 		# print("layer %d: " % (i))
		# 		# print(weight3D.shape)
		#
		# 		if weight3D.ndim == 2:
		# 			# weights
		# 			weight2D3D = np.zeros(weight3D.shape)
		# 			# order classes in flatten: 0 1 2 0 1 2 0 1 2
		# 			depth = 32  # hard coded but is always 32 (input size)
		# 			for j in range(3):
		# 				nbs = [((3 * k) + j) for k in range(depth)]
		# 				# print(nbs)
		# 				weight2D3D[nbs, j] = (1 / depth)
		# 		else:
		# 			# bias
		# 			weight2D3D = np.zeros(weight3D.shape)
		#
		# 	list.append(weight2D3D)
		#
		# model3D.set_weights(list)




	return model
