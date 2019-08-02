import numpy
from keras.models import Model
from keras.layers import Conv3D, AveragePooling3D
from keras.layers import Dropout, Input
from keras import backend as K, Model
from keras.optimizers import SGD
import tensorflow as tf
from keras.backend.tensorflow_backend import _to_tensor, epsilon

from src.inception_resnet_3d import InceptionResNet3D

# ----------------- LOSS ------------------ #

def weighted_focal_crossentropy(gamma=0.0, weights=(100 * numpy.array([0.445,0.545,0.01]))):
	"""
	weighted categorical crossentropy with focal loss parameter
	:param gamma: focal loss parameter gamma
	:param weights: weights for classes, based on RGB in gt, (dissimilar, similiar ,not of interest)
	:return:
	"""
	def internal(target, output, axis=-1):
		"""
		weighted categorical crossentropy with focal loss parameter
		:param target: gt
		:param output: prediction
		:param axis: axis to check
		:return:
		"""
		# scale preds so that the class probas of each sample sum to 1 -> already softmax
		output /= tf.reduce_sum(output, axis, True)
		# manual computation of crossentropy
		_epsilon = _to_tensor(epsilon(), output.dtype.base_dtype)
		output = tf.clip_by_value(output, _epsilon, 1. - _epsilon)
		# up to here copied from the keras files

		# focal part
		focal = tf.pow((1 - output),gamma)

		return - tf.reduce_sum(target * weights * focal * tf.log(output), axis)
	return internal

# accuracies for one class
def acc0(y_true,y_pred):
	return single_class_accuracy(y_true,y_pred,0)

def acc1(y_true,y_pred):
	return single_class_accuracy(y_true,y_pred,1)

def acc2(y_true,y_pred):
	return single_class_accuracy(y_true,y_pred,2)

def single_class_accuracy(y_true, y_pred,class_id):
	"""
	calculates the accuracy for only one given class
	:param y_true:
	:param y_pred:
	:param class_id: selected class
	:return:
	"""
	class_id_true = K.argmax(y_true, axis=-1)
	class_id_preds = K.argmax(y_pred, axis=-1)
	# Replace class_id_preds with class_id_true for recall here
	accuracy_mask = K.cast(K.equal(class_id_true, class_id), 'int32')
	class_acc_tensor = K.cast(K.equal(class_id_true, class_id_preds), 'int32') * accuracy_mask
	class_acc = K.sum(class_acc_tensor) / K.maximum(K.sum(accuracy_mask), 1)
	return class_acc

def mean_acc(y_true, y_pred):
	"""
	calculate the mean accuracy over the classes
	:param y_true:
	:param y_pred:
	:return:
	"""
	return K.mean(K.stack([acc0(y_true,y_pred), acc1(y_true, y_pred), acc2(y_true,y_pred)]))

# -------------------------------- MODEL ----------------------------------- #

def first_stage_model( input_shape):
	"""
	get first stage for the two stage model
	:param input_shape: input shape
	:return: model for feature extraction
	"""
	return InceptionResNet3D(input_shape=input_shape, include_top=False, pretrained=True)

def second_stage_model(out_size, loss):
	"""
	second stage of the two stage model
	:param out_size: output size
	:param loss: loss function
	:return:
	"""
	features = 1536
	num_classes = 3
	input_shape = (out_size,out_size,out_size,features)
	input = Input(shape=input_shape)
	x = Conv3D(1024, kernel_size=3, strides=1, padding='same')(input)
	x = Conv3D(512, kernel_size=3, strides=1, padding='same')(x)
	x = Conv3D(256, kernel_size=3, strides=1, padding='same')(x)
	x = AveragePooling3D(pool_size=3, strides=1, padding='same')(x)
	x = Dropout(0.6)(x)
	out = Conv3D(filters=num_classes, kernel_size=1, strides=1, activation='softmax')(x)

	model = Model(inputs=input, outputs=out)
	model.compile(SGD(lr=0.01, decay=0.000005), loss=loss, metrics=['acc',acc0,acc1,acc2, mean_acc])

	return model

