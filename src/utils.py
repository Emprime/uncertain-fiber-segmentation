from shutil import rmtree
import sys
import numpy as np
import os
import pickle
import time

from keras import backend as K

weights_cross = np.array([1,5,5])


def call_recursive(folder_path, path, function,params=None):
	"""
	call function on all subpath without anymore folders
	:param folder_path: main path
	:param path: sub path
	:param function: function to be called
	:param params: optional parameters
	:return:
	"""
	try:
		subdirs = next(os.walk(folder_path + "/" + path))[1]
		if len(subdirs) == 0:
			function(folder_path, path,params=params)
		else:
			subdirs =  sorted(subdirs)
			for subdir in subdirs:
				call_recursive(folder_path, path + "/" + subdir, function,params=params)
	except StopIteration:
		function(folder_path, path,params=params)



def get_nb(exact_number, batch_size):
	"""
	get rounded number for exact number and batchsize
	:param exact_number:
	:param batch_size:
	:return:
	"""
	number = ((int)(exact_number / batch_size)) * batch_size

	if exact_number - number > 0:
		print(("######################################################\n" +
			   "WARNING: you specified %d items, but with a batch size of %d you will only look at %d items\n"
			   + "######################################################") % (exact_number, batch_size, number))

	return number


def check_directory(file_path, delete_old=True):
	"""
	check if directory exists and create otherwise
	:param file_path: path
	:param delete_old: bool which specifies if an old directory should be deleted
	:return:
	"""
	# check if exists
	# if true and delete old delete and create
	# if false create

	delete = os.path.exists(file_path) and delete_old
	create = not os.path.exists(file_path) or delete

	if delete:
		rmtree(file_path)
	if create:
		os.makedirs(file_path)
	return create



def clear_session():
	"""
	clear tensorflow session between uses
	:return:
	"""
	# use this for multiple use of networks
	# for some reason destroys good networks sometimes
	K.clear_session()