import gzip
import os
import pickle
import numpy as np
from tqdm import tqdm

default_size = 1024
default_depth = 256
num_features = 1536
factor_depth2width = 4


def combine_features(input_directory, output_directory, depth, ):
	"""
	combine features of a gap layer to a new data block
	:param input_directory: directory of the features
	:param output_directory: directory of the combined feature
	:param depth: depth of the block from the features
	:return:
	"""
	widthHeight = depth * factor_depth2width

	# iterate over all files
	file_names = sorted(os.listdir(input_directory))
	length = len(file_names)
	# generate map for all files
	map = {}
	for i in range(0, length):
		# get identifier
		ident = file_names[i].split("part_")[0]

		# check map
		if ident not in map:
			map[ident] = []

		# add file names
		map[ident].append(file_names[i])

	# iterate over all idents
	print("combine features")
	for i, ident in enumerate(map):
		# print("ident: %s, length: %d" % (ident,len(map[ident])))

		x_y_dim = default_size // widthHeight

		matrix = np.zeros((x_y_dim,x_y_dim,x_y_dim,num_features)) # dont use zdim as dimension for same input size
		for i, file_name in enumerate(map[ident]):
			with gzip.GzipFile(os.path.join(input_directory ,file_name), 'rb') as f:
				matrix_part = pickle.load(f)

			pos = file_name.split("part_")[1].split(".gzipped_pickle")[0].split("_") # structure part_<num>_<num>_<num>.gzipped_pickle
			matrix[int(pos[0]),int(pos[1]),int(pos[2]),:] = matrix_part

		# save combined features
		with gzip.GzipFile(output_directory + ident + "_part_0_0_0.gzipped_pickle", 'wb') as f:
			pickle.dump(matrix, f)

