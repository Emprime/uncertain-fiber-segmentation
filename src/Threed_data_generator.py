import os
import gzip
import random
import keras
import numpy as np
import pickle


# augmentations
transformations = {'rot-0':"",'rot-90':"tv",'rot-180':"vh",'rot-270':"th",
				   'sp-rot-0': "v", 'sp-rot-90': "tvh", 'sp-rot-180': "h", 'sp-rot-270': "t",}
reversed = [False,True]

def flip_hor(block):
	return np.flip(block,axis=0)

def flip_vert(block):
	return np.flip(block, axis=1)

def reverse(block):
	return np.flip(block, axis=2)

def transp(block):
	return np.transpose(block,axes=(1,0,2,3))

def transform(codes, block):
	for code in codes:
		if code == 'h':
			func = flip_hor
		elif code == 'v':
			func = flip_vert
		elif code == 'r':
			func = reverse
		elif code == 't':
			func = transp
		else:
			func = None

		block = func(block)
	return block

class CombinedGenerator(keras.utils.Sequence):
	'Generates data for Keras'

	def __init__(self, gen1, gen2):
		self.gen1 = gen1
		self.gen2 = gen2

	def __len__(self):
		'Denotes the number of batches per epoch'
		return self.gen1.__len__()


	def __getitem__(self, index):
		return self.gen1.__getitem__(index), self.gen2.__getitem__(index)


	def on_epoch_end(self):
		'Updates indexes after each epoch'
		self.gen1.on_epoch_end()
		self.gen2.on_epoch_end()

class Generator3D(keras.utils.Sequence):
	'Generates data for Keras'
	def __init__(self, directory, batch_size=32, data_shape=(128, 128, 32), shuffle=True, seed=42, augment=True):
		'Initialization'
		self.data_shape = data_shape
		self.batch_size = batch_size
		self.directory = directory
		self.file_names = os.listdir(directory)
		self.shuffle = shuffle
		self.random = random.Random()
		self.random.seed(seed)
		self.augment = augment
		self.gzip = gzip # describes if the input is zipped
		self.on_epoch_end()

		print("GENERATOR: found %d data sets in %s" % (len(self.file_names),directory))
		# print(data_shape)


	def __len__(self):
		'Denotes the number of batches per epoch'
		return len(self.file_names) // self.batch_size

	def choice(self):
		rev_choice = self.random.choice(reversed)
		transform_choice = self.random.choice(list(transformations.keys()))

		name_appendix = "-rev" if rev_choice else ""
		codes_appendix = "r" if rev_choice else ""
		return transform_choice + name_appendix,transformations[transform_choice] + codes_appendix

	def __getitem__(self, index):
		'Generate one batch of data'
		# Generate indexes of the batch
		batch_file_names = self.file_names[index*self.batch_size:(index+1)*self.batch_size]

		X = np.zeros((self.batch_size, *self.data_shape,))

		for i, file_name in enumerate(batch_file_names):

			with gzip.GzipFile(os.path.join(self.directory , file_name), 'rb') as f:
				loaded = np.array(pickle.load(f)) / 255. # data are ints -> convert to range [0,1]

				if self.augment:
					name,codes = self.choice()
					loaded = transform(codes,loaded)

				X[i,:,:,:,:] = loaded


		return X

	def on_epoch_end(self):
		'Updates indexes after each epoch'
		if self.shuffle == True:
			self.random.shuffle(self.file_names)
