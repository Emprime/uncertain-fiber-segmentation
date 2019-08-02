import gzip
import os
import pickle
from tqdm import tqdm
from src import utils
from src.Threed_data_generator import Generator3D, CombinedGenerator

def predict(model, shape, input_directory, output_directory, features=False, batch_size=1):
	"""
	prediction for 3D data
	:param model: model used for for prediction
	:param shape: input shape the model
	:param input_directory: path to the images
	:param output_directory: save path to prediction
	:param features: output features instead of prediction
	:param batch_size: batch size
	:return:
	"""
	print("predict for %s" % (input_directory))

	image_generator = Generator3D(input_directory, batch_size=batch_size, data_shape=shape, augment=False, shuffle=False)

	# define images from data generators
	nb_samples = utils.get_nb(len(image_generator.file_names), batch_size)
	number_iterations = nb_samples // batch_size

	for i in tqdm(range(0, number_iterations)):

		image_batch = image_generator.__getitem__(i)

		preds = model.predict_on_batch(image_batch)

		# save predictions
		for j in range(0, batch_size):
			file_name = image_generator.file_names[i * batch_size + j]

			with gzip.GzipFile(os.path.join(output_directory, file_name), 'wb') as f:
				if features:
					# print(preds[j, :].shape)
					pickle.dump(preds[j, :], f)
				else:
					pickle.dump((preds[j, :, :, :, :] * 255).astype(int), f)



def train(name, model, shape, output_shape, epochs, batch_size,
		  train_image_path, train_mask_path, val_image_path, val_mask_path, callbacks, augment=True, ):
	print("train model for " + name)

	# define images from data generators
	shuffle = True  # shuffle not the validation
	train_image_generator = Generator3D(train_image_path, batch_size=batch_size, data_shape=shape, augment=augment,
										shuffle=shuffle)
	train_mask_generator = Generator3D(train_mask_path, batch_size=batch_size, data_shape=output_shape, augment=augment,
									   shuffle=shuffle,)
	shuffle = False  # shuffle not the validation
	val_image_generator = Generator3D(val_image_path, batch_size=batch_size, data_shape=shape, shuffle=shuffle, augment=False)
	val_mask_generator = Generator3D(val_mask_path, batch_size=batch_size, data_shape=output_shape, shuffle=shuffle,
									 augment=False,)

	# combine generators for mask and image
	train_generator = CombinedGenerator(train_image_generator, train_mask_generator)
	validation_generator = CombinedGenerator(val_image_generator, val_mask_generator)

	model.fit_generator(
		train_generator,
		epochs=epochs,
		validation_data=validation_generator,
		callbacks=callbacks
	)


