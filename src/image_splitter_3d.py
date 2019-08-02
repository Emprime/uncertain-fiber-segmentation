import gzip
import os
import pickle
from os.path import join
from scipy.ndimage import zoom
from tqdm import tqdm
import cv2
import numpy as np

from src import utils

scan_width_height = 1024
scan_depth = 256

def image_splitting(image_directory_parent, subfolder, params=None):
    """
    split the images for given directory and sub directory, input for first stage
    :param image_directory_parent:
    :param subfolder:
    :param params:
    :return:
    """


    # constants
    block_depth = params['block_depth']
    output_parent = params['output_parent']
    width_height = 4 * block_depth
    input_directory = image_directory_parent + subfolder
    unique_folder = "#".join(subfolder.split("/")[1:]) # create unique folder name from sub folder
    output_directory = join(output_parent ,unique_folder)

    print("split images from %s to %s with block size (%d, %d, %d)" % (input_directory, output_directory, width_height, width_height, block_depth))


    file_names = sorted(os.listdir(input_directory))
    chunked_file_names = [file_names[i:i + block_depth] for i in range(0, len(file_names), block_depth)] # chunk the images for better loading

    # create output directory
    utils.check_directory(output_directory,delete_old=False)

    # calculate iteration parameters
    iteration_depth = min(scan_depth // block_depth, len(chunked_file_names)) # maybe not enough depth information
    iteration_width_height = scan_width_height // width_height


    with tqdm(total=iteration_depth * iteration_width_height * iteration_width_height) as pbar:
        for k in range(iteration_depth):

            # load images in depth slices
            depth_slice = np.zeros((scan_width_height, scan_width_height,block_depth,3)) # slice of image with block depth

            for i, file_name in enumerate(chunked_file_names[k]): # select one image chunk

                img = cv2.imread(join(input_directory, file_name), -1) # load bgr image
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # expects rgb images at file location and cast to rgb
                depth_slice[:,:,i,:] = img


            # resize images so that generated blocks have a better input size
            factor = max(1,int(128 / width_height))
            resized_depth_slice = zoom(depth_slice, zoom=[factor, factor, factor, 1], order=1)  # bilinear interpolation
            resized_width_height = width_height * factor

            # create chunks in width and height
            for i in range(iteration_width_height):
                for j in range(iteration_width_height):

                    # get chunk of data
                    chunk = resized_depth_slice[i * resized_width_height:((i + 1) * resized_width_height),
                            j * resized_width_height:((j + 1) * resized_width_height),
                            :, :]

                    # save chunk
                    chunk_name = "part_%d_%d_%d.gzipped_pickle" % (i,j,k)
                    with gzip.GzipFile(join(output_directory, chunk_name), 'wb') as f:
                        f.write(pickle.dumps(chunk))

                    # logging
                    pbar.update()


def gt_splitting(gt_directory_parent, subfolder, params=None):
    """
      create gt based on scan for rough output
      :param gt_directory_parent:
      :param subfolder:
      :param params:
      :return:
      """

    # constants
    block_depth = params['block_depth']
    output_parent = params['output_parent']
    width_height = 4 * block_depth
    input_directory = gt_directory_parent + subfolder
    unique_folder = "#".join(subfolder.split("/")[1:])  # create unique folder name from sub folder
    file_end = "_part_0_0_0.gzipped_pickle"
    out_path = join(output_parent, unique_folder + file_end)

    file_names = sorted(os.listdir(input_directory))
    chunked_file_names = [file_names[i:i + block_depth] for i in
                          range(0, len(file_names), block_depth)]  # chunk the images for better loading


    # calculate iteration parameters
    iteration_depth = min(scan_depth // block_depth, len(chunked_file_names))  # maybe not enough depth information
    iteration_depth_raw = scan_depth // block_depth  # maybe not enough depth information
    iteration_width_height = scan_width_height // width_height

    out_size = (iteration_width_height, iteration_width_height, iteration_depth_raw,3)

    print("create gt output from %s to %s with out size %s" % (
    input_directory, out_path, out_size))

    gt_matrix = np.zeros(out_size)


    # print(iteration_depth, iteration_width_height)

    with tqdm(total=iteration_depth_raw * iteration_width_height * iteration_width_height) as pbar:
        for k in range(iteration_depth_raw):

            # load images in depth slices
            depth_slice = np.zeros(
                (scan_width_height, scan_width_height, block_depth, 3))  # slice of image with block depth
            depth_slice[:,:,:,2] = 255 # default not of interest

            for i, file_name in enumerate(chunked_file_names[k] if k < iteration_depth else []):  # select one image chunk
                    img = cv2.cvtColor(cv2.imread(join(input_directory, file_name), -1),
                                       cv2.COLOR_BGR2RGB)  # expects rgb images at file location and cast to rgb
                    depth_slice[:, :, i, :] = img

            # create chunks in width and height
            for i in range(iteration_width_height):
                for j in range(iteration_width_height):
                    # get chunk of data
                    chunk = depth_slice[i * width_height:((i + 1) * width_height),
                            j * width_height:((j + 1) * width_height),
                            :, :]

                    # cast to values [0,1]
                    chunk /= 255.

                    # count entries of classes
                    sum_classes = np.sum(np.sum(np.sum(chunk, axis=0), axis=0), axis=0)

                    gt_matrix[i,j,k,np.argmax(sum_classes)] = 255

                    pbar.update()

    with gzip.GzipFile(out_path, 'wb') as f:
        f.write(pickle.dumps(gt_matrix.astype(int)))