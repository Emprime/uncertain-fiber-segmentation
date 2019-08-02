import glob
import random
import shutil

import matplotlib
import time
import pickle
from keras.callbacks import TensorBoard, ModelCheckpoint

import src.models_loss
import src.utils
from src  import models_loss
from src  import feature_combiner_3d
from src import training_3d
from src import utils
import os
from tqdm import tqdm
import gzip
from os.path import join
import numpy as np

import matplotlib.pyplot as plt

import argparse

from src.image_splitter_3d import image_splitting, gt_splitting


def parse_arguments():
    """
    parse the arguments for the cross validation
    :return:
    """
    parser = argparse.ArgumentParser()

    # add paths
    parser.add_argument("--data_path", help="input path images, used for generating the input for the first stage",type=str,default="/data1/data/shg-ce-de")
    parser.add_argument("--gt_path", help="input path ground truth images, used for generating the ground truth for the second stage", type=str,default="/data1/data/shg-masks")
    parser.add_argument("--feature_path", help="out put path generated features, input for second stage", type=str,default="/data1/data/shg-features")
    parser.add_argument("--splits_path", help="path for the cross validation split directories", type=str,default="/data1/data/shg-cross-splits")
    parser.add_argument("--log_dir", help="directory for log entries for tensorboard and predictions", type=str,default="/data1/data/logs-predictions")


    # cross validation parameters
    parser.add_argument("--experiment_name", help="unique identifier for different trainings or reruns", type=str, default="default" )
    parser.add_argument("--num_splits", help="do num_splits times a cross validation", type=int, default=10 )
    parser.add_argument("--target_split_percentage",type=float, default=0.02,
                        help="percentage of each class that is ensured in a split, mind that high values aren't possible", )
    parser.add_argument("--first_stage", action="store_true", default=False,
                        help="tell the system to regenerate the splits and features, this might take some time, first stage",)
    parser.add_argument("--second_stage", action="store_true", default=False,
                        help="tell the system to retrain the networks to the given splits and predict, this might take some time, second stage",)
    parser.add_argument("--batch_size", type=int, default=2,
                        help="batch size for prediction in first stage and training in second stage", )
    parser.add_argument("--epochs", type=int, default=1000,
                        help="number of epochs the second stage will train if no early stopping occurs", )
    parser.add_argument("--block_depth", type=int, default=16,
                        help="depth of input blocks into the system, this defines also the width and height (4 x depth)", )


    config = parser.parse_args()
    return config

"""
cross validate the method two stage segmentation of the paper https://arxiv.org/abs/1907.12868
"""
class cross_validate:

    def __init__(self, config):
        print("CONFIG: ", config)
        # print(config.data_path)

        # store the different paths
        self.data_path = config.data_path
        self.gt_path = config.gt_path
        self.feature_path = config.feature_path
        self.splits_path = config.splits_path
        self.log_path = config.log_dir

        # experiment parameters
        self.name = config.experiment_name

        # define default parameters
        self.block_depth = config.block_depth # this defines the block depth, the algorithm expects a block of (4 * depth, 4* depth, depth)
        self.epochs = config.epochs
        self.batch_size = config.batch_size

        # input directories for second stage
        self.features_gt = join(self.feature_path, "gt")
        self.features_data = join(self.feature_path, "comb_feature")

    def first_stage(self, create_input_first_stage=True, create_output_first_stage=True, create_input_second_stage=True, create_gt_second_stage=True, delete_temporary=True):
        """
       run the first stage of the two stage segmentation
       :param create_input_first_stage: debug argument to start or not start this part of the first stage, should be True
       :param create_output_first_stage: debug argument to start or not start this part of the first stage, should be True
       :param create_input_second_stage: debug argument to start or not start this part of the first stage, should be True
       :param create_gt_second_stage: debug argument to start or not start this part of the first stage, should be True
       :return:
        """

        # This method splits into three sub tasks
        # 1. create image input blocks for first stage
        # 2. create features from input blocks (output first stage)
        # 3. combine features to 3D input for second stage (input second stage)
        # 4. create gt ouput for second stage

        # constants
        img_blocks_path = join(self.feature_path, "img-blocks") # input first stage
        features_path = join(self.feature_path , "features") # output first stage

        depth = self.block_depth  # depth of image blocks

        if create_input_first_stage:
            print("create img blocks as input for first stage")
            utils.call_recursive(self.data_path, "", image_splitting, params={'output_parent': img_blocks_path, 'block_depth': self.block_depth})

        # create features
        if create_output_first_stage:
            print("create features as output for first stage")
            batch_size = self.batch_size
            input_shape = (32 * 4, 32 * 4, 32, 3) # larger due to resizing in image splitting (manual set necessary)
            model = src.models_loss.first_stage_model(input_shape)

            scans = os.listdir(img_blocks_path)
            for scan in scans:
                utils.check_directory(join(features_path, scan))
                training_3d.predict(model, input_shape, join(img_blocks_path, scan), join(features_path, scan),
                                    features=True, batch_size=batch_size)

        # combine features
        if create_input_second_stage:
            print("combine features as input for second stage")
            utils.check_directory(self.features_data)
            for scan in tqdm(os.listdir(features_path)):
                feature_combiner_3d.combine_features(join(features_path, scan), join(self.features_data, scan),
                                                     depth=depth)
        # gt data
        if create_gt_second_stage:

            print("create gt data for second stage")
            # create output directory
            utils.check_directory(self.features_gt)

            utils.call_recursive(self.gt_path, "", gt_splitting,
                                 params={'output_parent': self.features_gt, 'block_depth': self.block_depth})

        # delete temporary files to save disk space
        if delete_temporary:
            print("delete temporary")
            shutil.rmtree(img_blocks_path)
            shutil.rmtree(features_path)


    def generate_splits(self, number_splits, target_percent, recreate_splits=True, verbose=False):
        """

        split the features and gt information into second_stage, val and test data multiple times
        :param number_splits: number of desired splits
        :param target_percent: percentage that is required of every class in every split
        :param recreate_splits: debug recreate items, should be True
        :param verbose: debug parameter for verbose output
        :return:
        """

        if recreate_splits:

            # show statistics on the data


            # calculate all statistics and store them in a map
            # map for statistics, structure: { mouse: {leg: {scan: [stats]} } }
            map = {}

            for file in sorted(os.listdir(self.features_data)):
                # print("Statistics for %s" % file)

                # load data
                # with gzip.GzipFile(join(self.features_data, file), 'rb') as f:
                #     data = pickle.load(f)
                with gzip.GzipFile(join(self.features_gt, file), 'rb') as f:
                    gt = pickle.load(f)

                countG = np.sum(gt[:, :, :, 0] == 255)
                countUG = np.sum(gt[:, :, :, 1] == 255)
                countN = np.sum(gt[:, :, :, 2] == 255)

                # create identifier for map
                split_file = file.split("#")  # [mouse, leg, scan]

                # add entries to map
                temp_map = map
                for i in range(3):
                    if not split_file[i] in temp_map:
                        temp_map[split_file[i]] = {}  # create map

                    if i < 2:
                        temp_map = temp_map[split_file[i]]

                temp_map[split_file[2]] = [countG, countUG, countN]

                if verbose:
                    print("Distribution of classes: G: %d (%0.02f) UG: %d (%0.02f) N: %d (%0.02f)" % (countG, countG/4096, countUG, countUG/4096, countN, countN/4096))

            stats_number = 3
            spacing = 10

            # calculate possible splits and visualize map
            stas = np.zeros((stats_number))
            possible_splits = []
            for mouse in map:
                map_mouse = map[mouse]
                mouse_stats = np.zeros((stats_number))
                print(" " * (0 * spacing), mouse)
                for leg in map_mouse:
                    map_scan = map_mouse[leg]
                    leg_stats = np.zeros((stats_number))
                    print(" " * (1 * spacing), leg)
                    for scan in map_scan:
                        data = map_scan[scan]
                        # print and add data
                        print(" " * (2 * spacing), scan, "G: %d (%0.02f) UG: %d (%0.02f) N: %d (%0.02f)" % (
                        data[0], data[0] / np.sum(data), data[1], data[1] / np.sum(data), data[2],
                        data[2] / np.sum(data)))
                        leg_stats += data

                    print(" " * (2 * spacing), "Sum", "G: %d (%0.02f) UG: %d (%0.02f) N: %d (%0.02f)" % (
                    leg_stats[0], leg_stats[0] / np.sum(leg_stats), leg_stats[1], leg_stats[1] / np.sum(leg_stats),
                    leg_stats[2], leg_stats[2] / np.sum(leg_stats)))
                    mouse_stats += leg_stats

                    # save possilbe split
                    possible_splits.append([mouse, leg, leg_stats[0], leg_stats[1], leg_stats[2]])

                print(" " * (1 * spacing), "sum", "G: %d (%0.02f) UG: %d (%0.02f) N: %d (%0.02f)" % (
                mouse_stats[0], mouse_stats[0] / np.sum(mouse_stats), mouse_stats[1],
                mouse_stats[1] / np.sum(mouse_stats), mouse_stats[2], mouse_stats[2] / np.sum(mouse_stats)))
                stas += mouse_stats
            print(" " * (0 * spacing), "sum", "G: %d (%0.02f) UG: %d (%0.02f) N: %d (%0.02f)" % (
            stas[0], stas[0] / np.sum(stas), stas[1], stas[1] / np.sum(stas), stas[2], stas[2] / np.sum(stas)))

            # split the data into training (50%) , validation (25%) and test (25%)
            # mind that each dataset has at least target percent of each target type
            print("generating splits")
            count_splits = 0
            splits = []  # structure: [[mouse, leg], ..]
            num_pos_splits = len(possible_splits)  # 22
            while count_splits < number_splits: # WARNING: Possible endless loop

                ## generate a split
                # create a list of assingments [0,0,0, ...1, 1, ... ,2,2 ...]
                num_train = (num_pos_splits // 2)
                num_val = (num_pos_splits // 4)
                num_test = num_pos_splits - num_train - num_val
                indices = ([0] * num_train) + ([1] * num_val) + ([2] * num_test)

                # shuffle them
                random.shuffle(indices)

                # calculate the splits and the class distributions
                temp_split = [[], [], []]
                temp_distribution = np.zeros((3, 3))  # shape: second_stage, val , test x classes
                for i, ind in enumerate(indices):
                    temp_split[ind].append(possible_splits[i])
                    temp_distribution[ind, :] += possible_splits[i][2:5]

                # check if split meets criteria
                min_distribution_percent = np.min(temp_distribution / np.sum(temp_distribution, axis=1, keepdims=True))
                if min_distribution_percent > target_percent:
                    splits.append(temp_split)
                    print(min_distribution_percent,
                          temp_distribution / np.sum(temp_distribution, axis=1, keepdims=True))
                    count_splits += 1

            # create directories with the split structure
            print("copy splits")
            for i in tqdm(range(number_splits)):
                split_folder = "split%02d" % i
                split = splits[i]

                for img_gt in ["data","gt"]:

                    # create directories
                    for j, dir in enumerate(["training","validation","test"]):
                        folder_location = join(self.splits_path,split_folder,img_gt,dir)
                        utils.check_directory(folder_location)

                        # copy files
                        for mouse_and_leg in split[j]:
                            files = glob.glob(join(self.features_data if img_gt == "data" else self.features_gt,"%s#%s#*" % (mouse_and_leg[0],mouse_and_leg[1])))

                            for file in files:
                                shutil.copy(file,join(folder_location,file.split("/")[-1])) # copy file to location with file after last /

    def second_stage(self, split_number):
        """
        second stage of two stage segmentation -> train and predict for given split number
        :param split_number:
        :return:
        """

        print("Train for split %d" % split_number)

        # pathes
        split_path = join(self.splits_path, "split%02d" % split_number)

        train_image_path = join(split_path, "data", "training")
        train_mask_path = join(split_path, "gt", "training")
        val_image_path = join(split_path, "data", "validation")
        val_mask_path = join(split_path, "gt", "validation")
        test_imgage_path = join(split_path, "data", "test")
        test_mask_path = join(split_path, "gt", "test")

        loss = models_loss.weighted_focal_crossentropy()

        # paramter
        epochs = self.epochs
        batch_size = self.batch_size
        in_out_size = 16
        in_shape = (in_out_size, in_out_size, in_out_size, 1536)
        out_shape = (in_out_size, in_out_size, in_out_size, 3)
        model = models_loss.second_stage_model(in_out_size, loss)

        # define logging
        log_dir = join(self.log_path, self.name ,"run%02d-TIME-%s" % (split_number, time.strftime("%d-%m-%Y-%H-%M-%S")))
        utils.check_directory(log_dir, delete_old=False)
        callbacks_array = [TensorBoard(log_dir=log_dir, batch_size=batch_size),
                           ModelCheckpoint(join(log_dir, "weights.h5"), monitor='val_loss',
                                           save_best_only=True, save_weights_only=True, verbose=1), ]

        name = "run%02d" % split_number
        training_3d.train(name, model, in_shape, out_shape, epochs, batch_size,
                       train_image_path, train_mask_path, val_image_path, val_mask_path, augment=True,
                       callbacks=callbacks_array)

        # evaluate the model
        print("Predict for split %d" % split_number)
        for img_path, gt_path in zip([train_image_path, val_image_path, test_imgage_path],
                                     [train_mask_path, val_mask_path, test_mask_path]):
            training_3d.predict(model, in_shape, img_path, log_dir, features=False, batch_size=1)

        src.utils.clear_session()

    def acc(self,diff_matrix, index):
        """
        get the accuracy for one class and handle no entries as a perfect prediction
        :param diff_matrix: diffusion matrix
        :param index: index of the class in the matrix
        :return:
        """
        count = diff_matrix[index,index]
        sum = np.sum(diff_matrix[:,index])

        return count / sum if sum > 0 else 1


    def calc_stats(self, list, verbose=False):
        """
        calculate two different versions of mean accuracy
        :param list: list of diffusion matrices for every scan
        :param verbose: print optional infos
        :return:
        """
        num_classes = 3
        diff_matrix = np.zeros((num_classes, num_classes))
        for entry in list:
            diff_matrix += entry

        acc0, acc1, acc2 = self.acc(diff_matrix,0), self.acc(diff_matrix,1), self.acc(diff_matrix,2)

        # mean over accuracies based on all scans
        added_mean = (acc0+acc1+acc2) / 3
        if verbose:
            print("added diff %0.02f,%0.02f,%0.02f - %0.02f" % (acc0,acc1,acc2,added_mean))

        # mean over means over accuracies based on one scan
        # -> this leads to slightly different results,
        # depending on the implementation of acc and the handeling of no entries
        accs0, accs1, accs2, means = [], [], [], []
        for diff_entry in list:
            acc0, acc1, acc2 = self.acc(diff_entry,0), self.acc(diff_entry,1), self.acc(diff_entry,2)
            mean = (acc0+acc1+acc2) / 3
            accs0.append(acc0)
            accs1.append(acc1)
            accs2.append(acc2)
            means.append(mean)

        if verbose:
            print("mean diff %0.02f,%0.02f,%0.02f - %0.02f" % (np.mean(accs0),np.mean(accs1),np.mean(accs2),np.mean(means)))

        return added_mean, np.mean(means)

    def evaluate(self, verbose=False):
        """
       evaluate the results based on the prediction for all data splits and gt,
       plots the results
       :param verbose: use optional printouts
       :return:
       """

        # init
        log_dir = join(self.log_path, self.name)
        map = {}
        dirs = os.listdir(log_dir)

        # iterate over second_stage, val, test
        for dat in ["training", "validation", "test"]:
            print(dat)
            added_means = []

            # iterate over all runs and calculate a mean accuracy
            for dir in sorted(dirs):
                split_number = int(dir.split("run")[1].split("-TIME-")[0])  # format runXX-TIME-...
                split_name = "split%02d" % split_number

                if verbose:
                    print("analyze split %d" % split_number)

                diff_matrices = []
                # iterate overall scans and calculate acc based on diffusion matrices
                for scan in os.listdir(join(self.splits_path, split_name, "data", dat)):
                    # get gt
                    with gzip.GzipFile(join(self.splits_path, split_name, "gt", dat, scan), "rb") as f:
                        gt = pickle.load(f)
                    # get pred
                    with gzip.GzipFile(join(log_dir, dir, scan), "rb") as f:
                        pred = pickle.load(f)

                    argmax_gt = np.argmax(gt, axis=-1).flatten()
                    argmax_pred = np.argmax(pred, axis=-1).flatten()

                    # generate diffusion matrix between classes
                    num_classes = 3
                    diff_matrix = np.zeros((num_classes, num_classes))
                    for gt_index in range(num_classes):
                        where_gt = np.where(argmax_gt == gt_index)
                        for pred_index in range(num_classes):
                            diff_matrix[pred_index, gt_index] = np.sum(argmax_pred[where_gt] == pred_index)

                    # save list
                    diff_matrices.append(diff_matrix)

                    if scan not in map:
                        map[scan] = np.zeros((3,3))

                    map[scan] += diff_matrix

                added_mean, _ = self.calc_stats(diff_matrices, verbose=verbose)
                added_means.append(added_mean)

            # show stats per dat
            print("added mean: mean - %0.04f, std - %0.04f" % (np.mean(added_means),np.sqrt(np.var(added_means))))

            # show results plot

            # set figure size
            font = {
                'size': 10 # 20 for good horizontal graphics
            }
            matplotlib.rc('font', **font)

            # show runs
            x = np.arange(0,len(added_means))
            plt.plot(x,added_means, label="cross validation run", marker="o")
            plt.xticks(x)
            plt.yticks(np.arange(0,1,step=0.1))
            plt.ylim(0,1)
            # show average and human performance
            plt.plot(x, [np.mean(added_means)] * len(added_means), label='mean = %0.02f' % np.mean(added_means), linestyle='--')
            plt.plot(x, [0.7828] * len(added_means), label='human = 0.78', linestyle='--')
            plt.legend()
            plt.xlabel("run id")
            plt.ylabel("accuracy")
            plt.show()



if __name__ == "__main__":
    # main program

    # init
    config = parse_arguments()
    cross = cross_validate(config)

    # run the first stage of the cross validation
    if config.first_stage:
        cross.first_stage()
        cross.generate_splits(number_splits=config.num_splits, target_percent=config.target_split_percentage,)

    # second_stage of the cross validation
    if config.second_stage:
        for split_number in range(config.num_splits):
            cross.second_stage(split_number)

    cross.evaluate()
