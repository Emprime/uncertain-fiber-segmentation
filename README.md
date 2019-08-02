# Uncertain-Fiber-Segmentation

This implementation is the 10 fold cross validation of the paper "2D and 3D Segmentation of uncertain local collagen fiber orientations in SHG microscopy".

### What can this code be used for?

We investigated the orientation of collagen fibers in mice bones of SHG scans.
Due to noise and personal differences fiber orirentations can only be segmented and classified in small regions.
We propose in our paper a two stage network with state of the art backbone in order to semantic segment local regions in a complete scan.

Out of the box this implementation supports the above mentioned 10 fold cross validation. 
However, the architecture is kept quite general to support any kind of 3D image input.
Especially the transformation of pretrained ImageNet weights from 2D to 3D can be used in a variety of architectures. 

### Paper

**Titel**  
2D and 3D Segmentation of uncertain local collagen fiber orientations in SHG microscopy  

**Abstract**  
_Collagen fiber orientations in bones, visible with Second Harmonic Generation (SHG) microscopy, represent the inner structure and its alteration due to influences like cancer. While analyses of these orientations are valuable for medical research, it is not feasible to analyze the needed large amounts of local orientations manually. Since we have uncertain borders for these local orientations only rough regions can be segmented instead of a pixel-wise segmentation. We analyze the effect of these uncertain borders on human performance by a user study. Furthermore, we compare a variety of 2D and 3D methods such as classical approaches like Fourier analysis with state-of-the-art deep neural networks for the classification of local fiber orientations. We present a general way to use pretrained 2D weights in 3D neural networks, such as Inception-ResNet-3D a 3D extension of Inception-ResNet-v2. In a 10 fold cross-validation our two stage segmentation based on Inception-ResNet-3D and transferred 2D ImageNet weights achieves a human comparable accuracy._

- The paper is accepted to GCPR 2019
- The paper will be published by Springer
- The paper is available at [Arxiv](https://arxiv.org/abs/1907.12868).
- A free copy of the paper and the supplementary material is also available [in this repository](./material).

## Installation

Clone this project to your personal workspace.
Install all requirements mentioned below and download the appropriate data.

### Requirements 

We encourage you to use a conda environment for python package management.
We will describe the setup of a new environment for a Titan X.
If you have a different hardware setup check that you change the appropriate commands below.
We expect that miniconda3 is correctly installed and ready to use.

```
conda create -n uncertain-fiber-segmentation python=3.7
conda activate uncertain-fiber-segmentation
conda install keras tensorflow-gpu=1.12 opencv scikit-learn=0.20 tqdm pillow 
conda install cudatoolkit=9.0 # needed due to hardware driver issues
```

If you have matplotlib not install please install it with

```
pip install matplotlib
```

Used versions:
```
keras - 2.2.4
tensorflow-gpu - 1.12.0
opencv - 3.4.2
tqdm - 4.29.1
pilllow - 5.3.0
cudatoolkit - 9.0
matplotlib - 3.0.2
```

### Data Download

The used data is uploaded at [Zenodo](TODO). 

The repository at Zenodo contains 5 zip files:
- `shg-ce-de`  contains the enhanced and denoised scans as image slices, the scans are sorted by mice (wt wildtyp, het ill mice), scan location and individual scan
- `shg-masks` contains the ground truth masks for the three different classes (similar - Green, dissimilar - Red, not of interest - blue)
- `shg-featues` contains the input and gt for the second stage of the proposed two stage segmentation
- `shg-cross-splits` contains the 10 random splits for the 10 fold cross validation
- `logs-prediction` contains the 10 tensorboard logs, weights and predictions for the 10 fold cross validations

If you just want to verify the results of the cross validation or/and to retrain the second stage a download of `shg-cross-splits` and `logs-prediction`. If you want to recalculate the first stage and/or choose different splits you need to download all zip files.

## Execution

Go to the project directory and execute:
```
python  -m src.main
```

If you want to recreate the data for the first or second stage use the following commands.
Be aware that the data generation / first stage will take time and due to tempory files a lot of disk space (about 12 hourse and 300 GB).
Be aware that the training of the second stage some times takes depending on your backend (about 10 hours on a Titan X).
```
python  -m src.main --first_stage
python  -m src.main --second_stage
```

In case of any problems make sure set up the conda environment correctly, downloaded the necessary data and setup the correct paths to the data.
The program support command line arguments to configure the desired behavior and paths. 
Use the arguments ```--help``` for further information:

```
>> python -m src.main --help
Using TensorFlow backend.
usage: main.py [-h] [--data_path DATA_PATH] [--gt_path GT_PATH]
               [--feature_path FEATURE_PATH] [--splits_path SPLITS_PATH]
               [--log_dir LOG_DIR] [--experiment_name EXPERIMENT_NAME]
               [--num_splits NUM_SPLITS]
               [--target_split_percentage TARGET_SPLIT_PERCENTAGE]
               [--first_stage] [--second_stage] [--batch_size BATCH_SIZE]
               [--epochs EPOCHS] [--block_depth BLOCK_DEPTH]

optional arguments:
  -h, --help            show this help message and exit
  --data_path DATA_PATH
                        input path images, used for generating the input for
                        the first stage
  --gt_path GT_PATH     input path ground truth images, used for generating
                        the ground truth for the second stage
  --feature_path FEATURE_PATH
                        out put path generated features, input for second
                        stage
  --splits_path SPLITS_PATH
                        path for the cross validation split directories
  --log_dir LOG_DIR     directory for log entries for tensorboard and
                        predictions
  --experiment_name EXPERIMENT_NAME
                        unique identifier for different trainings or reruns
  --num_splits NUM_SPLITS
                        do num_splits times a cross validation
  --target_split_percentage TARGET_SPLIT_PERCENTAGE
                        percentage of each class that is ensured in a split,
                        mind that high values aren't possible
  --first_stage         tell the system to regenerate the splits and features,
                        this might take some time, first stage
  --second_stage        tell the system to retrain the networks to the given
                        splits and predict, this might take some time, second
                        stage
  --batch_size BATCH_SIZE
                        batch size for prediction in first stage and training
                        in second stage
  --epochs EPOCHS       number of epochs the second stage will train if no
                        early stopping occurs
  --block_depth BLOCK_DEPTH
                        depth of input blocks into the system, this defines
                        also the width and height (4 x depth)

```


### TODO

- review in progress of readme
- enter link to zenodo
- add how to cite