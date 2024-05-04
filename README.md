# A UNet-based Cloud Segmentation Algorithm For Satellite Imagery

This repo contains the source code for 24Sp BME548 final project for Zion.

Author: Zion Sheng

Version: v1.0

Language: Python 3.10.12



## File Navigator

```
38-clouds-UNet
 ┣ extra_data											# Indices of non-empty patches in training and testing set
 ┃ ┣ non_empty_test_indices.txt
 ┃ ┗ non_empty_train_indices.txt
 ┣ model												# Saved model checkpoints
 ┃ ┣ 38-cloud-unet-ep10-lr0p005.pth						# Core model with 10 epochs and lr=0.005
 ┃ ┣ 38-cloud-unet-ep20-lr0p005.pth						# Core model with 20 epochs and lr=0.005
 ┃ ┣ 38-cloud-unet-no-clamp-ep20-lr0p005.pth			# Variance #1 (uncontraint physical layer) with 20 epochs and lr=0.005
 ┃ ┣ 38-cloud-unet-no-clamp-ep20-lr0p01.pth				# Variance #1 with 20 epochs and lr=0.01
 ┃ ┣ 38-cloud-unet-no-clamp-ep50-lr-stepLR.pth			# Variance #1 with 50 epochs and stepLR learning rate scheduler
 ┃ ┗ 38-cloud-unet-nophy-ep20-lr0p005.pth				# Variance #2 (no physical layer) with 20 epochs and lr=0.005
 ┣ 38-Clouds-UNet-cmp1.ipynb							# Control experiment #1: Train Variance #1 with 20 epochs and lr=0.005
 ┣ 38-Clouds-UNet-cmp2.ipynb							# Control experiment #2: Train Variance #1 with 20 epochs and lr=0.01
 ┣ 38-Clouds-UNet-cmp3.ipynb							# Control experiment #3: Train Variance #1 with 50 epochs and stepLR
 ┣ 38-Clouds-UNet-cmp4.ipynb							# Control experiment #4: Train Variance #2 with 20 epochs and lr=0.005
 ┣ 38-Clouds-UNet-train.ipynb							# Train core model with 20 epochs and lr=0.005
 ┣ UNet.py												# UNet model and UNet_no_physical model implementation
 ┣ cloud_utils.py										# Utility functions
 ┣ patchify_entire_gt.py								# The script to patchify ground truth mask in testing set
 ┣ testing.ipynb										# Code to evaluate model performance on testing set
 ┣ requirements.txt										# Required packages and thier version
 ┗ training_result.txt									# Summarization of model's performance
```



## Required Packages

```
matplotlib==3.5.1
numpy==1.21.5
patchify==0.2.3
Pillow==10.3.0
torch==2.2.1
torchvision==0.17.1
tqdm==4.66.2
```

To run the code, make sure you have all the required packages installed.



## Required Hardware

The code is run and tested on 2 NVIDIA TITAN RTX. Each has 24576MiB memory. Readers are recommended to use equivalent or better GPU devices. A rough estimation to complete running `38-Clouds-UNet-train.ipynb` is about 100 minutes.



## Source Data

Due to the huge size (~13GB) of the source data, it's impossible to attach them with the source code together in a small ZIP file. Still, readers can easily download them from this [link](https://www.kaggle.com/datasets/sorour/38cloud-cloud-segmentation-in-satellite-images), which will direct to this dataset on Kaggle. Readers can also choose to read this [README](https://github.com/SorourMo/38-Cloud-A-Cloud-Segmentation-Dataset?tab=readme-ov-file) file provided by the original creator of this dataset. It has some useful information about the data and how this dataset is created. Note that they didn't upload the patches for ground-truth masks in the testing set. Thus, readers need to use my self-written script, `patchify_entire_gt.py`, to divide those giant masks into patches first. The script has been verified to work well by carefully matching the mask patch and image patch.



## Model Checkpoints

To save some space, we only remain `38-cloud-unet-ep20-lr0p005.pth` as it is the best checkpoint of our core model. To download more checkpoints, please go to this Google Drive [folder](https://drive.google.com/drive/folders/13xSJvE3dgJcIbYDOofFXEsAbc2yySZB4?usp=drive_link).



## Contact

If you have any questions, please contact Zion Sheng (zs144@duke.edu)! Happy to share and discuss! 🤗

