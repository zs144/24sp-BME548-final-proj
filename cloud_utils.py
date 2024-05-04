import numpy as np
from numpy.typing import ArrayLike
from pathlib import Path
import cv2

import torch
from torch.nn import functional as F
from torch.utils.data import Dataset

from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class CloudDataset(Dataset):
    def __init__(self, r_dir, g_dir, b_dir, nir_dir, gt_dir, pytorch=True):
        super().__init__()

        self.files = [self.combine_files(f, g_dir, b_dir, nir_dir, gt_dir)
                      for f in r_dir.iterdir() if not f.is_dir()]
        self.pytorch = pytorch

    def combine_files(self, r_file: Path, g_dir, b_dir, nir_dir, gt_dir):
        # Combine file paths for different spectral bands into a dictionary
        files = {'red': r_file,
                 'green': g_dir / r_file.name.replace('red', 'green'),
                 'blue': b_dir / r_file.name.replace('red', 'blue'),
                 'nir': nir_dir / r_file.name.replace('red', 'nir'),
                 'gt': gt_dir / r_file.name.replace('red', 'gt')}
        return files

    def __len__(self):
        # Return the number of files in the dataset
        return len(self.files)

    def open_as_array(self, idx, invert=False, include_nir=False):
        # Open image files as arrays, optionally including NIR channel
        raw_rgb = np.stack([np.array(Image.open(self.files[idx]['red'])),
                            np.array(Image.open(self.files[idx]['green'])),
                            np.array(Image.open(self.files[idx]['blue'])),
                           ], axis=2)

        if include_nir:
            nir = np.expand_dims(np.array(Image.open(self.files[idx]['nir'])), 2)
            raw_rgb = np.concatenate([raw_rgb, nir], axis=2)

        if invert:
            raw_rgb = raw_rgb.transpose((2, 0, 1))

        # Normalize pixel values
        return (raw_rgb / np.iinfo(raw_rgb.dtype).max)

    def open_mask(self, idx, add_dims=False):
        # Open ground truth mask as an array
        raw_mask = np.array(Image.open(self.files[idx]['gt']))
        raw_mask = np.where(raw_mask == 255, 1, 0)

        return np.expand_dims(raw_mask, 0) if add_dims else raw_mask

    def __getitem__(self, idx):
        # Get an item from the dataset (image and mask)
        x = torch.tensor(self.open_as_array(idx, invert=self.pytorch, include_nir=True),
                         dtype=torch.float32)
        y = torch.tensor(self.open_mask(idx, add_dims=False), dtype=torch.int64)

        return x, y

    def open_as_pil(self, idx):
        # Open an image as a PIL image
        arr = 256 * self.open_as_array(idx)
        return Image.fromarray(arr.astype(np.uint8), 'RGB')

    def export_image(self, idx, path):
        cloud_image = (self.open_as_array(idx)*255).astype('uint8')
        im = Image.fromarray(cloud_image)
        im.save(path)

    def __repr__(self):
        # Return a string representation of the dataset
        s = 'Dataset class with {} files'.format(self.__len__())
        return s


def show_mask(mask, ax, random_color=False, alpha=1):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * torch.tensor(color.reshape(1, 1, -1))
    ax.imshow(mask_image, alpha=alpha)


def show_one_image_and_mask_info(cloud_image: torch.tensor, cloud_mask: torch.tensor):
	_, ax = plt.subplots(1, 2, figsize=(8, 8))
	ax[0].set_title('Cloud image')
	ax[0].imshow(cloud_image)

	ax[1].set_title('Ground truth mask')
	ax[1].imshow(cloud_mask, cmap='gray')

	plt.show()


def compute_IoU_per_image(pred_mask: torch.tensor, true_mask: torch.tensor) -> float:
    """Compute the IoU score between prediction and ground_truth"""
    if (true_mask.max() == 0): # no clouds in the image
        # use pixel accuracy as the metric
        total_pixels = true_mask.shape[0] * true_mask.shape[1]
        acc = torch.sum(pred_mask == true_mask) / total_pixels
        # iou_score = acc
        iou_score = acc
    else: # clouds are present
        TP = torch.logical_and(pred_mask, true_mask)
        FP = torch.logical_and(pred_mask, torch.logical_not(true_mask))
        FN = torch.logical_and(torch.logical_not(pred_mask), true_mask)
        iou_score = torch.sum(TP) / (torch.sum(TP) + torch.sum(FP) + torch.sum(FN) + 0.001)
    return iou_score


def compute_IoU(pred_outputs, true_masks):
    avg_IoU_score = torch.tensor(0.0).to(device=pred_outputs.device)
    num_images = len(pred_outputs)
    for pred, true_mask in zip(pred_outputs, true_masks):
        pred = pred.unsqueeze(0)
        pred_mask = F.softmax(pred[0], 0).argmax(0)
        one_IoU_score = compute_IoU_per_image(pred_mask, true_mask)
        avg_IoU_score += one_IoU_score
    avg_IoU_score /= num_images
    return avg_IoU_score


def compute_all_metrics(pred_outputs, true_masks):
    avg_IoU_score = torch.tensor(0.0).to(device=pred_outputs.device)
    avg_precision = torch.tensor(0.0).to(device=pred_outputs.device)
    avg_recall = torch.tensor(0.0).to(device=pred_outputs.device)
    avg_specificity = torch.tensor(0.0).to(device=pred_outputs.device)
    avg_accuracy = torch.tensor(0.0).to(device=pred_outputs.device)
    num_images = len(pred_outputs)
    for pred, true_mask in zip(pred_outputs, true_masks):
        pred = pred.unsqueeze(0)
        pred_mask = F.softmax(pred[0], 0).argmax(0)
        TP = torch.logical_and(pred_mask, true_mask)
        FP = torch.logical_and(pred_mask, torch.logical_not(true_mask))
        FN = torch.logical_and(torch.logical_not(pred_mask), true_mask)
        TN = torch.logical_and(torch.logical_not(pred_mask), torch.logical_not(true_mask))
        if (true_mask.max() == 0): # no clouds in the image
            # use pixel accuracy as the metric
            total_pixels = true_mask.shape[0] * true_mask.shape[1]
            acc = torch.sum(pred_mask == true_mask) / total_pixels
            one_IoU_score = acc
            one_precision = acc
            one_recall = acc
            one_specificity = acc
            one_accuracy = acc
        else: # clouds are present
            one_IoU_score = torch.sum(TP) / (torch.sum(TP) + torch.sum(FP) + torch.sum(FN) + 0.001)
            one_precision = torch.sum(TP) / (torch.sum(TP) + torch.sum(FP) + 0.001)
            one_recall = torch.sum(TP) / (torch.sum(TP) + torch.sum(FN) + 0.001)
            one_specificity = torch.sum(TN) / (torch.sum(TN) + torch.sum(FP) + 0.001)
            one_accuracy = (torch.sum(TP) + torch.sum(TN)) / (torch.sum(TP) + torch.sum(FP) + torch.sum(FN) + torch.sum(TN) + 0.001)
        avg_IoU_score += one_IoU_score
        avg_precision += one_precision
        avg_recall += one_recall
        avg_specificity += one_specificity
        avg_accuracy += one_accuracy
    avg_IoU_score /= num_images
    avg_precision /= num_images
    avg_recall /= num_images
    avg_specificity /= num_images
    avg_accuracy /= num_images
    return avg_IoU_score, avg_precision, avg_recall, avg_specificity, avg_accuracy


def plot_mask_comparison(pred_mask: torch.tensor, true_mask: torch.tensor,
                         image: torch.tensor, ax, has_legend=True) -> None:
    """Plot the predicted mask and ground truth mask over the image, with the true
    positive area colored as green, the true negatives as red, and the false negatives
    as yellow. The orginal image is shown in the background."""
    TP = torch.logical_and(pred_mask, true_mask)
    FP = torch.logical_and(pred_mask, torch.logical_not(true_mask))
    FN = torch.logical_and(torch.logical_not(pred_mask), true_mask)

    mask_comparison = np.zeros((image.shape[0], image.shape[1], 3))
    mask_comparison[TP.squeeze(), :] =  [0, 1, 0] # green
    mask_comparison[FP.squeeze(), :] =  [1, 0, 0] # red
    mask_comparison[FN.squeeze(), :] =  [1, 1, 0] # yellow

    # create legend and add it to the right of the image (outside)
    green_patch = mpatches.Patch(color='green', label='TP')
    red_patch = mpatches.Patch(color='red', label='FP')
    yellow_patch = mpatches.Patch(color='yellow', label='FN')
    if has_legend:
        ax.legend(handles=[green_patch, red_patch, yellow_patch],
                  loc='center left', bbox_to_anchor=(1, 0.5))

    ax.imshow(image)
    ax.imshow(mask_comparison, alpha=0.5)


def show_prediction_results(pred_mask: torch.tensor, true_mask: torch.tensor,
                            image: torch.tensor):
	_, ax = plt.subplots(1, 3, figsize=(12, 12))
	ax[0].imshow(image)
	show_mask(true_mask, ax[0])
	ax[0].set_title('Original mask')
	ax[0].axis('off')
	ax[1].imshow(image)
	show_mask(pred_mask, ax[1])
	real_score = compute_IoU_per_image(pred_mask, true_mask)
	ax[1].set_title(f"Predicted mask\nReal IoU Score: {real_score:.3f}")
	ax[1].axis('off')
	plot_mask_comparison(pred_mask, true_mask, image, ax[2])
	ax[2].set_title('Comparison')
	ax[2].axis('off')
	plt.show()