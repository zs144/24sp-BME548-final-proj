# This python is used to patchify the entire ground truth mask into 384x384 patches.
# The original gt mask may not be divisible by 384, so we pad the mask to be divisible
# by adding edges (pixel values = 0) to the original mask.
# One thing to note is that, in the output mask patches, the background pixel value
# is 0 and the cloud pixel value is 255. But, in the original gt mask, the cloud pixel
# value is 1. So, we need to multiply the patch by 255 to get the correct pixel value.

import numpy as np
from pathlib import Path
from patchify import patchify
from PIL import Image

# Set PATH variables.
current_path = Path.cwd()
# print(current_path)
testset_path = current_path/'data/38-cloud/38-Cloud_test'
testset_gt_path = testset_path/'Entire_scene_gts'
patch_len = 384 # patch size should be (384, 384)

# Iterate through each entire gt mask and patchify it.
for gt_file in testset_gt_path.iterdir():
	prefix = 'edited_corrected_gts_'
	postfix = '.TIF'
	image_index = gt_file.name[len(prefix):-len(postfix)]
	gt_mask = np.array(Image.open(gt_file))
	# pad gt_mask to be divisible by 384
	h, w = gt_mask.shape
	pad_h = patch_len - h % patch_len
	pad_w = patch_len - w % patch_len
	up_padding, down_padding = pad_h//2, pad_h - pad_h//2
	left_padding, right_padding = pad_w//2, pad_w - pad_w//2
	gt_mask = np.pad(gt_mask, ((up_padding, down_padding), (left_padding, right_padding)),
					mode='constant', constant_values=0)
	patches = patchify(gt_mask, (patch_len, patch_len), patch_len)
	for r in range(patches.shape[0]):
		for c in range(patches.shape[1]):
			patch = patches[r, c]
			patch = patch.astype(np.uint8) * 255 # background: 0, cloud: 255
			im_patch = Image.fromarray(patch)
			num = r * patches.shape[1] + (c+1)
			patch_name = f'gt_patch_{num}_{(r+1)}_by_{(c+1)}_{image_index}.TIF'
			im_patch.save(testset_path/'test_gt'/patch_name)