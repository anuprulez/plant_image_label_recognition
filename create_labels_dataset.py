import os
import sys
import random
import math
import re
import time
import numpy as np
import cv2
import glob
import matplotlib
import matplotlib.pyplot as plt

import logging

logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Root directory of the project
ROOT_DIR = os.path.abspath("mrcnn")

# Directory of images to run detection on
IMAGE_DIR = "data/crops/martius/"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


class LabelsDataset(utils.Dataset):
    """Create the labels dataset.
    """

    def load_shapes(self):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        n_max = 5
        file_names = glob.glob(IMAGE_DIR + "*.jpg")
        # Add classes
        self.add_class("labels", 1, "rectangle")
        
        for i, img in enumerate(file_names):
            image = cv2.imread(img, 0) #skimage.io.imread(img)
            shape = image.shape
            self.add_image("labels", image_id=i, path=img, width=shape[0], height=shape[1])
            if i > n_max:
                break

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        print(info)
        return info

    '''def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "shapes":
            return info["shapes"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        shapes = info['shapes']
        count = len(shapes)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
                                                shape, dims, 1)
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([self.class_names.index(s[0]) for s in shapes])
        return mask.astype(np.bool), class_ids.astype(np.int32)'''
        
        
# Training dataset
'''n_tr_images = 500
n_te_images = 50
dataset_train = ShapesDataset()
dataset_train.load_shapes(n_tr_images, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(n_te_images, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()'''

dataset = LabelsDataset()
dataset.load_shapes()
dataset.prepare()
img_ids = dataset.image_ids
for ids in img_ids:
    dataset.load_image(ids)


