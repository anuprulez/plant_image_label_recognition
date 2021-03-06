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

MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)
else:
    print("Trained model present")

# Directory of images to run detection on
TR_IMAGE_DIR = "data/crops/martius/tr/"
TE_IMAGE_DIR = "data/crops/martius/val/"
TR_IMAGE_SI_DIR = "data/crops/martius/tr_si/"
TE_IMAGE_SI_DIR = "data/crops/martius/val_si/"
TR_IMAGE_BACKGROUND_DIR = "data/crops/martius/original_background/"

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log
from mrcnn.utils import extract_bboxes
from mrcnn.visualize import display_instances


def remove_files():
    dirs = [TR_IMAGE_SI_DIR, TE_IMAGE_SI_DIR]
    for dr in dirs:
        bg_file_names = glob.glob(dr + "*.jpg")
        for f in bg_file_names:
            os.remove(f)

class LabelsConfig(Config):
    """Configuration for training on the toy shapes dataset.
    Derives from the base Config class and overrides values specific
    to the toy shapes dataset.
    """
    # Give the configuration a recognizable name
    NAME = "labels"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # background + 1 shape

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 50
    
    N_TR_IMAGES = 5
    N_TE_IMAGES = 1

    AUG_SIZE = 2


class LabelsDataset(utils.Dataset):
    """Create the labels dataset.
    """

    def load_labels(self, image_dir, train, aug_size, count):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        file_names = glob.glob(image_dir + "*.jpg")
        channels = 3
        # Add classes
        self.add_class("dataset", 1, "rectangle")
        for i, img in enumerate(file_names):
            if i < count:
                image = cv2.imread(img)
                self.augment_dataset(image, i, channels, aug_size, train)

    def augment_dataset(self, image, file_idx, channels, augment_size=2, train=True, min_w=128, max_w=512, min_h=128, max_h=512, te_wt=1024, te_ht=1024):
        for aug_idx in range(augment_size):
            width = np.random.randint(min_w, max_w)
            height = np.random.randint(min_h, max_h)
            resized_image = cv2.resize(image, (height, width))
            rand_w = np.random.randint(1, te_ht - width - 1)
            rand_h = np.random.randint(1, te_wt - height - 1)
            bg_file_names = glob.glob(TR_IMAGE_BACKGROUND_DIR + "*.jpg")
            np.random.shuffle(bg_file_names)
            bg_image = cv2.imread(bg_file_names[0])
            resized_bg_image = cv2.resize(bg_image, (te_ht, te_wt))
            resized_bg_image = cv2.cvtColor(resized_bg_image, cv2.COLOR_BGR2RGB)
            resized_bg_image[rand_w:rand_w + width, rand_h:rand_h + height, :] = resized_image
            file_prefix = "{}_{}".format(file_idx, aug_idx)
            si_file_name = "{}.jpg".format(file_prefix)
            if train is True:
                si_img_path = os.path.join(TR_IMAGE_SI_DIR, si_file_name)
            else:
                si_img_path = os.path.join(TE_IMAGE_SI_DIR, si_file_name)
            cv2.imwrite(si_img_path, resized_bg_image)
            self.add_image("dataset", image_id=file_prefix, path=si_img_path, width=te_wt, height=te_ht, x1=rand_w, x2=rand_w+width, y1=rand_h, y2=rand_h+height)

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        image = cv2.imread(info["path"])
        return image

    def image_reference(self, image_id):
        """Return the label data of the image."""
        info = self.image_info[image_id]
        return info['path']

    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        image = self.load_image(image_id)
        shapes = np.array(["rectangle"])
        mask = np.zeros([info["height"], info["width"], len(shapes)], dtype=np.uint8)
        class_ids = list()
        # only one rectangle is considered
        for i in range(len(shapes)):
            row_s = info["x1"]
            row_e = info["x2"]
            col_s = info["y1"]
            col_e = info["y2"]
            mask[row_s:row_e, col_s:col_e, i] = 1
            # Map class names to class IDs.
            class_ids.append(self.class_names.index('rectangle'))
        return mask, np.asarray(class_ids, dtype='int32')

    def draw_shape(self, image, val=100, color=(255,0,0)):
        threshold = val
        src_gray = image
        canny_output = cv2.Canny(src_gray, threshold, threshold * 2)

        kernel = np.ones((5, 5),np.uint8)

        canny_output = cv2.morphologyEx(canny_output, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(canny_output, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contours_poly = [None]*len(contours)
        boundRect = [None]*len(contours)
        centers = [None]*len(contours)
        radius = [None]*len(contours)
        for i, c in enumerate(contours):
            contours_poly[i] = cv2.approxPolyDP(c, 6, True)
            boundRect[i] = cv2.boundingRect(contours_poly[i])    
        drawing = np.zeros((canny_output.shape[0], canny_output.shape[1], 3), dtype=np.uint8)
        max_area = 0
        max_x1 = 0
        max_x2 = 0
        max_y1 = 0
        max_y2 = 0
        for i in range(len(contours)):
            x1 = int(boundRect[i][0])
            y1 = int(boundRect[i][1])
            x2 = int(boundRect[i][0] + boundRect[i][2])
            y2 = int(boundRect[i][1] + boundRect[i][3])
            area = (x2 - x1) * (y2 - y1)
            if area > max_area:
                max_area = area
                max_x1 = x1
                max_y1 = y1
                max_x2 = x2
                max_y2 = y2
        w = max_x2 - max_x1
        h = max_y2 - max_y1
        coors = [max_x1, max_y1, max_x2, max_y2]
        return coors, w, h

config = LabelsConfig()

print("Deleting old files...")
remove_files()

print("Creating train datasets...")

tr_dataset = LabelsDataset()
tr_dataset.load_labels(TR_IMAGE_DIR, True, config.AUG_SIZE, config.N_TR_IMAGES)
tr_dataset.prepare()

print("Creating test datasets...")

te_dataset = LabelsDataset()
te_dataset.load_labels(TE_IMAGE_DIR, False, config.AUG_SIZE, config.N_TE_IMAGES)
te_dataset.prepare()

np.random.shuffle(tr_dataset.image_ids)
np.random.shuffle(te_dataset.image_ids)

print("Top masks for training dataset")

for image_id in tr_dataset.image_ids:
    image = tr_dataset.load_image(image_id)
    mask, class_ids = tr_dataset.load_mask(image_id)
    bbox = extract_bboxes(mask)
    display_instances(image, bbox, mask, class_ids, tr_dataset.class_names)
    #visualize.display_top_masks(image, mask, class_ids, tr_dataset.class_names)

print("Top masks for test dataset")
for image_id in te_dataset.image_ids:
    image = te_dataset.load_image(image_id)
    mask, class_ids = te_dataset.load_mask(image_id)
    bbox = extract_bboxes(mask)
    display_instances(image, bbox, mask, class_ids, te_dataset.class_names)
    #visualize.display_top_masks(image, mask, class_ids, te_dataset.class_names)

################# Train model

'''print("Loading pretrained model...")
model = modellib.MaskRCNN(mode="training", config=config, model_dir=MODEL_DIR)
# Load weights trained on MS COCO, but skip layers that
# are different due to the different number of classes
# See README for instructions to download the COCO weights
model.load_weights(COCO_MODEL_PATH, by_name=True, exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",  "mrcnn_bbox", "mrcnn_mask"])

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
print("Training heads...")
model.train(tr_dataset, te_dataset, learning_rate=config.LEARNING_RATE, epochs=5, layers='heads')'''
