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

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log


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
    STEPS_PER_EPOCH = 5
    
    N_TR_IMAGES = 5
    N_TE_IMAGES = 2
    
    

class LabelsDataset(utils.Dataset):
    """Create the labels dataset.
    """

    def load_labels(self, image_dir, count_images, train=True, width=256, height=256, te_wt=512, te_ht=1024):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        file_names = glob.glob(image_dir + "*.jpg")
        channels = 3
        
        # Add classes
        self.add_class("dataset", 1, "rectangle")
        for i, img in enumerate(file_names):
            image = cv2.imread(img)
            shape = image.shape
            resized_image = cv2.resize(image, (width, height))
            rand_w = np.random.randint(1, te_ht - width - 1)
            rand_h = np.random.randint(1, te_wt - height - 1)
            background_image = np.zeros([te_ht, te_wt, channels], dtype=np.uint8)
            background_image[rand_w:rand_w + width, rand_h:rand_h + height, :] = resized_image
            si_file_name = "{}.jpg".format(i)
            if train is True:
                si_img_path = os.path.join(TR_IMAGE_SI_DIR, si_file_name)
            else:
                si_img_path = os.path.join(TE_IMAGE_SI_DIR, si_file_name)
            cv2.imwrite(si_img_path, background_image)
            self.add_image("dataset", image_id=i, path=si_img_path, width=te_wt, height=te_ht)
            if i > count_images:
                break

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
        box, w, h = self.draw_shape(image)
        mask = np.zeros([info["height"], info["width"], len(shapes)], dtype=np.uint8)
        class_ids = list()
        # only one rectangle is considered
        for i in range(len(shapes)):
            row_s = box[1]
            row_e = box[3]
            col_s = box[0]
            col_e = box[2]
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

width = 256
height = 256

print("Creating train datasets...")

tr_dataset = LabelsDataset()
# TR_IMAGE_DIR
tr_dataset.load_labels(TR_IMAGE_DIR, config.N_TR_IMAGES, True, width, height)
tr_dataset.prepare()

print("Creating test datasets...")
te_dataset = LabelsDataset()
te_dataset.load_labels(TE_IMAGE_DIR, config.N_TE_IMAGES, False, width, height)
te_dataset.prepare()


'''te_image_ids = np.random.choice(te_dataset.image_ids, config.N_TE_IMAGES)

tr_image_ids = np.random.choice(tr_dataset.image_ids, config.N_TR_IMAGES)

print("Top masks for training dataset")

for image_id in tr_image_ids:
    image = tr_dataset.load_image(image_id)
    mask, class_ids = tr_dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, tr_dataset.class_names)
   
print("Top masks for test dataset")
for image_id in te_image_ids:
    image = te_dataset.load_image(image_id)
    mask, class_ids = te_dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, te_dataset.class_names)'''

################# Train model

print("Loading pretrained model...")

model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc", 
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last(), by_name=True)

# Train the head branches
# Passing layers="heads" freezes all layers except the head
# layers. You can also pass a regular expression to select
# which layers to train by name pattern.
print("Training heads...")
model.train(tr_dataset, te_dataset, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1,
            layers='heads')

print("Training all, fine tuning...")          
# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(tr_dataset, te_dataset, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=1, 
            layers="all")
