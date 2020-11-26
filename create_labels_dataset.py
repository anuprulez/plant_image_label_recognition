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
    print("File present")

# Directory of images to run detection on
TR_IMAGE_DIR = "data/crops/martius/"
TE_IMAGE_DIR = "data/full_images/"

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
    NAME = "shapes"

    # Train on 1 GPU and 8 images per GPU. We can put multiple images on each
    # GPU because the images are small. Batch size is 8 (GPUs * images/GPU).
    GPU_COUNT = 1
    IMAGES_PER_GPU = 8

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # background + 3 shapes

    # Use small images for faster training. Set the limits of the small side
    # the large side, and that determines the image shape.
    IMAGE_MIN_DIM = 128
    IMAGE_MAX_DIM = 128

    # Use smaller anchors because our image and objects are small
    RPN_ANCHOR_SCALES = (8, 16, 32, 64, 128)  # anchor side in pixels

    # Reduce training ROIs per image because the images are small and have
    # few objects. Aim to allow ROI sampling to pick 33% positive ROIs.
    TRAIN_ROIS_PER_IMAGE = 32

    # Use a small epoch since the data is simple
    STEPS_PER_EPOCH = 20

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 1
    
    N_TR_IMAGES = 3
    N_TE_IMAGES = 5
    
    IMAGE_CHANNEL_COUNT = 3
    
    

class LabelsDataset(utils.Dataset):
    """Create the labels dataset.
    """

    def load_labels(self, image_dir, count_images, width, height):
        """Generate the requested number of synthetic images.
        count: number of images to generate.
        height, width: the size of the generated images.
        """
        file_names = glob.glob(image_dir + "*.jpg")
        # Add classes
        self.add_class("labels", 1, "rectangle")
        for i, img in enumerate(file_names):
            image = cv2.imread(img, 0)
            shape = image.shape
            
            self.add_image("labels", image_id=i, path=img, width=width, height=height)
            if i > count_images:
                break

    def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        image = cv2.imread(info["path"], 0)
        return image
        
    def image_reference(self, image_id):
        """Return the label data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "labels":
            return info["labels"]
        else:
            super(self.__class__).image_reference(self, image_id)
        
    def load_mask(self, image_id):
        """Generate instance masks for shapes of the given image ID.
        """
        info = self.image_info[image_id]
        #labels = #[(1,'rectangle')] #info['labels']
        count = 1 #len(labels)
        mask = np.zeros([info['height'], info['width'], count], dtype=np.uint8)
        '''for i, (shape, _, dims) in enumerate(info['shapes']):
            mask[:, :, i:i+1] = self.draw_shape(mask[:, :, i:i+1].copy(),
                                                shape, dims, 1)'''
        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        # Map class names to class IDs.
        class_ids = np.array([1]) #np.array([self.class_names.index(s[0]) for s in labels])
        return mask.astype(np.bool), class_ids.astype(np.int32)
       
config = LabelsConfig()

width = 128
height = 128


tr_dataset = LabelsDataset()
tr_dataset.load_labels(TE_IMAGE_DIR, config.N_TE_IMAGES, width, height)
tr_dataset.prepare()


te_dataset = LabelsDataset()
te_dataset.load_labels(TE_IMAGE_DIR, config.N_TE_IMAGES, width, height)
te_dataset.prepare()


image_ids = np.random.choice(te_dataset.image_ids, config.N_TE_IMAGES)

'''for image_id in image_ids:
    image = tr_dataset.load_image(image_id)
    mask, class_ids = tr_dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, tr_dataset.class_names)'''

################# Train model

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
model.train(tr_dataset, tr_dataset, 
            learning_rate=config.LEARNING_RATE, 
            epochs=1, 
            layers='heads')

print("Training all, fine tuning...")          
# Fine tune all layers
# Passing layers="all" trains all layers. You can also 
# pass a regular expression to select which layers to
# train by name pattern.
model.train(dataset_train, dataset_val, 
            learning_rate=config.LEARNING_RATE / 10,
            epochs=2, 
            layers="all")


############################################# Inference

'''class InferenceConfig(LabelsConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

inference_config = InferenceConfig()

print("Creating a model...")

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference", 
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = "/home/kumara/image_segmentation_plants/plant_image_label_recognition/mrcnn/logs/shapes20201126T1039/mask_rcnn_shapes_0002.h5" #model.find_last() 
#"/home/ubuntu/data/plants_image/plant_image_label_recognition/mrcnn/logs/shapes20201123T2239/mask_rcnn_shapes_0002.h5" #model.find_last()

# Load trained weights
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)



print("Choosing a random test image...")

# Test on random images
image_ids = np.random.choice(dataset_val.image_ids, num_images)


for image_id in image_ids:
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask = modellib.load_image_gt(dataset_val, inference_config, image_id, use_mini_mask=False)
    
    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    #plt.imshow(original_image)


    #visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id, dataset_train.class_names, figsize=(8, 8))

    print("Detecting objects...")

    results = model.detect([original_image], verbose=1)

    r = results[0]

    print("Displaying objects...")
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'], 
                            dataset_val.class_names, r['scores'], ax=get_ax(), figsize=(8, 8))'''
                            
