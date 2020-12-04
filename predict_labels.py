import os
import sys
import random
import math
import numpy as np
import skimage.io
import glob
import matplotlib
import matplotlib.pyplot as plt
import cv2

# Root directory of the project
ROOT_DIR = os.path.abspath("mrcnn")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Import COCO config
#sys.path.append(os.path.join(ROOT_DIR, "coco/"))  # To find local version
#import coco

#%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs/")

model_path = "/home/kumara/image_segmentation_plants/plant_image_label_recognition/mrcnn/logs/labels20201204T1616//mask_rcnn_labels_0001.h5"

# Local path to trained weights file
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
#if not os.path.exists(COCO_MODEL_PATH):
#    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = "data/full_images/"


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
    
    N_TR_IMAGES = 100
    N_TE_IMAGES = 20


class InferenceConfig(LabelsConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

width = 1024
height = 2048

# Load a random image from the images folder
file_names = glob.glob(IMAGE_DIR + "*.jpg")
for fn in file_names:
    #image = skimage.io.imread(fn)
    image = cv2.imread(fn)
    shape = image.shape
    resized_image = cv2.resize(image, (width, height))
    # Run detection
    results = model.detect([resized_image], verbose=1)
    class_names = ["BG", "rectangle"]
    # Visualize results
    r = results[0]
    visualize.display_instances(resized_image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], figsize=(8, 8))
