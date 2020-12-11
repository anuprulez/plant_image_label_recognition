import os
import sys
import random
import math
import numpy as np
from numpy import expand_dims
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
from mrcnn.model import mold_image

# Import COCO config
#sys.path.append(os.path.join(ROOT_DIR, "coco/"))  # To find local version
#import coco

#%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs/")

model_path = "/home/kumara/image_segmentation_plants/plant_image_label_recognition/mrcnn/logs/labels20201210T1251/mask_rcnn_labels_0005.h5"
# labels20201210T1251
# labels20201209T2238
# labels20201209T1106
# labels20201207T0924
# labels20201209T1008

# Local path to trained weights file
# COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
# if not os.path.exists(COCO_MODEL_PATH):
#    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
#IMAGE_DIR = "data/full_images/"
IMAGE_DIR = "data/crops/martius/random_te/"

'''class LabelsConfig(Config):
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
    N_TE_IMAGES = 20'''


class InferenceConfig(Config):
    # Give the configuration a recognizable name
    NAME = "labels"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # background + 1 shape
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

config = InferenceConfig()
#config.display()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
print("Loading weights from ", model_path)
model.load_weights(model_path, by_name=True)

width = 840
height = 1024

def filter_rois(res):
    threshold = 0.95
    rois = list()
    masks = list()
    cls_ids = list()
    scores = list()
    remove_idx = list()
    for i, score in enumerate(res["scores"]):
        if score < threshold:
            remove_idx.append(i)
            break  
    if len(remove_idx) > 0:  
        res["scores"] = res["scores"][0:remove_idx[0]]
        res["class_ids"] = res["class_ids"][0:remove_idx[0]]    
        res["rois"] = res["rois"][0:remove_idx[0], :]
        res["masks"] = res["masks"][:, :, 0:remove_idx[0]]
    return res

# Load a random image from the images folder
file_names = glob.glob(IMAGE_DIR + "*.jpeg")
for fn in file_names:
    image = cv2.imread(fn)
    shape = image.shape
    image = cv2.resize(image, (width, height))
    # Run detection
    results = model.detect([image], verbose=1)
    class_names = ["BG", "rectangle"]
    # Visualize results
    r = results[0]
    res = filter_rois(r)
    #visualize.display_instances(resized_image, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'], figsize=(8, 8))
    visualize.display_instances(image, res['rois'], res['masks'], res['class_ids'], class_names, res['scores'], figsize=(8, 8))
