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
    IMAGES_PER_GPU = 2

    # Number of classes (including background)
    NUM_CLASSES = 1 + 1 # background + 1 shape

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
    STEPS_PER_EPOCH = 5

    # use small validation steps since the epoch is small
    VALIDATION_STEPS = 1
    
    N_TR_IMAGES = 6
    N_TE_IMAGES = 2
    
    #IMAGE_CHANNEL_COUNT = 3
    
    

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
        self.add_class("dataset", 1, "rectangle")
        for i, img in enumerate(file_names):
            image = cv2.imread(img)
            shape = image.shape
            w = shape[0]
            h = shape[1]
            print(image.astype(np.float32).shape)
            self.add_image("dataset", image_id=i, path=img, width=w, height=h)
            if i > count_images:
                break

    '''def load_image(self, image_id):
        """Generate an image from the specs of the given image ID.
        Typically this function loads the image from a file, but
        in this case it generates the image on the fly from the
        specs in image_info.
        """
        info = self.image_info[image_id]
        image = cv2.imread(info["path"], 0)
        resized_image = cv2.resize(image, (info["width"], info["height"]))
        print()
        return image'''
        
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
        mask = np.zeros([info["width"], info["height"], len(shapes)], dtype=np.uint8)
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
            print(info)
            print(image.shape)
            print(mask.shape)
            print(class_ids)
            print("-------------------")
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

width = 128
height = 128


tr_dataset = LabelsDataset()
tr_dataset.load_labels(TR_IMAGE_DIR, config.N_TR_IMAGES, width, height)
tr_dataset.prepare()


te_dataset = LabelsDataset()
te_dataset.load_labels(TE_IMAGE_DIR, config.N_TE_IMAGES, width, height)
te_dataset.prepare()


tr_image_ids = np.random.choice(tr_dataset.image_ids, config.N_TR_IMAGES)
te_image_ids = np.random.choice(te_dataset.image_ids, config.N_TE_IMAGES)

'''print("Top masks for training dataset")

for image_id in tr_image_ids:
    image = tr_dataset.load_image(image_id)
    mask, class_ids = tr_dataset.load_mask(image_id)
    print(class_ids)
    visualize.display_top_masks(image, mask, class_ids, tr_dataset.class_names)
   
print("Top masks for test dataset")
for image_id in te_image_ids:
    image = te_dataset.load_image(image_id)
    mask, class_ids = te_dataset.load_mask(image_id)
    visualize.display_top_masks(image, mask, class_ids, te_dataset.class_names)'''

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
model.train(tr_dataset, te_dataset, 
            learning_rate=config.LEARNING_RATE, 
            epochs=5,
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
                            
