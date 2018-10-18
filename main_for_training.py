#!/usr/bin/env python

import os
import numpy as np

from mrcnn.config import Config
import mrcnn.utils as utils
import mrcnn.model as modellib

from helpers.datasets import make_datasets

ROOT_DIR = os.path.abspath(".")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

if not os.path.exists(os.path.join(ROOT_DIR, 'pretraining_weights')):
    os.mkdir(os.path.join(ROOT_DIR, 'pretraining_weights'))
COCO_MODEL_PATH = os.path.join(ROOT_DIR, 'pretraining_weights', "mask_rcnn_coco.h5")
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)


class InvoiceConfig(Config):
    NAME = "invoices_"

    LEARNING_RATE = 0.001
    LEARNING_MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0001

    GPU_COUNT = 1
    IMAGES_PER_GPU = 1

    NUM_CLASSES = 1 + 1

    IMAGE_RESIZE_MODE = 'none'

    RPN_ANCHOR_STRIDE = 3
    RPN_ANCHOR_SCALES = (16, 32, 64, 128, 256)

    POST_NMS_ROIS_TRAINING = 2048
    POST_NMS_ROIS_INFERENCE = 1024
    RPN_TRAIN_ANCHORS_PER_IMAGE = 1024
    ROI_POSITIVE_RATIO = 0.33
    TRAIN_ROIS_PER_IMAGE = 768
    MAX_GT_INSTANCES = 410
    DETECTION_MAX_INSTANCES = 768

    MEAN_PIXEL = np.array([230.0, 230.0, 230.0])

    STEPS_PER_EPOCH = 200
    VALIDATION_STEPS = 10


def main():

    np.random.seed(0)
    dataset_train, dataset_val = make_datasets()

    config = InvoiceConfig()
    config.display()

    model = modellib.MaskRCNN(
        mode="training",
        config=config,
        model_dir=MODEL_DIR
    )
    model.load_weights(
        COCO_MODEL_PATH,
        by_name=True,
        exclude=[
            "mrcnn_class_logits",
            "mrcnn_bbox_fc",
            "mrcnn_bbox",
            "mrcnn_mask"
        ]
    )
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE,
        epochs=50,
        layers='heads'
    )
    model.train(
        dataset_train,
        dataset_val,
        learning_rate=config.LEARNING_RATE / 10,
        epochs=200,
        layers="all"
    )


if __name__ == '__main__':
    main()
