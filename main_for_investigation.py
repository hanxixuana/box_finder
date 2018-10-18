#!/usr/bin/env python

import os
import random
import numpy as np
import matplotlib.pyplot as plt

import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

from main_for_training import InvoiceConfig
from helpers.datasets import make_datasets


ROOT_DIR = os.path.abspath(".")
MODEL_DIR = os.path.join(ROOT_DIR, "logs")


class InferenceConfig(InvoiceConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def main():

    np.random.seed(0)
    dataset_train, dataset_val = make_datasets()

    inference_config = InferenceConfig()

    model = modellib.MaskRCNN(
        mode="inference",
        config=inference_config,
        model_dir=MODEL_DIR
    )

    model_path = model.find_last()

    print("Loading weights from ", model_path)
    model.load_weights(model_path, by_name=True)

    image_id = random.choice(dataset_val.image_ids)
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(
            dataset_val,
            inference_config,
            image_id,
            use_mini_mask=False
        )

    log("original_image", original_image)
    log("image_meta", image_meta)
    log("gt_class_id", gt_class_id)
    log("gt_bbox", gt_bbox)
    log("gt_mask", gt_mask)

    visualize.display_instances(
        original_image,
        gt_bbox,
        gt_mask,
        gt_class_id,
        dataset_train.class_names,
        figsize=(8, 8)
    )

    results = model.detect([original_image], verbose=1)

    r = results[0]
    visualize.display_instances(
        original_image,
        r['rois'],
        r['masks'],
        r['class_ids'],
        dataset_val.class_names,
        r['scores']
    )
    plt.show()


if __name__ == '__main__':
    main()
