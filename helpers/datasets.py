#!/usr/bin/env python

import os
import cv2
import numpy as np

from mrcnn import utils
from sklearn.model_selection import train_test_split


DEFAULT_PATH = 'D:\\datasets\\invoices'


def make_datasets(validate_ratio=0.2, data_set_path=DEFAULT_PATH):
    path_to_imgs_folder = os.path.join(
        data_set_path, 'imgs'
    )
    img_paths = [
        os.path.join(path_to_imgs_folder, item)
        for item in os.listdir(path_to_imgs_folder)
    ]

    train_img_paths, validate_img_paths = train_test_split(img_paths, test_size=validate_ratio, shuffle=True)

    train_set = InvoiceDataset()
    train_set.search_for_invoices(train_img_paths)
    train_set.prepare()

    validate_set = InvoiceDataset()
    validate_set.search_for_invoices(validate_img_paths)
    validate_set.prepare()

    return train_set, validate_set


class InvoiceDataset(utils.Dataset):

    def search_for_invoices(self, img_paths):

        self.add_class("invoices", 1, "wordbox")

        for idx, path in enumerate(img_paths):
            self.add_image(
                "invoices", image_id=idx, path=path,
                bg_color=np.array([255, 255, 255])
            )

    def load_image(self, image_id):
        info = self.image_info[image_id]
        image = cv2.imread(info['path'], flags=cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def load_mask(self, image_id):
        info = self.image_info[image_id]
        img_path = info['path']
        mask_path = img_path.replace('imgs', 'masks').replace('png', 'npy')
        mask = np.load(mask_path)
        return mask, np.ones(mask.shape[-1]).astype(np.int)
