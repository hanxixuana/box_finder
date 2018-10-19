#!/usr/bin/env python

import os
import cv2
import time
import numpy as np
import pandas as pd

from copy import deepcopy
from pathos.multiprocessing import ProcessPool


n_process = 10
in_path = 'D:\\datasets\\all_raw_ocr'
out_imgs_path = 'D:\\datasets\\invoices\\imgs'
out_debug_imgs_path = 'D:\\datasets\\invoices\\debug_imgs'
out_masks_path = 'D:\\datasets\\invoices\\masks'
out_img_height = 1024
out_img_width = 768
distance_threshold = 40
discard_threshold_for_width = 40
discard_threshold_for_height = 400
debug = True


def worker(folder_path, csv):

    csv_path = os.path.join(folder_path, csv)
    img_path = csv_path[:-3].replace('out', 'org') + 'png'

    batch_name = os.path.split(folder_path)[-1]

    out_img_path = (
        os.path.join(
            out_imgs_path,
            csv.replace(
                'out', batch_name + '_org'
            ).replace(
                'csv', 'png'
            )
        )
    )
    out_mask_path = (
        os.path.join(
            out_masks_path,
            csv.replace(
                'out', batch_name + '_org'
            ).replace(
                'csv', 'npy'
            )
        )
    )

    if os.path.isfile(out_img_path) and os.path.isfile(out_mask_path):
        print(
            '[%s] Already finished %s.' %
            (
                time.ctime(),
                csv_path
            )
        )
        return

    try:
        table = pd.read_csv(csv_path, sep=',')

        table['right'] = table['left'] + table['width']
        table['previous_right'] = table['right'].shift(1)
        table['distance_from_previous'] = table['left'] - table['previous_right']

        table.dropna(axis=0, inplace=True)
        table.reset_index(inplace=True)

        boxes = []
        box = None
        for idx, row in table.iterrows():
            if box is None:
                box = row[['left', 'top', 'width', 'height']]
            else:
                if 0 <= row['distance_from_previous'] <= distance_threshold:
                    box['width'] = row['left'] - box['left'] + row['width']
                    box['top'] = np.min([box['top'], row['top']])
                    box['height'] = np.max([box['height'], row['height']])
                else:
                    if (
                            (box['width'] >= discard_threshold_for_width)
                            and
                            (box['height'] <= discard_threshold_for_height)
                    ):
                        boxes.append(box)
                    box = row[['left', 'top', 'width', 'height']]

        img = cv2.imread(img_path)

        if debug:
            test_img = deepcopy(img)
            for box in boxes:
                test_img = cv2.rectangle(
                    test_img,
                    (box['left'], box['top']),
                    (box['left'] + box['width'], box['top'] + box['height']),
                    255,
                    3
                )
            cv2.imwrite(
                os.path.join(
                    out_debug_imgs_path,
                    csv.replace(
                        'out', batch_name + '_org'
                    ).replace(
                        'csv', 'png'
                    )
                ),
                cv2.resize(
                    test_img,
                    (out_img_width, out_img_height)
                )
            )

        height, width, _ = img.shape
        mask = np.zeros(
            [height, width, len(boxes)],
            dtype=np.uint8
        )
        for idx, box in enumerate(boxes):
            mask[:, :, idx] = cv2.rectangle(
                mask[:, :, idx].copy(),
                (box['left'], box['top']),
                (box['left'] + box['width'], box['top'] + box['height']),
                255,
                -1
            )

        img = cv2.resize(
            img, (out_img_width, out_img_height)
        )
        mask = cv2.resize(
            mask, (out_img_width, out_img_height)
        )

        cv2.imwrite(out_img_path, img)
        np.save(out_mask_path, mask.astype(np.bool))

        print(
            '[%s] Done with %s.' %
            (
                time.ctime(),
                csv_path
            )
        )

    except KeyboardInterrupt as error:
        print('Stopped by keyboard.')
        raise error
    except Exception as error:
        print(error)
        pass


def main():

    if not os.path.exists(os.path.join(*os.path.split(out_imgs_path)[:-1])):
        os.mkdir(os.path.join(*os.path.split(out_imgs_path)[:-1]))

    if not os.path.exists(out_imgs_path):
        os.mkdir(out_imgs_path)

    if not os.path.exists(out_debug_imgs_path):
        os.mkdir(out_debug_imgs_path)

    if not os.path.exists(out_masks_path):
        os.mkdir(out_masks_path)

    folders = [
        item for item in os.listdir(in_path)
        if (
                ('.' not in item)
                and
                (item[0] == 'B')
                and
                (len(item) < 4)
        )
    ]

    for folder in folders:

        folder_path = os.path.join(in_path, folder)

        csvs = [
            item for item in os.listdir(folder_path)
            if (
                    (item[-3:] == 'csv')
                    and
                    (item[:3] == 'out')
            )
        ]

        if n_process > 1:
            with ProcessPool(nodes=n_process) as pool:
                pool.map(
                    worker,
                    [folder_path] * len(csvs),
                    csvs
                )
        else:
            for csv in csvs:
                worker(folder_path, csv)


if __name__ == '__main__':
    main()
