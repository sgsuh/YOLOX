"""
Create: 2021.12.09
Author: SG.SUH
Python: 3.7
"""

import json
import numpy as np
import os
import glob
import cv2
import random
import argparse

from PIL import Image
from PIL import ImageDraw

class_ids_v3 = {
    
}

class Labelme2COCO(object):
    def __init__(self, root_path, labelme_json = [], save_json_path = './coco.json'):
        self.labelme_json = labelme_json
        self.save_json_path = save_json_path
        self.images = []
        self.categories = []
        self.annotations = []
        self.label = []
        self.annID = 1
        self.height = 0
        self.width = 0
        self.root_path = root_path

        self.save_json()

    def data_transfer(self):
        for num, json_file in enumerate(self.labelme_json):
            print(json_file)

            fold_name = json_file.split('/')[-3] + '/' + json_file.split('/')[-2]

            with open(json_file, 'r') as fp:
                data = json.load(fp)
                
                self.images.append(self.image(fold_name, data, num))

                if not 'shapes' in data:
                    continue

                for shapes in data['shapes']:
                    label = shapes['label']

                    if class_ids_v3[label] == -2:
                        continue 

                    if label not in self.label:
                        self.label.append(label)

                    x1 = max(0, min(shapes['points'][0][0], shapes['points'][1][0]))
                    x2 = min(self.width - 1, max(shapes['points'][0][0], shapes['points'][1][0]))

                    y1 = max(0, min(shapes['points'][0][1], shapes['points'][1][1]))
                    y2 = min(self.height - 1, max(shapes['points'][0][1], shapes['points'][1][1]))

                    points = [[x1, y1], [x2, y2]]

                    self.annotations.append(self.annotation(points, label, num))

                    self.annID += 1
        
        # Sort all Text Labels so they are in the Same Order across Data Splits
        self.label.sort()

        for label in self.label:
            if class_ids_v3[label] == -2:
                continue 

            self.categories.append(self.category(label))

        for annotation in self.annotations:
            annotation['category_id'] = self.getcatid(annotation['category_id'])

    def image(self, fold_name, data, num):
        image = {}
        
        height = data['imageHeight']
        width = data['imageWidth']

        image['height'] = height
        image['width'] = width
        image['id'] = num

        image['file_name'] = fold_name + '/' + data['imagePath']

        img = cv2.imread(self.root_path + '/' + image['file_name'])

        if img is None:
            if os.path.splitext(image['file_name'])[-1] == '.png':
                image['file_name'] = os.path.splitext(image['file_name'])[0] + '.jpg'
                
                assert os.path.isfile(self.root_path + '/' + image['file_name']), image['file_name']
            else:
                image['file_name'] = os.path.splitext(image['file_name'])[0] + '.png'

                assert os.path.isfile(self.root_path + '/' + image['file_name']), image['file_name']

        self.height = height
        self.width = width

        return image

    def category(self, label):
        category = {}

        category['supercategory'] = label
        
        category['id'] = class_ids_v3[label]
        category['name'] = label

        return category

    def annotation(self, points, label, num):
        annotation = {}
        contour = np.array(points)
        x = contour[:, 0]
        y = contour[:, 1]
        
        area = (max(x) - min(x)) * (max(y) - min(y))

        annotation['segmentation'] = [float(x) for x in list(np.asarray(points).flatten())]

        annotation['area'] = float(area)

        annotation['image_id'] = num

        annotation['bbox'] = list(map(float, self.getbbox(points)))

        annotation['category_id'] = label

        if class_ids_v3[label] == -1:
            annotation['iscrowd'] = 1
        else:
            annotation['iscrowd'] = 0

        annotation['id'] = self.annID

        return annotation

    def getcatid(self, label):
        for category in self.categories:
            if label == category['name']:
                return category['id']

        print('Label: {} not in Categories: {}'.format(label, self.categories))
        exit()

        return -1

    def getbbox(self, points):
        polygons = points
        mask = self.polygons_to_mask([self.height, self.width], polygons)

        return self.mask2box(mask)

    def mask2box(self, mask):
        index = np.argwhere(mask == 1)
        rows = index[:, 0]
        cols = index[:, 1]

        left_top_r = np.min(rows)
        left_top_c = np.min(cols)

        right_bottom_r = np.max(rows)
        right_bottom_c = np.max(cols)

        return [left_top_c, left_top_r, right_bottom_c - left_top_c, right_bottom_r - left_top_r]

    def polygons_to_mask(self, img_shape, polygons):
        mask = np.zeros(img_shape, dtype = np.uint8)
        mask = Image.fromarray(mask)
        xy = list(map(tuple, polygons))

        ImageDraw.Draw(mask).polygon(xy = xy, outline = 1, fill = 1)

        mask = np.array(mask, dtype = bool)

        return mask

    def data2coco(self):
        data_coco = {}
        data_coco['images'] = self.images
        data_coco['categories'] = self.categories
        data_coco['annotations'] = self.annotations

        return data_coco

    def save_json(self):
        print('Save COCO Json')

        self.data_transfer()

        self.data_coco = self.data2coco()

        print(self.save_json_path)

        os.makedirs(os.path.dirname(os.path.abspath(self.save_json_path)), exist_ok = True)
        json.dump(self.data_coco, open(self.save_json_path, 'w'), indent = 4)

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--root_path",
                        default="")

    return parser.parse_args()

if __name__ == '__main__':
    args = make_parser()

    root_path = args.root_path

    dst_train_path = root_path + '/train.json'
    dst_val_path = root_path + '/val.json'

    fold_list = glob.glob(root_path + '/*')
    fold_list.sort()

    train_json_list = []
    val_json_list = []

    for fold_path in fold_list:
        if not os.path.isdir(fold_path):
            continue

        json_fold_path = fold_path + '/images'

        if not os.path.isdir(json_fold_path):
            continue

        json_file_list = glob.glob(json_fold_path + '/*.json')
        json_file_list.sort()
        random.shuffle(json_file_list)

        train_num = int(len(json_file_list) * 0.8)
        
        for i, json_file_path in enumerate(json_file_list):
            if i < train_num:
                train_json_list.append(json_file_path)
            else:
                val_json_list.append(json_file_path)

    Labelme2COCO(root_path, train_json_list, dst_train_path)
    Labelme2COCO(root_path, val_json_list, dst_val_path)    
