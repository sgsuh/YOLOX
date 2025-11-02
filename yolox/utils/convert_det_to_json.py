"""
Create: 2021.10.05
Author: SG.SUH
Python: 3.7
"""

import base64
import os
import json

from yolox.data.datasets import AD_CLASSES_V2
from yolox.data.datasets import AD_CLASSES_V3
from yolox.data.datasets import AD_CLASSES_V4

def convert_det_to_json(image_path, det, save_path):
    content = {'version': {}, 'flags': {}, 'shapes': [], 'imageData': {}, 'imagePath': {}, 'imageHeight': {}, 'imageWidth': {}}

    content['version'] = '4.5.7'
        
    if len(det['box']) != 0:
        for i in range(len(det['box'])):
            shape = {'label': {}, 'points': {}, 'group_id': None, 'shape_type': 'rectangle', 'flags': {}}

            if det['class'][i] == 0:
                shape['label'] = 'TL'
            elif det['class'][i] == 1:
                shape['label'] = 'Pedestrian'
            elif det['class'][i] == 2:
                shape['label'] = 'Car'
            elif det['class'][i] == 3:
                shape['label'] = 'Cyclist'
            else:
                shape['label'] = 'Dontcare'

            x1 = det['box'][i][0]
            y1 = det['box'][i][1]
            x2 = det['box'][i][2]
            y2 = det['box'][i][3]

            shape['points'] = [[x1, y1], [x2, y2]]

            content['shapes'].append(shape)
    
    content['imageData'] = base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')
    content['imagePath'] = os.path.basename(image_path)
    content['imageHeight'] = 1080
    content['imageWidth'] = 1920

    with open(save_path, 'w') as fout:
        json.dump(content, fout, indent = 2)

def convert_det_cnn_to_json(image_path, det, tl, save_path):
    content = {'version': {}, 'flags': {}, 'shapes': [], 'imageData': {}, 'imagePath': {}, 'imageHeight': {}, 'imageWidth': {}}

    content['version'] = '4.5.7'
        
    if len(det['box']) != 0:
        tl_count=0
        for i in range(len(det['box'])):
            shape = {'label': {}, 'points': {}, 'group_id': None, 'shape_type': 'rectangle', 'flags': {}}

            if det['class'][i] == 0:
                shape['label'] = tl[tl_count]
                tl_count +=1
            elif det['class'][i] == 1:
                shape['label'] = 'Pedestrian'
            elif det['class'][i] == 2:
                shape['label'] = 'Car'
            elif det['class'][i] == 3:
                shape['label'] = 'Cyclist'
            else:
                shape['label'] = 'Dontcare'

            x1 = det['box'][i][0]
            y1 = det['box'][i][1]
            x2 = det['box'][i][2]
            y2 = det['box'][i][3]

            shape['points'] = [[x1, y1], [x2, y2]]

            content['shapes'].append(shape)
    
    content['imageData'] = base64.b64encode(open(image_path, 'rb').read()).decode('utf-8')
    content['imagePath'] = os.path.basename(image_path)
    content['imageHeight'] = 1080
    content['imageWidth'] = 1920

    with open(save_path, 'w') as fout:
        json.dump(content, fout, indent = 2)

def convert_yolox_output_to_labelme(image_path, json_path, outputs, img_info, conf, cls_name):
    content = {'version': {}, 'flags': {}, 'shapes': [], 'imageData': {}, 'imagePath': {}, 'imageHeight': {}, 'imageWidth': {}}

    content['version'] = '5.0.1'

    if outputs is not None:
        for output in outputs:
            shape = {'label': {}, 'points': {}, 'group_id': None, 'shape_type': 'rectangle', 'flags': {}}

            score = output[4] * output[5]

            if score < conf:
                continue

            shape['label'] = cls_name[int(output[6])]

            x1 = max(0, int(output[0] / img_info['ratio']))
            y1 = max(0, int(output[1] / img_info['ratio']))
            x2 = min(1919, int(output[2] / img_info['ratio']))
            y2 = min(1079, int(output[3] / img_info['ratio']))

            shape['points'] = [[x1, y1], [x2, y2]]
            content['shapes'].append(shape)

    content['imageData'] = None
    content['imagePath'] = os.path.basename(image_path)
    content['imageHeight'] = 1080
    content['imageWidth'] = 1920

    with open(json_path, 'w') as fout:
        json.dump(content, fout, indent = 2)
