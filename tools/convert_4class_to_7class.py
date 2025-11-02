"""
Create: 2022.02.03
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import glob
import os
import torch
import cv2
import json
import argparse

from yolox.exp import get_exp
from yolox.data.datasets import AD_CLASSES_V3
from test_log import Predictor

def make_parser():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--name",
                        default="yolox-s")
    parser.add_argument("--exp_file",
                        default=None)
    parser.add_argument("--ckpt_file",
                        default="weights/yolox_s.pth")
    parser.add_argument("--device",
                        default="gpu")
    parser.add_argument("--conf",
                        default=0.4)
    parser.add_argument("--nms",
                        default=0.45)
    parser.add_argument("--tsize",
                        default=(540, 960))
    parser.add_argument("--fp16",
                        default=True)
    parser.add_argument("--legacy",
                        default=False)
    parser.add_argument("--fuse",
                        default=False)
    parser.add_argument("--trt",
                        default=False)
    parser.add_argument("--num_class",
                        default=7)
    parser.add_argument("--root_fold",
                        default="")
    parser.add_argument("--dst_root_fold",
                        default="")
    parser.add_argument("--search_fold_list",
                        default=[""])
    
    return parser

def main():
    args = make_parser()

    name = args.name
    exp_file = args.exp_file
    ckpt_file = args.ckpt_file
    device = args.device
    conf = args.conf
    nms = args.nms
    tsize = args.tsize
    fp16 = args.fp16
    legacy = args.legacy
    fuse = args.fuse
    trt = args.trt
    num_class = args.num_class

    exp = get_exp(exp_file, name)
    exp.test_conf = conf
    exp.nmsthre = nms
    exp.test_size = tsize
    exp.num_classes = num_class
    model = exp.get_model()

    model.cuda()
    model.half()
    model.eval()

    ckpt = torch.load(ckpt_file, map_location = 'cpu')

    model.load_state_dict(ckpt['model'])

    trt_file = None
    decoder = None

    predictor = Predictor(model, exp, AD_CLASSES_V3, trt_file, decoder, device, fp16, legacy)

    root_fold = args.root_fold

    dst_root_fold = args.dst_root_fold
    search_fold_list = args.search_fold_list

    src_fold_list = glob.glob(root_fold + '/*')
    src_fold_list.sort()

    for src_fold_path in src_fold_list:
        if not os.path.isdir(src_fold_path):
            continue

        src_file_list = glob.glob(src_fold_path + '/*.png')
        src_file_list.sort()

        car_num = os.path.basename(src_fold_path).split('_')[0]

        dst_fold_path = dst_root_fold + '/a2z_' + car_num + '/images'

        for src_file_path in src_file_list:
            src_file_name = os.path.splitext(os.path.basename(src_file_path))[0]
            src_json_path = os.path.splitext(src_file_path)[0] + '.json'

            print(src_file_path)

            is_new = True

            for search_fold_name in search_fold_list:
                search_fold_path = dst_root_fold + '/' + search_fold_name + '/images'

                search_file_path = search_fold_path + '/' + src_file_name + '.json'

                if os.path.isfile(search_file_path):
                    is_new = False

                    break

            if is_new is False:
                continue

            im = cv2.imread(src_file_path)

            with open(src_json_path, 'r') as f:
                json_data = json.load(f)

            if not os.path.isdir(dst_fold_path):
                os.makedirs(dst_fold_path)

            dst_file_path = dst_fold_path + '/' + os.path.basename(src_file_path)
            dst_json_path = dst_fold_path + '/' + os.path.basename(src_json_path)

            cv2.imwrite(dst_file_path, im)

            with open(dst_json_path, 'w') as fout:
                json.dump(json_data, fout, indent = 2)

if __name__ == "__main__":
    main()

        
        