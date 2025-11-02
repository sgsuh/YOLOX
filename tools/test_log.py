"""
Create: 2022.01.06
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import argparse
import torch
import os
import cv2
import time
import glob
import onnxruntime as ort
import numpy as np

from loguru import logger

# Src Files
from yolox.exp import get_exp
from yolox.utils import get_model_info
from yolox.utils import fuse_model
from yolox.utils import postprocess
from yolox.utils import vis
from yolox.utils import convert_yolox_output_to_labelme
from yolox.data import LogLoader
from yolox.data.datasets import A2Z_CLASSES_V2
from yolox.data.datasets import A2Z_CLASSES_V3
from yolox.data.datasets import A2Z_CLASSES_V4
from yolox.data.datasets import A2Z_CLASSES_V5
from yolox.data.data_augment import ValTransform


def make_parser():
    parser = argparse.ArgumentParser('Yolox Demo')
    parser.add_argument('--experiment-name', type = str, default = None)
    parser.add_argument('--name', type = str, default = 'yolox-s')
    parser.add_argument('--path', default = '')
    parser.add_argument('--save_result', default = False, action = 'store_true')
    
    # Exp File
    parser.add_argument('--exp_file', type = str, default = None)
    parser.add_argument('--ckpt', type = str, default = 'weights/yolox_s.pth')
    parser.add_argument('--device', type = str, default = 'gpu')
    parser.add_argument('--conf', type = float, default = 0.3)
    parser.add_argument('--nms', type = float, default = 0.45)
    parser.add_argument('--tsize', type = int, default = (540, 960))
    parser.add_argument('--fp16', dest = 'fp16', default = True, action = 'store_true')
    parser.add_argument('--legacy', dest = 'legacy', default = False, action = 'store_true')
    parser.add_argument('--fuse', dest = 'fuse', default = False, action = 'store_true')
    parser.add_argument('--trt', dest = 'trt', default = False, action = 'store_true')

    parser.add_argument('--hl_model', type = str, default = '')
    parser.add_argument('--vl_model', type = str, default = '')
    parser.add_argument('--save_root', type = str, default = '')

    parser.add_argument('--num_class', type = int, default = 8)

    return parser

class Predictor(object):
    def __init__(self, model, exp, cls_names = A2Z_CLASSES_V2, trt_file = None, decoder = None, device = 'cpu', fp16 = False, legacy = False):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy = legacy)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()

            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {'id': 0}

        if isinstance(img, str):
            img_info['file_name'] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info['file_name'] = None
    
        height, width = img.shape[:2]
        img_info['height'] = height
        img_info['width'] = width
        img_info['raw_img'] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info['ratio'] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        
        img = img.float()

        if self.device == 'gpu':
            img = img.cuda()

            if self.fp16:
                img = img.half()        # To FP16
        
        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)

            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype = outputs.type())

            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre, class_agnostic = True)

            logger.info('Infer Time: {:.4f}s'.format(time.time() - t0))

        return outputs, img_info

    def visual(self, output, img_info, cls_conf = 0.35):
        ratio = img_info['ratio']
        img = img_info['raw_img']

        if output is None:
            return img

        output = output.cpu()

        bboxes = output[:, 0:4]

        # Preprocessing: Resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)

        return vis_res

def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    file_name = os.path.join(exp.output_dir, args.experiment_name)
    os.makedirs(file_name, exist_ok = True)

    vis_folder = None
    if args.save_result:
        vis_folder = os.path.join(file_name, 'vis_res')
        os.makedirs(vis_folder, exist_ok = True)

    if args.trt:
        args.device = 'gpu'

    logger.info('Args: {}'.format(args))

    if args.conf is not None:
        exp.test_conf = args.conf

    if args.nms is not None:
        exp.nmsthre = args.nms

    if args.tsize is not None:
        exp.test_size = (args.tsize[0], args.tsize[1])

    exp.num_classes = args.num_class
    model = exp.get_model()

    logger.info('Model Summary: {}'.format(get_model_info(model, exp.test_size)))

    if args.device == 'gpu':
        model.cuda()

        if args.fp16:
            model.half()        # to FP16

    model.eval()

    if not args.trt:
        if args.ckpt is None:
            ckpt_file = os.path.join(file_name, 'best_ckpt.pth.tar')
        else:
            ckpt_file = args.ckpt

        logger.info('Loading Checkpoint')

        ckpt = torch.load(ckpt_file, map_location = 'cpu')

        # Load the Model State Dict
        model.load_state_dict(ckpt['model'])
        logger.info('Loaded Checkpoint Done')

    if args.fuse:
        logger.info('Fusing Model...')

        model = fuse_model(model)

    if args.trt:
        assert (not args.fuse), 'TensorRT Model is not Support Model Fusing'

        trt_file = os.path.join(file_name, 'model_trt.pth')

        assert os.path.exists(trt_file), ('TensorRT Model is not Found\n Run python3 tools/trt.py First')

        model.head.decode_in_inference = False

        decoder = model.head.decode_outputs

        logger.info('Using TensorRT to Inference')
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, A2Z_CLASSES_V5, trt_file, decoder, args.device, args.fp16, args.legacy)
    current_time = time.localtime()

    # TL Classification
    hl_model = ort.InferenceSession(args.hl_model, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])
    vl_model = ort.InferenceSession(args.vl_model, providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider'])

    model_name = os.path.splitext(os.path.basename(args.ckpt))[0]

    file_list = glob.glob(args.path + '/*.dat')
    file_list.sort()

    if not os.path.isdir(args.save_root):
        os.makedirs(args.save_root)

    for file_path in file_list:
        dataset = LogLoader(file_path, yuv = False)

        log_name = os.path.splitext(os.path.basename(file_path))[0]
        save_log_fold = os.path.join(args.save_root, log_name)

        if not os.path.isdir(save_log_fold):
            os.makedirs(save_log_fold)

        save_det_fold = os.path.join(save_log_fold, model_name)

        if not os.path.isdir(save_det_fold):
            os.makedirs(save_det_fold)

        save_img_fold = os.path.join(save_log_fold, 'image')

        if not os.path.isdir(save_img_fold):
            os.makedirs(save_img_fold)

        for im, crop in dataset:
            if im is None:
                break

            proc_im = im.copy()
            outputs, img_info = predictor.inference(proc_im)


            result_image = predictor.visual(outputs[0], img_info, predictor.confthre)

            cv2.rectangle(result_image, (dataset.crop_offset_x, dataset.crop_offset_y), (dataset.crop_offset_x + dataset.roi_width, dataset.crop_offset_y + dataset.roi_height), (255, 255, 255), 2)
            cv2.imshow('demo', result_image)
            cv2.waitKey(1)

            save_det_path = os.path.join(save_det_fold, log_name + '_{:06d}.jpg'.format(dataset.frame_num))
            save_img_path = os.path.join(save_img_fold, log_name + '_{:06d}.png'.format(dataset.frame_num))

            print(save_img_path)

            cv2.imwrite(save_img_path, im)
            cv2.imwrite(save_det_path, result_image)

            save_json_path = os.path.splitext(save_img_path)[0] + '.json'

            convert_yolox_output_to_labelme(save_img_path, save_json_path, outputs[0], img_info, args.conf, A2Z_CLASSES_V5)

if __name__ == '__main__':
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)

    main(exp, args)