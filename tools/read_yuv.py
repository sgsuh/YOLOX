"""
Create: 2022.05.13
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""

import numpy as np
import cv2
import torch
import os
import time
import glob
import argparse

from loguru import logger

from yolox.data.datasets import AD_CLASSES_V3
from yolox.data.data_augment import ValTransform
from yolox.utils import postprocess
from yolox.utils import vis
from yolox.utils import convert_yolox_output_to_labelme
from yolox.exp import get_exp



def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold_path",
                        default="")
    parser.add_argument("--ckpt_path",
                        default="weights/yolox_s.pth")
    parser.add_argument("--fp16",
                        default=True)

    return parser.parse_args()

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=AD_CLASSES_V3,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)
        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(
                outputs, self.num_classes, self.confthre,
                self.nmsthre, class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):
        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu()

        bboxes = output[:, 0:4]

        # preprocessing: resize
        bboxes /= ratio

        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]

        vis_res = vis(img, bboxes, scores, cls, cls_conf, self.cls_names)
        return vis_res

if __name__ == "__main__":
    args = make_parser()

    fold_path = args.fold_path
    ckpt_path = args.ckpt_path
    fp16 = args.fp16

    exp = get_exp(None, 'yolox-s')
    exp.test_conf = 0.3
    exp.nmsthre = 0.45
    exp.test_size = (540, 960)
    exp.num_classes = 7

    model = exp.get_model()
    model.cuda()
    model.half()
    model.eval()

    ckpt = torch.load(ckpt_path, map_location = 'cpu')

    model.load_state_dict(ckpt['model'])

    predictor = Predictor(model, exp, AD_CLASSES_V3, None, None, 'gpu', True, False)

    cnt = 1

    file_list = glob.glob(fold_path + '/*.dat')
    file_list.sort()

    for file_path in file_list:
        file_name = os.path.splitext(os.path.basename(file_path))[0]

        save_fold_path = fold_path + '/' + file_name

        if not os.path.isdir(save_fold_path):
            os.makedirs(save_fold_path)

        with open(file_path, 'rb') as f:
            while 1:
                try:
                    data = f.read(1920 * 1080 * 2)
                except:
                    break

                im = np.frombuffer(data, dtype = np.uint8)
                im = im.reshape(1080, 1920, 2)

                bgr = cv2.cvtColor(im, cv2.COLOR_YUV2BGR_UYVY)

                proc_im = bgr.copy()
                outputs, img_info = predictor.inference(proc_im)

                save_img_path = save_fold_path + '/' + file_name + '_{:06d}.png'.format(cnt)
                save_json_path = os.path.splitext(save_img_path)[0] + '.json'

                cv2.imwrite(save_img_path, bgr)

                convert_yolox_output_to_labelme(save_img_path, save_json_path, outputs[0], img_info, 0.3)

                cnt += 1