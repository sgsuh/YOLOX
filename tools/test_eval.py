"""
Create: 2022.01.21
Author: SG.SUH
Python: 3.7
PyTorch: 1.8
"""
import torch
import argparse

from loguru import logger

from yolox.data import COCODataset
from yolox.data import ValTransform

from yolox.evaluators import COCOEvaluator

from yolox.exp import get_exp

from yolox.utils import get_model_info

def get_eval_loader(data_dir, val_ann, test_size, data_num_workers, batch_size):
    val_dataset = COCODataset(data_dir = data_dir, json_file = val_ann, name = 'image', img_size = test_size, preproc = ValTransform(legacy = False))

    sampler = torch.utils.data.SequentialSampler(val_dataset)

    dataloader_kwargs = {'num_workers': data_num_workers, 'pin_memory': True, 'sampler': sampler}
    dataloader_kwargs['batch_size'] = batch_size

    val_loader = torch.utils.data.DataLoader(val_dataset, **dataloader_kwargs)

    return val_loader

def get_evaluator(data_dir, val_ann, test_size, data_num_workers, batch_size, test_conf, nmsthre, num_classes):
    val_loader = get_eval_loader(data_dir, val_ann, test_size, data_num_workers, batch_size)
    evaluator = COCOEvaluator(dataloader = val_loader, img_size = test_size, confthre = test_conf, nmsthre = nmsthre, num_classes = num_classes, testdev = False)

    return evaluator

def eval(model, evaluator, is_distributed = False, half = False):
    return evaluator.evaluate(model, is_distributed, half)

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--val_data_dir",
                        default="")
    parser.add_argument("--num_class",
                        default=4)
    parser.add_argument("--batch_size",
                        default=1)
    parser.add_argument("--val_ann",
                        default="val.json")
    parser.add_argument("--test_size",
                        default=(540, 960))
    parser.add_argument("--data_num_workers",
                        default=4)
    parser.add_argument("--test_conf",
                        default=0.01)
    parser.add_argument("--nmsthre",
                        default=0.65)
    parser.add_argument("--exp_file",
                        default=None)
    parser.add_argument("--name",
                        default="yolox-s")
    parser.add_argument("--device",
                        default="cuda:0")
    parser.add_argument("--fp16",
                        default=True)
    parser.add_argument("--ckpt",
                        default="weights/yolox_s.pth")

    return parser.parse_args()

if __name__ == "__main__":
    args = make_parser()

    val_data_dir = args.val_data_dir
    num_class = args.num_class
    batch_size = args.batch_size
    val_ann = args.val_ann
    test_size = args.test_size
    data_num_workers = args.data_num_workers
    test_conf = args.test_conf
    nmsthre = args.nmsthre

    exp_file = args.exp_file
    name = args.name
    device = args.device
    fp16 = args.fp16
    ckpt = args.ckpt

    exp = get_exp(exp_file, name)
    model = exp.get_model()

    logger.info('Model Summary: {}'.format(get_model_info(model, test_size)))

    model.load_state_dict(torch.load(ckpt, map_location = 'cpu')['model'])
    model.to(device)

    evaluator = get_evaluator(val_data_dir, val_ann, test_size, data_num_workers, batch_size, test_conf, nmsthre, num_class)

    ap50_95, ap50, summary = eval(model, evaluator)

    logger.info("\n" + summary)