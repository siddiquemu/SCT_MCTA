"""Perform inference on one or more datasets."""

import argparse
import os
import pprint
import sys
import time
import _init_paths
import pdb
import _init_paths  # pylint: disable=unused-import
from core.config import cfg, merge_cfg_from_file, merge_cfg_from_list, assert_and_infer_cfg
from core.test_engine_tuned import run_inference_tuned
import utils.logging
import cv2
import torch

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--dataset',
        help='training dataset')
    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')

    parser.add_argument(
        '--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--output_dir',
        help='output directory to save the testing results. If not provided, '
             'defaults to [args.load_ckpt|args.load_detectron]/../test.')

    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file.'
             ' See lib/core/config.py for all options',
        default=[], nargs='*')

    parser.add_argument(
        '--range',
        help='start (inclusive) and end (exclusive) indices',
        type=int, nargs=2)
    parser.add_argument(
        '--multi-gpu-testing', help='using multiple gpus for inference',
        action='store_true')
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true')

    return parser.parse_args()

dataset = 'clasp2020'
multi_gpu_testing = True
vis = False
output_dir = '/home/siddique/PANet/PANet_evaluation/'
set_cfgs = None
cfg_file = '/home/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml'
load_ckpt = '/home/siddique/PANet/panet_mask_step179999.pth'  # baseline
load_ckpt = '/home/siddique/PANet/Outputs/e2e_panet_R-50-FPN_2x_mask/Mar28-11-29-12_siddiquemu_step/ckpt/model_step19999.pth'  # freeze_conv
load_ckpt = '/home/siddique/PANet/Outputs/e2e_panet_R-50-FPN_2x_mask/Apr04-15-07-49_siddiquemu_step/ckpt/model_step19999.pth'  # fine-tuned
#load_ckpt = '/home/siddique/PANet/Outputs/e2e_panet_R-50-FPN_2x_mask/Apr17-12-08-35_siddiquemu_step/ckpt/model_step19999.pth'  # fine-tuned 20 angle v1
#load_ckpt = '/home/siddique/PANet/Outputs/e2e_panet_R-50-FPN_2x_mask/Apr17-21-14-48_siddiquemu_step/ckpt/model_step9999.pth'  # finetuned 20 angle v2
load_detectron = False


if __name__ == '__main__':

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    logger = utils.logging.setup_logging(__name__)
    #logger.info(args)
    #args.multi_gpu_testing = True
    #pdb.set_trace()
    assert (torch.cuda.device_count() == 1) ^ bool(multi_gpu_testing)

    assert bool(load_ckpt) ^ bool(load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    if output_dir is None:
        ckpt_path = load_ckpt if load_ckpt else load_detectron
        output_dir = os.path.join(
            os.path.dirname(os.path.dirname(ckpt_path)), 'test')
        logger.info('Automatically set output directory to %s', output_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cfg.VIS = vis

    if cfg_file is not None:
        merge_cfg_from_file(cfg_file)
    if set_cfgs is not None:
        merge_cfg_from_list(set_cfgs)

    if dataset == "coco2017":
        cfg.TEST.DATASETS = ('coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 81
    if dataset == "clasp2020":
        cfg.TEST.DATASETS = ('clasp2_2020_test',) #'clasp2_2020_test' GT from finetuned model predictions
        cfg.MODEL.NUM_CLASSES = 2
    elif dataset == "keypoints_coco2017":
        cfg.TEST.DATASETS = ('keypoints_coco_2017_val',)
        cfg.MODEL.NUM_CLASSES = 2
    else:  # For subprocess call
        assert cfg.TEST.DATASETS, 'cfg.TEST.DATASETS shouldn\'t be empty'
    assert_and_infer_cfg()

    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    # For test_engine.multi_gpu_test_net_on_dataset
    test_net_file, _ = os.path.splitext(__file__)
    # manually set args.cuda
    cuda = True

    run_inference_tuned(
        load_ckpt,
        load_detectron,
        test_net_file,
        output_dir,
        ind_range = None,
        multi_gpu_testing=multi_gpu_testing,
        check_expected_results=True)
