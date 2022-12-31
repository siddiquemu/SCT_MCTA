#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
import pprint
import subprocess
from collections import defaultdict
from six.moves import xrange

# Use a non-interactive backend
#import matplotlib
#matplotlib.use('Agg')

import numpy as np
import cv2
import time
import torch
import imutils
import pycocotools.mask as mask_util
from skimage import measure
import imageio
import glob
#import imgaug.augmenters as iaa # TODO: to extract polygon section from image: train DAHE appearance model??
#import imgaug as ia
import torch.nn as nn
from torch.autograd import Variable
import _init_paths
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_all
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from utils.process_box_mask import get_box_mask
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer
from clasp2coco import define_dataset_dictionary, Write_To_Json, Write_ImagesInfo, Write_AnnotationInfo

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


# required inputs
data ='clasp2020' #
cfg_file = '/home/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml'
#image_dir = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp1/imgs/cam02exp1.mp4'
#load_ckpt = '/home/siddique/PANet/panet_mask_step179999.pth'
#load_ckpt = '/home/siddique/PANet/Outputs/e2e_panet_R-50-FPN_2x_mask/Apr04-15-07-49_siddiquemu_step/ckpt/model_step19999.pth' #fine-tuned: 10 angle
#load_ckpt = '/home/siddique/PANet/Outputs/e2e_panet_R-50-FPN_2x_mask/Apr17-12-08-35_siddiquemu_step/ckpt/model_step19999.pth' #fine-tuned 20 angle
load_ckpt = '/home/siddique/PANet/Outputs/e2e_panet_R-50-FPN_2x_mask/Apr17-21-14-48_siddiquemu_step/ckpt/model_step9999.pth'  # finetuned 20 angle v2
images = False
set_cfgs = None
load_detectron = None
cuda = True
#output_dir = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp1/Results/cam02exp1.mp4/panet'
merge_pdfs = False
'''
def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Demonstrate mask-rcnn results')
    parser.add_argument(
        '--dataset', required=True,
        help='training dataset')

    parser.add_argument(
        '--cfg', dest='cfg_file', required=True,
        help='optional config file')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='set config keys, will overwrite config in the cfg_file',
        default=[], nargs='+')

    parser.add_argument(
        '--no_cuda', dest='cuda', help='whether use CUDA', action='store_false')

    parser.add_argument('--load_ckpt', help='path of checkpoint to load')
    parser.add_argument(
        '--load_detectron', help='path to the detectron weight pickle file')

    parser.add_argument(
        '--image_dir',
        help='directory to load images for demo')
    parser.add_argument(
        '--images', nargs='+',
        help='images to infer. Must not use with --image_dir')
    parser.add_argument(
        '--output_dir',
        help='directory to save demo results',
        default="infer_outputs")
    parser.add_argument(
        '--merge_pdfs', type=distutils.util.strtobool, default=True)

    args = parser.parse_args()

    return args
'''


def main(folders,angleSet,dataset_clasp,output_dir,isTrain,imgResultDir):
    """main function"""

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    #args = parse_args()
    print('Called with args:')
    #print(args)


    if data == 'coco':
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif data == 'clasp2020':
        dataset = datasets.get_clasp_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif data=="keypoints_coco":
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = 2
    else:
        raise ValueError('Unexpected dataset name: {}'.format(data))

    print('load cfg from file: {}'.format(cfg_file))
    cfg_from_file(cfg_file)
    # set NMS
    cfg['TEST']['NMS'] = 0.3
    cfg['TEST']['SCORE_THRESH'] = 0.8
    if set_cfgs is not None:
        cfg_from_list(set_cfgs)

    assert bool(load_ckpt) ^ bool(load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
    assert_and_infer_cfg()

    maskRCNN = Generalized_RCNN()

    if cuda:
        maskRCNN.cuda()

    load_name = load_ckpt
    print("loading checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
    net_utils.load_ckpt(maskRCNN, checkpoint['model'],isTrain)

    if load_detectron:
        print("loading detectron weights %s" % load_detectron)
        load_detectron_weight(maskRCNN, load_detectron)

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True, device_ids=[0])  # only support single GPU

    maskRCNN.eval()
    #save all images in one folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # read imgaes from multi-camera clasp2 data folders
    cam_list = [folders[0], folders[2], folders[4], folders[5], folders[6]]
    fr = 1
    annIDcount = 1
    for cam_path in cam_list:
        image_dir = cam_path
        assert image_dir or images
        assert bool(image_dir) ^ bool(images)
        if image_dir:
            imglist = misc_utils.get_imagelist_from_dir(image_dir)
        else:
            imglist = images
        num_images = len(imglist)

        for i, im_name in enumerate(imglist):
            if fr%100==0:
                print('img', im_name)
                im = cv2.imread(im_name)
                if im is not None:
                    assert im is not None
                    for angle in angleSet:
                        print('Rotated by {}'.format(angle))
                        imgrot = imutils.rotate_bound(im, angle)

                        timers = defaultdict(Timer)
                        start_time = time.time()
                        if angle == 0:
                            imgrot = im
                        cls_boxes, cls_segms, cls_keyps, cls_segms_coarse = im_detect_all(maskRCNN, imgrot, timers=timers)
                        print("Execution time {} sec".format(time.time() - start_time))
                        im_name, _ = os.path.splitext(os.path.basename(imglist[i]))
                        imgIdnew = 1000*int('%06d'%fr) + angle
                        imgname = str(imgIdnew) + '.png'
                        # get box, mask
                        det, masks = get_box_mask(fr, cls_boxes, cls_segms, cls_keyps,thresh=0.8,dataset=dataset)
                        dataset_clasp = Write_ImagesInfo(imgrot, imgname, int(imgIdnew), dataset_clasp)

                        imf = cv2.cvtColor(imgrot, cv2.COLOR_BGR2RGB)
                        # TODO: save images only for new angles used in dataset: save training data preparation time
                        img_write_path = output_dir+'/'+imgname
                        if not os.path.exists(img_write_path):
                            print('Writing image {}'.format(imgname))
                            imageio.imwrite(img_write_path, imf)

                        if len(det)>0:
                            if fr%5000==0:
                                vis_utils.vis_clasp(
                                    fr,
                                    angle,
                                    imgrot[:, :, ::-1],  # BGR -> RGB for visualization
                                    imgname,
                                    imgResultDir,
                                    cls_boxes,
                                    cls_segms,
                                    cls_segms_coarse,
                                    cls_keyps,
                                    dataset=dataset,
                                    box_alpha=1,
                                    show_class=True,
                                    thresh=0.8,
                                    kp_thresh=2,
                                    ext='png',
                                )

                            # coco parameters
                            # Whether the model needs RGB, YUV, HSV etc.
                            # Should be one of the modes defined here, as we use PIL to read the image:
                            # https://pillow.readthedocs.io/en/stable/handbook/concepts.html#concept-modes
                            # with BGR being the one exception. One can set image format to BGR, we will
                            # internally use RGB for conversion and flip the channels over
                            #_C.INPUT.FORMAT = "BGR"
                            # The ground truth mask format that the model will use.
                            # Mask R-CNN supports either "polygon" or "bitmask" as ground truth.
                            #_C.INPUT.MASK_FORMAT = "polygon"  # alternative: "bitmask"
                            if (det is not None):
                                if (len(det) > 0):
                                    print('total detection: {}'.format(len(det)))
                                    for ib,box in enumerate(det):
                                        bboxfinal = [round(x, 2) for x in box[2:6]]
                                        mask = masks[ib]
                                        area = mask_util.area(mask)#polygon area
                                        catID = box[-1].astype('int')
                                        annID = 1000 * int('%06d' % (annIDcount)) + angle
                                        annIDcount+=1
                                        #box = mask_util.toBbox(mask)
                                        # convert rle mask into polygon mask
                                        bmask = mask_util.decode(mask)
                                        contours = measure.find_contours(bmask, 0.5)
                                        #contours = cv2.findContours(bmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                                        segmPolys = []
                                        for contour in contours:
                                            contour = np.flip(contour, axis=1)
                                            segmentation = contour.ravel().tolist()
                                            segmPolys.append(segmentation)
                                        dataset_clasp = Write_AnnotationInfo(bboxfinal, segmPolys, int(imgIdnew), int(annID), catID, int(area),dataset_clasp)
            fr+=1

            if merge_pdfs and num_images > 1:
                merge_out_path = '{}/results.pdf'.format(output_dir)
                if os.path.exists(merge_out_path):
                    os.remove(merge_out_path)
                command = "pdfunite {}/*.pdf {}".format(output_dir,
                                                        merge_out_path)
                subprocess.call(command, shell=True)
    return dataset_clasp

if __name__ == '__main__':
    isTrain = False
    savefilename = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/coco_images/coco_2017/train2020_CLASP_aug_annotation/test2020_CLASP_aug_annotation_det2.json'
    imgDir = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/coco_images/coco_2017/train2020_CLASP_aug'
    imgResultDir = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp1/Results/cam02exp1.mp4/panet_tuned_rotationv2'
    # clear path files
    if os.path.exists(imgResultDir):
        result_list = glob.glob(imgResultDir + '/*.png')
        result_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        for path in result_list:
            os.remove(path)

    angleSet =[0, 12, 84, 90, 96, 174, 180, 186, 192, 264, 270, 276, 348, 354]
    dataset_clasp = define_dataset_dictionary()

    benchmark_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp1/imgs/'
    folders = glob.glob(benchmark_path + '*')
    folders.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))

    dataset_clasp = main(folders,angleSet,dataset_clasp,imgDir,isTrain,imgResultDir)
    Write_To_Json(savefilename, dataset_clasp)
