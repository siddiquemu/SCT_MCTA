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
import random
import pandas as pd

# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)


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
def get_split(GT, train_split=0.8, cam=None):
    """

    :param GT:
    :param train_splt:
    :param cam:
    :return:
    """
    GTp = [gt for gt in GT if gt[9] == 1]
    GTb = [gt for gt in GT if gt[9] == 2]
    gt_frameset = np.unique(GT[:, 0].astype('int'))
    gt_len = len(gt_frameset)
    print('full set {}: Nfr {}, person {} bag {}'.format(cam, gt_len, len(GTp), len(GTb)))

    # random split: keep split similar for training and testing forever
    random.seed(42)
    train_subset = random.sample(list(gt_frameset), int(gt_len * train_split))
    print('random sample {}'.format(train_subset[2]))
    # print(subset)
    train_GTp = [gt for gt in GT if gt[0] in train_subset and gt[9] == 1]
    train_GTb = [gt for gt in GT if gt[0] in train_subset and gt[9] == 2]
    print('train split {}: Nfr {}, person {} bag {}'.format(cam, len(train_subset), len(train_GTp), len(train_GTb)))

    test_subset = np.array([t for t in gt_frameset if t not in train_subset])
    test_GTp = [gt for gt in GT if gt[0] not in train_subset and gt[9] == 1]
    test_GTb = [gt for gt in GT if gt[0] not in train_subset and gt[9] == 2]
    print(
        'test split {}: Nfr {}, person {} bag {}'.format(cam, len(test_subset), len(test_GTp), len(test_GTb)))
    print('-------------------------------------------------')
    return train_subset, test_subset

def main(folders, angleSet, dataset_clasp, output_dir, isTrain, imgResultDir, class_list=None, data_type='gt', save_data=False):
    """main function"""

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    #args = parse_args()
    print('Called with args:')
    #print(args)


    if data == 'coco':
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)

    elif data == 'clasp2020':
        dataset = datasets.get_clasp_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif data in ['clasp1_2021', 'clasp2_2021']:
        dataset = datasets.get_clasp1_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif data in ['PVD']:
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
        #dataset = datasets.get_PVD_dataset()
        #cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif data in ['AppleA_train']:
        dataset = datasets.get_AppleA_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif data=="keypoints_coco":
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = 2
    else:
        raise ValueError('Unexpected dataset name: {}'.format(data))

    print('load cfg from file: {}'.format(cfg_file))
    cfg_from_file(cfg_file)
    # set NMS
    cfg['TEST']['NMS'] = 0.4
    cfg['TEST']['SCORE_THRESH'] = 0.7
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
    if load_ckpt_frcnn:
        checkpoint_frcnn = torch.load(load_ckpt_frcnn, map_location=lambda storage, loc: storage)
        checkpoint_frcnn = checkpoint_frcnn['model']
    else:
        checkpoint_frcnn = None

    if load_ckpt:
        load_name = load_ckpt
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt_det(maskRCNN, checkpoint['model'])

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
    #cam_list = [folders[0],folders[2],folders[4],folders[5],folders[6]]
    fr = 1
    annIDcount = 1
    for cam_path in sorted(folders):
        # get training set
        image_dir = os.path.join(cam_path, 'img1')
        if data=='clasp1_2021':
            box_gt = np.loadtxt(os.path.join(cam_path, 'gt/gt.txt'), delimiter=',')
        elif data=='PVD':
            box_gt = np.loadtxt(os.path.join(cam_path, 'gt_tso/gt.txt'), delimiter=',')
        else:
            box_gt = np.loadtxt(os.path.join(cam_path, 'gt_sct/gt.txt'), delimiter=',')
        #train_set, _, _, train_set_bag = get_split(box_gt, train_split=0.8, cam=cam_path.split('/')[-1])
        if data in ['clasp1_2021', 'clasp2_2021']:
            cam_folder = cam_path.split('/')[-1]
            train_frames = pd.read_csv(os.path.join(cam_path,'train_frames/{}.csv'.format(cam_folder)))
            train_set = train_frames.values.squeeze()

            test_frames = pd.read_csv(os.path.join(cam_path, 'test_frames/{}.csv'.format(cam_folder)))
            test_set = test_frames.values.squeeze()
            assert len(set(train_set).intersection(set(test_set)))==0
        else:
            train_set = np.unique(box_gt[:,0])
            bb = np.append(box_gt, box_gt[:,7].reshape(box_gt[:,7].shape[0], 1), axis=1)
            bb[:,7] = box_gt[:,7]
            box_gt = bb

        assert image_dir or images
        assert bool(image_dir) ^ bool(images)
        if image_dir:
            imglist = sorted(glob.glob(os.path.join(image_dir, '*.png')))
            #imglist = misc_utils.get_imagelist_from_dir(image_dir)
        else:
            imglist = images
        num_images = len(imglist)

        for i, im_name in enumerate(imglist):
            fr_num = float(os.path.basename(im_name).split('.')[0])
            if fr_num in train_set and fr_num % 1 == 0:
                print('img', im_name)
                im = cv2.imread(im_name)
                assert im is not None
                if data_type=='gt':
                    fr_box_gt = box_gt[box_gt[:,0]==fr_num]
                #for box in fr_box_gt:
                    #box = box.astype('int')
                    #blankImg = np.zeros(shape=[1080, 1920, 3], dtype=np.uint8)
                   # blankImg[box[3]:box[3]+box[5], box[2]:box[2]+box[4]] = im[box[3]:box[3]+box[5], box[2]:box[2]+box[4]]
                blankImg = im
                for angle in angleSet:
                        if angle>0:
                            print('Rotated by {}'.format(angle))
                            imgrot = imutils.rotate_bound(blankImg, angle)
                        else:
                            imgrot = blankImg

                        timers = defaultdict(Timer)
                        start_time = time.time()
                        if data_type=='gt':
                            det = fr_box_gt
                        else:
                            cls_boxes, cls_segms, cls_keyps, cls_segms_coarse = im_detect_all(maskRCNN, imgrot,
                                                                                              timers=timers,
                                                                                              test_aug=False,
                                                                                              soft_nms=True)
                            print("Execution time {} sec".format(time.time() - start_time))

                            det, masks_rle = get_box_mask(fr, cls_boxes, cls_segms, cls_keyps, thresh=0.8,
                                                          class_list=class_list, dataset=dataset)
                            # imf = cv2.cvtColor(imgrot, cv2.COLOR_BGR2RGB)

                        if len(det)>0:
                            if fr_num%10000000000000==0:
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
                                    class_list=class_list,
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
                            #det = fr_box_gt
                            #apply MS on the augmented set to get less noisy examples for active learning
                        #here det will be the positive clusters
                        if (det is not None) and save_data:
                            if (len(det) > 0):
                                imgIdnew = 10000 * int('%06d' % fr) + angle
                                imgname = '{:08d}.png'.format(imgIdnew)
                                img_write_path = output_dir + '/' + imgname

                                if not os.path.exists(img_write_path) and save_data and len(det) > 0:
                                    dataset_clasp = Write_ImagesInfo(imgrot, imgname, int(imgIdnew), dataset_clasp)
                                    print('Writing image {}'.format(imgname))
                                    cv2.imwrite(img_write_path, imgrot)

                                print('Frame: {}, #detection: {}'.format(fr, len(det)))
                                #save image info
                                dataset_clasp = Write_ImagesInfo(imgrot, imgname, int(imgIdnew), dataset_clasp)

                                for ib,box in enumerate(det):
                                    bboxfinal = [round(x, 2) for x in box[2:6]]
                                    if data_type == 'test_aug':
                                        mask = masks_rle[ib]
                                        area = mask_util.area(mask)#polygon area
                                        assert len(box)==8
                                        #[fr, i, bbox[0], bbox[1], w, h, score, classes[i]]
                                        if box[7]==1:
                                            catID = 1
                                        else:
                                            catID = 2
                                    else:
                                        catID = int(box[9])
                                        area = box[4]*box[5]

                                    annID = 1000 * int('%06d' % (annIDcount)) + angle
                                    annIDcount+=1
                                    #box = mask_util.toBbox(mask)
                                    # convert rle mask into polygon mask
                                    #TODO: try to use rle format in coco annotation format
                                    segmPolys = []  # mask['counts'].decode("utf-8") #[]
                                    if data_type=='test_aug':
                                        bmask = mask_util.decode(mask)
                                        contours = measure.find_contours(bmask, 0.5)
                                        #contours = cv2.findContours(bmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                                        for contour in contours:
                                            contour = np.flip(contour, axis=1)
                                            segmentation = contour.ravel().tolist()
                                            segmPolys.append(segmentation)
                                    assert int(imgIdnew)== int(os.path.basename(img_write_path).split('.')[0])
                                    #save annotation infor for each image
                                    dataset_clasp = Write_AnnotationInfo(bboxfinal, segmPolys, int(imgIdnew), int(annID), catID, int(area), dataset_clasp)
                fr+=1
    return dataset_clasp

if __name__ == '__main__':
    """Script to generate COCO format JSON from 80% of total annotated frames
    CLASP1: 2802 frames annotated for bbox only, dir: /storage/tracking_wo_bnw/data/CLASP1/train_gt_det
    CLASP2: 1380 frames annotated for bbox only, dir: /storage/tracking_wo_bnw/data/CLASP/train_gt_det
    """
    isTrain = False
    # required inputs
    data_type='gt'#'gt' #'test_aug
    data = 'clasp1_2021'#'AppleA_train' #'clasp1_2021'#'clasp2_2021' #'clasp1_2021'  # 'clasp1_2021' #'clasp1_2021' #'clasp1_2021' #'coco' #'clasp2020' #
    database = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/'
    model_path = '/home/siddique/PANet/Outputs/'
    storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/'
    cfg_file = '/home/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_1x_det.yaml'

    if data == 'coco':
        load_ckpt = '/home/siddique/PANet/panet_mask_step179999.pth'
        class_list = [1, 25, 27, 29]

    if data == 'PVD':
        load_ckpt = '/home/siddique/PANet/panet_mask_step179999.pth'
        class_list = [1, 2] #1: PAX, 2: TSO

    if data == 'AppleA_train':
        load_ckpt = '/home/siddique/PANet/panet_mask_step179999.pth'
        class_list = [1] #1: flower

    if data=='clasp1_2021':
        load_ckpt = storage + 'PANet_Models/clasp1/modified_loss/iter7/ckpt/model_step19999.pth'
        class_list = [1, 2]
    if data=='clasp2_2021':
        load_ckpt = storage + 'PANet_Models/clasp1/modified_loss/iter7/ckpt/model_step19999.pth'  
        class_list = [1, 2]

    load_ckpt_frcnn = None  # model_path+'e2e_panet_R-50-FPN_1x_det/panet_box_tuned/ckpt/model_step25524.pth'
    images = False
    set_cfgs = None
    load_detectron = None
    cuda = True
    merge_pdfs = False

    save_data = True
    test_aug = 0

    if data == 'clasp2_2021':
        benchmark = database + 'CLASP/'
    if data == 'clasp1_2021':
        benchmark = database + 'CLASP1/'
    if data == 'PVD':
        benchmark = database + 'PVD/'
    if data == 'AppleA_train':
        benchmark = database + 'AppleA_train/'

    if data_type=='gt':
        imgResultDir = benchmark + 'train_gt_det'
        if data == 'clasp2_2021':
            savefilename = imgResultDir + '/clasp2_gt_det.json'
        if data == 'clasp1_2021':
            savefilename = imgResultDir + '/clasp1_gt_det.json'
        if data == 'PVD':
            savefilename = imgResultDir + '/pvd_gt_det.json'
        if data == 'AppleA_train':
            savefilename = imgResultDir + '/apple_gt_det.json'

        SaveImgDir = imgResultDir + '/img1'
    else:
        imgResultDir = benchmark + 'test_aug_gt'
        savefilename = imgResultDir + '/clasp1_test_aug2.json'
        SaveImgDir = imgResultDir + '/img1_2'

    # clear path files
    if os.path.exists(imgResultDir):
        result_list = glob.glob(imgResultDir + '/*.png')
        result_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        for path in result_list:
            os.remove(path)
    else:
        os.makedirs(imgResultDir)
    if test_aug:
        angleSet =[0, 12, 84, 90, 180, 186, 264, 270, 348, 354]
    else:
        angleSet = [0]
    dataset_clasp = define_dataset_dictionary()

    if data == 'clasp2_2021':
        benchmark_path = database + '/CLASP/train_gt_all/PB_gt/'
    if data == 'clasp1_2021':
        benchmark_path = database + '/CLASP1/train_gt/'
    if data == 'PVD':
        benchmark_path = database + 'PVD/HDPVD_new/train_gt/'
    if data == 'AppleA_train':
        benchmark_path = database + '//'

    folders = glob.glob(benchmark_path + '*')
    folders.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
    if data=='PVD':
        folders = [cam for cam in folders if cam.split('/')[-1] in ['C330', 'C360']]

    dataset_clasp = main(folders, angleSet, dataset_clasp, SaveImgDir, isTrain, imgResultDir, class_list, data_type, save_data)
    if save_data:
        Write_To_Json(savefilename, dataset_clasp)
