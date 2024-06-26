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
from core.test import im_detect_all, im_detect_regress
from modeling.model_builder import Generalized_RCNN
import datasets.dummy_datasets as datasets
import utils.misc as misc_utils
import utils.net as net_utils
import utils.vis as vis_utils
from utils.process_box_mask import get_box_mask
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer
from clasp2coco import define_dataset_dictionary, Write_To_Json, Write_ImagesInfo, Write_AnnotationInfo
from get_cluster_mode import Cluster_Mode
import random
import copy
import pandas as pd
import pdb
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
    return_items={}
    GTp = [gt for gt in GT if gt[9] == 1]
    GTb = [gt for gt in GT if gt[9] == 2]
    gt_frameset = np.unique(GT[:, 0].astype('int'))
    gt_len = len(gt_frameset)
    print('full set {}: Nfr {}, person {} bag {}'.format(cam, gt_len, len(GTp), len(GTb)))

    # random split: keep split similar for training and testing forever
    random.seed(42)
    train_subset = random.sample(list(gt_frameset), int(gt_len * train_split))
    return_items['train_subset'] = train_subset
    print('random sample {}'.format(train_subset[2]))
    # print(subset)
    train_GTp = [gt for gt in GT if gt[0] in train_subset and gt[9] == 1]
    train_GTb = [gt for gt in GT if gt[0] in train_subset and gt[9] == 2]
    train_GTb = np.array(train_GTb)
    return_items['bag_gt'] = train_GTb
    gt_frameset_b = np.unique(train_GTb[:,0].astype('int'))
    random.seed(42)
    return_items['semi_subset_b'] = random.sample(list(gt_frameset_b), int(len(gt_frameset_b) * 0.1))
    print('train split {}: Nfr {}, person {} bag {}'.format(cam, len(train_subset), len(train_GTp), len(train_GTb)))
    print('train split {}: Nfr {}, person {} 10% bag {}'.format(cam, len(train_subset), len(train_GTp), len(return_items['semi_subset_b'])))

    test_subset = np.array([t for t in gt_frameset if t not in train_subset])
    return_items['test_subset']= test_subset
    test_GTp = [gt for gt in GT if gt[0] not in train_subset and gt[9] == 1]
    test_GTb = [gt for gt in GT if gt[0] not in train_subset and gt[9] == 2]
    print(
        'test split {}: Nfr {}, person {} bag {}'.format(cam, len(test_subset), len(test_GTp), len(test_GTb)))
    print('-------------------------------------------------')
    return return_items

def configure_detector(data, det_thr):
    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")
    # configure detector
    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")
    torch.cuda.set_device(0)
    torch.set_num_threads(1)
    #args = parse_args()
    print('Called with args:')
    #print(args)

    if data == 'COCO':
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif data == 'clasp2020':
        dataset = datasets.get_clasp_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif data == 'clasp1':
        dataset = datasets.get_clasp1_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif data == 'MOT20':
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
    cfg['TEST']['SCORE_THRESH'] = det_thr

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
    net_utils.load_ckpt(maskRCNN, checkpoint['model'], ckpt_frcnn=checkpoint_frcnn,
                        isTrain=False, FreezeResnetConv=False)

    if load_detectron:
        print("loading detectron weights %s" % load_detectron)
        load_detectron_weight(maskRCNN, load_detectron)

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True, device_ids=[0])  # only support single GPU
    maskRCNN.eval()

    return maskRCNN, dataset

def convert_from_cls_format(cls_boxes, cls_segms=None,coarse_masks=None, cls_keyps=None):
    """Convert from the class boxes/segms/keyps format generated by the testing
    code.
    """
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_segms is not None:
       segms_coarse = [b for b in coarse_masks if len(b) > 0]
       coarse_masks_all = np.concatenate(segms_coarse)
    else:
        coarse_masks_all = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, coarse_masks_all, keyps, classes

def format_dets( cls_boxes, fr=None, dataset=None, class_list=None, thresh=0.5, class_id=None, angle=None):

    if isinstance(cls_boxes, list):
        #boxes, _,_, _, classes = convert_from_cls_format(cls_boxes)
        boxes = cls_boxes[class_id]
        classes = [class_id] * len(boxes)

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    dets = []
    objP=1
    for i in sorted_inds:
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        w = bbox[2] - bbox[0] + 1
        h = bbox[3] - bbox[1] + 1
        w = np.maximum(w, 1)
        h = np.maximum(h, 1)
        # not necessary to use tis condition if model is finetuned dor PAX
        if classes[i] in class_list:
            box_remapd = [fr, objP, bbox[0], bbox[1], w, h, score, classes[i], angle]
            objP+=1

            dets.append(box_remapd)
            print(dataset.classes[classes[i]], score)

    return np.array(dets)

def regress_dets(proposals=None, im=None, im_name=None, fr=None, pred_score=None, dataset=None,
                 class_list=None, timers=None, output_dir=None, vis=False, img_fmt='jpg', class_id=None):

    # TODO: use all remapped dets as proposals to select the best candidate (filter partial dets)
    detPB = []
    # How to format proposals variable for multi-claass
    dets_proposals =proposals[:,2:6]
    dets_proposals[:, 2:4] = dets_proposals[:, 0:2] + dets_proposals[:, 2:4]
    cls_boxes, cls_segms, \
    cls_keyps, cls_segms_coarse = im_detect_regress(maskRCNN, im,
                                                    box_proposals=dets_proposals,
                                                    timers=timers,
                                                    test_aug=0,
                                                    soft_nms=0,
                                                    score_thr=0)
    out_pathreg = os.path.join(output_dir, 'regress_aug')
    det_size = sum(len(cls_boxes[cl]) for cl in class_list)
    #assert len(class_list)==1, 'regression for augmented dets only workd for single class model.. mulit-class: full input proposals are used for all classes'
    #single class proposals distributed in the image for all net classes
    assert det_size//2==dets_proposals.shape[0], 'prediction size {}, proposals size {}'.format(det_size//2, dets_proposals.shape[0])
    if fr % 1 == 0 and vis and det_size//2 > 0:
        vis_utils.vis_clasp(
            fr,
            0,
            im[:, :, ::-1],  # BGR -> RGB for visualization
            im_name,
            out_pathreg,
            cls_boxes,
            dataset=dataset,
            class_list=class_list,
            box_alpha=1,
            show_class=True,
            thresh=pred_score,
            kp_thresh=2,
            ext=img_fmt,
            show_mask=0
        )

    if det_size > 0:
        dets = format_dets(cls_boxes, fr=fr, dataset=dataset, class_list=class_list, thresh=pred_score, class_id=class_id, angle=0)
        detPB = np.array(dets)
    return detPB

def main(maskRCNN, dataset, folders, angleSet, dataset_clasp, output_dir, isTrain, imgResultDir, soft_nms=False,
         class_list=None, data_type='gt', save_data=False, saveAugResult=False,  cluster_score_thr=[0,0], det_thr=None,
         all_scores=None, database=None, cam=None, fr_rate=None, regress_cluster=None, img_fmt='png', save_mask=False):
    """main function"""

    #save all images in one folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # read imgaes from multi-camera clasp2 data folders
    #cam_list = [folders[0],folders[2],folders[4],folders[5],folders[6]]
    fr = 1
    annIDcount = 1
    folders = sorted(folders)
    for cam_path in folders:
        # get training set
        image_dir = os.path.join(cam_path, 'img1')
        box_gt = np.loadtxt(os.path.join(cam_path, 'gt/gt.txt'), delimiter=',')
        #TODO: get 10% of bag annotations
        GT = get_split(box_gt, train_split=0.8, cam=cam_path.split('/')[-1])

        # open text files for saving detections
        save_dets_dict = {}

        #create and clean path for visualization
        result_path = os.path.join(imgResultDir, cam_path.split('/')[-1])
        if not os.path.exists(result_path):
            os.makedirs(result_path)
        else:
            delete_all(result_path, fmt='png')

        # to save final det from MI-MS
        save_dets_dict['dets_aug'] = open(
            os.path.join(result_path, '{}'.format(cam_path.split('/')[-1] + '_pb_1aug1nms.txt')), mode='w')

        # check image path
        assert image_dir or images
        assert bool(image_dir) ^ bool(images)
        if image_dir:
            imglist = sorted(glob.glob(os.path.join(image_dir, '*.png')))
            #imglist = misc_utils.get_imagelist_from_dir(image_dir)
        else:
            imglist = images
        num_images = len(imglist)

        # loop over all the annotated image
        for i, im_name in enumerate(imglist):
            print('img', im_name)
            im = cv2.imread(im_name)
            fr_num = float(os.path.basename(im_name).split('.')[0])
            #
            if im is not None and fr_num in GT['train_subset']:

                blankImg = im
                rot_imgs_dict = {}
                detPB = []
                maskPB_rot = []
                detPB_rot = []
                det_indexs = []

                for angle in angleSet:
                        if angle>0:
                            print('Rotated by {}'.format(angle))
                            imgrot = imutils.rotate_bound(blankImg, angle)
                        else:
                            imgrot = blankImg
                        rot_imgs_dict[angle] = copy.deepcopy(imgrot)

                        timers = defaultdict(Timer)
                        start_time = time.time()
                        cls_boxes, cls_segms, cls_keyps, cls_segms_coarse = im_detect_all(maskRCNN, imgrot,
                                                                                          timers=timers,
                                                                                          test_aug=False,
                                                                                          soft_nms=True)
                        print("Execution time {} sec".format(time.time() - start_time))
                        if cls_boxes is not None:
                            dets_rot, masks_rot = get_box_mask(fr, cls_boxes, cls_segms, cls_keyps, thresh=det_thr,
                                                           angle=angle, class_list=class_list, dataset=dataset)
                            det_size = sum([len(cls_boxes[cl]) for cl in class_list])
                            #get remapped dets and maintain the index of dets_rot
                            dets = []
                            if det_size > 0:
                                #remap only box without mask
                                dets = vis_utils.return_box(
                                    imgrot[:, :, ::-1],
                                    im[:, :, ::-1],
                                    fr,
                                    angle,
                                    cls_boxes,
                                    dataset=dataset,
                                    class_list=class_list,
                                    feat_type='box',
                                    save_coarse=save_mask,
                                    ext=img_fmt,
                                )
                            raw_det_len = sum([len(cls_boxes[id]) for id in class_list])
                            assert raw_det_len==len(dets) == len(dets_rot),\
                            'found #raw dets: {} #remap dets: {}, #dets_rot: {}'.format(
                                len(cls_boxes), len(dets),len(dets_rot))

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
                            #collect augmented dets to apply MI-MS
                            if (len(dets) > 0):
                                for i, box in enumerate(dets):
                                    # append detections for each orientation
                                    #TODO: obtain det and cluster score from thr score distribution
                                    # we are not using any GT information
                                    all_scores['dets']['det_score'].append(box[6])
                                    all_scores['dets']['class_id'].append(box[7])
                                    all_scores['dets']['frame'].append(fr)
                                    if box[6] >= det_thr and box[7] in class_list:
                                        detPB.append(box)
                                        #maskPB_rot.append(masks_rot[i])
                                        detPB_rot.append(dets_rot[i])


                #apply MS on the augmented det set to get less noisy examples for active learning
                detPB = np.array(detPB)
                detPB_rot = np.array(detPB_rot)
                assert len(detPB)==len(detPB_rot)

                if len(detPB) > 0 and saveAugResult:
                    # regress to reduce noise in remap
                    #apply cluster regression separately foreach class
                    det_cl = {}
                    for cl in class_list:
                        det_cl[cl] = detPB[detPB[:,7]==cl]
                        # regress to reduce noise in remap
                        if len(det_cl[cl])>0:
                            det_cl[cl] = regress_dets(proposals=det_cl[cl], im=im, im_name=im_name, fr=fr, pred_score=0, dataset=dataset,
                                         class_list=class_list, timers=timers, output_dir=output_dir, vis=0, img_fmt='jpg', class_id=cl)
                    #detPB will be updated using regressed dets whatever the objectness score: no filtering using det_thr or nms
                    box_list = [b for _, b in det_cl.items() if len(b) > 0]
                    regressed_dets = np.concatenate(box_list)
                    assert len(detPB) == len(regressed_dets)
                    detPB = regressed_dets
                    assert len(detPB) == len(detPB_rot)

                    #call cluster selection rutine on test-time augmented predictions
                    if fr_num % fr_rate[cam] == 0:
                        show_result = True
                    else:
                        show_result = False
                    MI_MS = Cluster_Mode(detPB, fr_num, angleSet, im,
                                    result_path, save_dets_dict, vis=show_result,
                                    save_modes=True, cluster_scores_thr=cluster_score_thr,
                                    nms=soft_nms, save_scores=all_scores, global_frame=fr, dataset=database)

                    det_indexs, all_scores = MI_MS.get_modes(im_name)
                    print("Execution time with augmentation {} sec".format(time.time() - start_time))

                #here det_indexs will be the position of the selected clusters dets in detPB
                if len(det_indexs) > 0 and len(detPB_rot)>0 and save_data:
                    detf = detPB_rot[det_indexs]
                    #maskf = [maskPB_rot[ind] for ind in det_indexs]
                    #assert len(detf)==len(maskf)
                    print('Frame: {}, #detection: {}'.format(fr, len(detf)))
                    #start loop for all unique angles and save corresponding image and detections
                    #[CXbox, CYbox, fr, detBox[1], x, y, w, h, score, classID, angle]
                    for theta in angleSet:
                        fr_det = detf[detf[:,-1]==theta]
                        # semi-supervision for bag
                        if theta==0 and fr_num in GT['semi_subset_b']:
                            bag_gt = GT['bag_gt'][GT['bag_gt'][:,0]==fr_num]
                            #update fr_det with bag_gt
                            fr_bag_gt = []
                            for i, gtb in enumerate(bag_gt):
                                fr_bag_gt.append([gtb[0], i, gtb[2], gtb[3], gtb[4], gtb[5], 1, 2, 0])
                            fr_det = np.concatenate((fr_det[fr_det[:,7]==1], fr_bag_gt))

                        #fr_mask = [maskf[ind] for ind in np.where(detf[:,-1]==theta)[0]]
                        if len(fr_det)>0:
                            #save image info
                            imgrot = rot_imgs_dict[theta]
                            imgIdnew = 10000 * int('%06d' % fr) + theta
                            imgname = '{:08d}.png'.format(imgIdnew)
                            img_write_path = output_dir + '/' + imgname

                            if not os.path.exists(img_write_path):
                                dataset_clasp = Write_ImagesInfo(imgrot, imgname, int(imgIdnew), dataset_clasp)
                                print('Writing image {}'.format(imgname))
                                cv2.imwrite(img_write_path, imgrot)
                            dataset_clasp = Write_ImagesInfo(imgrot, imgname, int(imgIdnew), dataset_clasp)

                            # save det info
                            for ib,box in enumerate(fr_det):
                                if (box[6]>=det_thr and box[7]==1) or (box[6]>=det_thr and box[7]!=1):
                                    bboxfinal = [round(x, 2) for x in box[2:6]]
                                    if data_type == 'test_augMS' :
                                        #mask = fr_mask[ib]
                                        #area = mask_util.area(mask)#polygon area
                                        assert len(box)==9, 'box {}'.format(box)
                                        #[fr, i, bbox[0], bbox[1], w, h, score, classes[i]]
                                        if box[7]==1:
                                            catID = 1
                                        else:
                                            catID = 2
                                        area = box[4] * box[5]
                                    else:
                                        catID = int(box[9])
                                        area = box[4]*box[5]

                                    annID = 1000 * int('%06d' % (annIDcount)) + theta
                                    annIDcount+=1
                                    #box = mask_util.toBbox(mask)
                                    # convert rle mask into polygon mask
                                    #TODO: try to use rle format in coco annotation format
                                    segmPolys = []  # mask['counts'].decode("utf-8") #[]
                                    #if data_type=='test_augMS':
                                        #bmask = mask_util.decode(mask)
                                        #contours = measure.find_contours(bmask, 0.5)
                                        #contours = cv2.findContours(bmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                                        #for contour in contours:
                                           # contour = np.flip(contour, axis=1)
                                           # segmentation = contour.ravel().tolist()
                                            #segmPolys.append(segmentation)
                                    assert int(imgIdnew)== int(os.path.basename(img_write_path).split('.')[0])
                                    #save annotation infor for each image
                                    dataset_clasp = Write_AnnotationInfo(bboxfinal, segmPolys, int(imgIdnew),
                                                                         int(annID), catID, int(area), dataset_clasp)
                    fr+=1
    return dataset_clasp, all_scores

def delete_all(demo_path, fmt='png'):
    filelist = glob.glob(os.path.join(demo_path, '*.' + fmt))
    if len(filelist) > 0:
        for f in filelist:
            os.remove(f)

if __name__ == '__main__':
    storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61'
    isTrain = False
    # required inputs
    data = 'clasp1'#'COCO'
    model_path = '/home/siddique/PANet/Outputs/'
    cfg_file = '/home/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml'

    if data == 'COCO':
        load_ckpt = '/home/siddique/PANet/panet_mask_step179999.pth'
        class_list = [1, 25, 27, 29]
    else:
        load_ckpt = model_path + 'e2e_panet_R-50-FPN_2x_mask/panet_box_mask_tunedMS_train08/ckpt/model_step19999.pth'
        class_list = [1, 2]

    load_ckpt_frcnn = None  # model_path+'e2e_panet_R-50-FPN_1x_det/panet_box_tuned/ckpt/model_step25524.pth'
    images = False
    set_cfgs = None
    load_detectron = None
    cuda = True
    merge_pdfs = False

    save_data = 1
    test_aug = 1
    soft_nms = 1
    regress_cluster=1

    exp = 10
    det_thr = 0.8
    fr_rate = {9:1,11:1}

    maskRCNN, dataset = configure_detector(data, det_thr)
    for cam in [9, 11]:
        # recommended: pax: 0.4, bag: 0.2
        # higher cluster score decrease the possibility of negative examples from the augmented dets...
        cluster_score_thr = [0.6, 0.4]
        data_type = 'test_augMS'
        if data_type=='test_augMS':
            benchmark = storage + '/tracking_wo_bnw/data/CLASP1/'
            benchmark_path = storage + '/tracking_wo_bnw/data/CLASP1/train_gt/'
            imgResultDir = benchmark + 'test_augMS_gt'
            savefilename = imgResultDir + '/clasp1_test_aug1_{}C{}.json'.format(exp, cam)
            SaveImgDir = imgResultDir + '/img1_{}C{}'.format(exp, cam)
            # to compute the det score and cluster score threshold
            all_scores = {'dets': {'det_score': [], 'class_id': [], 'frame': []},
                          'clusters': {'cluster_score': [], 'class_id': [], 'frame': []}}

            det_score_file = imgResultDir + '/det_scores_clasp1_test_aug1_{}C{}.csv'.format(exp, cam)
            cluster_score_file = imgResultDir + '/cluster_scores_clasp1_test_aug1_{}C{}.csv'.format(exp, cam)


        # clear path files
        if os.path.exists(imgResultDir):
            delete_all(imgResultDir, fmt='png')
        else:
            os.makedirs(imgResultDir)
        if test_aug:
            angleSet = [0, 6, 12, 78, 84, 90, 96, 102, 168, 174, 180, 186, 192, 258, 264, 270, 276, 342, 348, 354]
            saveAugResult=1

        else:
            angleSet = [0]
            saveAugResult = 0

        dataset_clasp = define_dataset_dictionary()
        #prepare self_supervised training data
        benchmark_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/CLASP1/train_gt/'
        folders = glob.glob(benchmark_path + '*{}'.format(cam))
        folders.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))

        dataset_clasp, all_scores = main(maskRCNN, dataset, folders, angleSet, dataset_clasp, SaveImgDir,
                                         isTrain, imgResultDir, soft_nms,class_list, data_type, save_data,
                                         saveAugResult, cluster_score_thr, det_thr, all_scores, database=data,
                                         cam=cam, fr_rate=fr_rate, regress_cluster=regress_cluster)
        if save_data:
            Write_To_Json(savefilename, dataset_clasp)

            Dframe_dets = pd.DataFrame(all_scores['dets'])
            Dframe_dets.to_csv(det_score_file, mode='w', index=False)

            Dframe_clusters = pd.DataFrame(all_scores['clusters'])
            Dframe_clusters.to_csv(cluster_score_file, mode='w', index=False)
        #if train_cam:
            #TODO: use train_net_step_online script to train on each camera in each iteration
            #train_net_step_online()
