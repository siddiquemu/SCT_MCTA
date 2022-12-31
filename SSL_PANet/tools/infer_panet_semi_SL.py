from __future__ import absolute_import
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
import pandas as pd
# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import cv2
import time
import torch
import glob
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
from utils.detectron_weight_helper import load_detectron_weight
from utils.timer import Timer
import imutils
import collections
from get_cluster_mode import Cluster_Mode
import random
import copy
from MI_MS_clasp2_annotations import get_annos_cluster_mode
import pdb
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

def camera_intrinsics(cam):
    # A: 3*3 camera matrix
    # dist_coeff: distortion coefficient k1, k2
    #Cxy = [cx,cy]: optical center
    if cam==9:
        A = np.array([[979.2825, 0, 969.2157],
                      [0, 977.9172, 600.3857],
                      [0.0, 0.0, 1.00]])
        dist_coeff = np.array([-0.2334, 0.0673, 0, 0, 0])  # why 5 coefficients are used in opencv????
    if cam==2:
        A = np.array([[633.2743, 0.0, 972.8379],
                      [0.0, 628.4946, 540.1200],
                      [0.0, 0.0, 1.00]])
        dist_coeff = np.array([-0.0904, 0.0097, 0, 0, 0])
    return dist_coeff, A

def undistort_image(img, cam=None):
    """
    This can straighten out an image given the intrinsic matrix (camera
    matrix) and the distortion coefficients.

    Parameters
    ----------
    img : image
        This is the image we wish to undistort
    camMatrix : 2dmatrix
        This is the intrinsic matrix of the camera (3x3)
    distCoeff : 1darray
        This is the distortion coefficient of the camera (1x5)

    """
    h,  w = img.shape[:2]
    distCoeff, camMatrix = camera_intrinsics(cam)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camMatrix,
                                                      distCoeff,
                                                      (w, h),
                                                      1,
                                                      (w, h)
                                                      )
    # undistort
    dst = cv2.undistort(img, camMatrix, distCoeff, None, newcameramtx)
    #x, y, w, h = roi
    #dst = dst[y:y+h, x:x+w]
    return dst

def save_coarse_mask_PANet(box_remapped_P, mask_remapped_P, output_path):
    data = {
        'info': {},
        'coarse_mask': []
    }
    for i, bbox in enumerate(box_remapped_P):
        # for each bounding box save mask-30*30 (train autoencoder) also TODO: save mask overlay on image 1080*1920
        # PersonDet = [fr, objP, x, y, w, h, score,classes[i],angle]
        info = {
            "class": bbox[7],
            "frame": bbox[0],
            "segmentation": mask_remapped_P[i],
            "angle": bbox[8],
            "box": bbox
        }
        new_dictP = collections.OrderedDict()
        new_dictP["class"] = info["class"]
        new_dictP["frame"] = info["frame"]
        new_dictP["segmentation"] = info["segmentation"]
        new_dictP["angle"] = info["angle"]
        new_dictP["box"] = info["box"]
        data['coarse_mask'].append(new_dictP)
    filename = output_path + 'box_mask_frame' + np.str(bbox[0])
    np.save(filename, data, allow_pickle=True, fix_imports=True)

def save_baseline_dets(file,det,angle=None):
    assert angle==0
    #[fr, objP, bbox[0], bbox[1], w, h, score, classes[i],angle]
    file.writelines(
            str(det[0]) + ',' + str(det[1]) + ',' + str(det[2]) + ',' + str(det[3]) + ',' + str(
                det[4]) + ',' + str(det[5]) + ',' + str(det[6]) + ',' + str(det[7]) + ',' + str(det[8]) + '\n')


def configure_detector(data, pred_score, gpu_id):
    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")
    torch.cuda.set_device(gpu_id)
    torch.set_num_threads(1)
    if data == 'coco':
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)

    elif data == 'clasp2020':
        dataset = datasets.get_clasp_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)

    elif data in ['clasp1_2021', 'clasp2_2021']:
        dataset = datasets.get_clasp1_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)

    elif data == 'mot_2020':
        dataset = datasets.get_clasp_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)

    elif data == 'AppleA_train':
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
    cfg['TEST']['NMS'] = 0.3
    cfg['TEST']['SCORE_THRESH'] = pred_score
    if set_cfgs is not None:
        cfg_from_list(set_cfgs)

    assert bool(load_ckpt) ^ bool(load_detectron), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
    assert_and_infer_cfg()

    maskRCNN = Generalized_RCNN()

    if cuda:
        maskRCNN.cuda()

    if load_ckpt:
        load_name = load_ckpt
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN, checkpoint['model'], ckpt_frcnn=False, isTrain=False)

    if load_detectron:
        print("loading detectron weights %s" % load_detectron)
        load_detectron_weight(maskRCNN, load_detectron, isTrain=False)

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True, device_ids=[gpu_id])  # only support single GPU

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

def regress_dets(maskRCNN, proposals=None, im=None, im_name=None, fr=None, pred_score=None, dataset=None,
                 class_list=None, timers=None, output_dir=None, vis=False, img_fmt='jpg', class_id=None):

    # TODO: use all remapped dets as proposals to select the best candidate (filter partial dets)
    # TODO: convert all dets into torch variable
    # How to format proposals variable for multi-claass
    dets_proposals = copy.deepcopy(proposals[:, 2:6])
    dets_proposals[:, 2:4] = dets_proposals[:, 0:2] + dets_proposals[:, 2:4]
    # TODO: apply det_thr on the regressed augmented detections to reduce noises
    cls_boxes = im_detect_regress(maskRCNN, im,
                                    box_proposals=dets_proposals,
                                    timers=timers,
                                    test_aug=0,
                                    soft_nms=0,
                                    score_thr=0)

    det_size = sum(len(cls_boxes[cl]) for cl in class_list)
    #assert len(class_list)==1, 'regression for augmented dets only workd for single class model.. mulit-class: full input proposals are used for all classes'
    assert det_size//len(class_list)==dets_proposals.shape[0], 'proposals size {}, prediction size {}'.format(det_size, dets_proposals)

    if det_size > 0:
        dets = format_dets(cls_boxes, fr=fr, dataset=dataset, class_list=class_list,
                           thresh=0, class_id=class_id, angle=0)
    det_class = np.array(dets)
    assert len(det_class)==len(proposals)
    return det_class



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

def main(maskRCNN, dataset, output_dir,image_dir,out_path, angleSet,
         saveAugResult=False,vis=False, img_fmt='png', class_list=None, pred_score=0.5, nms_thr=0.3,
         cam=None,test_time_aug=False,nms=False, database=None, save_mask=False,
         regress_augment=False, cluster_mode_vis=False, vis_rate=10):
    """main function"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    #args = parse_args()
    print('Called with args:')
    #print(args)

    assert image_dir or images
    assert bool(image_dir) ^ bool(images)
    if image_dir:
        #imglist = misc_utils.get_imagelist_from_dir(image_dir)
        imglist = glob.glob(image_dir + '/*')
        imglist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    else:
        imglist = images
    num_images = len(imglist)

    #open gt files
    if database in ['clasp1','clasp2']:
        GT = np.loadtxt(os.path.join(image_dir.split('img1')[0], 'gt/gt.txt'), delimiter=',')
        #train_set, test_set = get_split(GT, train_split=0.8, cam=cam)
        cam_folder = image_dir.split('/img1')[0].split('/')[-1]
        test_frames = pd.read_csv(os.path.join(image_dir.split('/img1')[0],'test_frames/{}.csv'.format(cam_folder)))
        test_set = test_frames.values.squeeze()
        print('cam {}, total frames: {}'.format(cam_folder, len(test_set)))


        train_frames = pd.read_csv(os.path.join(image_dir.split('/img1')[0],'train_frames/{}.csv'.format(cam_folder)))
        train_set = train_frames.values.squeeze()
        #save gt of test-set
        test_gt_path = os.path.join(out_path, '{}'.format(cam + '_test_gt.txt'))
        #if not os.path.exists(test_gt_path):
        test_GT = np.concatenate([GT[GT[:,0]==t] for t in test_set])
        np.savetxt(test_gt_path, test_GT, delimiter=',', fmt='%d')
    else:
        test_set = [int(os.path.basename(im_name).split('.')[0]) for im_name in imglist]

    #open text files for saving detections
    save_dets_dict = {}
    if nms:
        if test_time_aug:
            save_dets_dict['dets_aug'] = open(
                os.path.join(out_path, '{}'.format(cam + '_pb_1aug1nms.txt')), mode='w')
        else:
            save_dets_dict['dets'] = open(
                os.path.join(out_path, '{}'.format(cam + '_pb_0aug1nms.txt')), mode='w')
    else:
        if test_time_aug:
            save_dets_dict['dets_aug'] = open(
                os.path.join(out_path, '{}'.format(cam + '_pb_1aug0nms.txt')), mode='w')
        else:
            save_dets_dict['dets_aug'] = open(
                os.path.join(out_path, '{}'.format(cam + '_pb_0aug0nms.txt')), mode='w')

    PAXdet_all = []
    #fr=1
    for i in xrange(num_images):
        if database not in ['clasp-5k', 'DOTA', 'COCO', 'iSAID']:
            fr = int(os.path.basename(imglist[i]).split('.')[0])
        im_name = imglist[i]
        if fr in test_set:# and 0<=fr<=205:
            print('cam {} img {}'.format(cam, os.path.basename(imglist[i])))
            start_time = time.time()
            im = cv2.imread(imglist[i])
            if im is None:
                print('None image found')
                continue
            #assert im is not None
            #im = undistort_image(im, cam=2)
            PAXdet = []
            PAXmask = []
            detPB = []
            for angle in angleSet:
                if angle!=0:
                    print('Frame {} Rotated by {}'.format(fr, angle))
                    imgrot = imutils.rotate_bound(im, angle)
                else:
                    imgrot = im
                timers = defaultdict(Timer)
                cls_boxes, \
                cls_segms, \
                cls_keyps, \
                cls_segms_coarse = im_detect_all(maskRCNN, imgrot,
                                                 timers=timers,
                                                 test_aug=test_time_aug,
                                                 soft_nms=nms,
                                                 nms_thr=nms_thr,
                                                 score_thr=pred_score,
                                                 return_coarse=True
                                                 )
                imgIdnew = 1000*int('%06d'%fr) + angle
                imgname = os.path.basename(imglist[i])#str(imgIdnew) + '.'+img_fmt
                # use vis
                det_size = 0
                for class_id in class_list:
                    det_size += len(cls_boxes[class_id])


                if fr%vis_rate==0 and vis and det_size>0 and not saveAugResult:
                    vis_utils.vis_clasp(
                        fr,
                        angle,
                        imgrot[:, :, ::-1],  # BGR -> RGB for visualization
                        imgname,
                        output_dir,
                        cls_boxes,
                        cls_segms,
                        cls_segms_coarse,
                        cls_keyps,
                        dataset=dataset,
                        class_list=class_list,
                        box_alpha=1,
                        show_class=True,
                        thresh=pred_score,
                        kp_thresh=2,
                        ext=img_fmt,
                        show_mask=1
                    )
                if det_size>0:
                    dets,_,_ = vis_utils.return_box_mask(
                        imgrot[:, :, ::-1],
                        im[:,:,::-1],
                        fr,
                        angle,
                        cls_boxes,
                        cls_segms,
                        cls_segms_coarse,
                        dataset = dataset,
                        class_list=class_list,
                        save_coarse=save_mask,
                        thresh=pred_score,
                        ext=img_fmt,
                    )
                    if (dets is not None):
                        if (len(dets) > 0):
                            for i, box in enumerate(dets):
                                if (box[6] >= pred_score and box[7] == 1) or (box[6] >= pred_score and box[7] != 1): #0.8,0.5
                                    if angle == 0 and not saveAugResult:
                                        save_baseline_dets(save_dets_dict['dets'], box, angle=angle)
                                    #append detections for each orientation
                                    detPB.append(box)
                                    #PAXdet_all.append(box)
                                    #PAXmask.append(masks[i])
            if len(detPB)>0 and saveAugResult:
                detPB = np.array(detPB)
                if regress_augment:
                    # apply cluster regression separately for each class
                    det_cl = {}
                    for cl in class_list:
                        det_cl[cl] = detPB[detPB[:, 7] == cl]
                        # regress to reduce noise in remap
                        if len(det_cl[cl]) > 0:
                            det_cl[cl] = regress_dets(maskRCNN, proposals=det_cl[cl], im=im,
                                                      im_name=os.path.basename(im_name), fr=fr,
                                                      pred_score=0, dataset=dataset,
                                                      class_list=class_list, timers=timers, output_dir=output_dir,
                                                      vis=0, img_fmt='jpg', class_id=cl)
                    # detPB will be updated using regressed dets whatever the objectness score: no filtering using det_thr or nms
                    box_list = [b for _, b in det_cl.items() if len(b) > 0]
                    regressed_dets = np.concatenate(box_list)
                    assert len(detPB) == len(regressed_dets)
                    #use regressed score: re-scored the augmented det score using org frame features
                    detPB = regressed_dets
                    #wo using regressed score
                    #detPB[:, 0:6] = regressed_dets[:, 0:6]
                #assert len(detPB)==len(PAXmask)
                #save_coarse_mask_PANet(PAXdet,PAXmask,out_path)
                #call mode selection rutine on test-time augmented predictions
                if fr%vis_rate==0:
                    show_result = False
                else:
                    show_result = False

                MI_MS = Cluster_Mode(detPB, fr, angleSet, im,
                                     output_dir, save_dets_dict, vis=show_result,
                                     save_modes=show_result, cluster_scores_thr=[0.1, 0.1],
                                     nms=1, im_name=im_name)
                _, _, cluster_modes = MI_MS.get_modes()

                if cluster_mode_vis:
                    masks_0 = None
                    theta = 0
                    vis_annos_dir = os.path.join(output_dir, cam_folder, 'cluster_mode')
                    if not os.path.exists(vis_annos_dir):
                        os.makedirs(vis_annos_dir)
                    fr_det, fr_mask, _ = get_annos_cluster_mode(maskRCNN, cluster_modes, masks_0, imgrot=im,
                                                                class_list=class_list, im_name=im_name, fr_num=fr, pred_score=0,
                                                                dataset=dataset, vis_dir=vis_annos_dir, img_fmt='jpg',
                                                                angle=theta, vis_annos=1)
                    if len(fr_det) > 0:
                        print('collect training examples: frame: {}, #detection: {}, angle: {}'.format(fr,
                                                                                                       len(fr_det),
                                                                                                       theta))
                print("Execution time with augmentation {} sec".format(time.time() - start_time))
                start_time = time.time()
                #issue: cluster score when soft-NMS is not used
                #save predicted mode as final detection
        #fr+=1
    if not saveAugResult:
        save_dets_dict['dets'].close()
    #np.savetxt(os.path.join(output_dir, cam + '_person'), PAXdet_all, delimiter=',')

def delete_all(demo_path, fmt='png'):
    filelist = glob.glob(os.path.join(demo_path, '*.' + fmt))
    if len(filelist) > 0:
        for f in filelist:
            os.remove(f)


if __name__ == '__main__':
    #TODO: multiprocess for multicamera
    #TODO: Combine PANet with DHAE-SCT: jointly-learnable??
    NAS = '/media/siddique/RemoteServer/LabFiles/Walden/trainTestSplit/'
    base_dir = NAS+'test/dataFormattedProperly/splitImages4x3/'

    gpu_id = 0
    storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62/'
    # required inputs
    database = 'clasp2' #'clasp2_30fps' #'clasp1_30fps' #'clasp1' #'flower' # #'PVD' #'clasp1_30fps' #'clasp1' #'clasp2_30fps' #'clasp2' #'clasp1_30fps'#'MOT20'#'iSAID'#'clasp1'#'DAVIS'#'COCO' #'clasp1' #'AICity' #'PVD'#'clasp1' #'DOTA'#'clasp1'#'clasp1'#'clasp-5k'#'wild-track'#'clasp-5k'#'wild-track'#'logan'#'kri', 'wild-track'

    #model and loss
    iter = 7
    percent = 5

    baseline = 'ResNet50_box_mask_tuned' #'ResNet50'#'ResNet50DOTA'#'ResNet50_box_mask_tuned0' #'ResNet50AICity' #'ResNet50_box_mask_tuned0' # 'ResNet50DOTA'#'ResNet50_box_mask_tuned0'#'ResNet50_box_mask_tuned0'#'ResNet50_box_mask_tuned0'#'ResNet50'#'ResNet50_tuned'  # 'ResNeXt101_tuned'
    SSL_alpha_loss = 1
    SSL_alpha_loss_sigm = 0 # continue exp after finishing paper
    SSL_loss = 0
    SL_loss = 0

    #regression
    regress_augment = 1

    #test aug
    test_aug=0
    save_aug =test_aug

    soft_nms=1
    nms_thr=0.5
    save_mask = 0

    mask_model = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62/PANet_Models'
    det_model = '/home/siddique/PANet/Outputs/e2e_panet_R-50-FPN_1x_det'
    pred_score = 0.5

    #vis
    cluster_mode_vis = 1
    vis_rate = 100
    show_dets = True

    if baseline == 'ResNet50':
        cfg_file = '/home/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml'
        load_ckpt = '/home/siddique/PANet/panet_mask_step179999.pth'  #baseline
        data = 'coco'
        class_list = [1]#, 25, 27, 29] #list(np.arange(1,81)) #[1, 25, 27, 29]
        load_detectron = None


    if baseline == 'ResNet50_box_mask_tuned':
        class_list = [1, 2]
        cfg_file = '/home/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml'

        if database in ['clasp2', 'clasp2_30fps']:
            data = 'clasp2_2021'
            if SSL_alpha_loss:
                #SSL-alpha
                load_ckpt =  f'{mask_model}/clasp2/modified_loss_semi/{percent}_percent/iter{iter}/ckpt/model_step19999.pth'
                #load_ckpt = mask_model + '/Jul24-00-31-16_siddiquemu_step/ckpt/model_step19999.pth'
                print('alpha loss model: {}'.format(load_ckpt))

            elif SSL_alpha_loss_sigm:
                #load_ckpt =  mask_model + '/clasp2/loss_sigmoid/iter{}/ckpt/model_step19999.pth'.format(iter)
                load_ckpt = mask_model + '/Jul29-01-14-32_siddiquemu_step/ckpt/model_step9999.pth'
                print('sigmoid alpha loss model: {}'.format(load_ckpt))

            elif SL_loss:
                # SL loss
                load_ckpt = det_model + '/PANet_box_tuned/ckpt/model_step19999.pth'
                print('SL loss model: {}'.format(load_ckpt))

            elif SSL_loss:
                #SSL loss
                load_ckpt = mask_model + '/clasp2/wo_score_regress_loss/iter{}/ckpt/model_step19999.pth'.format(iter)
                print('default loss model: {}'.format(load_ckpt))
            else:
                print('base model read separately')



        if database in ['clasp1', 'clasp1_30fps']:
            data = 'clasp1_2021'
            if SSL_alpha_loss:
                load_ckpt =  f'{mask_model}/clasp1/modified_loss_semi/{percent}_percent/iter{iter}/ckpt/model_step19999.pth'
                #load_ckpt = mask_model + '/clasp1/modified_loss_wo_reg/iter{}/ckpt/model_step19999.pth'.format(iter)
                #load_ckpt =  mask_model + '/clasp1/modified_loss/iter{}/ckpt/model_step19999.pth'.format(iter)
                #load_ckpt = mask_model + '/Jul24-00-31-16_siddiquemu_step/ckpt/model_step19999.pth'
                print('alpha loss model: {}'.format(load_ckpt))
            elif SSL_alpha_loss_sigm:
                load_ckpt =  mask_model + '/clasp1/loss_sigmoid/iter{}/ckpt/model_step19999.pth'.format(iter)
                #load_ckpt = mask_model + '/Jul29-01-14-32_siddiquemu_step/ckpt/model_step4854.pth'
                print('sigmoid alpha loss model: {}'.format(load_ckpt))
            elif SSL_loss:
                load_ckpt = mask_model + '/clasp1/wo_score_regress_loss/iter{}/ckpt/model_step19999.pth'.format(iter)
            else:
                load_ckpt = mask_model + 'base model'.format(iter)


        if database=='PVD':
            load_ckpt = mask_model + '/PVD_tuned00/ckpt/model_step12898.pth'
            data = 'clasp1_2021'
            class_list = [1, 2]

        if database=='flower':
            load_ckpt = mask_model + '/AppleA_train/iter{}/ckpt/model_step19999.pth'.format(iter)
            data = 'AppleA_train'
            class_list = [1]

        if database=='MOT20':
            load_ckpt = mask_model + '/MOT20Det_iter1/ckpt/model_step29842.pth'
            data = 'mot_2020'
            class_list = [1]

        load_detectron = None

    if baseline == 'ResNet50DOTA':
        cfg_file = '/home/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml'
        load_ckpt = '/home/siddique/PANet/panet_mask_step179999.pth'  # baseline
        data = 'coco'
        class_list = list(np.arange(0, 81))
        load_detectron = None

    images = False
    set_cfgs = None
    cuda = True
    maskRCNN, dataset = configure_detector(data, pred_score, gpu_id)
    #cams = ['C3']#,'C2','C3', 'C4', 'C5', 'C6', 'C7']
    #cams = ['group_A', 'group_B'] , 'C4']
    if database=='MOTS':
        cams = ['0002']
    if database=='clasp1':
        cams = ['A_9', 'E_9', 'A_11', 'B_11', 'B_9', 'C_11', 'C_9', 'D_11', 'D_9', 'E_11']

    if database in ['clasp2_30fps']:
        cams = ['G_2', 'G_5', 'H_2', 'H_5', 'I_2', 'I_5']#, 'G_9', 'G_11', 'H_9', 'H_11', 'I_9', 'I_11'] #['G_2', 'G_5', 'H_2', 'H_5', 'I_2', 'I_5']
    if database in ['clasp2']:
        cams = ['G_9', 'G_11', 'H_9', 'H_11', 'I_9', 'I_11']

    if database=='PVD':
        cams =  ['C330']#,'C2','C3', 'C4', 'C5', 'C6']
    if database=='flower':
        cams =  ['flowersSplit']

    if database == 'clasp1_30fps':
        subset = 'exp9a'
        server = f'/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/ourdataset/{subset}/'
        #cams = glob.glob(server+'imgs/*')
        cams = [f'cam05{subset}.mp4', f'cam09{subset}.mp4', f'cam02{subset}.mp4', f'cam11{subset}.mp4']

    for cam in cams:
        if database == 'wild-track':
            #image_dir = storage+'tracking_wo_bnw/data/wild-track/imgs_30fps/' + cam
            image_dir = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/multicamera_wildtrack/wildtrack/Wildtrack_dataset/Image_subsets/' + cam
            output_dir = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/PANet_Results/wild-track/' + cam + '/resnet50-baseline'
            out_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/PANet_Results/wild-track/' + cam + '/resnet50-baseline-aug' # det result path
            img_fmt='png'

        if database == 'kri':
            image_dir = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp1/imgs/' + cam
            output_dir = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp1/Results/' + cam + '/panet_tuned_rotationv2_angle0_180'
            out_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp1/Results/' + cam + '/box_mask_aug_panetv2/'
        if database=='logan':
            save_dir = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d6/tracking_wo_bnw'
            image_dir = save_dir + '/data/logan-data/exp1-train/' + cam
            output_dir = save_dir + '/output/PANet-det/logan/' + cam + '/panet_tuned_rotationv2_angle0'
            out_path = save_dir + '/output/PANet-det/logan/' + cam + '/box_mask_aug_panetv2/'
        if database=='clasp-5k':
            save_dir = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw'
            image_dir = save_dir + '/data/5k-data/'+cam
            output_dir = save_dir + '/output/PANet-det/5k-data/' +cam+ '/panet_tuned_rotationv2_angle0'
            out_path = save_dir + '/output/PANet-det/5k-data/'+cam+ '/box_mask_aug_panetv2/'
            img_fmt='jpg'

        if database=='clasp1':
            image_dir = storage + 'tracking_wo_bnw/data/CLASP1/train_gt/' + cam + '/img1'
            if test_aug:
                output_dir = storage + 'PANet_Results/CLASP1/PANet_mask/' + cam + '/resnet50-aug'
            else:
                output_dir = storage + 'PANet_Results/CLASP1/PANet_mask/' + cam + '/resnet50-infer'
            result_path = storage + 'PANet_Results/CLASP1/PANet_mask' # det result path
            img_fmt='png'


        if database in ['clasp2', 'clasp2_30fps']:
            #image_dir = storage + 'tracking_wo_bnw/data/CLASP/train_gt_all/PB_no_gt/' + cam + '/img1'
            image_dir = storage + 'tracking_wo_bnw/data/CLASP/train_gt_all/PB_gt/' + cam + '/img1'
            if test_aug:
                output_dir = storage + 'PANet_Results/CLASP2/PANet_mask/' + cam + '/resnet50-aug'
            else:
                output_dir = storage + 'PANet_Results/CLASP2/PANet_mask/' + cam + '/resnet50-tunedms'
            result_path = storage + 'PANet_Results/CLASP2/PANet_mask' # det result path

            if database=='clasp2_30fps':
                if SSL_alpha_loss:
                    output_dir = storage + 'PANet_Results/CLASP2/PANet_mask/SSL_alpha_dets_30fps_semi/' + cam
                elif SSL_loss:
                    output_dir = storage + 'PANet_Results/CLASP2/PANet_mask/SSL_dets_30fps/' + cam
                else:
                    output_dir = storage + 'PANet_Results/CLASP1/PANet_mask/Base_dets_30fps/' + cam

                result_path = output_dir
                print('results will be svaed to: {}'.format(output_dir))

            img_fmt='jpg'

        if database=='clasp1_30fps':
            image_dir = os.path.join(server, 'imgs', cam)
            if SSL_alpha_loss:
                #output_dir = storage + 'PANet_Results/CLASP1/PANet_mask/SSL_alpha_dets_30fps/'+ cam
                #output_dir = storage + 'PANet_Results/CLASP1/PANet_mask/SSL_wo_reg_alpha_dets_30fps/' + cam
                output_dir = storage + 'PANet_Results/CLASP1/PANet_mask/SSL_alpha_dets_30fps_semi/'+ cam
            elif SSL_loss:
                output_dir = storage + 'PANet_Results/CLASP1/PANet_mask/SSL_dets_30fps/' + cam
            else:
                output_dir = storage + 'PANet_Results/CLASP1/PANet_mask/base_dets_30fps/' + cam
            result_path = output_dir
            img_fmt='png'

        if database=='PVD':
            image_dir = storage + 'tracking_wo_bnw/data/PVD/HDPVD_new/train_gt/' + cam + '/img1'
            if test_aug:
                output_dir = storage + 'PANet_Results/PVD/HDPVD_new/PANet_mask/' + cam + '/resnet50-base'
            else:
                output_dir = storage + 'PANet_Results/PVD/HDPVD_new/PANet_mask/' + cam + '/resnet50-aug'
            result_path = storage + 'PANet_Results/PVD/HDPVD_new/PANet_mask' # det result path
            img_fmt='png'

        if database=='flower':
            image_dir = base_dir + cam
            if test_aug:
                output_dir = storage + 'PANet_Results/AppleA_train/PANet_mask/' + cam + '/resnet50-base'
            else:
                output_dir = storage + 'PANet_Results/AppleA_train/PANet_mask/' + cam + '/resnet50-aug'
            result_path = storage + 'PANet_Results/AppleA_train/PANet_mask' # det result path
            img_fmt='png'

        if database=='AICity':
            image_dir = storage + 'tracking_wo_bnw/data/AICityChallenge/AIC21_Track3_MTMC_Tracking/' \
                                  'AIC21_Track3_MTMC_Tracking/validation/S02/' + cam + '/img1'
            if test_aug:
                output_dir = storage + 'PANet_Results/AICity/PANet_mask/' + cam + '/resnet50-base'
            else:
                output_dir = storage + 'PANet_Results/AICity/PANet_mask/' + cam + '/resnet50-tuned'
            result_path = storage + 'PANet_Results/AICity/PANet_mask' # det result path
            img_fmt='png'

        if database=='COCO':
            image_dir = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/coco_images/coco_2017/' + cam
            if test_aug:
                output_dir = storage + 'PANet_Results/COCO/PANet_mask/' + cam + '/resnet50-base'
            else:
                output_dir = storage + 'PANet_Results/COCO/PANet_mask/' + cam + '/resnet50-tuned'
            result_path = storage + 'PANet_Results/COCO/PANet_mask' # det result path
            img_fmt='jpg'

        if database=='DAVIS':
            image_dir = storage + 'tracking_wo_bnw/data/DAVIS/DAVIS/JPEGImages/480p/'+cam
            if test_aug:
                output_dir = storage + 'PANet_Results/DAVIS/PANet_mask/' + cam + '/resnet50-base'
            else:
                output_dir = storage + 'PANet_Results/DAVIS/PANet_mask/' + cam + '/resnet50-tuned'
            result_path = storage + 'PANet_Results/DAVIS/PANet_mask' # det result path
            img_fmt='jpg'

        if database=='iSAID':
            image_dir = storage+'tracking_wo_bnw/data/iSAID/'+cam+'/images'
            if test_aug:
                output_dir = storage + 'PANet_Results/iSAID/PANet_mask/' + cam + '/resnet50-base'
            else:
                output_dir = storage + 'PANet_Results/iSAID/PANet_mask/' + cam + '/resnet50-tuned'
            result_path = storage + 'PANet_Results/iSAID/PANet_mask' # det result path
            img_fmt='png'

        if database=='MOTS':
            image_dir = storage+'Mask_Instance_Clustering/USC_MOTS_bitbucket/usc_mots/data/MOT17/imgs/'+cam
            if test_aug:
                output_dir = storage + 'PANet_Results/MOT20/PANet_mask/' + cam + '/resnet50-aug'
            else:
                output_dir = storage + 'PANet_Results/MOTS/PANet_mask/' + cam + '/resnet50-tuned'
            result_path = storage + 'PANet_Results/MOTS/PANet_mask' # det result path
            img_fmt='jpg'

        # clear path files
        if os.path.exists(output_dir):
            delete_all(output_dir, fmt=img_fmt)
        # if path not exist: create path
        if not os.path.exists(result_path):
            os.makedirs(result_path)

        if test_aug:
            if database in ['clasp1', 'clasp2']:
                #angleSet = [0, 6, 12, 78, 84, 90, 96, 102, 168, 174, 180, 186, 192, 258, 264, 270, 276, 342, 348, 354] #[0, 6, 12, 90, 96, 180, 186, 270, 348, 354]#
                #angleSet = [0, 6, 12, 18, 24, 72, 78, 84, 90, 96, 102, 162, 168, 174, 180, 186, 192, 252, 258, 264, 270, 276, 272,336, 342, 348, 354]
                angleSet = [0, 6, 12, 78, 84, 90, 96, 102, 168, 174, 180, 186, 192, 258, 264, 270, 276, 342, 348, 354]
                #angleSet = list(np.arange(0, 360, 3))
            elif database=='MOT20':
                angleSet = [0, 3, 6, 9, 12, 15, 18, 21, 336, 339, 342, 345, 348, 351, 354]
                angleSet = [0, 6, 12, 18, 24, 78, 174, 180, 186, 192, 330, 336, 342, 348, 354]
            else:
                angleSet = [0, 6, 12, 78, 84, 90, 96, 102, 168, 174, 180, 186, 192, 258, 264, 270, 276, 342, 348, 354]
        else:
            angleSet = [0]
        main(maskRCNN, dataset, output_dir, image_dir, result_path, angleSet,
             saveAugResult=save_aug, vis=show_dets, cam=cam, img_fmt=img_fmt, class_list=class_list,
             pred_score=pred_score, nms_thr=nms_thr, test_time_aug=test_aug, nms=soft_nms, database=database,
             save_mask=save_mask, regress_augment=regress_augment, cluster_mode_vis=cluster_mode_vis,
             vis_rate=vis_rate)
