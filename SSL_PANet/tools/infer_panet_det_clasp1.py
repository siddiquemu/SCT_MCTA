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

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

import numpy as np
import random
import cv2
import time
import torch
import glob
import torch.nn as nn
from torch.autograd import Variable

import _init_paths
import nn as mynn
from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test import im_detect_box, im_detect_regress
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
import copy
import pandas as pd
# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

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

def configure_detector(data, det_thr=0.5, cuda_id=0, nms_thr=0.3):
    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")

    torch.cuda.set_device(cuda_id)
    torch.set_num_threads(1)

    if data == 'coco':
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)

    elif data == 'PVD_train':
        dataset = datasets.get_PVD_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)

    elif data in ['clasp1_2021', 'clasp2_2021']:
        dataset = datasets.get_clasp1_dataset()
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
    cfg['TEST']['NMS'] = nms_thr
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

    if load_ckpt:
        load_name = load_ckpt
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt_det(maskRCNN, checkpoint['model'])

    if load_detectron:
        print("loading detectron weights %s" % load_detectron)
        load_detectron_weight(maskRCNN, load_detectron, isTrain=False)

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True, device_ids=[cuda_id])  # only support single GPU

    maskRCNN.eval()

    return maskRCNN, dataset

def main(maskRCNN, dataset, output_dir, image_dir, out_path,
         angleSet, det_thr=None, class_list=None, saveAugResult=False,vis=False, img_fmt='png',
         cam=None,test_time_aug=False,nms=False, database=None,
         save_mask=False, regress_aug=False, nms_thr=0.5, pred_score=0.5):
    """main function"""
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
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    #open gt files
    if database in ['clasp1','clasp2']:
        GT = np.loadtxt(os.path.join(image_dir.split('img1')[0], 'gt/gt.txt'), delimiter=',')
        #train_set, test_set = get_split(GT, train_split=0.8, cam=cam)
        cam_folder = image_dir.split('/img1')[0].split('/')[-1]
        test_frames = pd.read_csv(os.path.join(image_dir.split('/img1')[0],'test_frames/{}.csv'.format(cam_folder)))
        test_set = test_frames.values.squeeze()
        print('cam {}, total frames: {}'.format(cam_folder, len(test_set)))
        #save gt of test-set
        test_gt_path = os.path.join(out_path, '{}'.format(cam + '_test_gt.txt'))
        #if not os.path.exists(test_gt_path):
        test_GT = np.concatenate([GT[GT[:,0]==t] for t in test_set])
        np.savetxt(test_gt_path, test_GT, delimiter=',', fmt='%d')
    else:
        test_set = [i for i in range(1, len(imglist)+1)]

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
        if database!='clasp-5k':
            fr = int(os.path.basename(imglist[i]).split('.')[0])
        im_name = imglist[i]
        if fr%1990==0 and fr in test_set: #evaluate only test-set frames
            print('cam {} img {}'.format(cam, os.path.basename(imglist[i])))
            start_time = time.time()
            im = cv2.imread(imglist[i])
            assert im is not None, 'None image found'
            #assert im is not None
            #im = undistort_image(im, cam=2)
            PAXdet = []
            detPB = []
            for angle in angleSet:
                if angle!=0:
                    print('Frame {} Rotated by {}'.format(fr, angle))
                    imgrot = imutils.rotate_bound(im, angle)
                else:
                    imgrot = im
                timers = defaultdict(Timer)
                cls_boxes = im_detect_box(maskRCNN, imgrot, timers=timers,
                                        test_aug=test_time_aug, soft_nms=nms, nms_thr=nms_thr, score_thr=pred_score)
                imgIdnew = 1000*int('%06d'%fr) + angle
                imgname = os.path.basename(imglist[i])#str(imgIdnew) + '.'+img_fmt
                # use vis
                if fr%1==0 and vis and len(cls_boxes[1])>0 and not saveAugResult:
                    vis_utils.vis_clasp(
                        fr,
                        angle,
                        imgrot[:, :, ::-1],  # BGR -> RGB for visualization
                        imgname,
                        output_dir,
                        cls_boxes,
                        dataset=dataset,
                        class_list=class_list,
                        box_alpha=1,
                        show_class=True,
                        thresh=det_thr,
                        kp_thresh=2,
                        ext=img_fmt,
                    )
                if len(cls_boxes[1])>0:
                    dets = vis_utils.return_box(
                        imgrot[:, :, ::-1],
                        im[:,:,::-1],
                        fr,
                        angle,
                        cls_boxes,
                        dataset = dataset,
                        class_list=class_list,
                        feat_type='box',
                        save_coarse=save_mask,
                        ext=img_fmt,
                    )
                    if (dets is not None):
                        if (len(dets) > 0):
                            for i, box in enumerate(dets):
                                if angle == 0 and not saveAugResult:
                                    save_baseline_dets(save_dets_dict['dets'], box, angle=angle)
                                #append detections for each orientation
                                detPB.append(box)
                                #PAXdet_all.append(box)

            if len(detPB)>0 and saveAugResult:
                detPB = np.array(detPB)
                #assert len(detPB)==len(PAXmask)
                #save_coarse_mask_PANet(PAXdet,PAXmask,out_path)
                if regress_aug:
                    # TODO: use all remapped dets as proposals to select the best candidate (filter partial dets)
                    # TODO: convert all dets into torch variable
                    # How to format proposals variable for multi-claass
                    dets_proposals = copy.deepcopy(detPB[:,2:6])
                    dets_proposals[:, 2:4] = dets_proposals[:, 0:2] + dets_proposals[:, 2:4]
                    cls_boxes, cls_segms, \
                    cls_keyps, cls_segms_coarse= im_detect_regress(maskRCNN, im,
                                              box_proposals=dets_proposals,
                                              timers=timers,
                                              test_aug=0,
                                              soft_nms=0,
                                              score_thr=0.5)
                    out_pathreg = os.path.join(output_dir, 'regress_aug')
                    if fr % 100 == 0 and vis and len(cls_boxes[1])+len(cls_boxes[2])> 0 and saveAugResult:
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
                            thresh=det_thr,
                            kp_thresh=2,
                            ext=img_fmt,
                            show_mask=0
                        )

                    if len(cls_boxes[1])+len(cls_boxes[2]) > 0:
                        dets = vis_utils.return_box(
                            imgrot[:, :, ::-1],
                            im[:, :, ::-1],
                            fr,
                            0,
                            cls_boxes,
                            dataset=dataset,
                            class_list=class_list,
                            feat_type='box',
                            save_coarse=save_mask,
                            ext=img_fmt,
                        )
                    detPB = np.array(dets)
                    #assert len(cls_boxes[1])+len(cls_boxes[2])==len(detPB)
                #call mode selection rutine on test-time augmented predictions
                if fr%1==0:
                    show_result=True
                else:
                    show_result = False

                MI_MS = Cluster_Mode(detPB, fr, angleSet, im,
                                     output_dir, save_dets_dict, vis=show_result,
                                     save_modes=True, cluster_scores_thr=[0.4, 0.2],
                                     nms=1, dataset=database)
                det_indexs, _, _ = MI_MS.get_modes(im_name)
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
    storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62/'
    # required inputs
    database = 'clasp2' #'clasp1_30fps'#'clasp2_30fps' #'clasp1_30fps'#'clasp1'#'clasp-5k'#'wild-track'#'clasp-5k'#'wild-track'#'logan'#'kri', 'wild-track'
    baseline = 'ResNet50PANet_det_tuned'#'ResNet50' #'ResNet50'#'ResNet50PANet_det_tuned'#'ResNet50'#'ResNet50_tuned'  # 'ResNeXt101_tuned'
    regress_aug = 0
    test_aug=0
    soft_nms=1
    save_aug =test_aug
    save_mask = 0
    cuda_id =0
    nms_thr=0.5
    # ResNet50
    if baseline == 'ResNet50':
        cfg_file = '/home/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml'
        load_ckpt = '/home/siddique/PANet/panet_mask_step179999.pth'  # baseline
        data = 'coco'
        load_detectron = None
        class_list = [1, 25, 27, 29]
        det_thr = 0.5

    if baseline == 'ResNet50PANet_det_tuned':
        if database in ['clasp1', 'clasp1_30fps']:
            cfg_file = '/home/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_1x_det.yaml'
            load_ckpt = '/home/siddique/PANet/Outputs/e2e_panet_R-50-FPN_1x_det/clasp1/PANet_box_tuned/ckpt/model_step19999.pth'#'/home/siddique/PANet/Outputs/e2e_panet_R-50-FPN_1x_det/panet_box_tuned/ckpt/model_step25524.pth'
            data = 'clasp1_2021'

        if database in ['clasp2', 'clasp2_30fps']:
            cfg_file = '/home/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_1x_det.yaml'
            load_ckpt = f'{storage}/PANet_Models/e2e_panet_R-50-FPN_1x_det/clasp2/PANet_box_tuned/ckpt/model_step19999.pth'  # '/home/siddique/PANet/Outputs/e2e_panet_R-50-FPN_1x_det/panet_box_tuned/ckpt/model_step25524.pth'
            data = 'clasp2_2021'
        if database in ['PVD']:
            cfg_file = '/home/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_1x_det.yaml'
            load_ckpt = '/home/siddique/PANet/Outputs/e2e_panet_R-50-FPN_1x_det/PVD/PANet_box_tuned/ckpt/model_step19999.pth'  # '/home/siddique/PANet/Outputs/e2e_panet_R-50-FPN_1x_det/panet_box_tuned/ckpt/model_step25524.pth'
            data = 'PVD_train'

        load_detectron = None
        class_list = [1,2]
        det_thr = 0.6

    images = False
    set_cfgs = None
    cuda = True
    maskRCNN, dataset = configure_detector(data, det_thr=det_thr,
                                           cuda_id=cuda_id, nms_thr=nms_thr)
    #cams = ['C3']#,'C2','C3', 'C4', 'C5', 'C6', 'C7']
    #cams = ['group_A', 'group_B']
    if database == 'clasp1':
        cams = ['A_11', 'A_9', 'A_11', 'B_9', 'B_11', 'C_9', 'C_11', 'D_9', 'D_11', 'E_9', 'E_11']
    if database == 'clasp2':
        cams = ['H_9']#['G_9', 'G_11', 'H_9', 'H_11', 'I_9', 'I_11']
    if database in ['clasp2_30fps']:
        #cams = ['G_9', 'G_11',  'H_11', 'I_9', 'I_11']
        cams = ['G_2' , 'G_5', 'H_2', 'H_5', 'I_2', 'I_5']
    if database in ['PVD']:
        cams = ['C330' , 'C360']
    if database == 'clasp1_30fps':
        server = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/' \
                 'ourdataset/exp5a/'
        #cams = glob.glob(server+'imgs/*')
        cams = ['cam05exp5a.mp4', 'cam09exp5a.mp4', 'cam02exp5a.mp4', 'cam11exp5a.mp4']

    for cam in cams:
        if database=='wild-track':
            #image_dir = storage+'tracking_wo_bnw/data/wild-track/imgs_30fps/' + cam
            image_dir = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/multicamera_wildtrack/wildtrack/Wildtrack_dataset/Image_subsets/' + cam
            output_dir = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/PANet_Results/wild-track/' + cam + '/resnet50-baseline'
            out_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/PANet_Results/wild-track/' + cam + '/resnet50-baseline-aug' # det result path
            img_fmt='png'

        if database=='clasp1':
            image_dir = storage + 'tracking_wo_bnw/data/CLASP1/train_gt/' + cam + '/img1'
            #output_dir = storage + 'PANet_Results/CLASP1/PANet_det/' + cam + '/resnet50-baseline'
            output_dir = storage + 'PANet_Results/CLASP1/PANet_det/' + cam + '/resnet50-tuned'
            result_path = storage + 'PANet_Results/CLASP1/PANet_det/'
            img_fmt='png'

        if database=='clasp1_30fps':
            image_dir = os.path.join(server, 'imgs', cam)
            #output_dir = storage + 'PANet_Results/CLASP1/PANet_det/dets_base_30fps/'+ cam
            output_dir = storage + 'PANet_Results/CLASP1/PANet_det/SL_dets_30fps/' + cam
            result_path = output_dir
            img_fmt='png'

        if database in ['clasp2', 'clasp2_30fps']:
            image_dir = storage + 'tracking_wo_bnw/data/CLASP/train_gt_all/PB_gt/' + cam + '/img1'
            if test_aug:
                output_dir = storage + 'PANet_Results/CLASP2/PANet_det/' + cam + '/resnet50-aug'
            else:
                output_dir = storage + 'PANet_Results/CLASP2/PANet_det/' + cam + '/resnet50-infer'
            result_path = storage + 'PANet_Results/CLASP2/PANet_det' # det result path
            print('results will be svaed to: {}'.format(output_dir))
            
            if database=='clasp2_30fps':
                result_folder = 'SL_dets_30fps'
                result_path = storage + 'PANet_Results/CLASP2/PANet_det/'+result_folder
                output_dir = result_path + '/' + cam
                print('results will be svaed to: {}'.format(output_dir))
            img_fmt = 'png'

        if database=='PVD':
            image_dir = storage + 'tracking_wo_bnw/data/PVD/HDPVD_new/train_gt/' + cam + '/img1'
            result_folder = 'PVD'
            result_path = storage + 'PANet_Results/PVD/PANet_det/'+result_folder
            output_dir = result_path + '/' + cam
            print('results will be svaed to: {}'.format(output_dir))

            img_fmt='png'

        # clear path files
        if os.path.exists(output_dir):
            delete_all(output_dir, fmt=img_fmt)
        # if path not exist: create path

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if test_aug:
            angleSet = [0, 6, 12, 78, 84, 90, 96, 102, 168, 174, 180, 186, 192, 258, 264, 270, 276, 342, 348, 354]
        else:
            angleSet = [0]

        main(maskRCNN, dataset, output_dir, image_dir, result_path, angleSet, det_thr=det_thr, class_list=class_list,
             saveAugResult=save_aug, vis=True, cam=cam, img_fmt=img_fmt, test_time_aug=test_aug,
             nms=soft_nms, database=database, save_mask=save_mask, regress_aug=regress_aug, nms_thr=nms_thr, pred_score=det_thr)
