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
coderoot = os.path.dirname(os.path.realpath(__file__)).split('SSL_PANet')[0] + 'SSL_PANet'
print(f'coderoot:{coderoot}')
sys.path.insert(0, f"{coderoot}")
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
from pathlib import Path
import pdb
import shutil
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


def save_baseline_dets(file, det, angle=None):
    assert angle == 0
    # [fr, objP, bbox[0], bbox[1], w, h, score, classes[i],angle]
    file.writelines(
        str(det[0]) + ',' + str(det[1]) + ',' + str(det[2]) + ',' + str(det[3]) + ',' + str(
            det[4]) + ',' + str(det[5]) + ',' + str(det[6]) + ',' + str(det[7]) + ',' + str(det[8]) + '\n')


def configure_detector(configs):
    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")
    torch.cuda.set_device(gpu_id)
    torch.set_num_threads(1)
    if configs['data'] == 'coco':
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)

    elif configs['data'] == 'clasp2020':
        dataset = datasets.get_clasp_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)

    elif configs['data'] in ['clasp1_2021', 'clasp2_2021']:
        dataset = datasets.get_clasp1_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)

    elif configs['data'] == 'mot_2020':
        dataset = datasets.get_clasp_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)

    elif configs['data'] == 'AppleA_train':
        dataset = datasets.get_AppleA_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)

    elif configs['data'] == "keypoints_coco":
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = 2
    else:
        raise ValueError('Unexpected dataset name: {}'.format(configs['data']))

    print('load cfg from file: {}'.format(configs['cfg_file']))
    cfg_from_file(configs['cfg_file'])
    # set NMS
    cfg['TEST']['NMS'] = configs['nms_thr']
    cfg['TEST']['SCORE_THRESH'] = configs['pred_score']
    if configs['set_cfgs'] is not None:
        cfg_from_list(configs['set_cfgs'])

    assert bool(configs['load_ckpt']) ^ bool(configs['load_detectron']), \
        'Exactly one of --load_ckpt and --load_detectron should be specified.'
    cfg.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS = False  # Don't need to load imagenet pretrained weights
    assert_and_infer_cfg()

    maskRCNN = Generalized_RCNN()

    if configs['cuda']:
        maskRCNN.cuda()

    if configs['load_ckpt']:
        load_name = configs['load_ckpt']
        print("loading checkpoint %s" % (load_name))
        checkpoint = torch.load(load_name, map_location=lambda storage, loc: storage)
        net_utils.load_ckpt(maskRCNN, checkpoint['model'], ckpt_frcnn=False, isTrain=False)

    if configs['load_detectron']:
        print("loading detectron weights %s" % configs['load_detectron'])
        load_detectron_weight(maskRCNN, configs['load_detectron'], isTrain=False)

    maskRCNN = mynn.DataParallel(maskRCNN, cpu_keywords=['im_info', 'roidb'],
                                 minibatch=True, device_ids=[gpu_id])  # only support single GPU

    maskRCNN.eval()

    return maskRCNN, dataset


def convert_from_cls_format(cls_boxes, cls_segms=None, coarse_masks=None, cls_keyps=None):
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


def format_dets(cls_boxes, fr=None, dataset=None, class_list=None, thresh=0.5, class_id=None, angle=None):
    if isinstance(cls_boxes, list):
        # boxes, _,_, _, classes = convert_from_cls_format(cls_boxes)
        boxes = cls_boxes[class_id]
        classes = [class_id] * len(boxes)

    if boxes is None or boxes.shape[0] == 0 or max(boxes[:, 4]) < thresh:
        return

    # Display in largest to smallest order to reduce occlusion
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_inds = np.argsort(-areas)

    dets = []
    objP = 1
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
            objP += 1

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
    # assert len(class_list)==1, 'regression for augmented dets only workd for single class model.. mulit-class: full input proposals are used for all classes'
    assert det_size // len(class_list) == dets_proposals.shape[0], 'proposals size {}, prediction size {}'.format(
        det_size, dets_proposals)

    if det_size > 0:
        dets = format_dets(cls_boxes, fr=fr, dataset=dataset, class_list=class_list,
                           thresh=0, class_id=class_id, angle=0)
    det_class = np.array(dets)
    assert len(det_class) == len(proposals)
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


def main(maskRCNN, dataset, output_dir, image_dir, out_path, angleSet,
         saveAugResult=False, vis=False, img_fmt='png', class_list=None, pred_score=0.5,
         nms_thr=0.3, cam=None, test_time_aug=False, nms=False, database=None, save_mask=False,
         regress_augment=False, cluster_mode_vis=False, vis_rate=10, detector_infer_time=None, detector_test_aug_time=None):
    """main function"""

    # args = parse_args()
    print('Called with args:')
    # print(args)
    image_dir = f'{image_dir}'
    assert image_dir
    if image_dir:
        # imglist = misc_utils.get_imagelist_from_dir(image_dir)
        imglist = glob.glob(f'{image_dir}/*')
        imglist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    else:
        print(f'need valid image dir but found {image_dir}')
    num_images = len(imglist)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # open gt files
    if database in ['clasp1', 'clasp2']:
        GT = np.loadtxt(os.path.join(image_dir.split('img1')[0], 'gt/gt.txt'), delimiter=',')
        # train_set, test_set = get_split(GT, train_split=0.8, cam=cam)
        cam_folder = image_dir.split('/img1')[0].split('/')[-1]
        test_frames = pd.read_csv(os.path.join(image_dir.split('/img1')[0], 'test_frames/{}.csv'.format(cam_folder)))
        test_set = test_frames.values.squeeze()
        print('cam {}, total frames: {}'.format(cam_folder, len(test_set)))

        train_frames = pd.read_csv(os.path.join(image_dir.split('/img1')[0], 'train_frames/{}.csv'.format(cam_folder)))
        train_set = train_frames.values.squeeze()
        # save gt of test-set
        test_gt_path = os.path.join(out_path, '{}'.format(cam + '_test_gt.txt'))
        # if not os.path.exists(test_gt_path):
        test_GT = np.concatenate([GT[GT[:, 0] == t] for t in test_set])
        np.savetxt(test_gt_path, test_GT, delimiter=',', fmt='%d')
    else:
        test_set = [int(os.path.basename(im_name).split('.')[0]) for im_name in imglist]

    # open text files for saving detections
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
    for i in xrange(num_images):
        fr = int(os.path.basename(imglist[i]).split('.')[0])
        im_name = imglist[i]
        if fr % 1 == 0 and fr in test_set:  #fr==7480: #
            print(f'cam {cam} img {os.path.basename(imglist[i])}')
            start_time = time.time()
            im = cv2.imread(imglist[i])
            if im is None:
                print('None image found')
                continue
            # assert im is not None
            # im = undistort_image(im, cam=2)
            PAXdet = []
            PAXmask = []
            detPB = []
            for angle in angleSet:
                if angle != 0:
                    print('Frame {} Rotated by {}'.format(fr, angle))
                    imgrot = imutils.rotate_bound(im, angle)
                else:
                    imgrot = im
                timers = defaultdict(Timer)
                cls_boxes, \
                cls_segms, \
                cls_keyps, \
                cls_segms_coarse = im_detect_all(maskRCNN, imgrot, timers=timers,
                                                 test_aug=test_time_aug, soft_nms=nms,
                                                 nms_thr=nms_thr,
                                                 score_thr=pred_score
                                                 )
                detector_infer_time.append(time.time() - start_time)
                print("Average execution time without augmentation {} sec".format(np.average(detector_infer_time)))


                imgIdnew = 1000 * int('%06d' % fr) + angle
                imgname = os.path.basename(imglist[i])  # str(imgIdnew) + '.'+img_fmt
                # use vis
                det_size = 0
                for class_id in class_list:
                    det_size += len(cls_boxes[class_id])

                if fr % vis_rate == 0 and vis and det_size > 0 and not saveAugResult:
                    print(f' >> save to {output_dir}')
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
                if det_size > 0:
                    dets, _, _ = vis_utils.return_box_mask(
                        imgrot[:, :, ::-1],
                        im[:, :, ::-1],
                        fr,
                        angle,
                        cls_boxes,
                        cls_segms,
                        cls_segms_coarse,
                        dataset=dataset,
                        class_list=class_list,
                        save_coarse=save_mask,
                        thresh=pred_score,
                        ext=img_fmt,
                    )
                    if (dets is not None):
                        if (len(dets) > 0):
                            for i, box in enumerate(dets):
                                if (box[6] >= pred_score and box[7] == 1) or (
                                        box[6] >= pred_score and box[7] != 1):  # 0.8,0.5
                                    if angle == 0 and not saveAugResult:
                                        save_baseline_dets(save_dets_dict['dets'], box, angle=angle)
                                    # append detections for each orientation
                                    detPB.append(box)
                                    # PAXdet_all.append(box)
                                    # PAXmask.append(masks[i])
            if len(detPB) > 0 and saveAugResult:
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
                    # use regressed score: re-scored the augmented det score using org frame features
                    detPB = regressed_dets
                    # wo using regressed score
                    # detPB[:, 0:6] = regressed_dets[:, 0:6]
                # assert len(detPB)==len(PAXmask)
                # save_coarse_mask_PANet(PAXdet,PAXmask,out_path)
                # call mode selection rutine on test-time augmented predictions
                if fr % vis_rate == 0:
                    show_result = vis
                else:
                    show_result = False

                MI_MS = Cluster_Mode(detPB, fr, angleSet, im,
                                     output_dir, save_dets_dict, vis=show_result,
                                     save_modes=True, cluster_scores_thr=[0.1, 0.1],
                                     nms=1, im_name=im_name)
                _, _, cluster_modes = MI_MS.get_modes()
                #print(cluster_modes)

                if cluster_mode_vis:
                    masks_0 = None
                    theta = 0
                    vis_annos_dir = os.path.join(output_dir, cam_folder, 'cluster_mode')
                    if not os.path.exists(vis_annos_dir):
                        os.makedirs(vis_annos_dir)
                    fr_det, fr_mask, _ = get_annos_cluster_mode(maskRCNN, cluster_modes, masks_0, imgrot=im,
                                                                class_list=class_list, im_name=im_name, fr_num=fr,
                                                                pred_score=0,
                                                                dataset=dataset, vis_dir=vis_annos_dir, img_fmt='jpg',
                                                                angle=theta, vis_annos=1)
                    if len(fr_det) > 0:
                        print('collect training examples: frame: {}, #detection: {}, angle: {}'.format(fr,
                                                                                                       len(fr_det),
                                                                                                       theta))
                detector_test_aug_time.append(time.time() - start_time)
                print("Execution time with augmentation {} sec".format(np.average(detector_test_aug_time)))

    if not saveAugResult:
        save_dets_dict['dets'].close()
    # np.savetxt(os.path.join(output_dir, cam + '_person'), PAXdet_all, delimiter=',')


def delete_all(demo_path, fmt='png'):
    filelist = glob.glob(os.path.join(demo_path, '*.' + fmt))
    if len(filelist) > 0:
        for f in filelist:
            os.remove(f)

def parse_args():
    """Parse input arguments"""
    parser = argparse.ArgumentParser(description='Test SSL PANet models')
    
    parser.add_argument(
        '--gpu_id',
        help='select inference gpu id',
        default=0, type=int)
    
    parser.add_argument(
        '--dataset', type=str, default='clasp1',
        help='Dataset to test')
    
    parser.add_argument(
        '--data_dir', type=str, default='/media/6TB_local/tracking_wo_bnw/data/CLASP1/train_gt',
        help='Data directory to test')
    parser.add_argument(
        '--model_dir', type=str, default='/media/6TB_local/PANet_Models/clasp1/modified_loss',
        help='Data directory to test')
    parser.add_argument(
        '--set', dest='set_cfgs',
        help='Set config keys. Key value sequence seperate by whitespace.'
             'e.g. [key] [value] [key] [value]',
        default=[], nargs='+')
    parser.add_argument(
        '--ssl_iter',
        help='SSL iteration index to make sure the pretrained model is loaded properly',
        default=0, type=int)
    
    # parser.add_argument('--model_type', type=str, default='segm')
    # parser.add_argument('--learning_type', type=str, default='semi')
    # parser.add_argument(
    #     '--working_dir', help='checkpoint path to load and save')
    # parser.add_argument(
    #     '--label_percent', help='percent of manual annotations in semi-SL')

    return parser.parse_args()

if __name__ == '__main__':
    from setup_infer_config import get_infer_config
    args = parse_args()
    print('Called with args:')
    print(args)

    if not torch.cuda.is_available():
        sys.exit("Need a CUDA device to run the code.")
        
    gpu_id = args.gpu_id
    storage = coderoot #'/media/6TB_local/SSL_PANet'
    database = args.dataset 
    #model related args 
    configs = get_infer_config(args)
    #init model
    maskRCNN, dataset_obj = configure_detector(configs) 
    
    detector_infer_time = []
    detector_test_aug_time = []
    for cam in configs['cams']:
        if args.dataset == 'clasp1':
            configs['image_dir'] = Path(args.data_dir)/cam/'img1'
            if configs['test_aug']:
                configs['output_dir'] = storage + '/Results/CLASP1/PANet_mask/' + cam + '/resnet50-aug'
            else:
                configs['output_dir'] = storage + '/Results/CLASP1/PANet_mask/' + cam + '/resnet50-infer'
            configs['result_path'] = f"{storage}/Results/CLASP1/PANet_mask"  # det result path
            # det result path for rotation experiment
            #configs['result_path'] = f"{storage}/Results/CLASP1/PANet_mask/rotations_{len(configs['angleSet'])}"
            configs['img_fmt'] = 'png'

        if database in ['clasp2', 'clasp2_30fps']:
            configs['image_dir'] = Path(args.data_dir)/cam/'img1'
            if configs['test_aug']:
                configs['output_dir'] = storage + '/Results/CLASP2/PANet_mask/' + cam + '/resnet50-aug'
            else:
                configs['output_dir'] = storage + '/Results/CLASP2/PANet_mask/' + cam + '/resnet50-infer'
            configs['result_path'] = storage + '/Results/CLASP2/PANet_mask'  # det result path
            #result_path = f"{storage}PANet_Results/CLASP2/PANet_mask/rotations_{len(configs['angleSet'])}"  # det result path for rotation experiment

            if database == 'clasp2_30fps':
                if configs['SSL_alpha_loss']:
                    configs['output_dir'] = storage + '/Results/CLASP2/PANet_mask/SSL_alpha_dets_30fps/' + cam
                elif configs['SSL_loss']:
                    configs['output_dir'] = storage + '/Results/CLASP2/PANet_mask/SSL_dets_30fps/' + cam
                else:
                    configs['output_dir'] = storage + '/Results/CLASP1/PANet_mask/Base_dets_30fps/' + cam

                configs['result_path'] = configs['output_dir']
                print('results will be svaed to: {}'.format(configs['output_dir']))

            configs['img_fmt'] = 'jpg'

        if args.dataset == 'clasp1_30fps':
            configs['image_dir'] = os.path.join(storage, 'data', 'imgs', cam)
            if configs['SSL_alpha_loss']:
                # output_dir = storage + 'PANet_Results/CLASP1/PANet_mask/SSL_alpha_dets_30fps/'+ cam
                output_dir = storage + '/Results/CLASP1/PANet_mask/SSL_wo_reg_alpha_dets_30fps/' + cam
            elif configs['SSL_loss']:
                configs['output_dir'] = storage + '/Results/CLASP1/PANet_mask/SSL_dets_30fps/' + cam
            else:
                configs['output_dir'] = storage + '/Results/CLASP1/PANet_mask/base_dets_30fps/' + cam
            configs['result_path'] = configs['output_dir']
            configs['img_fmt'] = 'png'

        # clear path files
        if os.path.exists(configs['output_dir']):
            delete_all(configs['output_dir'], fmt=configs['img_fmt'])
        # if path not exist: create path
        try:
            shutil.rmtree(configs['result_path'])
        except:
            pass
        if not os.path.exists(configs['result_path']):
            os.makedirs(configs['result_path']) #

        main(maskRCNN, dataset_obj, configs['output_dir'], configs['image_dir'], configs['result_path'],configs['angleSet'],
             saveAugResult=configs['save_aug'], vis=configs['vis'], cam=cam, img_fmt=configs['img_fmt'], class_list=configs['class_list'],
             pred_score=configs['pred_score'], nms_thr=configs['nms_thr'], test_time_aug=configs['test_aug'], nms=configs['soft_nms'], 
             database=database, save_mask=configs['save_mask'], regress_augment=configs['regress_augment'], 
             cluster_mode_vis=configs['cluster_mode_vis'], vis_rate=configs['vis_rate'], detector_infer_time=detector_infer_time, 
             detector_test_aug_time=detector_test_aug_time)
