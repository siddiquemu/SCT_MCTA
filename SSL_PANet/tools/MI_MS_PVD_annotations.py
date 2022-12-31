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

    if data == 'coco':
        dataset = datasets.get_coco_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif data == 'clasp2020':
        dataset = datasets.get_clasp_dataset()
        cfg.MODEL.NUM_CLASSES = len(dataset.classes)
    elif data == 'clasp1_2021':
        dataset = datasets.get_clasp1_dataset()
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

def main(maskRCNN, dataset, folders, angleSet, dataset_clasp, output_dir, isTrain, imgResultDir, soft_nms=False,
         class_list=None, data_type='gt', save_data=False, saveAugResult=False,
         cluster_score_thr=[0,0], det_thr=0.7, all_scores=None):
    """main function"""

    #save all images in one folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # read imgaes from multi-camera clasp2 data folders
    #cam_list = [folders[0],folders[2],folders[4],folders[5],folders[6]]
    fr = 1
    annIDcount = 1
    #folders = sorted(folders)
    for cam_path in [folders]:

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
        image_dir = cam_path
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
            if im is not None:
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
                            #get remapped dets and maintain the index of dets_rot
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
                                save_coarse=False,
                                thresh=det_thr
                                )
                            raw_det_len = sum([len(cls_boxes[id]) for id in class_list])
                            assert raw_det_len==len(dets) == len(masks_rot) == len(dets_rot),\
                            'found #raw dets: {} #remap dets: {}, #masks_rot: {}, #dets_rot: {}'.format(
                                len(cls_boxes), len(dets),len(masks_rot),len(dets_rot))

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
                                    if (box[6] >= 0.8 and box[7] == 1) or (box[6] >= 0.5 and box[7] != 1):
                                        detPB.append(box)
                                        maskPB_rot.append(masks_rot[i])
                                        detPB_rot.append(dets_rot[i])


                #apply MS on the augmented det set to get less noisy examples for active learning
                detPB = np.array(detPB)
                detPB_rot = np.array(detPB_rot)
                assert len(detPB)==len(detPB_rot)

                if len(detPB) > 0 and saveAugResult:
                    # save_coarse_mask_PANet(PAXdet,PAXmask,out_path)
                    #call cluster selection rutine on test-time augmented predictions
                    if fr % 1 == 0:
                        show_result = True
                    else:
                        show_result = False
                    MI_MS = Cluster_Mode(detPB, fr, angleSet, im,
                                    result_path, save_dets_dict, vis=show_result,
                                    save_modes=True, cluster_scores_thr=cluster_score_thr,
                                    nms=soft_nms, save_scores=all_scores, global_frame=fr)
                    det_indexs, all_scores = MI_MS.get_modes(im_name)
                    print("Execution time with augmentation {} sec".format(time.time() - start_time))

                #here det_indexs will be the position of the selected clusters dets in detPB
                if len(det_indexs) > 0 and len(detPB_rot)>0 and save_data:
                    detf = detPB_rot[det_indexs]
                    maskf = [maskPB_rot[ind] for ind in det_indexs]
                    assert len(detf)==len(maskf)
                    print('Frame: {}, #detection: {}'.format(fr, len(detf)))
                    #start loop for all unique angles and save corresponding image and detections
                    #[CXbox, CYbox, fr, detBox[1], x, y, w, h, score, classID, angle]
                    for theta in angleSet:
                        fr_det = detf[detf[:,-1]==theta]
                        fr_mask = [maskf[ind] for ind in np.where(detf[:,-1]==theta)[0]]
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
                                if (box[6]>=0.8 and box[7]==1) or (box[6]>=0.5 and box[7]!=1):
                                    bboxfinal = [round(x, 2) for x in box[2:6]]
                                    if data_type == 'test_augMS' :
                                        mask = fr_mask[ib]
                                        area = mask_util.area(mask)#polygon area
                                        assert len(box)==9, 'box {}'.format(box)
                                        #[fr, i, bbox[0], bbox[1], w, h, score, classes[i]]
                                        if box[7]==1:
                                            catID = 1
                                        else:
                                            catID = 2
                                    else:
                                        catID = int(box[9])
                                        area = box[4]*box[5]

                                    annID = 1000 * int('%06d' % (annIDcount)) + theta
                                    annIDcount+=1
                                    #box = mask_util.toBbox(mask)
                                    # convert rle mask into polygon mask
                                    #TODO: try to use rle format in coco annotation format
                                    segmPolys = []  # mask['counts'].decode("utf-8") #[]
                                    if data_type=='test_augMS':
                                        bmask = mask_util.decode(mask)
                                        contours = measure.find_contours(bmask, 0.5)
                                        #contours = cv2.findContours(bmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

                                        for contour in contours:
                                            contour = np.flip(contour, axis=1)
                                            segmentation = contour.ravel().tolist()
                                            segmPolys.append(segmentation)
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
    data = 'coco'#'clasp1_2021' #'coco'#  # 'clasp1_2021' #'clasp1_2021' #'clasp1_2021' #'coco' #'clasp2020' #
    model_path = '/home/siddique/PANet/Outputs/'
    cfg_file = '/home/siddique/PANet/configs/panet/e2e_panet_R-50-FPN_2x_mask.yaml'

    if data == 'coco':
        load_ckpt = '/home/siddique/PANet/panet_mask_step179999.pth'
        class_list = [1, 27, 25, 29]
    else:
        load_ckpt = model_path + 'e2e_panet_R-50-FPN_2x_mask/panet_box_mask_tunedMS_train07/ckpt/model_step19999.pth'  # 'e2e_panet_R-50-FPN_2x_mask/panet_box_mask_tuned3k/ckpt/model_step9999.pth'
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

    exp = 1
    det_thr = 0.5 #0.5 #6 - 0.6
    maskRCNN, dataset = configure_detector(data, det_thr)
    for cam in [2, 3, 4]:
        # recommended: pax: 0.4, bag: 0.2
        # higher cluster score decrease the possibility of negative examples from the augmented dets...
        cluster_score_thr = [0.25, 0.25] #0.5,0.4
        data_type = 'test_augMS'

        if data_type=='test_augMS':
            benchmark = storage + '/tracking_wo_bnw/data/PVD/'
            benchmark_path = benchmark
            imgResultDir = benchmark + 'test_augMS_gt'
            savefilename = imgResultDir + '/PVD_test_aug1_{}C{}.json'.format(exp, cam)
            SaveImgDir = imgResultDir + '/img1_{}C{}'.format(exp, cam)
            #to compute the det score and cluster score threshold
            all_scores = {'dets':{'det_score':[], 'class_id':[], 'frame':[]},
                          'clusters':{'cluster_score':[], 'class_id':[], 'frame':[]}}

            det_score_file = imgResultDir + '/det_scores_PVD_test_aug1_{}C{}.csv'.format(exp, cam)
            cluster_score_file = imgResultDir + '/cluster_scores_PVD_test_aug1_{}C{}.csv'.format(exp, cam)


        # clear path files
        if os.path.exists(imgResultDir):
            delete_all(imgResultDir, fmt='png')
        else:
            os.makedirs(imgResultDir)
        if test_aug:
            #angleSet =[0, 12, 84, 90, 180, 186, 264, 270, 348, 354]
            angleSet = [0, 6, 12, 78, 84, 90, 96, 102, 168, 174, 180, 186, 192, 258, 264, 270, 276, 342, 348, 354]
            saveAugResult=1

        else:
            angleSet = [0]

        #prepare self_supervised training data
        dataset_clasp = define_dataset_dictionary()
        folders = glob.glob(benchmark_path + 'train/*')
        folders.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))

        dataset_clasp, all_scores = main(maskRCNN, dataset, folders[cam-1], angleSet, dataset_clasp, SaveImgDir,
                                         isTrain, imgResultDir, soft_nms,class_list, data_type, save_data,
                                         saveAugResult, cluster_score_thr, det_thr, all_scores)
        if save_data:
            Write_To_Json(savefilename, dataset_clasp)

            Dframe_dets = pd.DataFrame(all_scores['dets'])
            Dframe_dets.to_csv(det_score_file, mode='w', index=False)

            Dframe_clusters = pd.DataFrame(all_scores['clusters'])
            Dframe_clusters.to_csv(cluster_score_file, mode='w', index=False)
        if train_cam:
            #TODO: use train_net_step_online script to train on each camera in each iteration
            train_net_step_online()
