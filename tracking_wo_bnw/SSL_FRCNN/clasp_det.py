import multiprocessing as mp
import os
from pathlib import Path
import csv
import time
from os import path as osp
from tkinter import image_names
import psutil
import tracemalloc
import sys
import glob
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.ops.boxes import clip_boxes_to_image, nms
from PIL import Image
import motmetrics as mm
import copy
import imutils

import torchvision
import yaml
import pandas as pd
from tqdm import tqdm
from frcnn_fpn import FRCNN_FPN
from tracktor.utils import delete_all
import pdb
import logging
# from read_config import Configuration
from clustering.get_cluster_mode import Cluster_Mode
from train_step import SST_Model
import configparser
import random
from get_all_params import init_all_params

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  #add ROOT to PATH
# ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


class SSL(object):
    def __init__(self, _config=None, init_flags=None,
                 cam=None, output_dir=None, num_class=2, nms_thr=None,
                 det_pred_thr=0.7, angleSet=None,
                 save_data=None, gpu_test=0, gpu_train=1, dataset=None):
        self._config = _config
        self.cam = cam
        self.num_class = num_class
        self.nms_thr = nms_thr
        self.test_aug_dets = []
        self.test_aug_imgs = []
        self.detection_person_thresh = det_pred_thr

        self.time_total = 0
        self.num_frames = 0
        self.transforms = ToTensor()
        self.out_dir = output_dir
        self.init_flags = init_flags
        self.aug_dets_fr = []
        self.remap_box_fr = []
        self.angle_set = angleSet
        self.save_data = save_data
        self.gpu_test = gpu_test
        self.gpu_train = gpu_train
        self.dataset = dataset
        self.GT = None
        self.gt_vis_path = os.path.join(self.out_dir, 'gt_vis')
        if not os.path.exists(self.gt_vis_path):
            os.makedirs(self.gt_vis_path)
        else:
            delete_all(self.gt_vis_path)

    @staticmethod
    def write_det_results(all_dets, file=None, fr=None, cam=None, appearance=None):
        # this function is used when caresults are save to do offline MCTA association
        all_dets = all_dets.cpu().numpy()
        for i, bb in enumerate(all_dets):
            x1 = bb[0]
            y1 = bb[1]
            x2 = bb[2]
            y2 = bb[3]
            score = bb[4]
            label = bb[5]
            if appearance is None:
                # [fr,id,x,y,w,h]
                t_feature = [fr, i, x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, score, label, cam]
                file.writerow(t_feature)
            else:
                # currently appearance is not used
                app = bb[4::]
                t_feature = [fr, i, x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, score, label, cam]
                file.writerow(t_feature + app.tolist())

    @staticmethod
    def append_det_results(all_dets, pb_aug=None, fr=None, cam=None, appearance=None, angle=None):
        # this function is used to append the augmented detections
        all_dets = all_dets.cpu().numpy()
        for i, bb in enumerate(all_dets):
            x1 = bb[0]
            y1 = bb[1]
            x2 = bb[2]
            y2 = bb[3]
            score = bb[4]
            label = bb[5]
            if appearance is None:
                # format for cluster mode class: [fr, objP, bbox[0], bbox[1], w, h, score, classes[i], angle]
                pb_aug.append([fr, i, x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, score, label, angle])
        return pb_aug

    @staticmethod
    def format_dets(all_dets, fr_num=None, cam=None, appearance=None, angle=None, labels=None, img_HW=None):
        # this function is used to append the augmented detections
        all_dets = all_dets.cpu().numpy()
        labels = labels.cpu().numpy()
        pb_aug = []
        for i, bb in enumerate(all_dets):
            x1 = max(min(img_HW[1], bb[0]), 0)
            y1 = max(min(img_HW[0], bb[1]), 0)
            x2 = max(min(img_HW[1], bb[2]), 0)
            y2 = max(min(img_HW[0], bb[3]), 0)

            score = bb[4]
            if appearance is None:
                # format for cluster mode class: [fr, objP, bbox[0], bbox[1], w, h, score, classes[i], angle]
                pb_aug.append([fr_num, i, x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, score, labels[i], angle])
        return np.array(pb_aug)
    
    @staticmethod
    def rotated_boxes(boxs, img=None, img_rot=None, angle=None):
        # rotated boxes are axis aligned
        #rot_boxes = []
        for i, bb in enumerate(boxs): 
            bb = bb[2:6].astype('int')   
            bImg = np.zeros(shape=[img.shape[0], img.shape[1]], dtype=np.uint8)
            bImg[bb[1]:bb[1]+bb[3], bb[0]:bb[0]+bb[2]] = 255
            mask_rot = imutils.rotate_bound(bImg, angle)
           
            assert mask_rot.shape == img_rot.shape[0:2]
            H,W = img_rot.shape[:2]
            [x, y, w, h] = cv2.boundingRect(mask_rot)
            x = min(W, x)
            x = max(0, x)
            y = min(H, y)
            y = max(0, y)
            boxs[i,2:6] = np.array([x, y, w, h])
        return boxs

    @staticmethod
    def box_image_remap(boxs, remap_aug=None, img_org=None, img_rot=None, fr=None, angle=None):
        # TODO: verify bbox remapping without using segm
        boxs = boxs.cpu().numpy()
        for i, box in enumerate(boxs):
            if angle > 0:
                bb = box[0:4].astype('int')
                bImg = np.zeros(shape=[img_rot.shape[0], img_rot.shape[1]], dtype=np.uint8)
                bImg[bb[1]:bb[3], bb[0]:bb[2]] = 255
                imgrerot = imutils.rotate_bound(bImg, -angle)  # mask_image
                Hrot, Wrot = imgrerot.shape[0] // 2, imgrerot.shape[1] // 2
                H, W = img_org.shape[0] // 2, img_org.shape[1] // 2
                mask_org = imgrerot[Hrot - H: Hrot + H, Wrot - W:Wrot + W]
                [x, y, w, h] = cv2.boundingRect(mask_org)
                x = min(2 * W, x)
                x = max(0, x)
                y = min(2 * H, y)
                y = max(0, y)

                # save masks at multiple inference
                assert (2 * H, 2 * W) == mask_org.shape
                remap_aug.append([fr, i, x, y, w, h, box[4], box[5], angle])
            else:
                remap_aug.append(
                    [fr, i, box[0] + 1, box[1] + 1, box[2] - box[0] + 1, box[3] - box[1] + 1, box[4], box[5], angle])
        return remap_aug

    @staticmethod
    def get_random_angles(ranges=None, factor=None):
        angleSet = [0, 180]
        for intvl in ranges:
            angleSet += random.sample(range(intvl[0], intvl[1]), factor)
        return angleSet

    def vis_gt(self, im, boxs, gt_vis_path=None, imname=None, cam_path=None):
        class_label = {1:'PAX', 2:'TSO'}
        for bb in boxs:
            im = cv2.rectangle(im, (np.int(bb[2]), np.int(bb[3])), (np.int(bb[2]+bb[4]), np.int(bb[3]+bb[5])), (0,255,0), 4)
            if self.num_class==3:
                cat_id = self.get_target_labels(cam_path, bb[1])
            else:
                cat_id = 1

            cv2.putText(im, class_label[cat_id], (int(bb[2]+bb[4]/2), int(bb[3]+bb[5]/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 0), 2)
        cv2.imwrite(os.path.join(gt_vis_path, os.path.basename(imname)), im)

    def get_manual_labels(self, cam_path):
        gt_path = os.path.join(self.init_flags['gt_dir'], cam_path.split('/')[-1], 'gt/gt.txt')
        #gt_path = os.path.join(self.init_flags['gt_dir'], cam_path.split('/')[-1], 'gt_tso/gt.txt')
        print('read gt from: {}'.format(gt_path))
        if os.path.exists(gt_path):
            self.GT = np.loadtxt(gt_path, delimiter=',')
            if self.init_flags['server_loc']=='KRI_exp2_train':
                self.GT[:,0] = self.GT[:,0] + self.init_flags['fr_offset'][cam_path.split('/')[-1]]
            print('gt frames {}'.format(np.unique(self.GT[:,0])))
        else:
            self.GT=[0]

    def init_det_model(self, model_path):
        # object detection

        self.obj_detect = FRCNN_FPN(num_classes=self.num_class,
                                    box_nms_thresh=self.init_flags['nms_thr'],
                                    backbone_type=self.init_flags['backbone'],
                                    pretrained=False)

        if self.init_flags['backbone'] == 'ResNeXt101FPN' and self.init_flags['server_loc'] == 'CLASP2':
            self.obj_detect.load_state_dict(torch.load(self._config['SSL']['obj_detect_model_R101'],
                                                       map_location=lambda storage, loc: storage))

        elif self.init_flags['backbone'] == 'ResNet101FPN' and self.init_flags['server_loc'] == 'CLASP1':
            self.obj_detect.load_state_dict(torch.load(self._config['SSL']['obj_detect_model_R101_clasp1'],
                                                       map_location=lambda storage, loc: storage))
        elif self.init_flags['backbone'] == 'ResNet50FPN' and self.init_flags['server_loc'] == 'MOT20':
            print('load pretrained R50FPN-FRCNN COCO')
            
        elif self.init_flags['backbone'] == 'ResNet34FPN':
            print('num_classes: {},  pretrained model will load before changing the last layers in train_step script'.format(self.num_class))
        
        elif self.num_class<=2:
            self.obj_detect.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
       
        else:
            print('num_classes: {},  pretrained model will load before changing the last layers in train_step script'.format(self.num_class))

        self.obj_detect.eval()
        self.obj_detect.cuda()  # device=self.gpu

        return self.obj_detect

    def predict(self, model, imgrot):
        # original images list
        raw_inputs = [{"image": imgrot, "height": imgrot.shape[0], "width": imgrot.shape[1]}]
        # transformed inputs for model
        image = self.transform_gen.get_transform(imgrot).apply_image(imgrot)
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        net_inputs = [{"image": image, "height": image.shape[1], "width": image.shape[2]}]

        net_img = model.preprocess_image(net_inputs)
        features = model.backbone(img.tensor)
        proposals, _ = model.proposal_generator(img, features, None)
        # images.tensor.shape
        # TODO: proposals willbe replaced by clusters
        # proposals[0].get_fields()['proposal_boxes'][0]
        # cfg.MODEL.PROPOSAL_GENERATOR.NAME
        results, _ = model.roi_heads(net_img, features, proposals)
        outputs = model._postprocess(results, raw_inputs, net_img.image_sizes)

    def get_img_blob(self, path_i, dets=[]):
        """Return the ith image converted to blob"""
        img_org = Image.open(path_i).convert("RGB")
        W, H = img_org.size
        # img = img.resize((540,960))
        img = self.transforms(img_org)

        sample = {}
        sample['img_org'] = cv2.imread(path_i)  # np.asarray(img_org)
        sample['img'] = torch.reshape(img, (1, 3, H, W))
        sample['dets'] = torch.tensor([det[:4] for det in dets])
        sample['img_path'] = path_i
        sample['gt'] = {}
        sample['vis'] = {}

        return sample

    def init_score_saver(self, iteration):
        # to compute the det score and cluster score threshold
        all_scores = {'dets': {'det_score': [], 'class_id': [], 'frame': []},
                      'clusters': {'cluster_score': [], 'class_id': [], 'frame': []}}

        det_score_file = os.path.join(self.out_dir, '/det_scores_iter{}.csv'.format(iteration))
        cluster_score_file = os.path.join(self.out_dir, '/cluster_scores_iter{}.csv'.format(iteration))
        return all_scores, det_score_file, cluster_score_file

    def init_online_dataloader(self, dataset, img_path, fr_num, data_loc, angle=0):
        '''

        :param dataset: 'exp2training'
        :param img_path:
        :param frIndex:
        :return: return a batch of frames to SCT_clasp
        '''
        if data_loc == 'CLASP1':
            if self.cam == 9:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path)
            if self.cam == 2:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path)
            if self.cam == 5:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path)
            if self.cam == 11:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path)
            path_i = os.path.join(img_path, '{:06d}.png'.format(fr_num))
            assert os.path.exists(path_i), '{} does not exist'.format(path_i)

        if data_loc == 'LOGAN':
            path_i = os.path.join(img_path, '{:05d}.jpg'.format(fr_num))
            assert os.path.exists(path_i), '{} does not exist'.format(path_i)

        if data_loc == 'MOT20':
            path_i = os.path.join(img_path, '{:06d}.jpg'.format(fr_num))
            assert os.path.exists(path_i), '{} does not exist'.format(path_i)

        if data_loc == 'CLASP2':
            path_i = os.path.join(img_path, 'img1/{:05d}.jpg'.format(fr_num))

        if data_loc == 'KRI_exp2_train':
            path_i = os.path.join(img_path, 'img1/{:05d}.jpg'.format(fr_num))

        if data_loc == 'PVD':
            path_i = os.path.join(img_path, 'img1/{:06d}.png'.format(int(fr_num)))

        if not osp.exists(path_i):
            return None

        return self.get_img_blob(path_i)

    def get_gt_theta0(self, cam_path, GT=None, frame=None, angle=None):
        if GT is not None:
            gts_i = GT[GT[:,0]==frame]

            if len(gts_i) > 0:
                for i, det in enumerate(gts_i):
                    gts_i[i, 2] = min(self.init_flags['img_HW'][1], gts_i[i, 2])
                    gts_i[i, 2] = max(0, gts_i[i, 2])
                    gts_i[i, 3] = min(self.init_flags['img_HW'][0], gts_i[i, 3])
                    gts_i[i, 3] = max(0, gts_i[i, 3])
        else:
            gts_i = []
        cam = cam_path.split('/')[-3]
        if len(gts_i)==0 and self.init_flags['PANet_detector']:
            dets_i = self.init_flags['PANet_dets'][cam][self.init_flags['PANet_dets'][cam][:, 0] == frame]
            #for tracktor results
            #array([  1,   1,  92, 202, 404, 233,   1,   1, 330])
            if dets_i.shape[1]<7:
                dets_i = np.insert(dets_i, -1, np.ones(len(dets_i)), axis=1)
                dets_i = np.insert(dets_i, -1, np.ones(len(dets_i)), axis=1)
            #for raw PANet dets
            dets_i = dets_i[dets_i[:,7]==self.init_flags['class_id']]
            # convert to x1y1x2y2
            if len(dets_i) > 0:
                print('read PANet dets')
                for i, det in enumerate(dets_i):
                    dets_i[i, 2] = min(self.init_flags['img_HW'][1], dets_i[i, 2])
                    dets_i[i, 2] = max(0, dets_i[i, 2])
                    dets_i[i, 3] = min(self.init_flags['img_HW'][0], dets_i[i, 3])
                    dets_i[i, 3] = max(0, dets_i[i, 3])
        else:
            print('read gts')
            if cam in ['C5']:
                gts_i = np.insert(gts_i, -1, np.ones(len(gts_i)), axis=1)
                gts_i = np.insert(gts_i, -1, np.ones(len(gts_i)), axis=1)
                dets_i = gts_i
            else:
                dets_i = gts_i
        return dets_i

    def save_test_results(self, cam_path, iteration):

        # define camera paths to save test data and update for each camera
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        save_dir = os.path.join(self.out_dir, 'iter{}_{}'.format(iteration, cam_path.split('/')[-1]))

        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            delete_all(save_dir, fmt='jpg')

        # open text files for saving detections
        save_dets_dict = {}
        if self.init_flags['soft_nms']:
            if self.init_flags['test_aug']:
                save_dets_dict['dets_aug'] = open(
                    os.path.join(self.out_dir, 'iter{}_{}'.format(iteration, self.dataset + '_pb_1aug1nms.txt')),
                    mode='w')
            else:
                save_dets_dict['dets'] = open(
                    os.path.join(self.out_dir, 'iter{}_{}'.format(iteration, self.dataset + '_pb_0aug1nms.txt')),
                    mode='w')
        else:
            if self.init_flags['test_aug']:
                save_dets_dict['dets_aug'] = open(
                    os.path.join(self.out_dir, 'iter{}_{}'.format(iteration, self.dataset + '_pb_1aug0nms.txt')),
                    mode='w')
            else:
                save_dets_dict['dets_aug'] = open(
                    os.path.join(self.out_dir, 'iter{}_{}'.format(iteration, self.dataset + '_pb_0aug0nms.txt')),
                    mode='w')
        return save_dir, save_dets_dict

    def get_prev_model(self, iteration):
        if iteration > 0:
            model_path = os.path.join(self.out_dir.split(self.out_dir.split('/')[-1])[0],
                                      'iter{}/models/model_epoch_{}.model'.format(
                                          iteration - 1, self.init_flags['num_epochs'])) #self.init_flags['num_epochs']
            assert os.path.exists(model_path)
            print('load {} for test time augmentation'.format(model_path))
            self.obj_detect = self.init_det_model(model_path)
        else:
            if self.init_flags['server_loc'] in ['LOGAN', 'CLASP1']:
                model_path = os.path.join(ROOT.parent, self._config['SSL']['obj_detect_model_logan'])
                assert os.path.exists(model_path)
                print('load {} at iteration {}'.format(model_path, iteration))
                self.obj_detect = self.init_det_model(model_path)

            if self.init_flags['server_loc'] in ['CLASP2', 'PVD', 'MOT20']:
                model_path = os.path.join(ROOT.parent, self._config['SSL']['obj_detect_model_base'])
                assert os.path.exists(model_path)
                print('load {} at iteration {}'.format(model_path, iteration))
                self.obj_detect = self.init_det_model(model_path)

            if self.init_flags['server_loc'] in ['KRI_exp2_train']:
                model_path = os.path.join(ROOT.parent, self._config['SSL']['obj_detect_model_kri_exp2_train'])
                assert os.path.exists(model_path)
                print('load {} at iteration {}'.format(model_path, iteration))
                self.obj_detect = self.init_det_model(model_path)

    def modes_by_nms(self, det_pos, det_scores, det_labels):
        keep = nms(det_pos, det_scores, self.init_flags['nms_thr'])
        det_pos = det_pos[keep]
        det_scores = det_scores[keep]
        det_labels = det_labels[keep]
        det_labels = torch.tensor(det_labels.cpu().numpy(), dtype=torch.float32).cuda()  # long to float
        camfr_results = torch.cat((det_pos,
                                   det_scores.resize(det_scores.shape[0], 1),
                                   det_labels.resize(det_labels.shape[0], 1)), dim=1)
        return camfr_results

    def get_target_labels(self, cam_path, target_id):
        """To split PAX and TSO class ids
        """
        if cam_path.split('/')[-1] in ['cam02exp2.mp4']:
            if target_id in [2,13,1]:
                class_id = 2
            else:
                class_id = 1
        if cam_path.split('/')[-1] in ['cam09exp2.mp4']:
            if target_id in [1]:
                class_id = 2
            else:
                class_id = 1
        if cam_path.split('/')[-1] in ['cam05exp2.mp4']:
            if target_id in [2,22,8,1]:
                class_id = 2
            else:
                class_id = 1
        if cam_path.split('/')[-1] in ['cam11exp2.mp4']:
            if target_id in [21,22,23]:
                class_id = 2
            else:
                class_id = 1
        if cam_path.split('/')[-1] in ['cam13exp2.mp4']:
            if target_id in [21,1,23]:
                class_id = 2
            else:
                class_id = 1
        if cam_path.split('/')[-1] in ['cam14exp2.mp4']:
            if target_id in [20,4]:
                class_id = 2
            else:
                class_id = 1
        return class_id

    def regress_proposals(self, remap_box_fr, fr_num, angle=0):

        print(f'regressing noisy clusters for theta={angle}...')
        proposals = copy.deepcopy(remap_box_fr[:, 2:6])
        proposals[:, 2:4] = proposals[:, 0:2] + proposals[:, 2:4]
        prop = torch.tensor([[det[0], det[1], det[2], det[3]] for det in proposals])
        prop = prop.squeeze(dim=0)
        self.frame['img_org'] = self.transforms(Image.fromarray(self.rot_imgs_dict[angle]))
        H, W = self.rot_imgs_dict[angle].shape[0:2]
        self.frame['img'] = torch.reshape(self.frame['img_org'], (1, 3, H, W))

        self.obj_detect.load_image(self.frame['img'])
        if len(prop.shape) == 1:
            prop = prop.unsqueeze(dim=0)
        pred_boxs, scores, labels = self.obj_detect.predict_boxes(prop)
        remap_box_fr = torch.cat((pred_boxs,
                                       scores.resize(scores.shape[0], 1)), dim=1)
        assert prop.shape[0] == remap_box_fr.shape[0] # not NMS or thresholding
        # TODO: Instead of clustering, apply nms and low regression score threshold (instead of zero regression score)
        remap_box_fr = self.format_dets(remap_box_fr, fr_num=fr_num, cam=self.cam, angle=angle, labels=labels, img_HW=[H,W])

        return remap_box_fr


    def test_aug(self, dataset=None, server='local', last_frame=2000,
                 iteration=None, vis=False, cluster_score_thr=None):

        #do not need to load model when either PANet dets available
        #in current version we use PANet dets or GT (few shot learning)
        if not self.init_flags['PANet_detector']:
            #load test model for predicting pseudo labels in SSL
            torch.cuda.set_device(self.gpu_test)
            torch.set_num_threads(1)
            self.get_prev_model(iteration)
        else:
            torch.cuda.set_device(self.gpu_train)
            torch.set_num_threads(1)
        # init augmented images path which will use in SSL
        save_aug_imgs = os.path.join(self.out_dir, 'iter{}_imgs'.format(iteration))
        if not os.path.exists(save_aug_imgs):
            os.makedirs(save_aug_imgs)
        else:
            delete_all(save_aug_imgs, fmt='png')

        # init score variables: currently unused
        all_scores, det_score_file, cluster_score_file = self.init_score_saver(iteration)
        fr=1
        for cam_path in self.init_flags['folders']:
            # open text files for saving detections for iter evaluation
            save_dir, save_dets_dict = self.save_test_results(cam_path, iteration)
            # TODO: select X% training frames randomly for few shot learning
            if self.init_flags['GT']:
                self.get_manual_labels(cam_path)
            # all images
            imglist = sorted(glob.glob(os.path.join(cam_path, 'img1/*')))
            print(f'found total {len(imglist)} in {cam_path}')
            # loop over all the annotated image
            for i, im_name in enumerate(imglist):
                fr_num = int(os.path.basename(im_name).split('.')[0])
                # search training set frames for augmented detections
                # train: use only gt frames (each 100th) for SL model, for ssemi-SL: gt + eatch 60th
                assert self.GT is not None
                if fr_num in np.unique(self.GT[:,0]):# or fr_num%60 ==0:
                    self.frame = self.init_online_dataloader(dataset, cam_path, fr_num, data_loc=server)
                    
                    print('cam: {}, img: {}'.format(im_name.split('/')[-3], os.path.basename(im_name)))
                    #iteration=0; no augmented labels; initial model
                    if iteration>0:
                        self.angle_set = [0,90,180]#self.get_random_angles(ranges=self.init_flags['angle_ranges'], factor=2)
                    print(f'random angle set: {self.angle_set}')
                    start_time = time.time()
                    self.rot_imgs_dict = {}
                    self.test_aug_fr = []  # update this attribute for each frame during MI
                    self.remap_box_fr = []

                    for angle in self.angle_set:
                        
                        if angle > 0:
                            print('Rotated by {}'.format(angle))
                            self.frame['img_org'] = imutils.rotate_bound(self.frame['img_org'], angle)
                        
                        self.rot_imgs_dict[angle] = copy.deepcopy(self.frame['img_org'])

                        if fr_num not in np.unique(self.GT[:, 0]): # to predict aug proposals and pseudo labels

                            H, W = self.frame['img_org'].shape[0:2]
                            # convert ot pil image
                            self.frame['img_org'] = self.transforms(Image.fromarray(self.frame['img_org']))
                            self.frame['img'] = torch.reshape(self.frame['img_org'], (1, 3, H, W))

                            #use network prediction when GT or PANet dets are not available
                            #otherwise use only GT or PANet dets
                            self.obj_detect.load_image(self.frame['img'])
                            if iteration == 0:
                                boxes, scores, labels = self.obj_detect.detect(self.frame['img'])
                            else:
                                boxes, scores, labels = self.obj_detect.detect(self.frame['img'])
                            # print('Predicted box: ', boxes)
                            # print('Predicted scores: ', scores)
                            if boxes.nelement() > 0:
                                boxes = clip_boxes_to_image(boxes, self.frame['img'].shape[-2:])
                                # boxes, scores = self.obj_detect.predict_boxes(boxes)
                                # Filter out tracks that have too low person score
                                inds = torch.gt(scores, self.detection_person_thresh).nonzero().view(-1)
                            else:
                                inds = torch.zeros(0).cuda()

                            if inds.nelement() > 0:
                                det_pos = boxes[inds]

                                det_scores = scores[inds]
                                det_labels = labels[inds]
                            else:
                                det_pos = torch.zeros(0).cuda()
                                det_scores = torch.zeros(0).cuda()
                                det_labels = torch.zeros(0).cuda()

                            if det_pos.nelement() > 0:
                                #fr = float(os.path.basename(frame['img_path']).split('.')[0])
                                keep = nms(det_pos, det_scores, self.init_flags['nms_thr'])
                                det_pos = det_pos[keep]
                                det_scores = det_scores[keep]
                                det_labels = det_labels[keep]
                                det_labels = torch.tensor(det_labels.cpu().numpy(), dtype=torch.float32).cuda()  # long to float
                                camfr_results = torch.cat((det_pos,
                                                            det_scores.resize(det_scores.shape[0], 1),
                                                            det_labels.resize(det_labels.shape[0], 1)), dim=1)

                                self.append_det_results(camfr_results, pb_aug=self.test_aug_fr, fr=fr_num, cam=self.cam, angle=angle)
                                self.box_image_remap(camfr_results, remap_aug=self.remap_box_fr, img_org=self.rot_imgs_dict[0],
                                                        img_rot=self.rot_imgs_dict[angle], fr=fr_num, angle=angle)

                    self.test_aug_fr = np.array(self.test_aug_fr)
                    self.remap_box_fr = np.array(self.remap_box_fr)

                    if self.init_flags['regress_remap']:
                        if len(self.remap_box_fr) > 0:
                            self.remap_box_fr = self.regress_proposals(self.remap_box_fr, fr_num, angle=0)
                        else:
                            print(f'no remapped detection found at {im_name}')

                    if len(self.remap_box_fr) > 0:
                        if fr_num % 1 == 0:  # num_frame==fr
                            show_result = True
                        else:
                            show_result = False
                        # apply clustering on augmented dets
                        #TODOL use nms instead of clustering, the n we can skip regression for clustering modes
                        if self.init_flags['modes_by_nms']:
                            if det_pos.nelement() > 0:
                                cluster_modes = self.modes_by_nms(det_pos, det_scores, det_labels)
                            else:
                                cluster_modes = []
                        else:
                            MI_MS = Cluster_Mode(detPB=self.remap_box_fr, frame=fr_num, angle_set=self.angle_set,
                                                 img=copy.deepcopy(self.rot_imgs_dict[0]), out_path=save_dir,
                                                 save_dets_dict=save_dets_dict, bw_type='estimated',
                                                 vis=show_result, save_modes=True, verbose=True,
                                                 cluster_scores_thr=self.init_flags['cluster_score_thr'],
                                                 nms=self.init_flags['soft_nms'], save_scores=all_scores, global_frame=fr)

                            det_indexs, all_scores, cluster_modes = MI_MS.get_modes(self.frame['img_path'])
                    else:
                        det_indexs = []
                        all_scores = []
                        cluster_modes = []

                    # collect selected clusters as training example
                    if len(cluster_modes) > 0 or fr_num in self.GT[:,0]:
                        print('Frame: {}, #unlabeled modes: {}'.format(fr, len(cluster_modes)))
                        # start loop for all unique angles and save corresponding image and detections
                        # [CXbox, CYbox, fr, detBox[1], x, y, w, h, score, classID, angle]
                        #TODO: apply a set of random orientation
                        # self.pseudo_angles = random.sample(self.angle_set, len(self.angle_set)//3)
                        # self.pseudo_labels_angles = [ang for ang in self.pseudo_angles if ang not in [0, 180]]
                        # self.pseudo_labels_angles.extend([0, 180])
                        # self.pseudo_labels_angles = random.sample(self.pseudo_labels_angles, len(self.pseudo_labels_angles))
                        self.pseudo_labels_angles = self.angle_set
                        print(f'found random angles for pseudo labels: {self.pseudo_labels_angles}')

                        if fr_num in self.GT[:,0]:
                            print('get PANet/GT predictions...')
                            cluster_modes = self.get_gt_theta0(im_name, GT=self.GT, frame = fr_num, angle=0)
                            
                        assert len(cluster_modes)>0, f'cluster modes should be found from labeled or unlabeled frames'
                        for theta in self.pseudo_labels_angles:

                            imgrot = self.rot_imgs_dict[theta]
                            if theta!=0:
                                modes_rot = self.rotated_boxes(copy.deepcopy(cluster_modes), \
                                                img=self.rot_imgs_dict[0], img_rot=imgrot, angle=theta)
                                # regress cluster modes
                                fr_det = self.regress_proposals(modes_rot, fr_num, angle=theta)
                                #keep box id for pax and tso
                                if fr_num in self.GT[:,0]:
                                    fr_det[:,1] = cluster_modes[:,1]
                            else:
                                if fr_num in self.GT[:,0]:
                                    fr_det = copy.deepcopy(cluster_modes)
                                else:
                                    fr_det = self.regress_proposals(copy.deepcopy(cluster_modes), fr_num, angle=theta)

                            if len(fr_det) > 0:
                                # save image info
                                imgIdnew = 10000 * int('%06d' % fr) + theta
                                imgname = '{:08d}.png'.format(imgIdnew)
                                img_write_path = os.path.join(save_aug_imgs, imgname)
                                #images dir will be used for training
                                self.test_aug_imgs.append(img_write_path)

                                print('Writing image {}'.format(imgname))
                                cv2.imwrite(img_write_path, imgrot)
                                if fr_num in self.GT[:,0]:
                                    self.vis_gt(self.rot_imgs_dict[theta], fr_det, gt_vis_path=self.gt_vis_path, imname=img_write_path, cam_path=cam_path)
                                # save det info
                                for ib, box in enumerate(fr_det):
                                    #TODO: apply regressed score before selecting the pseudo labels from the unlabeled frames
                                    #if box[6] >= self.init_flags['det_selection_thr']:
                                        #assert len(box) == 9, 'box {}'.format(box)
                                        # [fr, i, bbox[0], bbox[1], w, h, score, classes[i]]
                                    #define class_id for pax and TSO
                                    if self.num_class==3:
                                        catID = self.get_target_labels(cam_path, box[1])
                                    else:
                                        catID = 1
                                    assert catID==1 or catID==2
                                    # if box[7] == 1:
                                    #     catID = 1
                                    # else:
                                    #     catID = 2
                                    print('writing dets for catID {}'.format(catID))
                                    self.test_aug_dets.append(
                                        [imgIdnew, box[1]] + [round(x, 2) for x in box[2:6]] + [1, 1, 1, catID])

                        assert len(self.test_aug_imgs)==len(np.unique(np.array(self.test_aug_dets)[:,0])), \
                                f'found {len(self.test_aug_imgs)}!={np.unique(np.array(self.test_aug_dets)[:,0])}'
                        fr+=1
                        
                        # fps_var.append(1 / (time.time() - start_time))
                        # print('avg speed in {}: {:2f} Hz'
                        #   .format(cam_path.split('/')[-1], np.mean(fps_var)))

        if self.save_data:
            Dframe_dets = pd.DataFrame(all_scores['dets'])
            Dframe_dets.to_csv(det_score_file, mode='w', index=False)

            Dframe_clusters = pd.DataFrame(all_scores['clusters'])
            Dframe_clusters.to_csv(cluster_score_file, mode='w', index=False)

        if self.init_flags['isTrain']:
            
            model_save_path = os.path.join(self.init_flags['output_dir'], 'models')
            if not os.path.exists(model_save_path):
                os.makedirs(model_save_path)
                
            if iteration>0:
                pretrained_model_path = os.path.join(
                    self.init_flags['output_dir'].split('iter{}'.format(iteration))[0],
                    'iter{}'.format(iteration-1), 'models', 'model_epoch_{}.model'.format(self.init_flags['num_epochs'])
                )
                print(f'For training at current iter {iteration}')
                print(f'load iter{iteration - 1} model: {pretrained_model_path}')

            else:
                if self.init_flags['backbone'] in ['ResNet50FPN', 'ResNet101FPN']:
                    pretrained_model_path = self._config['SSL']['obj_detect_model_base']
                else:
                    pretrained_model_path = None
                print('load base model for SSL: {}'.format(pretrained_model_path))

            SST = SST_Model(num_classes=self.num_class, nms_thresh=self.nms_thr, load_iter_checkpoint=iteration,
                            train_gpu=self.gpu_train, backbone=self.init_flags['backbone'],
                            pretrained_model_path=pretrained_model_path,
                            model_save_path=model_save_path, train_data=np.array(self.test_aug_dets),
                            train_imgs=self.test_aug_imgs,
                            num_epochs=self.init_flags['num_epochs'], cam=self.cam)
            SST.train_epochs()


def update_params(storage, benchmark, outdir, init_flags, iteration):

    # 1FPS: GT frames
    if init_params['server_loc']!='PVD':
        # config_file = os.path.join(benchmark, init_flags['dataset'], 'seqinfo.ini')
        # assert os.path.exists(config_file), \
        #     'Path does not exist: {}'.format(config_file)
        # config = configparser.ConfigParser()
        # config.read(config_file)

        #init_flags['img_path'] = os.path.join(benchmark, init_flags['dataset'], 'img1')
        folders = glob.glob(os.path.join(benchmark, '*'))
        folders.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
        init_flags['folders'] = folders
        init_flags['batch_end'] = 13000
        init_flags['gt_dir'] = benchmark


    else:
        #init_flags['img_path'] = os.path.join(benchmark, init_flags['dataset'])
        # test image paths
        folders = glob.glob(os.path.join(benchmark, '*'))
        folders.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
        #1:330, 3:360
        init_flags['folders'] = [folders[i] for i,_ in enumerate(folders) if i in init_flags['train_cams']]
        init_flags['fr_factor'] = {'C360': 1, 'C330': 1}
        #init_flags['folders'] = folders
        #init_flags['fr_factor'] = {'C360': 1, 'C300': 1, 'C340': 1, 'C361': 1, 'C440': 1, 'C330': 1} #{'C1': 30, 'C3': 30, 'C5': 30, 'C8': 40}
        init_flags['gt_dir'] = benchmark

        if init_flags['PANet_detector']:
            init_flags['PANet_dets'] = {}
            # init_params['PANet_dets_dir'] = os.path.join(storage, 'PANet_Results/CLASP1/PANet_det/dets_30fps')
            # init_params['output_dir'] = storage + 'tracking_wo_bnw/output/panet_supervised_clasp1/'
            #init_flags['PANet_dets_dir'] = os.path.join(storage, 'PANet_Results/PVD/PANet_mask')
            init_flags['PANet_dets_dir'] = os.path.join(storage, 'PANet_Results/PVD/PANet_det')
            #init_flags['PANet_dets_dir'] = os.path.join(storage, 'PANet_Results/PVD/HDPVD/PANet_mask')
            #init_flags['PANet_dets_dir'] = os.path.join(storage, 'tracking_wo_bnw/data/PVD/HDPVD_new/train_gt')
            #init_flags['output_dir'] = storage + 'tracking_wo_bnw/output/mrcnn/supervised_clasp1/'
            for cam in init_flags['fr_factor'].keys():
                init_flags['PANet_dets'][cam] = np.loadtxt(os.path.join(init_params['PANet_dets_dir'],
                                                                         cam + '_pb_0aug1nms.txt'), delimiter=',')
                #tracking results
                #init_flags['PANet_dets'][cam] = np.loadtxt(os.path.join(init_params['PANet_dets_dir'],
                                                                         #'{}/1_{}.txt'.format(cam, cam)), delimiter=',')

    #init_flags['dataset'] = 'C{}'.format(cam)
    init_flags['output_dir'] = os.path.join(outdir, 'iter{}'.format(iteration))

    # clear path
    if not osp.exists(init_params['output_dir']):
        os.makedirs(init_params['output_dir'])
    else:
        #TODO: delete all folder
        delete_all(init_params['output_dir'], fmt='jpg')
    if iteration>0:
        init_flags['num_class'] = 2
    else:
        init_flags['num_class'] = 2#81
    return init_flags


if __name__ == '__main__':

    # initialize parameters
    # loop over all datasets
    server = '/media/abubakar/PhD_Backup'
    storage = '/media/abubakar/PhD_Backup'
    database = 'PVD' #'KRI_exp2_train' #'CLASP2' #'PVD' #'CLASP2' #'MOT20'  # 'LOGAN' #'PVD'

    init_params = {}
    #Semisupervised: use both gt and precomputed/augmented detections
    #TODO: apply color jitter and other effective augmentation using more gpus
    init_params['PANet_detector'] = 0
    init_params['onlySL'] = 0
    init_params['GT'] = 1
    init_params['modes_by_nms'] = 0
    init_params['test_aug'] = 0

    init_params, benchmark, out_dir, camlist = init_all_params(init_params, database, storage)
    with open(osp.join(ROOT, 'cfg/SSL.yaml')) as file:
        _config = yaml.load(file, Loader=yaml.FullLoader)
    print(init_params)
    # generate training example for self-supervision
    # start iteration for self-supervision PVD:8-cam330, 9cam360, clasp2: use SL model as initial model
    iteration = 0
    while iteration < 1:
        # if iteration==0:init_params['num_class'] = 81
        # if iteration>0:init_params['num_class'] = 2
        init_params = update_params(storage, benchmark, out_dir, init_params, iteration)

        SelfSupervision = SSL(_config=_config,
                              init_flags=init_params,
                              cam=init_params['cam'],
                              output_dir=init_params['output_dir'],
                              num_class=init_params['num_class'],
                              det_pred_thr=init_params['pred_thr'],
                              angleSet=init_params['angleSet'],
                              save_data=init_params['save_scores'],
                              gpu_test=init_params['gpu_test'],
                              gpu_train=init_params['gpu_train'],
                              dataset=init_params['dataset']
                              )
        #TODO: apply mulitple precess to generate pseudo labels 
        SelfSupervision.test_aug(dataset=init_params['dataset'],
                                 server=init_params['server_loc'],
                                 last_frame=init_params['batch_end'],
                                 iteration=iteration,
                                 vis=True,
                                 )
        #TODO: separate member function for training
        #TODO: implement multi-gpu training
        # TODO: qualitative test after each iteration
        # TODO: release memory for next iter computation
        init_params['cluster_score_thr'][0] += 0.005
        init_params['pred_thr'] += 0.005
        init_params['det_selection_thr'] += 0.005
        iteration += 1


