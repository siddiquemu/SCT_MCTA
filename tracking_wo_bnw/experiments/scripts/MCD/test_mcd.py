import multiprocessing as mp
import os
import csv
import time
from os import path as osp
import psutil
import tracemalloc
import sys
sys.path.insert(0,'/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/experiments/cfgs')
sys.path.insert(1,'/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/experiments/scripts/MCTA')
sys.path.insert(2,'//media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/experiments/src')
sys.path.insert(3,'/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/experiments/scripts')
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchvision.ops.boxes import clip_boxes_to_image, nms
from PIL import Image
import motmetrics as mm

mm.lap.default_solver = 'lap'

import torchvision
import yaml
import pandas as pd
from tqdm import tqdm
import sacred
from sacred import Experiment
from frcnn_fpn_mc import FRCNN_FPN
from tracktor.config import get_output_dir
from tracktor.reid.resnet import resnet50
from tracktor.utils import interpolate, plot_sequence, plot_scanned_track, \
    filter_tracklets, delete_all, get_mot_accum, \
    evaluate_mot_accums, plot_dets
import pdb
import logging
from read_config import Configuration
from project_3D import *
import copy
# TODO:
# 1. replace data loader with clasp batch image reader or online image reader
# 2.
class SCD_Clasp(object):
    def  __init__(self, _config, _log,
                 cam, output_dir, num_class=2,
                 batch=None, gpu=0, min_trklt_size=30,
                init_flags=None):
        self.cam = cam
        self.num_class = num_class
        self.time_total = 0
        self.num_frames = 0
        self.scan_intrvl = batch
        self.gpu = gpu
        self.transforms = ToTensor()
        self.out_dir = output_dir
        self.detector = _config['tracktor']
        self.reid = _config['reid']
        self._config = _config
        self._log = _log
        self.reid_patience = 20
        self.init_flags = init_flags
        self.obj_detect = None
        self.mcd = True
        self.aux_list = [2,3,4,5,6,7]
        self.cam_projection = Projection()
        self.cam_indexs = [1, 2, 3, 4, 5, 6, 7]

        # get camera calibration params
        self.cam_params = {}

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
            #writer.writerow([frame+1, i+1, x1+1, y1+1, x2-x1+1, y2-y1+1, -1, -1, -1, -1])
            #f.write("\n".join(" ".join(map(str, x)) for x in (a, b)))
            if appearance is None:
                #[fr,id,x,y,w,h]
                t_feature = [fr, i , x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, score, cam ]
                file.writerow(t_feature)
            else:
                #currently appearance is not used
                app = bb[4::]
                t_feature = [frame, i , x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, score, cam]
                file.writerow(t_feature+app.tolist())

    def get_cam_params(self):
        for c in self.cam_indexs:
            self.cam_params['C{}'.format(c)] = {}
            self.cam_params['C{}'.format(c)]['A'], self.cam_params['C{}'.format(c)]['rvec'], \
            self.cam_params['C{}'.format(c)]['tvec'], self.cam_params['C{}'.format(c)][
                'dist_coeff'] = self.cam_projection.cam_params_getter(cam=c, isDistorted=False)

    def init_det_model(self):
        # object detection
        self._log.info("Initializing object detector.")

        self.obj_detect = FRCNN_FPN(num_classes=self.num_class)
        self.obj_detect.load_state_dict(torch.load(self._config['tracktor']['obj_detect_wildtrack_model'],
                                                   map_location=lambda storage, loc: storage))
        self.obj_detect.eval()
        self.obj_detect.cuda()#device=self.gpu

    def get_img_blob(self, path_i, dets=[]):
        """Return the ith image converted to blob"""
        img = Image.open(path_i).convert("RGB")
        #img = img.resize((540,960))
        img = self.transforms(img)

        sample = {}
        sample['img'] = torch.reshape(img, (1, 3, int(self.init_flags['img_HW'][0]),
                                            int(self.init_flags['img_HW'][1])))
        sample['dets'] = torch.tensor([det[:4] for det in dets])
        sample['img_path'] = path_i
        sample['gt'] = {}
        sample['vis'] = {}

        return sample

    def init_online_dataloader(self, dataset, cam, img_path, fr_num, data_loc):
        '''

        :param dataset: 'exp2training'
        :param img_path:
        :param frIndex:
        :return: return a batch of frames to SCT_clasp
        '''
        if data_loc == 'wild-track':
            if cam == 1:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'C1')
            if cam == 2:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'C2')
            if cam == 3:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'C3')
            if cam == 4:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'C4')
            if cam == 5:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'C5')
            if cam == 6:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'C6')
            if cam == 7:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'C7')
            path_i = os.path.join(img_path, '{:08d}.png'.format(fr_num))

        if not osp.exists(path_i):
            return None

        return self.get_img_blob(path_i)

    def projected_boxes(self, boxes, box_features, ca=None):
        boxes = boxes.cpu().numpy()
        keep = []
        proj_boxes = []
        for i, bb in enumerate(boxes):
            #project (x1,y1)
            _,  x1y2= self.cam_projection.projectCentroid(copy.deepcopy([bb[0], bb[3]]), self.cam_params, ca=ca, cp=self.cam)

            #project (x2,y2)
            _, x2y2 = self.cam_projection.projectCentroid(copy.deepcopy([bb[2], bb[3]]), self.cam_params, ca=ca, cp=self.cam)

            if 0<=x1y2[0]<=1920 and 0<=x2y2[0]<=1920 and 0<=x2y2[1]<=1080:
                bb[0] = x1y2[0]
                bb[1] = x2y2[1]- (bb[3]-bb[1])
                bb[2] = x2y2[0]
                bb[3] = x2y2[1]
                keep.append(True)
                proj_boxes.append(bb)
            else:
                keep.append(False)

        proj_boxes = torch.tensor(np.array(proj_boxes)).cuda()
        box_features = box_features[keep]
        return proj_boxes, box_features

    def threshold_by_score(self, boxes, scores, frame):
        if boxes.nelement() > 0:
            boxes = clip_boxes_to_image(boxes, frame['img'].shape[-2:])
            # Filter out tracks that have too low person score
            # Computes input>other\text{input} > \text{other}input>other element-wise.
            inds = torch.gt(scores, self.detector['tracker']['detection_person_thresh']).nonzero().view(-1)
        else:
            inds = torch.zeros(0).cuda()

        if inds.nelement() > 0:
            det_pos = boxes[inds]

            det_scores = scores[inds]
        else:
            det_pos = torch.zeros(0).cuda()
            det_scores = torch.zeros(0).cuda()
        return det_pos, det_scores

    def threshold_by_nms(self, boxes, scores):
        if boxes.nelement() > 0:
            keep = nms(boxes, scores, self.detector['tracker']['detection_nms_thresh'])
            det_pos = boxes[keep]
            det_scores = scores[keep]
        return det_pos, det_scores


    def test_SCD(self, start_frame=1, dataset=None,
                 img_path=None, server='local', last_frame=2000, vis=False):

        #init single camera det model
        self.init_det_model()
        #get all camera params
        self.get_cam_params()

        # function to complete tracking by detection for a camera
        # start frame use to do synchronization in mc system
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        save_dir = os.path.join(self.out_dir, 'C{}'.format(self.cam))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        #file to write frame by frame detections
        file = open(osp.join(self.out_dir, f'C{self.cam}.txt'), "w")
        file = csv.writer(file, delimiter=',')

        num_frame = start_frame
        while num_frame <= last_frame:
            frame = self.init_online_dataloader(dataset, self.cam, img_path, num_frame, data_loc=server)
            if frame is None:
                #print('None frame found at {}'.format(num_frame))
                continue
                #break
            print('processing {} in C{}'.format(os.path.basename(frame['img_path']),self.cam))
            if frame is not None:
                mc_proposals = torch.tensor([]).cuda()
                mc_features = torch.tensor([]).cuda()

                self.obj_detect.load_image(frame['img'])
                boxes, scores = self.obj_detect.detect(frame['img'])
                boxes, scores = self.threshold_by_score(boxes, scores, frame)
                boxes, scores = self.threshold_by_nms(boxes, scores)
                mc_proposals = torch.cat((mc_proposals, boxes), dim=0)
                box_features = self.obj_detect.update_mc_features(boxes)
                mc_features = torch.cat((mc_features, box_features), dim=0)
                # print('Predicted box: ', boxes)
                # print('Predicted scores: ', scores)
                if self.mcd:
                    #prepare auxiliary mc-proposals and features
                    #loop for auxiliary cameras: parallel process is recommended
                    aux_frame = {}
                    for ca in self.aux_list:
                        aux_frame[ca] = self.init_online_dataloader(dataset, ca, img_path, num_frame, data_loc=server)
                        self.obj_detect.load_image(aux_frame[ca]['img'])
                        boxes, scores = self.obj_detect.detect(aux_frame[ca]['img'])
                        boxes, scores = self.threshold_by_score(boxes, scores, aux_frame[ca])
                        boxes, scores = self.threshold_by_nms(boxes, scores)
                        #TODO: map the corresponding FPN feature according to the projected proposals
                        #use interpolation to resize the FPN feature
                        box_features = self.obj_detect.update_mc_features(boxes)
                        #TODO: project (x1,y1) and (x2,y2) to primary before augmenting proposals
                        #ca>world>cp
                        boxes, box_features = self.projected_boxes(boxes, box_features, ca)
                        if boxes.nelement()>0:
                            mc_proposals = torch.cat((mc_proposals, boxes), dim=0)
                            mc_features = torch.cat((mc_features, box_features), dim=0)
                    #after finishing the proposals and feature preparation we do the prediction for current frame
                    self.obj_detect.load_image(frame['img'])
                    boxes, scores = self.obj_detect.predict_boxes(mc_proposals, mc_features)
                    boxes, scores = self.threshold_by_score(boxes, scores, frame)
                    det_pos, det_scores = self.threshold_by_nms(boxes, scores)
                    camfr_results = torch.cat((det_pos, det_scores.resize(det_scores.shape[0], 1)), dim=1)
                    fr = float(os.path.basename(frame['img_path']).split('.')[0])
                    self.write_det_results(camfr_results, file, fr, cam=self.cam, appearance=None)

            if det_pos.nelement()>0 and num_frame % 15 == 0:
                plot_dets(num_frame, camfr_results, frame['img_path'], save_dir,cam=self.cam)
            num_frame += 5


if __name__ == '__main__':
    # initialize parameters
    storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/'
    benchmark = storage+'tracking_wo_bnw/data/wild-track/'
    init_params = {}
    init_params['dataset'] = 'wild-track'
    init_params['server_loc'] = 'wild-track'
    init_params['print_stats'] = False
    init_params['global_mcta_graph'] = {}
    init_params['single_cam_multi_asso'] = False
    init_params['isSCT'] = False
    init_params['sequential_id'] = False

    init_params['gpu_id'] = [1, 0, 1]
    init_params['num_class'] = 2

    init_params['img_HW'] = [1080.0, 1920.0]
    init_params['cams'] = [1,2,3]
    # TODO: design cam graph and use BFS to get the connected cams at each batch
    init_params['cam_graph'] = {1: [2]}
    if init_params['server_loc'] == 'wild-track':
        init_params['only_SCTs'] = True
        init_params['img_path'] = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/multicamera_wildtrack/wildtrack/Wildtrack_dataset/Image_subsets'#benchmark+'imgs_30fps'
    #init_params['output_dir'] = storage+'tracking_wo_bnw/output/tracktor_wildtrack/SCT'
    init_params['output_dir'] = storage+'tracking_wo_bnw/output/tracktor-detpropose'

    init_params['last_batch_ids'] = {}
    init_params['current_batch_ids'] = {}
    init_params['new_ids'] = {}
    init_params['prev_affinity'] = {}
    init_params['global_track_id'] = 0
    init_params['keep_raw_id'] = {}
    init_params['batch_size'] = None
    init_params['min_trklt_size'] = 10
    init_params['batch_start'] = 1
    init_params['start_frame'] = 0#1
    init_params['batch_end'] = 1995 #6120
    init_params['save_imgs'] = 1
    init_params['vis_rate'] = 1
    if len(init_params['cams']) ==2:
        init_params['vis_pair'] = 1
    else:
        init_params['vis_pair']=0


    #clear path
    if not osp.exists(init_params['output_dir']):
        os.makedirs(init_params['output_dir'])
    else:
        delete_all(init_params['output_dir'],fmt='jpg')
    #get config
    configuration = []
    with open(osp.join(os.getcwd().split('scripts')[0],'cfgs/tracktor.yaml')) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        _config = yaml.load(file, Loader=yaml.FullLoader)
        with open(osp.join(os.getcwd().split('experiments')[0],_config['tracktor']['reid_config'])) as file1:
            _config.update(yaml.load(file1, Loader=yaml.FullLoader))
    gpu_index = 0
    _log = logging
    single_cam_tracker = SCD_Clasp( _config,
                                   _log,
                                   init_params['cams'][gpu_index],
                                   init_params['output_dir'],
                                   num_class=init_params['num_class'],
                                   batch=init_params['batch_size'],
                                   gpu=init_params['gpu_id'][gpu_index],
                                   min_trklt_size=init_params['min_trklt_size'],
                                    init_flags= init_params
                                    )

    #run single camera detector
    single_cam_tracker.test_SCD(start_frame=init_params['start_frame'],
                                dataset=init_params['dataset'],
                                img_path=init_params['img_path'],
                                server=init_params['server_loc'],
                                last_frame=init_params['batch_end'],
                                vis=True)