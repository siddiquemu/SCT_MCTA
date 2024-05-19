import multiprocessing as mp
import os
import csv
import time
from os import path as osp
import psutil
import tracemalloc
import sys
sys.path.insert(0,'/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62/tracking_wo_bnw/experiments/cfgs_clasp')
sys.path.insert(1,'/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62/tracking_wo_bnw/experiments/scripts/MCTA')
sys.path.insert(2,'/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62/tracking_wo_bnw/src')
sys.path.insert(3,'/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62/tracking_wo_bnw/experiments/scripts/MCD')

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
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.config import get_output_dir
#from tracktor.datasets.factory import Datasets
from tracktor.oracle_tracker import OracleTracker
from tracktor.tracker import Tracker
##from tracktor.tracker_reid_ot import Tracker
from tracktor.reid.resnet import resnet50
from tracktor.utils import interpolate, plot_sequence, plot_scanned_track, \
    filter_tracklets, delete_all, get_mot_accum, \
    evaluate_mot_accums, plot_dets
import pdb
import logging
import random
random.seed(1234)
from read_config import Configuration
from MCTA.MCTA_Online import MCTA
from MCD.get_cluster_mode import Cluster_Mode
import configparser
# from pypapi import events, papi_high as high



# TODO:
# 1. replace data loader with clasp batch image reader or online image reader
# 2.
class SCT_Clasp(object):
    def  __init__(self, _config, _log,
                 cam, output_dir, num_class=2,
                 batch=None, gpu=0, min_trklt_size=30,
                  global_track_num=None, global_track_start_time=None, init_flags=None):
        self.cam = cam
        self.num_class = num_class
        self.time_total = 0
        self.num_frames = 0
        self.scan_intrvl = batch
        self.gpu = gpu
        self.transforms = ToTensor()
        self.out_dir = output_dir
        self.tracktor = _config['tracktor']
        self.reid = _config['reid']
        self._config = _config
        self.global_track_num = global_track_num
        self.global_track_start_time = global_track_start_time
        self._log = _log
        self.reid_patience = 20
        self.min_trklt_size = min_trklt_size
        self.init_flags = init_flags
        self.precomputed_dets = []
    @staticmethod
    def _format_global_batch(global_batch, cam_list, velocity=None, appearance=None):
        #We generally use static methods to create utility functions
        global_feature = []
        for cam in cam_list:
            for i, track in global_batch[cam].items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    score = bb[4]
                    if velocity and appearance is not None:
                        v_cx = bb[5]
                        v_cy = bb[6]
                        app = bb[7::]
                    # [fr,id,x,y,w,h]
                    global_feature.append([frame + 1, i, x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, cam])
        return np.array(global_feature)

    @staticmethod
    def write_cam_results(all_tracks, output_dir, dataset=None, class_id=None, min_track_size=10, cam=None, appearance=None):
        # this function is used when caresults are save to do offline MCTA association
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file = osp.join(output_dir, dataset, f'{class_id}_C{cam}.txt')

        print("[*] Writing tracks to: {}".format(file))

        with open(file, "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():

                if len(list(track.keys()))>min_track_size:
                    for frame, bb in track.items():
                        x1 = bb[0]
                        y1 = bb[1]
                        x2 = bb[2]
                        y2 = bb[3]
                        score = bb[4]
                        #writer.writerow([frame+1, i+1, x1+1, y1+1, x2-x1+1, y2-y1+1, -1, -1, -1, -1])
                        #f.write("\n".join(" ".join(map(str, x)) for x in (a, b)))
                        if appearance is None:
                            #[fr,id,x,y,w,h,score,cam]
                            t_feature = [frame+1, i , x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, cam ]
                            writer.writerow(t_feature)
                        else:
                            v_cx = bb[5]
                            v_cy = bb[6]
                            app = bb[7::]
                            t_feature = [frame, i , x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, -1, -1, -1, -1, v_cx, v_cy, cam]
                            writer.writerow(t_feature+app.tolist())
        print('[*] writing finished ...')

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
            #writer.writerow([frame+1, i+1, x1+1, y1+1, x2-x1+1, y2-y1+1, -1, -1, -1, -1])
            #f.write("\n".join(" ".join(map(str, x)) for x in (a, b)))
            if appearance is None:
                #[fr,id,x,y,w,h]
                t_feature = [fr, i , x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, score, label, cam ]
                file.writerow(t_feature)
            else:
                #currently appearance is not used
                app = bb[4::]
                t_feature = [frame, i , x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, score, label, cam]
                file.writerow(t_feature+app.tolist())

    def init_det_model(self):
        # object detection
        self._log.info("Initializing object detector.")

        self.obj_detect = FRCNN_FPN(num_classes=self.num_class,
                                    box_nms_thresh=self.init_flags['nms_thr'],
                                    backbone_type=self.init_flags['backbone'],
                                    class_id=self.init_flags['class_id'])

        if self.init_flags['backbone']=='ResNet101FPN' and self.init_flags['server_loc']=='CLASP2':
            self.obj_detect.load_state_dict(torch.load(self._config['tracktor']['obj_detect_model_R101'],
                                                   map_location=lambda storage, loc: storage))

        elif self.init_flags['server_loc']=='CLASP1_30fps':
            if self.init_flags['backbone']=='ResNet101FPN':
                self.obj_detect.load_state_dict(torch.load(self._config['tracktor']['obj_detect_model_R101_clasp1'],
                                                   map_location=lambda storage, loc: storage))
            else:
                # self.obj_detect.load_state_dict(torch.load(self._config['tracktor']['obj_detect_model_R50_clasp1'],
                #                                        map_location=lambda storage, loc: storage))
                # print('load {}'.format(self._config['tracktor']['obj_detect_model_R50_clasp1']))
                self.obj_detect.load_state_dict(torch.load(self._config['tracktor']['obj_detect_model_base'],
                                                   map_location=lambda storage, loc: storage))
                print('load {}'.format(self._config['tracktor']['obj_detect_model_base']))

        elif self.init_flags['server_loc']=='CLASP2':
            if self.init_flags['backbone']=='ResNet101FPN':
                self.obj_detect.load_state_dict(torch.load(self._config['tracktor']['obj_detect_model_R101_clasp1'],
                                                   map_location=lambda storage, loc: storage))
            else:
                if self.init_flags['num_class']==3:
                    self.obj_detect.load_state_dict(torch.load(self._config['tracktor']['obj_detect_model_R50_clasp1'],
                                                       map_location=lambda storage, loc: storage))
                    print('load {}'.format(self._config['tracktor']['obj_detect_model_R50_clasp2']))
                else:
                    # self.obj_detect.load_state_dict(torch.load(self._config['tracktor']['obj_detect_model_R50_clasp2'],
                    #                                    map_location=lambda storage, loc: storage))
                    self.obj_detect.load_state_dict(torch.load(self._config['tracktor']['obj_detect_model_base'],
                                                       map_location=lambda storage, loc: storage))
                    print('load {}'.format(self._config['tracktor']['obj_detect_model_base']))

        elif self.init_flags['server_loc']=='PVD':
            self.obj_detect.load_state_dict(torch.load(self._config['tracktor']['obj_detect_model_tso_iter7'],
                                                   map_location=lambda storage, loc: storage))

            print('load {}'.format(self._config['tracktor']['obj_detect_model_tso_iter7']))
        else:
            self.obj_detect.load_state_dict(torch.load(self._config['tracktor']['obj_detect_model'],
                                                   map_location=lambda storage, loc: storage))
            print('load {}'.format(self._config['tracktor']['obj_detect_model']))
        self.obj_detect.eval()
        self.obj_detect.cuda()#device=self.gpu
        return self.obj_detect

    def init_reid_model(self):
        self.reid_network = resnet50(pretrained=False, **self.reid['cnn'])
        self.reid_network.load_state_dict(torch.load(osp.join(os.getcwd().split('experiments')[0],
                                                              self.tracktor['reid_weights']),
                                                               map_location=lambda storage, loc: storage))
        self.reid_network.eval()
        self.reid_network.cuda()#device=self.gpu
        return self.reid_network

    def init_tracktor(self):
        tracker = Tracker(self.init_det_model(),
                          self.init_reid_model(),
                          self.tracktor['tracker'],
                          self.global_track_num,
                          self.global_track_start_time,
                          class_id=self.init_flags['class_id'],
                          start_frame=self.init_flags['start_frame'],
                          det_time_counter=self.init_flags['det_time_counter'])
        return tracker

    def get_precomputed_dets(self, fr=None):
        assert  len(self.precomputed_dets)>0, 'precomputed dets should not empty'
        return self.precomputed_dets[self.precomputed_dets[:,0]==fr]

    def get_img_blob(self, path_i, dets=[]):
        """Return the ith image converted to blob"""
        #st = time.time()
        #https://towardsdatascience.com/what-library-can-load-image-in-python-and-what-are-their-difference-d1628c6623ad
        img = Image.open(path_i).convert("RGB")
        #print('img read time: {:2f}'.format(time.time()-st))
        self.init_flags['img_HW'][0] = img.size[1]
        self.init_flags['img_HW'][1] = img.size[0]
        #img = img.resize((540,960))
        img = self.transforms(img)

        sample = {}
        sample['img'] = torch.reshape(img, (1, 3, int(self.init_flags['img_HW'][0]),
                                            int(self.init_flags['img_HW'][1])))
        sample['dets'] = torch.tensor([[det[0], det[1], det[2], det[3]] for det in dets])
        sample['img_path'] = path_i
        sample['gt'] = {}
        sample['vis'] = {}

        return sample

    def init_online_dataloader(self, dataset, img_path, fr_num, data_loc):
        '''

        :param dataset: 'exp2training'
        :param img_path:
        :param frIndex:
        :return: return a batch of frames to SCT_clasp
        '''
        dets_i = []
        if data_loc == 'PVD':
            if self.cam == 1:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'C{}'.format(self.cam))
            if self.cam == 2:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'C{}'.format(self.cam))
            if self.cam == 360:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'C{}/img1'.format(self.cam))
            if self.cam == 4:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'C{}'.format(self.cam))
            if self.cam == 5:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'C{}'.format(self.cam))
            if self.cam == 6:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'C{}'.format(self.cam))
            if self.cam == 7:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'C{}'.format(self.cam))
            if self.cam == 8:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'C{}'.format(self.cam))
            if self.cam == 9:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'C{}'.format(self.cam))
            if self.cam == 10:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'C{}'.format(self.cam))

            path_i = os.path.join(img_path, '{:06d}.png'.format(fr_num))
            if os.path.exists(path_i) and self.init_flags['PANet_detector']:
                dets_i =  self.init_flags['PANet_dets'][self.cam][self.init_flags['PANet_dets'][self.cam][:,0]==fr_num]
                dets_i = dets_i[dets_i[:,7]==self.init_flags['class_id']]
                #convert to x1y1x2y2
                if len(dets_i)>0:
                    for i, det in enumerate(dets_i):
                        dets_i[i, 2] = min(self.init_flags['img_HW'][1], dets_i[i, 2])
                        dets_i[i, 2] = max(0, dets_i[i, 2])
                        dets_i[i, 3] = min(self.init_flags['img_HW'][0], dets_i[i, 3])
                        dets_i[i, 3] = max(0, dets_i[i, 3])
                    dets_i[:, 4:6] = dets_i[:, 4:6] + dets_i[:, 2:4]
                    dets_i = dets_i[:, 2:6]
            else:
                dets_i = []
            assert os.path.exists(path_i), '{} does not exist'.format(path_i)

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

        if data_loc == 'CLASP1_30fps':
            path_i = os.path.join(img_path, 'cam{:02d}{}.mp4/{:06d}.png'.format(self.cam, self.init_flags['dataset'], fr_num))

            if os.path.exists(path_i) and self.init_flags['PANet_detector']:
                dets_i =  self.init_flags['PANet_dets'][self.cam][self.init_flags['PANet_dets'][self.cam][:,0]==fr_num]
                dets_i = dets_i[dets_i[:,7]==self.init_flags['class_id']]
                #convert to x1y1x2y2
                if len(dets_i)>0:
                    for i, det in enumerate(dets_i):
                        dets_i[i, 2] = min(self.init_flags['img_HW'][1], dets_i[i, 2])
                        dets_i[i, 2] = max(0, dets_i[i, 2])
                        dets_i[i, 3] = min(self.init_flags['img_HW'][0], dets_i[i, 3])
                        dets_i[i, 3] = max(0, dets_i[i, 3])
                    dets_i[:, 4:6] = dets_i[:, 4:6] + dets_i[:, 2:4]
                    dets_i = dets_i[:, 2:6]
            else:
                dets_i = []
            assert os.path.exists(path_i), '{} does not exist'.format(path_i)


        if data_loc == 'CLASP2':
            if self.cam == 9:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, '{}_{}/img1'.format(dataset, self.cam))

            if self.cam == 2:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, '{}_{}/img1'.format(dataset, self.cam))

            if self.cam == 5:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, '{}_{}/img1'.format(dataset, self.cam))

            if self.cam == 11:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, '{}_{}/img1'.format(dataset, self.cam))
            if dataset=='G':
                if self.cam in [9, 11]:
                    path_i = os.path.join(img_path, '{:05d}.jpg'.format(fr_num))
                else:
                    path_i = os.path.join(img_path, '{:06d}.png'.format(fr_num))
            else:
                path_i = os.path.join(img_path, '{:06d}.png'.format(fr_num))

            if os.path.exists(path_i) and self.init_flags['PANet_detector']:
                dets_i =  self.init_flags['PANet_dets'][self.cam][self.init_flags['PANet_dets'][self.cam][:,0]==fr_num]
                dets_i = dets_i[dets_i[:,7]==self.init_flags['class_id']]
                #apply detection threshold
                dets_i = dets_i[dets_i[:, 6] >= self.init_flags['PANet_det_thr']]
                #convert to x1y1x2y2
                if len(dets_i)>0:
                    for i, det in enumerate(dets_i):
                        dets_i[i, 2] = min(self.init_flags['img_HW'][1], dets_i[i, 2])
                        dets_i[i, 2] = max(0, dets_i[i, 2])
                        dets_i[i, 3] = min(self.init_flags['img_HW'][0], dets_i[i, 3])
                        dets_i[i, 3] = max(0, dets_i[i, 3])
                    dets_i[:, 4:6] = dets_i[:, 4:6] + dets_i[:, 2:4]
                    dets_i = dets_i[:, 2:6]
            else:
                dets_i = []

        if not osp.exists(path_i):
            return None
        # for PANet and MRCNN get the precomputed dets for the corresponding frames
        return self.get_img_blob(path_i, dets=dets_i)

    def test_SCT(self, tracktor=None, start_frame=1, fr_step=1, dataset=None,
                 img_path=None, server='local', queue_batch=None, last_frame=2000, vis=False):
        # function to complete tracking by detection for a camera
        # start frame use to do synchronization in mc system
        save_dir = os.path.join(self.out_dir, self.init_flags['dataset'],
                                '{}_C{}'.format(self.init_flags['class_id'], self.cam))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            delete_all(save_dir, fmt='png')

        num_frame = start_frame
        avg_sct_time = []
        while num_frame <= last_frame:

            sct_time = time.time()
            frame = self.init_online_dataloader(dataset, img_path, num_frame, data_loc=server)
            if frame is None:
                print('None frame found at {}'.format(num_frame))
                #continue
                break
            
            print('processing {} in C{}'.format(os.path.basename(frame['img_path']), self.cam))
            if frame is not None:
                with torch.no_grad():
                    tracktor.step(frame, self.cam)
            avg_sct_time.append(time.time()-sct_time)
            print(f'avg sct time: {np.average(avg_sct_time)}')

            #at last frame return all tracking history in camera
            if num_frame%last_frame==0:
                cam_results = tracktor.get_results()
                self.write_cam_results(cam_results, self.out_dir,
                                       dataset=self.init_flags['dataset'],
                                       class_id=self.init_flags['class_id'],
                                       min_track_size=self.min_trklt_size,
                                       cam=self.cam, appearance=None)
                queue_batch.put(cam_results)

            if num_frame % self.init_flags['vis_rate'] == 0:
                cam_results = tracktor.get_results()
                plot_scanned_track(num_frame, cam_results, frame['img_path'], save_dir,cam=self.cam)
            num_frame += 1

    def test_SCD(self, tracktor=None, start_frame=1, fr_step=1, dataset=None,
                 img_path=None, server='local', queue_batch=None, last_frame=2000, vis=False):
        # function to complete tracking by detection for a camera
        # start frame use to do synchronization in mc system
        if not os.path.exists(self.out_dir):
            os.makedirs(self.out_dir)
        save_dir = os.path.join(self.out_dir, 'C{}'.format(self.cam))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        else:
            delete_all(save_dir, fmt='png')

        #file to write frame by frame detections
        if self.init_flags['skip_nms']:
            file = open(osp.join(self.out_dir, f'C{self.cam}_0nms0aug.txt'), "w")
        else:
            file = open(osp.join(self.out_dir, f'C{self.cam}_1nms0aug.txt'), "w")
        file = csv.writer(file, delimiter=',')
        fps_var = []
        num_frame = start_frame
        while num_frame <= self.init_flags['batch_end']:
            start_time = time.time()
            frame = self.init_online_dataloader(dataset, img_path, num_frame, data_loc=server)
            if frame is None:
                print('None frame found at {}'.format(num_frame))
                print('Average speed {} FPS'.format(np.mean(fps_var)))
                #continue
                break
            print('processing {} in C{}'.format(os.path.basename(frame['img_path']),self.cam))
            if frame is not None:
                self.obj_detect.load_image(frame['img'])
                boxes, scores, labels = self.obj_detect.detect_clasp1(frame['img'])

                # print('Predicted box: ', boxes)
                # print('Predicted scores: ', scores)
                if boxes.nelement() > 0:
                    boxes = clip_boxes_to_image(boxes, frame['img'].shape[-2:])
                    #boxes, scores = self.obj_detect.predict_boxes(boxes)
                    # Filter out tracks that have too low person score
                    # Computes input>other\text{input} > \text{other}input>other element-wise.
                    inds = torch.gt(scores, self.tracktor['tracker']['detection_person_thresh']).nonzero().view(-1)
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
                    fr = float(os.path.basename(frame['img_path']).split('.')[0])

                    if not self.init_flags['skip_nms']:
                        keep = nms(det_pos, det_scores, self.tracktor['tracker']['detection_nms_thresh'])
                        det_pos = det_pos[keep]
                        det_scores = det_scores[keep]
                        det_labels = det_labels[keep]
                        det_labels = torch.tensor(det_labels.cpu().numpy(), dtype=torch.float32).cuda() #long to float
                        camfr_results = torch.cat((det_pos,
                                                   det_scores.resize(det_scores.shape[0], 1),
                                                   det_labels.resize(det_labels.shape[0], 1)), dim=1)
                        self.write_det_results(camfr_results, file, fr, cam=self.cam, appearance=None)
                        if num_frame % 5 == 0:
                            plot_dets(num_frame, camfr_results, frame['img_path'], save_dir,cam=self.cam)

                    else:# cluster mode prediction class update the multi-class/single-class prediction
                        print('number of dets before nms {}'.format(det_pos.shape))
                        camfr_results = torch.cat((det_pos, det_scores.resize(det_scores.shape[0], 1)), dim=1)
                        pred_cluster_mode = Cluster_Mode(detPB=camfr_results, frame=fr, img=frame['img_path'],
                                                         out_path=save_dir, save_file=file, num_class=1, vis=True, skip_nms=self.init_flags['skip_nms'])
                        camfr_results, _ = pred_cluster_mode.get_modes()
                    fps_var.append(1/(time.time() - start_time))
                    print('avg speed in cam {}: {:2f} Hz'
                          .format(self.cam, np.mean(fps_var)))

            num_frame += fr_step


def SCT_main(q, init_params, gpu_index, _config, global_track_num, global_track_start_time):
    # ***Args: gpu, cam_x_batch, model
    # ***Return: batch tracking results: keep active/inactive tracks for associating with next batch
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(init_params['gpu_id'][gpu_index])
    torch.cuda.set_device(init_params['gpu_id'][gpu_index])
    torch.set_num_threads(1)
    _log = logging
    print('process start with gpu {} cam {}'.format(init_params['gpu_id'][gpu_index], init_params['cams'][gpu_index]))

    single_cam_tracker = SCT_Clasp( _config,
                                   _log,
                                   init_params['cams'][gpu_index],
                                   init_params['output_dir'],
                                   num_class=init_params['num_class'],
                                   batch=init_params['batch_size'],
                                   gpu=init_params['gpu_id'][gpu_index],
                                   min_trklt_size=init_params['min_trklt_size'],
                                   global_track_num=global_track_num,
                                   global_track_start_time = global_track_start_time,
                                    init_flags= init_params
                                    )

    #run single camera tracker
    tracker = single_cam_tracker.init_tracktor()
    single_cam_tracker.test_SCT(tracktor=tracker,
                                start_frame=init_params['start_frame'],
                                fr_step = init_params['fr_step'],
                                dataset=init_params['dataset'],
                                img_path=init_params['img_path'],
                                server=init_params['server_loc'],
                                queue_batch = q,
                                last_frame=init_params['batch_end'],
                                vis=True)
    #save results in queue
    #q.put(batch_results)
def update_params(init_params, storage, kri_data=None, only_det=False):
    if kri_data=='clasp1':
        server = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/' \
                 'ourdataset/'
        batch_end = {'exp6a':7100, 'exp7a':6500, 'exp10a': 6100, 'exp5a':4090, 'exp9a':6000}


        init_params['server_loc'] = 'CLASP1_30fps'
        init_params['img_HW'] = [1080.0, 1920.0]
        if init_params['class_id'] == 1:
            init_params['cams'] = [9, 11]#[2, 5, 9, 11]
            init_params['gpu_id'] = [0, 1]
        else:
            init_params['cams'] = [9, 11]
            init_params['gpu_id'] = [0, 1]

        # 1FPS: GT frames
        # init_params['img_path'] = os.path.join(benchmark, init_params['dataset'], 'img1')
        init_params['batch_end'] = batch_end[init_params['dataset']]

        init_params['fr_step'] = 1
        init_params['batch_start'] = 1
        init_params['start_frame'] = 1

        if only_det:
            benchmark = storage + 'tracking_wo_bnw/data/CLASP1/train_gt'
            init_params['img_path'] = benchmark
            init_params['num_class'] = 3
            init_params['output_dir'] = storage + 'tracking_wo_bnw/output/tracktor-clasp1-det/' + init_params[
                'dataset']
        else:
            init_params['num_class'] =2# 3#2
            init_params['img_path'] = os.path.join(server, init_params['dataset'], 'imgs')
            init_params['output_dir'] = storage + 'tracking_wo_bnw/output/clasp1/{}_tracks'.format(init_params['model'])


            if init_params['PANet_detector']:
                init_params['PANet_dets'] = {}
                if init_params['model']=='SL':
                    init_params['PANet_dets_dir'] = os.path.join(storage,
                                                                 'PANet_Results/CLASP1/PANet_det/{}_dets_30fps'.format(
                                                                     init_params['model']))
                else:
                    init_params['PANet_dets_dir'] = os.path.join(storage,
                                                                 'PANet_Results/CLASP1/PANet_mask/{}_dets_30fps'.format(
                                                                     init_params['model']))

                print('load PANet detections for {} model'.format(init_params['model']))
                for cam in init_params['cams']:
                    seq = 'cam{:02d}{}.mp4'.format(cam, init_params['dataset'])
                    seq_dets    = np.loadtxt(os.path.join(init_params['PANet_dets_dir'],
                                                          '{}/{}_pb_0aug1nms.txt'.format(seq, seq)), delimiter=',')
                    seq_dets[seq_dets[:, 7] >= 2, 7] = 2
                    init_params['PANet_dets'][cam] = seq_dets
                    print('read seq {} dets: {}'.format(seq,  init_params['PANet_dets'][cam].shape))

    if kri_data == 'PVD':
        benchmark = storage + 'tracking_wo_bnw/data/PVD/HDPVD_new/train_gt/'
        init_params['dataset'] = 'C'
        init_params['frame_rate'] = '10FPS'
        init_params['server_loc'] = 'PVD'
        init_params['img_HW'] = [800.0, 1280.0]
        init_params['cams'] = [360]

        batch_end = {'C': 6002}
        init_params['gpu_id'] = [0]
        init_params['only_SCTs'] = True
        # 1FPS: GT frames
        init_params['img_path'] = benchmark
        init_params['fr_step'] = 1
        init_params['batch_start'] = 1
        init_params['start_frame'] = 1
        init_params['num_class'] = 3
        init_params['batch_end'] = batch_end[init_params['dataset']]

        if init_params['PANet_detector']:
            init_params['PANet_dets'] = {}
            # init_params['PANet_dets_dir'] = os.path.join(storage, 'PANet_Results/CLASP1/PANet_det/dets_30fps')
            # init_params['output_dir'] = storage + 'tracking_wo_bnw/output/panet_supervised_clasp1/'
            # init_params['PANet_dets_dir'] = os.path.join(storage, 'PANet_Results/PVD/PANet_mask')
            init_params['PANet_dets_dir'] = os.path.join(storage, 'PANet_Results/PVD/PANet_mask/SSL_alpha_dets_30fps_semi')
            for cam in init_params['cams']:
                seq = '{}{}'.format(init_params['dataset'], cam)
                init_params['PANet_dets'][cam] = np.loadtxt(os.path.join(init_params['PANet_dets_dir'],
                                                                         seq + '_pb_0aug1nms.txt'), delimiter=',')
        # for det
        if only_det:
            init_params['output_dir'] = storage + 'tracking_wo_bnw/output/tracktor-PVD-det/' + init_params[
                'dataset']
        else:
            init_params['output_dir'] = storage + 'tracking_wo_bnw/output/PVD/'

    if kri_data == 'clasp2':
        batch_end = {'G': 13430, 'H': 10400, 'I': 11100}
        init_params['frame_rate'] = '30FPS'
        init_params['server_loc'] = 'CLASP2'
        init_params['img_HW'] = [1080.0, 1920.0]

        if init_params['class_id'] == 1:
            #for clasp2 run 9, 11 and 2, 5 separately
            init_params['cams'] = [9, 11] #2, 5,
            init_params['gpu_id'] = [0, 1]
        else:
            init_params['cams'] = [9, 11]
            init_params['gpu_id'] = [0, 1]

        init_params['only_SCTs'] = True
        if 2 in init_params['cams']:
            benchmark = storage + 'tracking_wo_bnw/data/CLASP/train_gt_all/PB_no_gt'
        else:
            benchmark = storage + 'tracking_wo_bnw/data/CLASP/train_gt_all/PB_gt'
        # 1FPS: GT frames
        init_params['img_path'] = benchmark
        init_params['fr_step'] = 1
        init_params['batch_start'] = 1
        init_params['start_frame'] = 1
        init_params['num_class'] = 2#3 #2 for clasp2 person only model
        init_params['batch_end'] = batch_end[init_params['dataset']]
        # for det
        if only_det:
            init_params['output_dir'] = storage + 'tracking_wo_bnw/output/tracktor-clasp2-det/' + init_params[
                'dataset']
        else:
            init_params['img_path'] = benchmark #os.path.join(benchmark, init_params['dataset'], '_{}/imgs')
            init_params['output_dir'] = storage + 'tracking_wo_bnw/output/clasp2/{}_tracks'.format(init_params['model'])

            if init_params['PANet_detector']:
                init_params['PANet_dets'] = {}
                if init_params['model']=='SL':
                    init_params['PANet_dets_dir'] = os.path.join(storage,
                                                                 'PANet_Results/CLASP2/PANet_det/{}_dets_30fps'.format(
                                                                     init_params['model']))
                else:
                    init_params['PANet_dets_dir'] = os.path.join(storage,
                                                                 'PANet_Results/CLASP2/PANet_mask/{}_dets_30fps'.format(
                                                                     init_params['model']))
                #init_params['PANet_dets_dir'] = os.path.join(storage,
                 #                                            'PANet_Results/CLASP2/PANet_mask/ssl_alpha_dets_30fps')
                #init_params['output_dir'] = storage + 'tracking_wo_bnw/output/panet_supervised_clasp1/'
                #init_params['PANet_dets_dir'] = os.path.join(storage, 'Detectron2/gt_det_model/outputs_clasp_30fps/iter_1')
                #init_params['output_dir'] = storage + 'tracking_wo_bnw/output/mrcnn/supervised_clasp1/'
                print('load PANet detections for {} model'.format(init_params['model']))
                for cam in init_params['cams']:
                    seq = '{}_{}'.format(init_params['dataset'], cam)
                    seq_dets = np.loadtxt(os.path.join(init_params['PANet_dets_dir'],
                                            seq + '_pb_0aug1nms.txt'), delimiter=',')

                    seq_dets[seq_dets[:, 7] >= 2, 7] = 2
                    init_params['PANet_dets'][cam] = seq_dets
                    print('read seq {} dets: {}'.format(seq,  init_params['PANet_dets'][cam].shape))

    return init_params

if __name__ == '__main__':
    #dir params
    storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62/'
    kri_data = 'clasp1' #'PVD' #'clasp2', 'clasp1'
    model = 'tracktor' #'SSL_alpha'
    cam_type = 'pri'
    #from the optimal point of PR curve: for C2 and C5
    if cam_type=='pri':
        # for C9 and C11
        pred_score = {'tracktor':{'clasp1':{1:0.5, 2:0.5}, 'clasp2':{1:0.5, 2:0.5}},
                      'SSL_alpha_semi':{'clasp1':{1:0.8, 2:0.75}, 'clasp2':{1:0.8, 2:0.6}},
                      'SSL_alpha':{'clasp1':{1:0.88, 2:0.89}, 'clasp2':{1:0.8, 2:0.6}},
                      'SSL':{'clasp1':{1:0.85, 2:0.89}, 'clasp2':{1:0.65, 2:0.5}},
                      'SL':{'clasp1':{1:0.8, 2:0.75}, 'clasp2':{1:0.75, 2:0.7}},
                      'Base':{'clasp1':{1:0.8, 2:0.5}, 'clasp2':{1:0.5, 2:0.5}}}
    else:
        # for C2 and C5
        pred_score = {'SSL_alpha_semi':{'clasp1':{1:0.8, 2:0.75}, 'clasp2':{1:0.8, 2:0.6}},
                      'SSL_alpha': {'clasp1': {1: 0.8, 2: 0.8}, 'clasp2': {1: 0.8, 2: 0.6}},
                      'SSL': {'clasp1': {1: 0.85, 2: 0.89}, 'clasp2': {1: 0.65, 2: 0.5}},
                      'SL': {'clasp1': {1: 0.8, 2: 0.75}, 'clasp2': {1: 0.75, 2: 0.7}},
                      'Base': {'clasp1': {1: 0.8, 2: 0.5}, 'clasp2': {1: 0.5, 2: 0.5}}}

    only_det = 0
    init_params = {}
    init_params['model'] = model
    init_params['class_id'] = 1 # mrcnn: 0:person, 1:bag, 1, panet: 1:person, 2:bag
    init_params['PANet_detector'] = 0

    if kri_data=='PVD':
        init_params['PANet_det_thr'] = 0.5
    else:
        init_params['PANet_det_thr'] = pred_score[model][kri_data][init_params['class_id']]
    #TODO: loop over all datasets
    for dataset in ['exp10a']:
        init_params['dataset'] = dataset
        init_params = update_params(init_params, storage, kri_data=kri_data, only_det=only_det)

        #mcta params unused in this script
        init_params['print_stats'] = False
        init_params['global_mcta_graph'] = {}
        init_params['single_cam_multi_asso'] = False
        init_params['isSCT'] = False
        init_params['sequential_id'] = False
        init_params['last_batch_ids'] = {}
        init_params['current_batch_ids'] = {}
        init_params['new_ids'] = {}
        init_params['prev_affinity'] = {}
        init_params['global_track_id'] = 0
        init_params['keep_raw_id'] = {}
        init_params['batch_size'] = None
        init_params['min_trklt_size'] = 0

        #model params
        init_params['backbone'] = 'ResNet50FPN'
        init_params['skip_nms'] = 0
        if init_params['skip_nms']:
            init_params['nms_thr'] = 1.0
        else:
            init_params['nms_thr'] = 0.3
        init_params['save_imgs'] = 1
        init_params['vis_rate'] = 30
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
        print(os.getcwd())
        with open(osp.join(os.getcwd(),'experiments/cfgs_clasp/tracktor.yaml')) as file:
            # The FullLoader parameter handles the conversion from YAML
            # scalar values to Python the dictionary format
            _config = yaml.load(file, Loader=yaml.FullLoader)
            with open(osp.join(os.getcwd().split('experiments')[0],_config['tracktor']['reid_config'])) as file1:
                _config.update(yaml.load(file1, Loader=yaml.FullLoader))

        #set detection threshold externally for baggage class
        if init_params['class_id'] == 2:
            _config['tracktor']['tracker']['detection_person_thresh'] = init_params['PANet_det_thr']
        else:
            _config['tracktor']['tracker']['detection_person_thresh'] = init_params['PANet_det_thr']

        if init_params['PANet_detector']:
            _config['tracktor']['tracker']['public_detections'] = True
            if init_params['class_id'] == 2:
                _config['tracktor']['tracker']['detection_person_thresh'] = 0.0
            else:
                _config['tracktor']['tracker']['detection_person_thresh'] = 0.0
        else:
            _config['tracktor']['tracker']['public_detections'] = False

        # start multiprocess for multi-camera
        #https://pytorch.org/docs/stable/multiprocessing.html
        mp.set_start_method('spawn')
        # global variable for sequential global tracks
        #global_track_num = []
        manager = mp.Manager()
        global_track_num = manager.list()
        global_track_start_time = manager.dict()
        init_params['det_time_counter'] = manager.dict()
        #set of gpus
        num_gpus = torch.cuda.device_count()
        #list of processes
        q=list(range(len(init_params['cams'])))
        p = {}
        
        #assign process to each gpu
        for i in range(len(init_params['cams'])):
            q[i] = mp.Queue()
            # Pass GPU number through q
            init_params['det_time_counter'][init_params['cams'][i]] = manager.list()
            p[i] = mp.Process(target=SCT_main, args=(q[i], init_params, i, _config, global_track_num, global_track_start_time))
            p[i].start()

        #terminate process
        for i in range(len(init_params['cams'])):
            p[i].join()
            p[i].terminate()