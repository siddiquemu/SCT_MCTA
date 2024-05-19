import copy
import multiprocessing as mp
import os
import csv
import random
import time
from os import path as osp
import psutil
import tracemalloc
import sys
sys.path.insert(0,'/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62/experiments/cfgs')
sys.path.insert(1,'/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62/tracking_wo_bnw/experiments/scripts/MCTA')
sys.path.insert(2,'/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62/tracking_wo_bnw/src')
import numpy as np
import torch
import cv2
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
from PIL import Image, ImageDraw
import motmetrics as mm
mm.lap.default_solver = 'lap'

import torchvision
import yaml
import pandas as pd
from tqdm import tqdm
import sacred
from sacred import Experiment
#from tracktor.datasets.factory import Datasets
#base
#from tracktor.frcnn_fpn import FRCNN_FPN
#from tracktor.tracker import Tracker
#batch det+sct reid
#from tracktor.frcnn_fpn_clasp import FRCNN_FPN
#from tracktor.tracker_batches import Tracker
#sct reid
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.tracker_reid_rot import Tracker

from tracktor.reid.resnet import resnet50
from tracktor.utils import plot_scanned_track, filter_tracklets_app, delete_all
import pdb
import logging
from read_config import Configuration
#from MCTA.MCTA_Online import MCTA
from MCTA.MCTA_KRI_app import MCTA
import torchvision.transforms as T
import utils
import torch.nn as nn
from collections import deque
from read_meta_data import *
random.seed(42)

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)
# TODO:
# 1. replace data loader with clasp batch image reader or online image reader
# 2.
class batch_loader(Dataset):
    def __init__(self, batch_start=1, batch_size=40, transforms=None, cam=None, img_path=None):
        #batch_start will be: 1, 41, 81, ...
        self.transforms = transforms
        self._classes = ('background', 'passenger')
        self._img_paths = []
        self.batch_size = batch_size
        self.batch_start = batch_start
        self.batch_end = self.batch_start + self.batch_size
        self.cam = cam

        for i in range(self.batch_start, self.batch_end + 1):

            path_i = os.path.join(img_path, 'cam{:02d}exp2.mp4/cam{}_{:06d}.jpg'.format(self.cam, self.cam, i))
            assert os.path.exists(path_i), \
                'Path does not exist: {}'.format(path_i)
            self._img_paths.append(path_i)

    #@property decorator applied to the num_classes() method. The num_classes() method
    # returns the len private instance attribute value self._classes. So, we can now use the
    # num_classes()() method as a property to get the value of the len of self._classes attribute
    @property
    def num_classes(self):
        return len(self._classes)

    def __getitem__(self, idx):
        # load images
        img_path = self._img_paths[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transforms is not None:
            img = self.transforms(img)
        return img

    def __len__(self):
        return len(self._img_paths)


class SCT_Clasp(object):
    def  __init__(self, _config, _log,
                 cam, output_dir, num_class=2,
                 batch=120, gpu=0, min_trklt_size=30, global_track_count=None,
                  global_track_num=None, global_track_start_time=None, p_id=0,
                  reid_rads_patience=None, init_flags=None, prev_batches=3):
        self.cam = cam
        self.num_class = num_class
        self.time_total = 0
        self.num_frames = 0
        self.scan_intrvl = batch
        self.prev_batches = prev_batches
        self.gpu = gpu
        self.transforms = ToTensor()
        self.out_dir = output_dir
        self.tracktor = _config['tracktor']
        self.reid = _config['reid']
        self._config = _config
        self.global_track_count = global_track_count
        self.global_track_num = global_track_num
        self.p_id = p_id
        self.global_track_start_time = global_track_start_time
        self.reid_rads_patience = reid_rads_patience
        self._log = _log
        self.reid_patience = 20
        self.min_trklt_size = min_trklt_size
        self.init_flags = init_flags
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
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
    def write_cam_results(all_tracks, output_dir, dataset=None, class_id=None, cam=None, appearance=None):
        # this function is used when caresults are save to do offline MCTA association
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file = osp.join(output_dir, dataset, f'{class_id}_C{cam}.txt')

        print("[*] Writing tracks to: {}".format(file))

        with open(file, "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():
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


    def get_transform(self, train):
        transforms = []
        # converts the image, a PIL image, into a PyTorch Tensor
        transforms.append(T.ToTensor())
        if train:
            # during training, randomly flip the training images
            # and ground-truth for data augmentation
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)

    def init_det_model(self):
        # object detection
        self._log.info("Initializing object detector.")
        self.obj_detect = FRCNN_FPN(num_classes=self.num_class, backbone_type=self.init_flags['backbone'])
        if self.init_flags['backbone'] == 'ResNet50FPN':
            #train camera specific models: all will fit in SSL
            self.obj_detect.load_state_dict(torch.load(self._config['tracktor']['obj_detect_model'],
                                                       map_location=lambda storage, loc: storage))
        if self.init_flags['backbone'] == 'ResNet34FPN':
                #train camera specific models: all will fit in SSL
            self.obj_detect.load_state_dict(torch.load(self._config['tracktor']['obj_detect_model_R34'],
                                                       map_location=lambda storage, loc: storage))
            print('load {} model: {}'.format(self.init_flags['backbone'], self._config['tracktor']['obj_detect_model_R34']))
        
        if self.init_flags['backbone'] == 'ResNet101FPN':
            self.obj_detect.load_state_dict(torch.load(self._config['tracktor']['obj_detect_model_R101'],
                                                   map_location=lambda storage, loc: storage))
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
                          self.global_track_count,
                          self.global_track_num,
                          self.global_track_start_time,
                          self.reid_rads_patience,
                          class_id=1,
                          start_frame=self.init_flags['start_frame'],
                          det_time_counter=self.init_flags['det_time_counter'],
                          isSCT_ReID=self.init_flags['isSCT_ReID'],
                          p_id = self.p_id,
                          cam=self.cam)
        return tracker

    def get_img_blob(self, path_i, dets=[]):
        """Return the ith image converted to blob"""
        #st = time.time()
        #https://towardsdatascience.com/what-library-can-load-image-in-python-and-what-are-their-difference-d1628c6623ad
        img = Image.open(path_i).convert("RGB")
        if self.cam==10:
            ImageDraw.Draw(img).rectangle([(0, 0), (1280, 100)], fill='black')
            ImageDraw.Draw(img).rectangle([(0, 0), (100, 800)], fill='black')
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
        if data_loc == 'local':
            if self.cam==2:
                fr_num=fr_num+7#15#46
                img_path = os.path.join(img_path, 'cam02exp2.mp4')
            if self.cam==9:
                fr_num = fr_num+0
                img_path = os.path.join(img_path, 'cam09exp2.mp4')
            if self.cam==5:
                fr_num = fr_num+0#10:9#30:28
                img_path = os.path.join(img_path, 'cam05exp2.mp4')
            if self.cam==11:
                fr_num = fr_num+3+44
                img_path = os.path.join(img_path, 'cam11exp2.mp4')
            if self.cam==13:
                fr_num = fr_num-44+44
                img_path = os.path.join(img_path, 'cam13exp2.mp4')
            if self.cam==14:
                fr_num = fr_num-41+44
                img_path = os.path.join(img_path, 'cam14exp2.mp4')
            path_i = img_path + '/%06d' % fr_num + '.png'

        if data_loc == 'remote':
            if self.cam==2:
                path = os.path.join(img_path, 'cam02exp2.mp4')
            if self.cam==9:
                path = os.path.join(img_path, 'cam09exp2.mp4')
            if self.cam==5:
                path = os.path.join(img_path, 'cam05exp2.mp4')
            if self.cam==4:
                path = os.path.join(img_path, 'cam04exp2.mp4')
            if self.cam==11:
                path = os.path.join(img_path, 'cam11exp2.mp4')
            if self.cam==13:
                path = os.path.join(img_path, 'cam13exp2.mp4')
            if self.cam==14:
                path = os.path.join(img_path, 'cam14exp2.mp4')

            path_i = os.path.join(path, 'cam{}_{:06d}.jpg'.format(self.cam, fr_num))

        if data_loc=='remote_30fps':
            timeoffset = '{:.2f}'.format((fr_num) / 30.0)
            if self.cam==2:#cam2_0_00.jpg
                path_i = os.path.join(img_path, 'cam{}_{}_{}.jpg'.format(self.cam, timeoffset.split('.')[0], timeoffset.split('.')[1]))
            if self.cam==9:
                path_i = os.path.join(img_path, 'cam{}_{}_{}.jpg'.format(self.cam, timeoffset.split('.')[0], timeoffset.split('.')[1]))
            if self.cam==5:
                path_i = os.path.join(img_path, 'cam{}_{}_{}.jpg'.format(self.cam, timeoffset.split('.')[0], timeoffset.split('.')[1]))
            if self.cam==11:
                path_i = os.path.join(img_path, 'cam{}_{}_{}.jpg'.format(self.cam, timeoffset.split('.')[0], timeoffset.split('.')[1]))
            if self.cam==13:
                path_i = os.path.join(img_path, 'cam{}_{}_{}.jpg'.format(self.cam, timeoffset.split('.')[0], timeoffset.split('.')[1]))
            if self.cam==14:
                path_i = os.path.join(img_path, 'cam{}_{}_{}.jpg'.format(self.cam, timeoffset.split('.')[0], timeoffset.split('.')[1]))

        if data_loc == 'Pease':
            path_i = os.path.join(img_path, 'C{}/img1/{:06d}.png'.format(self.cam, fr_num))

            if not osp.exists(path_i):
                #print('None frame found at {}'.format(path_i))
                return None
            #else:
              #print('reading frames for cam {} at {}'.format(self.cam, timeoffset))

        if data_loc == 'wild-track':
            if self.cam == 1:
                img_path = os.path.join(img_path, 'cam1') #exp1
            if self.cam == 2:
                img_path = os.path.join(img_path, 'cam2') #exp2-C7
            if self.cam == 3:
                img_path = os.path.join(img_path, 'cam3')
            if self.cam == 4:
                img_path = os.path.join(img_path, 'cam4')
            path_i = os.path.join(img_path, 'cam{}_{:06d}.jpg'.format(self.cam, fr_num))

        if data_loc == 'logan':
            if self.cam == 1:
                img_path = os.path.join(img_path, 'C7') #exp1
            if self.cam == 2:
                img_path = os.path.join(img_path, 'C8') #exp2-C7
            if self.cam == 3:
                img_path = os.path.join(img_path, 'cam3')
            if self.cam == 4:
                img_path = os.path.join(img_path, 'cam4')
            path_i = os.path.join(img_path, '{:04d}.jpg'.format(fr_num))

        dets_i = []
        if data_loc == 'PVD':
            if self.cam == 300:
                fr_num = fr_num + 23
                img_path = os.path.join(img_path, 'C{}/img1'.format(self.cam))
            if self.cam == 440:
                fr_num = fr_num + 23
                img_path = os.path.join(img_path, 'C{}/img1'.format(self.cam))
            if self.cam == 340:
                fr_num = fr_num + 17
                img_path = os.path.join(img_path, 'C{}/img1'.format(self.cam))
            if self.cam == 360:
                fr_num = fr_num + 6
                img_path = os.path.join(img_path, 'C{}/img1'.format(self.cam))
            if self.cam == 441:
                fr_num = fr_num + 6
                img_path = os.path.join(img_path, 'C{}/img1'.format(self.cam))
            if self.cam == 330:
                fr_num = fr_num +33
                img_path = os.path.join(img_path, 'C{}/img1'.format(self.cam))
            if self.cam == 361:
                fr_num = fr_num + 6
                img_path = os.path.join(img_path, 'C{}/img1'.format(self.cam))

            if self.cam == 1:
                fr_num = fr_num # 13
                img_path = os.path.join(img_path, 'C{}/img1'.format(self.cam))
            if self.cam == 2:
                fr_num = fr_num + 21
                img_path = os.path.join(img_path, 'C{}/img1'.format(self.cam))
            if self.cam == 3:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'C{}/img1'.format(self.cam))
            if self.cam == 4:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'C{}/img1'.format(self.cam))
            if self.cam == 5:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'C{}/img1'.format(self.cam))
            if self.cam == 6:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'C{}/img1'.format(self.cam))
            if self.cam == 8:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'C{}/img1'.format(self.cam))
            if self.cam == 10:
                fr_num = fr_num + 16
                img_path = os.path.join(img_path, 'C{}/img1'.format(self.cam))

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

        if not osp.exists(path_i):
            return None

        return self.get_img_blob(path_i, dets=dets_i)

    def test_SCT(self, tracktor=None, start_frame=None, dataset=None,
                 img_path=None, server='local', queue_batch=None, vis=False):

        num_frame = copy.deepcopy(start_frame)
        read_time = []
        tracking_time = []
        batches_frames = np.arange(num_frame - self.prev_batches * self.scan_intrvl + 1, num_frame + 1)
        
        while num_frame <= self.init_flags['batch_end']:
            read_start = time.time()
            frame = self.init_online_dataloader(dataset, img_path, num_frame, data_loc=server)
            read_time.append(time.time()-read_start)

            if frame is None:
                continue

            if frame is not None:
                with torch.no_grad():
                    tracktor.step(frame, self.cam, batches_frames)
            #print(f'at {num_frame} SCT time: {time.time()-read_start}')
            #at scan interval return all tracking history in camera
            if num_frame%self.scan_intrvl==0:
                all_results = {}
                all_results['pose'] = tracktor.get_results()

                all_results['appearance'] = tracktor.get_tracks_appearance()
                batches_frames = np.arange(num_frame - self.prev_batches * self.scan_intrvl + 1, num_frame + 1)

                all_results, self.global_track_start_time, unused_tracks = filter_tracklets_app(num_frame,
                                                           trackers = all_results,
                                                           start_frame=self.init_flags['start_frame'],
                                                            batch_length=self.scan_intrvl,
                                                            min_trklt_size=self.min_trklt_size,
                                                            look_back_batches=self.prev_batches,
                                                            reid_patience=self.reid_patience,
                                                            track_start_time=self.global_track_start_time)

                #TODO: pass unused tracks to tracker step to return less information
                queue_batch.put(all_results)

            if self.init_flags['only_SCTs']:
                if num_frame % self.init_flags['batch_end'] == 0:
                    cam_results = tracktor.get_results()
                    self.write_cam_results(cam_results, self.out_dir,
                                           dataset=self.init_flags['dataset'],
                                           class_id=self.init_flags['class_id'],
                                           cam=self.cam, appearance=None)
                if num_frame%self.scan_intrvl==0: #self.scan_intrvl
                    cam_out = os.path.join(self.out_dir, 'C{}'.format(self.cam))
                    if not os.path.exists(cam_out):
                        os.makedirs(cam_out)
                    plot_scanned_track(num_frame, batch_results, frame['img_path'],
                                       cam_out, cam=self.cam, reid_rads=self.reid_rads_patience['rads'])
            num_frame += 1

    def test_SCT_batch(self, tracktor=None, start_frame=None, dataset=None,
                 img_path=None, server='local', queue_batch=None, vis=False):

        num_frame = start_frame
        start_time = time.time()
        read_time = []
        tracking_time = []
        det_time = []
        batch_frames = []
        frame_indexs = []
        steps_det = [10, 10, 10, 10]
        step_ind = 0
        while num_frame <= self.init_flags['batch_end']:
            time_offset = '{:.2f}'.format((num_frame)/30.0) #-1

            read_start = time.time()
            frame = self.init_online_dataloader(dataset, img_path, num_frame, data_loc=server)
            read_time.append(time.time()-read_start)

            if frame is None:
                continue
                print('frame {} not found'.format(frame['img_path']))
                #break
            else:
                batch_frames.append(frame['img'])
                frame_indexs.append(num_frame)
            #at scan interval return all tracking history in camera
            if len(batch_frames)==steps_det[step_ind]:
                #batch detections in 2/3 steps
                #compute backbone features
                with torch.no_grad():

                    det_start = time.time()
                    images = torch.cat(tuple(batch_frames))
                    # get raw detections
                    detections, features = self.obj_detect.detect_batches(images)
                    print('det time {:.2f} sec for #frames: {}'.format((time.time() - det_start), steps_det[step_ind] ))
                    det_time.append(time.time() - det_start)
                    step_ind+=1

                # apply batch detections to update trackers
                det_batch_fr = 0
                for fr in frame_indexs:

                    track_start = time.time()
                    with torch.no_grad():
                        frame['img'] = batch_frames[det_batch_fr]
                        frame['features'] = {key:feat[det_batch_fr].unsqueeze(dim=0) for key, feat in features.items()}
                        tracktor.step(frame, detections[det_batch_fr], self.cam)
                        tracking_time.append(time.time() - track_start)

                    det_batch_fr+=1

                batch_frames = []
                frame_indexs = []

            if num_frame%self.scan_intrvl==0:
                batch_results = tracktor.get_results()

                batch_results = filter_tracklets(num_frame, batch_results, start_frame=start_frame,
                                                 batch_length=self.scan_intrvl, min_trklt_size=self.min_trklt_size,
                                                 reid_patience=self.reid_patience)

                print('at {} batch ids {}'.format(num_frame, batch_results.keys()))
                print('cam: {}, batch: {}, read time: avg {:.2f} sec, max {:.2f}'.format(self.cam, num_frame//self.scan_intrvl,
                                                                       np.mean(read_time)*self.scan_intrvl, np.max(read_time)*self.scan_intrvl))
                print('cam: {}, batch: {}, tracking time avg {:.2f} sec, max {:.2f}'.format(self.cam, num_frame//self.scan_intrvl,
                                                                       np.mean(tracking_time)*self.scan_intrvl, np.max(tracking_time)*self.scan_intrvl))
                #print('cam: {}, batch: {}, det time: avg {:.2f} sec, max {:.2f}'.format(self.cam, num_frame//self.scan_intrvl,
                                                                       #np.mean(self.init_flags['det_time_counter'][self.cam])*self.scan_intrvl,
                                                                        #np.max(self.init_flags['det_time_counter'][self.cam])*self.scan_intrvl))

                print('cam: {}, batch: {}, detection time avg {:.2f} sec, max {:.2f}'.format(self.cam, num_frame//self.scan_intrvl,
                                                                       np.mean(det_time)*len(steps_det), np.max(det_time)*len(steps_det)))
                print('cam: {}, batch: {}, batch time: avg {:.2f} sec'.format(self.cam, num_frame // self.scan_intrvl,
                                                                         (np.mean(tracking_time)
                                                                         +np.mean(read_time))
                                                                         *self.scan_intrvl
                                                                         +np.mean(det_time)*len(steps_det)))
                print('batch tracking speed in cam {}: {:2f} Hz'
                      .format(self.cam, self.scan_intrvl/(time.time()-start_time)))

                #plot_scanned_track(num_frame, batch_results, frame['img_path'], self.out_dir,cam=self.cam)
                queue_batch.put(batch_results)

                start_time = time.time()
                step_ind = 0

            num_frame += 1


def SCT_main(q, init_params, gpu_index, _config, global_track_num,
             global_track_count, global_track_start_time, reid_rads_patience):
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
                                   global_track_count=global_track_count,
                                   global_track_num=global_track_num,
                                   global_track_start_time = global_track_start_time,
                                    reid_rads_patience = reid_rads_patience,
                                    p_id=gpu_index,
                                    init_flags= init_params
                                    )

    #run single camera tracker
    tracker = single_cam_tracker.init_tracktor()
    single_cam_tracker.test_SCT(tracktor=tracker,
                                start_frame=init_params['start_frame'],
                                dataset=init_params['dataset'],
                                img_path=init_params['img_path'],
                                server=init_params['server_loc'],
                                queue_batch = q,
                                vis=True)
    #save results in queue
    #q.put(batch_results)
def init_cam2cam_pvd(init_params):
    one2one_graph = {361:360, 360:330, 340:360}
    init_params['keep_id_pvd'] = {}
    for cp, ca in one2one_graph.items():
        init_params['keep_id_pvd']['C{}C{}'.format(ca,cp)] = {}
        init_params['keep_id_pvd']['C{}C{}'.format(ca, cp)][ca] = {}
        init_params['keep_id_pvd']['C{}C{}'.format(ca, cp)][cp] = {}
    return init_params['keep_id_pvd']

if __name__ == '__main__':
    onlyPair = 1

    # initialize parameters
    storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62/'
    init_params = {}
    init_params['PANet_detector'] = 0
    init_params['class_id'] = 1
    init_params['backbone'] = 'ResNet34FPN'
    init_params['isSCT_ReID'] = 0
    #PVD
    init_params['only_lane3_ids'] = 0
    init_params['reid_rads_patience'] = {}

    init_params['dist_correct'] = 0
    init_params['print_stats'] = False
    init_params['global_mcta_graph'] = {}
    init_params['assodist_observer'] = []
    init_params['single_cam_multi_asso'] = False
    init_params['isSCT'] = False

    init_params['update_dist_thr'] = False
    init_params['update_frechet_dist'] = False
    init_params['sequential_id'] = False
    init_params['multi_match'] = False

    # TODO: 7 gpus for 7 cameras simultaneously
    init_params['dataset'] = 'clasp1' #'Pease'#'logan'#'exp2training'#'wild-track'#'exp2training'

    if init_params['dataset'] == 'exp2training':
        init_params['only_SCTs'] = False
        init_params['img_HW'] = [720.0, 1080.0] #[1080.0, 1920.0]
        if onlyPair:
            init_params['cams'] = [2, 9] #[1,3]
            init_params['cam_graph'] = {2: [9]}
            init_params['gpu_id'] = [1, 1]
        else:
            init_params['cams'] = [2, 9,4, 5, 11]#
            init_params['gpu_id'] = [0, 1, 0, 1, 1]
            # TODO: design cam graph and use BFS to get the connected cams at each batch
            #init_params['cam_graph'] = {5:[11], 2:[5, 9]}
            init_params['cam_graph'] = {2: [5, 9, 4],  4: [5], 5: [11] }
        batch_end = {'exp2training': 4500}
        init_params['cam_graph'] = {i:init_params['cam_graph'][i] for i in reversed(sorted(init_params['cam_graph'].keys()))}

        init_params['server_loc'] = 'remote'#'remote_30fps'  # 'local'  # remote
        benchmark = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/wrapper_imgs'#storage + 'tracking_wo_bnw/data/CLASP/imgs_30fps'
        #'/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/wrapper_imgs'
        # '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/10FPS'
        init_params['output_dir'] = storage+'tracking_wo_bnw/output/tracktor/online_SCT'

    elif init_params['dataset']=='clasp1':
        init_params['img_HW'] = [1080.0, 1920.0]
        init_params['cams'] = [2, 9]  # , 13, 14]
        # TODO: design cam graph and use BFS to get the connected cams at each batch
        init_params['server_loc'] = 'clasp1'
        init_params['only_SCTs'] = 0
        if onlyPair:
            init_params['cams'] = [2, 9] #[1,3]
            init_params['cam_graph'] = {2: [9]}
            init_params['gpu_id'] = [1, 1]
        else:
            init_params['cams'] = [2, 9,4, 5, 11]#
            init_params['gpu_id'] = [0, 1, 0, 1, 1]
        benchmark = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/' \
                                  'multicamera_wildtrack/wildtrack/Wildtrack_dataset/imgs'
        init_params['output_dir'] = storage+'tracking_wo_bnw/output/tracktor/online_SCT'
        batch_end = {'clasp1': 5000}

    elif init_params['dataset']=='PVD':
        init_params['only_SCTs'] = 0
        benchmark = storage + 'tracking_wo_bnw/data/PVD/HDPVD_new/train_gt/'
        init_params['frame_rate'] = '10FPS'
        init_params['server_loc'] = 'PVD'
        init_params['img_HW'] = [800.0, 1280.0]
        if onlyPair:
            init_params['cams'] = [360, 330]
            init_params['cam_graph'] = {360:[330]} #{340:[300]} #{340:[300]} #{440:[340]} #{340:[300]}
        else:
            init_params['cams'] = [340, 361, 330, 360]
            init_params['cam_graph'] = {361:[360], 360:[330], 340:[360]}#{340:[300], 440: [361, 300], 361:[360], 360:[330], 330:[340]} #340:[360]

        batch_end = {'PVD': 5720} #6002-21 #3590-1712
        if onlyPair:
            init_params['gpu_id'] = [1, 1]
        else:
            init_params['gpu_id'] = [0, 1, 0, 1]


    elif init_params['dataset'] == 'Pease':
        init_params['only_SCTs'] = 1
        benchmark = storage + 'tracking_wo_bnw/data/Pease/train_gt/'
        init_params['dataset'] = 'C'
        init_params['frame_rate'] = '10FPS'
        init_params['server_loc'] = 'Pease'
        init_params['img_HW'] = [576.0, 1024.0]
        if onlyPair:
            init_params['cams'] = [759, 860]
            init_params['cam_graph'] = {759: [860]}  # {340:[300]} #{340:[300]} #{440:[340]} #{340:[300]}
        else:
            init_params['cams'] = [300, 340, 440, 361, 330, 360]
            init_params['cam_graph'] = {340: [300], 440: [361, 300], 361: [360], 360: [330]}  # 340:[360]

        batch_end = {'C': 5720}  # 6002-21 #3590-1712
        if onlyPair:
            init_params['gpu_id'] = [0, 1]
        else:
            init_params['gpu_id'] = [0, 1, 0, 1, 0, 1]

    # 1FPS: GT frames
    init_params['img_path'] = benchmark
    init_params['fr_step'] = 1
    init_params['num_class'] = 2 #3: including TSO, 2: only pax
    init_params['batch_end'] = batch_end[init_params['dataset']]

    if init_params['only_SCTs']:
        init_params['output_dir'] = storage + 'tracking_wo_bnw/output/{}'.format(init_params['dataset'])
    else:
        init_params['output_dir'] = storage + 'tracking_wo_bnw/output/{}/mcta'.format(init_params['dataset'])

    #define data structure for AIT and Metal detector association
    init_params['keep_pvd_id'] = init_cam2cam_pvd(init_params)

    if init_params['PANet_detector']:
        init_params['PANet_dets'] = {}
        # init_params['PANet_dets_dir'] = os.path.join(storage, 'PANet_Results/C]
        # LASP1/PANet_det/dets_30fps')
        # init_params['output_dir'] = storage + 'tracking_wo_bnw/output/panet_supervised_clasp1/'
        init_params['PANet_dets_dir'] = os.path.join(storage, 'PANet_Results/{}/PANet_det'.format(init_params['dataset']))
        for cam in init_params['cams']:
            seq = '{}{}'.format('C', cam)
            init_params['PANet_dets'][cam] = np.loadtxt(os.path.join(init_params['PANet_dets_dir'],seq
                                                                     + '_pb_0aug1nms.txt'), delimiter=',')


    init_params['last_batch_ids'] = {}
    init_params['current_batch_ids'] = {}
    init_params['lane3_ids'] = []
    init_params['raw_meta_map'] = {}

    init_params['new_ids'] = {}
    init_params['prev_affinity'] = {}
    init_params['global_track_id'] = 0
    init_params['keep_raw_id'] = {}  # TODO: keep camera info as well: camera info is not needed since we define unique global tracks
    init_params['keep_tu_id'] = {}

    init_params['batch_size']=40 #20#40
    init_params['min_trklt_size'] = 10#30#10
    init_params['batch_start'] = 1#2001#1#1801#1 #3701
    init_params['start_frame'] = init_params['batch_start']
    init_params['batch_end'] = 4400#13800#4500#13800
    init_params['save_imgs'] = 1
    init_params['vis_rate'] = 1
    if len(init_params['cams']) ==2:
        init_params['vis_pair'] = 1
    else:
        init_params['vis_pair']=0

    init_params['meta_data'] = read_metadata(storage+'tracking_wo_bnw/data/CLASP/Exp2_10fps_People_metadata.xlsx')
    init_params['use_metadata'] = True

    init_params['save_global_tracks'] = True
    init_params['global_full_track_file'] = init_params['output_dir'] + '/global_tracks_kri_10FPS_R50.csv'.format(init_params['cams'][0],init_params['cams'][1])
    init_params['result_fields'] = {'frame':[], 'id':[], 'x1':[], 'y1':[], 'x2':[],'y2':[],
                                    'cam':[], 'timeoffset':[], 'firstused':[], 'tu':[]
                                    }
    init_params['keep_batch_hist'] = {}


    if not osp.exists(init_params['output_dir']):
        os.makedirs(init_params['output_dir'])
    else:
        delete_all(init_params['output_dir'],fmt='jpg')
    print('output will save to : {}'.format(init_params['output_dir']))
    #get config
    configuration = []
    with open(osp.join(os.getcwd().split('scripts')[0],'cfgs/tracktor.yaml')) as file:
        # The FullLoader parameter handles the conversion from YAML
        _config = yaml.load(file, Loader=yaml.FullLoader)
        with open(osp.join(os.getcwd().split('experiments')[0],_config['tracktor']['reid_config'])) as file1:
            _config.update(yaml.load(file1, Loader=yaml.FullLoader))

    # start multiprocess for multi-camera
    #https://pytorch.org/docs/stable/multiprocessing.html
    mp.set_start_method('spawn')
    # global variable for sequential global tracks
    #global_track_num = []
    reid_rads_patience = init_params['reid_rads_patience']
    manager = mp.Manager()
    global_track_num = manager.list()
    global_track_count = manager.list()
    global_track_count.append(0)
    global_track_start_time = manager.dict()
    init_params['det_time_counter'] = manager.dict()
    reid_rads_patience['rads'] = manager.dict()
    reid_rads_patience['patience'] = manager.dict()

    #set of gpus
    num_gpus = torch.cuda.device_count()
    #list of processes
    q=list(range(len(init_params['cams'])))

    #assign process to each gpu
    p = {}
    for i in range(len(q)):
        q[i] = mp.Queue()
        # Pass GPU number through q
        init_params['det_time_counter'][init_params['cams'][i]] = manager.list()
        p[i] = mp.Process(target=SCT_main, args=(q[i], init_params, i, _config, global_track_num,
                                              global_track_count, global_track_start_time, reid_rads_patience))
        p[i].start()

    # video frames: process synchronized frames from 6 cameras
    fr = init_params['start_frame']#+1
    in_loop = True
    max_gpu = []

    while fr<init_params['batch_end']:
        if (fr)%init_params['batch_size']==0:
            if in_loop:
                print('waiting for batch result at frame {}'.format(fr))

            check_list = [not q[i].empty() for i in range(len(q))]
            if all(check_list):

                print('all queue status {}'.format(check_list))
                print('at {} batch started for MCTA'.format(fr))
                assert len(set(global_track_num)) == len(global_track_num), f"{len(set(global_track_num))}"

                global_batch={}
                for i in range(len(q)):
                    global_batch[init_params['cams'][i]] = q[i].get()
                #call MCTA if at least one camera has tracks
                if not init_params['only_SCTs']:
                    start_time = time.time()
                    timeoffset = '{:.2f}'.format(init_params['batch_start'] / 10.0)

                    matches, global_track_results = MCTA(mc_track_segments=global_batch,
                                                        all_params=init_params,
                                                        global_track_start_time=global_track_start_time,
                                                        timeoffset=timeoffset).get_global_matches(vis_pair=init_params['vis_pair'])

                    #print(f"Current GPU memory usage max {max(max_gpu) / 10 ** 6}GB; min was {min(max_gpu) / 10 ** 6}GB")
                    if init_params['print_stats']:
                        print('fr {} found matches: {}'.format(fr, matches))
                        print('MCTA batch completion time {} sec'.format(time.time() - start_time))

                    #visualize global tracks
                    init_params['batch_start']+=init_params['batch_size']
                    in_loop = True

                    if fr>=4400:
                        if init_params['save_global_tracks']:
                            # dump batch results to csv file
                            #Dframe = pd.DataFrame(global_track_results)
                            #Dframe.to_csv(init_params['global_full_track_file'], mode='w',index=False)
                            global_tracker_file = open(init_params['output_dir'] +
                                                                        '/global_tracks_pvd_10FPS_R50.txt'.format(
                                                                            init_params['cams'][0],
                                                                            init_params['cams'][1]), 'w')
                            for fr,x1,y1,x2,y2,id,cam in zip(global_track_results['frame'], global_track_results['x1'], global_track_results['y1'],
                                                global_track_results['x2'], global_track_results['y2'],global_track_results['id'], global_track_results['cam']):
                                camera = str(cam)
                                event_type = 'LOC:'
                                type = 'PAX'
                                pax_id = str(id)
                                if pax_id.isdigit():
                                    pax_id = 'P'+pax_id
                                global_tracker_file.writelines(
                                    event_type + ' ' + 'type: ' + type + ' ' + 'camera-num: ' + camera + ' ' + 'frame: ' + str(fr)
                                    + ' '+ 'time-offset: ' + '{:.2f}'.format(fr / 10.0) + ' '
                                    + 'BB: ' + str(int(x1)) + ', ' + str(int(y1)) + ', '
                                    + str(int(x2)) + ', ' + str(int(y2)) + ' ' + 'ID: ' + pax_id + '\n')
                        break
            else:
                in_loop = False
                    #break
                continue
        fr += 1
    #end all the running processes
    for i in list(range(len(init_params['cams']))):
        p[i].join()
        p[i].terminate()