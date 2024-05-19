import os
import time
from os import path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from PIL import Image
import motmetrics as mm
mm.lap.default_solver = 'lap'

import torchvision
import yaml
from tqdm import tqdm
import sacred
from sacred import Experiment
from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.config import get_output_dir
from tracktor.datasets.factory import Datasets
from tracktor.oracle_tracker import OracleTracker
from tracktor.tracker import Tracker
from tracktor.reid.resnet import resnet50
from tracktor.utils import interpolate, plot_sequence, plot_scanned_track, filter_tracklets, delete_all, get_mot_accum, evaluate_mot_accums
import pdb

#TODO:
#1. replace data loader with clasp batch image reader or online image reader
#2.
class SCT_Clasp(object):
    def __init__(self,tracktor, reid, _config, _log, _run,cam,num_class,output_dir,batch=120,gpu=0):
        self.cam=cam
        self.num_class=num_class
        self.time_total = 0
        self.num_frames = 0
        self.scan_intrvl = batch
        self.gpu=gpu
        self.transforms = ToTensor()
        self.out_dir = output_dir
        self.tracktor = tracktor
        self.reid = reid
        self._config = _config
        self._log = _log
        self._run = _run

    def init_det_model(self):
        # object detection
        self._log.info("Initializing object detector.")

        self.obj_detect = FRCNN_FPN(num_classes=self.num_class)
        self.obj_detect.load_state_dict(torch.load(self._config['tracktor']['obj_detect_model'],
                                              map_location=lambda storage, loc: storage))
        self.obj_detect.eval()
        self.obj_detect.cuda()
        return self.obj_detect

    def init_reid_model(self):
        self.reid_network = resnet50(pretrained=False, **self.reid['cnn'])
        self.reid_network.load_state_dict(torch.load(self.tracktor['reid_weights'],
                                                map_location=lambda storage, loc: storage))
        self.reid_network.eval()
        self.reid_network.cuda()
        return self.reid_network

    def init_tracktor(self):
        tracker = Tracker(self.init_det_model(), self.init_reid_model(), self.tracktor['tracker'])
        return tracker

    def get_img_blob(self, path_i, dets=[]):
        """Return the ith image converted to blob"""
        img = Image.open(path_i).convert("RGB")
        img = self.transforms(img)

        sample = {}
        sample['img'] = torch.reshape(img,(1,3,1080,1920))
        sample['dets'] = torch.tensor([det[:4] for det in dets])
        sample['img_path'] = path_i
        sample['gt'] = {}
        sample['vis'] = {}

        return sample

    def init_online_dataloader(self,dataset,img_path,batch_start, data_loc):
        '''

        :param dataset: 'exp2training'
        :param img_path:
        :param frIndex:
        :return: return a batch of frames to SCT_clasp
        '''
        '''
        if data_loc='remote_server'
            myurl = 'http://127.0.0.1:5000/frames'
    
            # Prepare all the inputs
            # Examples
            dataset = 'exp2training'
            cameralist = '9,11,13'
            timeoffset = '440.00'
            duration = 3
            duration = "{0:.2f}".format(float(duration))
            while True:
                inputParams = {"dataset": dataset, "cameralist": cameralist, "timeoffset": timeoffset, "duration": duration,
                               "filesize": "1920x1080"}
                jsoninputParams = json.dumps(inputParams)
                jsonParams = {"APIParams": jsoninputParams}
                response = requests.get(url=myurl, params=jsonParams)
                if response.text == 'No More Frames':
                    print('Reached end of video!')
                    break
                else:
                    zf = zipfile.ZipFile(io.BytesIO(response.content))
                    zf.extractall('<Output Path>')  # replace with the actual output path
                    # get all batch images from the server
                    filebytes = io.BytesIO(response.content)
                    myzipfile = zipfile.ZipFile(filebytes)
                    for name in myzipfile.namelist():
                        [ call online SCT_clasp to track targets of a batch and update tracker state. If necessary return the tracker state to video feed server ]
                    timeoffset = float(timeoffset) + float(duration)
        '''
        frames=[]
        if data_loc=='local':
            for fr in range(batch_start, batch_start+self.scan_intrvl): #100 ot 220
                path_i = img_path + '/%06d' % fr + '.png'
                frames.append(self.get_img_blob(path_i))
        return frames

    def test_SCTA_clasp(self,fr_start,fr_end,dataset=None,img_path=None,server='local',vis=False):
        tracker = self.init_tracktor()
        num_frame = fr_start
        for batch_start in range(fr_start, fr_end, self.scan_intrvl):
            #frames are collected for each batch: for all cameras??
            frames = self.init_online_dataloader(dataset, img_path, batch_start, data_loc=server)
            for frame in frames:
                print('Frame',num_frame)
                num_frame+=1
                # pdb.set_trace()
                # Scanned tracks result should contain the followings;
                # 1. pos: [x1,y1,x2,y2]
                # 2. motion: [v_cx,v_cy]
                # 3. appearance: [128D_app_descriptor]
                # 4. identity: ID
                # 5. Frame index: fr
                # 6. score: track score is optional
                with torch.no_grad():
                    tracker.step(frame)

            batch_results = tracker.get_results()
            batch_results = filter_tracklets(num_frame-1, batch_results, min_len=self.scan_intrvl, reid_patience=10)
            plot_scanned_track(num_frame-1, batch_results, frame['img_path'], self.out_dir)
        return batch_results, tracker