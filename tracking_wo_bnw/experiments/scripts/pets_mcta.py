import multiprocessing as mp
import os
import csv
import time
from os import path as osp
import psutil
import tracemalloc
import sys
sys.path.insert(0,'/home/marquetteu/MCTA/tracking_wo_bnw/experiments/cfgs')
sys.path.insert(1,'/home/marquetteu/MCTA/tracking_wo_bnw/experiments/scripts/MCTA')
sys.path.insert(2,'/home/marquetteu/MCTA/tracking_wo_bnw/src')

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
from tracktor.reid.resnet import resnet50
from tracktor.utils import interpolate, plot_sequence, plot_scanned_track, \
    filter_tracklets, delete_all, get_mot_accum, \
    evaluate_mot_accums, plot_dets
import pdb
import logging
from read_config import Configuration
from MCTA.MCTA_Online import MCTA

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
        self.nms_thr = self.tracktor['tracker']['detection_nms_thresh']
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
    def write_cam_results(all_tracks, output_dir, cam=None, appearance=None):
        # this function is used when caresults are save to do offline MCTA association
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file = osp.join(output_dir, f'C{cam}.txt')

        print("[*] Writing to: {}".format(file))

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
                        #[fr,id,x,y,w,h]
                        t_feature = [frame, i , x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, cam ]
                        writer.writerow(t_feature)
                    else:
                        v_cx = bb[5]
                        v_cy = bb[6]
                        app = bb[7::]
                        t_feature = [frame, i , x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, -1, -1, -1, -1, v_cx, v_cy, cam]
                        writer.writerow(t_feature+app.tolist())
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

    def init_det_model(self):
        # object detection
        self._log.info("Initializing object detector.")
#'obj_detect_wildtrack_model'
        self.obj_detect = FRCNN_FPN(num_classes=self.num_class)
        self.obj_detect.load_state_dict(torch.load(self._config['tracktor']['obj_detect_model_base'],
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
        tracker = Tracker(self.init_det_model(), self.init_reid_model(), self.tracktor['tracker'],
                          self.global_track_num, self.global_track_start_time, self.init_flags['start_frame'])
        return tracker

    def get_img_blob(self, path_i, dets=[]):
        """Return the ith image converted to blob"""
        img = Image.open(path_i).convert("RGB")
        if img.size[1]==self.init_flags['img_HW'][0]:
            img = img.resize((int(self.init_flags['img_HW'][1]), int(self.init_flags['img_HW'][0])))
        img = self.transforms(img)

        sample = {}
        sample['img'] = torch.reshape(img, (1, 3, int(self.init_flags['img_HW'][0]),
                                            int(self.init_flags['img_HW'][1])))
        sample['dets'] = torch.tensor([det[:4] for det in dets])
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
        if data_loc == 'pets2009':
            if self.cam == 1:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'View_001')
            if self.cam == 2:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'View_002')
            if self.cam == 3:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'View_003')
            if self.cam == 4:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'View_004')
            if self.cam == 5:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'View_005')
            if self.cam == 6:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'View_006')
            if self.cam == 7:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'View_007')
            if self.cam == 8:
                fr_num = fr_num - 0
                img_path = os.path.join(img_path, 'View_008')
            path_i = os.path.join(img_path, 'frame_{:04d}.jpg'.format(fr_num))
        if not osp.exists(path_i):
            return None

        return self.get_img_blob(path_i)

    def test_SCT(self, tracktor=None, start_frame=1, fr_step = 1, dataset=None,
                 img_path=None, server='local', queue_batch=None, last_frame=2000, vis=False):
        # function to complete tracking by detection for a camera
        # start frame use to do synchronization in mc system

        num_frame = start_frame
        while num_frame <= last_frame:
            frame = self.init_online_dataloader(dataset, img_path, num_frame, data_loc=server)
            if frame is None:
                #print('None frame found at {}'.format(num_frame))
                continue
                #break
            #print('Frame', frame['img_path'])
            # pdb.set_trace()
            # tracks result should contain the followings;
            # 1. pos: [x1,y1,x2,y2]
            # 2. motion: [v_cx,v_cy]
            # 3. appearance: [128D_app_descriptor]
            # 4. identity: ID
            # 5. Frame index: fr
            # 6. score: track score is optional
            print('processing {} in C{}'.format(os.path.basename(frame['img_path']),self.cam))
            if frame is not None:
                with torch.no_grad():
                    tracktor.step(frame, self.cam)
            #at last frame return all tracking history in camera
            if num_frame%last_frame==0:
                cam_results = tracktor.get_results()
                self.write_cam_results(cam_results, self.out_dir, cam=self.cam, appearance=None)
                queue_batch.put(cam_results)

            if num_frame % 15 == 0 and num_frame>0:
                save_dir = os.path.join(self.out_dir,'C{}'.format(self.cam))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                cam_results = tracktor.get_results()
                plot_scanned_track(num_frame, cam_results, frame['img_path'], save_dir,
                                   im_wh=(int(self.init_flags['img_HW'][1]), int(self.init_flags['img_HW'][0])),
                                   cam=self.cam)
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

        #file to write frame by frame detections
        file = open(osp.join(self.out_dir, f'C{self.cam}.txt'), "w")
        file = csv.writer(file, delimiter=',')

        num_frame = start_frame
        while num_frame <= last_frame:
            frame = self.init_online_dataloader(dataset, img_path, num_frame, data_loc=server)
            if frame is None:
                print('None frame found at {}'.format(num_frame))
                continue
                #break
            print('processing {} in C{}'.format(os.path.basename(frame['img_path']),self.cam))
            if frame is not None:
                self.obj_detect.load_image(frame['img'])
                boxes, scores = self.obj_detect.detect(frame['img'])

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
                else:
                    det_pos = torch.zeros(0).cuda()
                    det_scores = torch.zeros(0).cuda()


                if det_pos.nelement() > 0:
                    print('number of dets before nms {}'.format(det_pos.shape))
                    if not self.init_flags['skip_nms']:
                        keep = nms(det_pos, det_scores, self.nms_thr)
                        det_pos = det_pos[keep]
                        det_scores = det_scores[keep]

                    camfr_results = torch.cat((det_pos, det_scores.resize(det_scores.shape[0], 1)), dim=1)
                    fr = float(os.path.basename(frame['img_path']).split('.')[0][-4:])
                    self.write_det_results(camfr_results, file, fr, cam=self.cam, appearance=None)

            if det_pos.nelement()>0 and num_frame % 15 == 0:
                plot_dets(num_frame, camfr_results, frame['img_path'], save_dir,
                          im_wh=(int(self.init_flags['img_HW'][1]), int(self.init_flags['img_HW'][0])), cam=self.cam)
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

if __name__ == '__main__':
    # initialize parameters
    storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/'
    benchmark = storage+'tracking_wo_bnw/data/pets2009/S2_L1/Crowd_PETS09/S2/L1/Time_12-34'

    init_params = {}
    init_params['dataset'] = 'pets2009'
    init_params['server_loc'] = 'pets2009'
    init_params['print_stats'] = False
    init_params['global_mcta_graph'] = {}
    init_params['single_cam_multi_asso'] = False
    init_params['isSCT'] = False
    init_params['sequential_id'] = False

    init_params['gpu_id'] = [1,0, 1, 0]
    init_params['num_class'] = 2

    init_params['img_HW'] = [576.0, 768.0,]
    init_params['cams'] = [1,3,4]#[5,6,7,8]#[1,3,4]#[1,2,3,4]#
    # TODO: design cam graph and use BFS to get the connected cams at each batch
    init_params['cam_graph'] = {1: [2]}
    if init_params['server_loc'] == 'pets2009':
        init_params['only_SCTs'] = False
        init_params['img_path'] = benchmark
        init_params['fr_step'] = 1
    #for tracks
    init_params['output_dir'] = storage+'tracking_wo_bnw/output/tracktor_pets/SCT'
    #for det
    #init_params['output_dir'] = storage+'tracking_wo_bnw/output/tracktor-det-pets-wo-nms'

    init_params['skip_nms'] = True
    init_params['last_batch_ids'] = {}
    init_params['current_batch_ids'] = {}
    init_params['new_ids'] = {}
    init_params['prev_affinity'] = {}
    init_params['global_track_id'] = 0
    init_params['keep_raw_id'] = {}
    init_params['batch_size'] = None
    init_params['min_trklt_size'] = 5
    init_params['batch_start'] = 0
    init_params['start_frame'] = 0
    init_params['batch_end'] = 794
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

    # start multiprocess for multi-camera
    #https://pytorch.org/docs/stable/multiprocessing.html
    mp.set_start_method('spawn')
    # global variable for sequential global tracks
    #global_track_num = []
    manager = mp.Manager()
    global_track_num = manager.list()
    global_track_start_time = manager.dict()
    #set of gpus
    num_gpus = torch.cuda.device_count()
    #list of processes
    q=list(range(len(init_params['cams'])))
    #assign process to each gpu
    for i in range(len(q)):
        q[i] = mp.Queue()
        # Pass GPU number through q
        p = mp.Process(target=SCT_main, args=(q[i], init_params, i, _config, global_track_num, global_track_start_time))
        p.start()
    # video frames: process synchronized frames from 6 cameras
    p.join()
    p.terminate()