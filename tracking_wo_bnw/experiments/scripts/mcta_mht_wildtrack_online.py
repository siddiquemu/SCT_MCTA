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
from PIL import Image
import motmetrics as mm

import torchvision
import yaml
import pandas as pd
import pdb
import logging
import random
from read_config import Configuration
from MCTA.MCTA_combined_new import MCTA

# TODO:
# 1. replace data loader with clasp batch image reader or online image reader
# 2.
class SCT_Clasp(object):
    def  __init__(self, _config, _log,
                 cam, output_dir, num_class=2,
                 batch=120, gpu=0, min_trklt_size=30,
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
        self.debug = init_flags['print_stats']
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

if __name__ == '__main__':
    from read_meta_data import *
    onlyPair = 0
    # initialize parameters
    server = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/' \
             'multicamera_wildtrack/wildtrack/Wildtrack_dataset/'
    storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/'
    benchmark = storage+'tracking_wo_bnw/data/wild-track/'

    init_params = {}
    init_params['colors'] = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)]) for i in
                  range(5000)]
    init_params['print_stats'] = False
    init_params['global_mcta_graph'] = {}
    init_params['assodist_observer'] = []
    init_params['single_cam_multi_asso'] = False
    init_params['isSCT'] = False
    init_params['update_dist_thr'] = False
    init_params['update_frechet_dist'] = False
    init_params['sequential_id'] = False

    init_params['gpu_id'] = [0, 1, 0, 1]
    init_params['num_class'] = 2
    init_params['dist_metric'] ='frechet'#'hausdorff'# 'frechet'#'hausdorff'
    # TODO: 7 gpus for 7 cameras simultaneously
    init_params['dataset'] = 'wild-track'#'logan'#'exp2training'#'wild-track'#'exp2training'
    if init_params['dataset'] == 'exp2training':
        init_params['only_SCTs'] = False
        init_params['img_HW'] = [720.0, 1080.0]  # [720.0, 1080.0] #[1080.0, 1920.0]
        if onlyPair:
            init_params['cams'] = [2, 4]
            init_params['cam_graph'] = {2: [4]}
            init_params['gpu_id'] = [0, 1]
        else:
            init_params['cams'] = [2,9,5,4, 11]#
            # TODO: design cam graph and use BFS to get the connected cams at each batch
            init_params['cam_graph'] = {5:[11], 4: [5], 2:[5, 9, 4]}

        init_params['cam_graph'] = {i:init_params['cam_graph'][i] for i in reversed(sorted(init_params['cam_graph'].keys()))}

        init_params['server_loc'] = 'remote'  # 'local'  # remote
        init_params['img_path'] = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/wrapper_imgs'
        # '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/10FPS'
        init_params[
            'output_dir'] = storage+'tracking_wo_bnw/output/tracktor/online_SCT'
    else:
        init_params['img_HW'] = [1080.0, 1920.0]#[960.0, 1280.0]#[1080.0, 1920.0]
        init_params['cams'] = [1, 2, 3, 4]  # , 13, 14]
        # TODO: design cam graph and use BFS to get the connected cams at each batch
        init_params['cam_graph'] = {1: [2,3,4],2:[3,4],3:[4]}
        init_params['server_loc'] = 'wild-track'#'logan'#'wild-track'  # 'local'  # remote
        if init_params['server_loc'] == 'wild-track':
            init_params['only_SCTs'] = False
            init_params['img_path'] = server+'sync_frames/imgs_30fps'
            # for tracks
            init_params['output_dir'] = storage + 'tracking_wo_bnw/output/tracktor_wildtrack/global_vis'
            if not os.path.exists(init_params['output_dir']):
                os.makedirs(init_params['output_dir'])
        else:
            init_params['only_SCTs'] = True
            init_params['img_path']=storage+'tracking_wo_bnw/data/logan-data/exp1-train'
            init_params['output_dir'] = storage+'tracking_wo_bnw/output/tracktor/online_SCT'

    init_params['last_batch_ids'] = {}
    init_params['current_batch_ids'] = {}
    init_params['new_ids'] = {}
    init_params['prev_affinity'] = {}
    init_params['global_track_id'] = 0
    init_params['keep_raw_id'] = {}  # TODO: keep camera info as well: camera info is not needed since we define unique global tracks
    init_params['batch_size']= 120
    init_params['min_trklt_size'] = 10
    init_params['batch_start'] = 1
    init_params['start_frame'] = 1
    init_params['start_gt_frame'] = 0
    init_params['batch_end'] = 6001
    init_params['save_imgs'] = 0
    init_params['vis_rate'] = 1
    if len(init_params['cams']) ==2:
        init_params['vis_pair'] = 1
    else:
        init_params['vis_pair']=0
    init_params['save_global_tracks'] = True
    init_params['global_full_track_file'] = init_params['output_dir'] + '/global_track_wildtrack.csv'

    init_params['result_fields'] = {'frame': [], 'id':[], 'x':[], 'y':[], 'w':[],'h':[], 'cam': []}



    if not osp.exists(init_params['output_dir']):
        os.makedirs(init_params['output_dir'])
    else:
        delete_all(init_params['output_dir'],fmt='png')

    # video frames: process synchronized frames from 6 cameras
    fr = init_params['start_frame']#+1
    in_loop = True

    max_gpu = []
    while fr<init_params['batch_end']:

        if (fr)%init_params['batch_size']==0:
            #prepare global batch from mht tracks of all camera
            #use batch frame range:1-120,121-240,.. and cams:1,2,3,4
            global_batch={}
            for i in init_params['cams']:
                #currently we get all previous frames for a track at a batch interval: only batch may split the track
                global_batch[init_params['cams'][i]] = q[i].get()

            if init_params['print_stats']:
                print('cam {} ids: {}'.format(init_params['cams'][0], global_batch[init_params['cams'][0]].keys()))
                print('cam {} ids: {}'.format(init_params['cams'][1], global_batch[init_params['cams'][1]].keys()))
                print('cam {} ids: {}'.format(init_params['cams'][2], global_batch[init_params['cams'][2]].keys()))
                print('cam {} ids: {}'.format(init_params['cams'][3], global_batch[init_params['cams'][3]].keys()))

            #keep track of all ids so that we can know about new ids in a batch: new id will search corresponding cameras to find association
            #call MCTA if at least one camera has tracks
            if not init_params['only_SCTs']:
                start_time = time.time()
                tracemalloc.start()
                timeoffset = '{:.2f}'.format(init_params['batch_start'] / 30.0)
                global_track_results = MCTA(mc_track_segments=global_batch,
                                            all_params=init_params,
                                            global_track_start_time=global_track_start_time,
                                            ).get_global_matches()

                current, peak = tracemalloc.get_traced_memory()

               #print(global_track_results)
                if init_params['print_stats']:
                    print(f"Current memory usage is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
                tracemalloc.stop()
                #gpu memory
                #for gpuid in init_params['gpu_id']:
                    #max_gpu.append(torch.cuda.max_memory_allocated(device=torch.device(gpuid)))

                #print(f"Current GPU memory usage max {max(max_gpu) / 10 ** 6}GB; min was {min(max_gpu) / 10 ** 6}GB")
                if init_params['print_stats']:
                    print('MCTA batch completion time {} sec'.format(time.time() - start_time))

                #observe asso dist
                #visualize global tracks
                init_params['batch_start']+=init_params['batch_size']

                if init_params['print_stats']:
                    print('global_track'.format(global_track_num))
                in_loop = True
                if init_params['save_global_tracks'] and fr>=6000:#10000:
                    # dump batch results to csv file
                    Dframe = pd.DataFrame(global_track_results)
                    Dframe.to_csv(init_params['global_full_track_file'], mode='w',index=False)
            else:
                in_loop = False
                    #break
                continue

        fr += 1