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
from tracktor.utils import interpolate, plot_sequence, plot_scanned_track, filter_tracklets, delete_all, get_mot_accum, \
    evaluate_mot_accums
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

    def init_det_model(self):
        # object detection
        self._log.info("Initializing object detector.")

        self.obj_detect = FRCNN_FPN(num_classes=self.num_class)
        self.obj_detect.load_state_dict(torch.load(self._config['tracktor']['obj_detect_wildtrack_model'],
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
                img_path = os.path.join(img_path, 'cam02exp2.mp4')
            if self.cam==9:
                img_path = os.path.join(img_path, 'cam09exp2.mp4')
            if self.cam==5:
                img_path = os.path.join(img_path, 'cam05exp2.mp4')
            if self.cam==4:
                img_path = os.path.join(img_path, 'cam04exp2.mp4')
            if self.cam==11:
                img_path = os.path.join(img_path, 'cam11exp2.mp4')
            if self.cam==13:
                img_path = os.path.join(img_path, 'cam13exp2.mp4')
            if self.cam==14:
                img_path = os.path.join(img_path, 'cam14exp2.mp4')

            path_i = os.path.join(img_path, 'cam{}_{:06d}.jpg'.format(self.cam, fr_num))

        if data_loc == 'wild-track':
            if self.cam == 1:
                img_path = os.path.join(img_path, 'C1') #exp1
            if self.cam == 2:
                img_path = os.path.join(img_path, 'C2') #exp2-C7
            if self.cam == 3:
                img_path = os.path.join(img_path, 'C3')
            if self.cam == 4:
                img_path = os.path.join(img_path, 'C4')
            path_i = os.path.join(img_path, '{:08d}.png'.format(fr_num))
            #print(path_i)

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

        if not osp.exists(path_i):
            return None

        return self.get_img_blob(path_i)

    def test_SCT(self, tracktor=None, start_frame=None, dataset=None,
                     img_path=None, server='local', queue_batch=None, vis=False):
            if self.debug:
                tracemalloc.start()
                process = psutil.Process(os.getpid())
                print(process.pid)
                print(str(round(process.memory_info()[0] / (1024 * 1024))) + ' MB')
            num_frame = start_frame
            start_time = time.time()
            while num_frame < 6002:
                time_offset = '{:.2f}'.format((num_frame)/30.0) #-1
                frame = self.init_online_dataloader(dataset, img_path, num_frame, data_loc=server)
                if frame is None:
                    #print('None frame found at {}'.format(num_frame))
                    continue
                    #break
                #print('Frame', frame['img_path'])
                # pdb.set_trace()
                # Scanned tracks result should contain the followings;
                # 1. pos: [x1,y1,x2,y2]
                # 2. motion: [v_cx,v_cy]
                # 3. appearance: [128D_app_descriptor]
                # 4. identity: ID
                # 5. Frame index: fr
                # 6. score: track score is optional
                if frame is not None:
                    with torch.no_grad():
                        tracktor.step(frame, self.cam)
                #at scan interval return all tracking history in camera

                if num_frame%self.scan_intrvl==0:
                    if self.debug:
                        tracemalloc.start()
                        process = psutil.Process(os.getpid())
                        print(process.pid)
                        print(str(round(process.memory_info()[0] / (1024 * 1024))) + ' MB')
                    batch_results = tracktor.get_results()

                    batch_results = filter_tracklets(num_frame, batch_results, start_frame=start_frame,
                                                     batch_length=self.scan_intrvl, min_trklt_size=self.min_trklt_size,
                                                     reid_patience=self.reid_patience)

                    #plot_scanned_track(num_frame, batch_results, frame['img_path'], self.out_dir,cam=self.cam)
                    queue_batch.put(batch_results)
                    if self.debug:
                        print('at {} batch ids {}'.format(num_frame, batch_results.keys()))
                        print('batch tracking speed in cam {}: {:2f} Hz'
                          .format(self.cam, self.scan_intrvl/(time.time()-start_time)))
                    start_time = time.time()
                num_frame += 1

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
                                dataset=init_params['dataset'],
                                img_path=init_params['img_path'],
                                server=init_params['server_loc'],
                                queue_batch = q,
                                vis=True)
    #save results in queue
    #q.put(batch_results)

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

    # for total ram usage
    tracemalloc.start()
    process = psutil.Process(os.getpid())
    print(process.pid)
    print(str(round(process.memory_info()[0] / (1024 * 1024))) + ' MB')
    tracemalloc.stop()
    #assign process to each gpu
    for i in range(len(q)):
        q[i] = mp.Queue()
        # Pass GPU number through q
        p = mp.Process(target=SCT_main, args=(q[i], init_params, i, _config, global_track_num, global_track_start_time))
        p.start()
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
                global_batch={}
                for i in range(len(q)):
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
    p.join()
    p.terminate()