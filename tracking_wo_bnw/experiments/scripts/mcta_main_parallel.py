import multiprocessing as mp
import os
import time
from os import path as osp
import psutil
import tracemalloc

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
from read_config import Configuration
from MCTA.MCTA_parallel import MCTA

# TODO:
# 1. replace data loader with clasp batch image reader or online image reader
# 2.
class SCT_Clasp(object):
    def  __init__(self, _config, _log,
                 cam, output_dir, num_class=2,
                 batch=120, gpu=0, min_trklt_size=30,
                global_track_num=None, batch_finished_cams=None):
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
        self._log = _log
        self.reid_patience = 30
        self.min_trklt_size = min_trklt_size
        self.batch_finished_cams = batch_finished_cams
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
    def read_flags(init_params):
        with open(osp.join(init_params['flags_mu']['path'],'Flags.yaml')) as file:
            server_flags = yaml.load(file, Loader=yaml.FullLoader)
            if server_flags is not None:
                if server_flags['People_Processed'] == 'TRUE':
                    init_params['flags_mu']['People_Processed'] = 'TRUE'
                else:
                    init_params['flags_mu']['People_Processed'] = 'FALSE'

            else:
                init_params['flags_mu']['People_Processed'] = 'FALSE'
        return init_params

    @staticmethod
    def write_flags(init_params):
        #with open(osp.join(init_params['flags_mu']['path'],'Flags_MU.yaml')) as file:
            #server_flags = yaml.load(file, Loader=yaml.FullLoader)
        if init_params['flags_mu']['People_Processed']=='TRUE':
            flag_value = {'People_Processed':'TRUE'}
        else:
            flag_value = {'People_Processed':'FALSE'}
        with open(osp.join(init_params['flags_mu']['path'],'Flags.yaml'), 'w') as flags_file:
            yaml.dump(flag_value, flags_file)

    def init_det_model(self):
        # object detection
        self._log.info("Initializing object detector.")

        self.obj_detect = FRCNN_FPN(num_classes=self.num_class)
        self.obj_detect.load_state_dict(torch.load(self._config['tracktor']['obj_detect_model'],
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
        tracker = Tracker(self.init_det_model(), self.init_reid_model(), self.tracktor['tracker'], self.global_track_num)
        return tracker

    def get_img_blob(self, path_i, dets=[]):
        """Return the ith image converted to blob"""
        img = Image.open(path_i).convert("RGB")
        #img = img.resize((540,960))
        img = self.transforms(img)

        sample = {}
        sample['img'] = torch.reshape(img, (1, 3, 1080,1920))
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
            if not osp.exists(path_i):
                return None

        return self.get_img_blob(path_i)

    def test_SCT(self, tracktor=None, start_frame=None, dataset=None,
                 img_path=None, server='local', queue_batch=None, vis=False):
        tracemalloc.start()
        process = psutil.Process(os.getpid())
        print(process.pid)
        print(str(round(process.memory_info()[0] / (1024 * 1024))) + ' MB')
        num_frame = start_frame
        start_time = time.time()
        while num_frame < 13800:

            time_offset = '{:.2f}'.format((num_frame)/30.0) #-1
            frame = self.init_online_dataloader(dataset, img_path, num_frame, data_loc=server)
            if frame is None:
                print('None frame found at {}'.format(num_frame))
                continue
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
                tracemalloc.start()
                process = psutil.Process(os.getpid())
                print(process.pid)
                print(str(round(process.memory_info()[0] / (1024 * 1024))) + ' MB')
                batch_results = tracktor.get_results()
                batch_results = filter_tracklets(num_frame , batch_results, start_frame=start_frame,
                                                 batch_length=self.scan_intrvl, reid_patience=self.reid_patience)
                print('at {} batch ids {}'.format(num_frame, batch_results.keys()))
                #plot_scanned_track(num_frame - 1, batch_results, frame['img_path'], self.out_dir,cam=self.cam)
                queue_batch.put(batch_results)
                # to keep track which camera is finished and which camera is unfinished
                self.batch_finished_cams.append(self.cam)
                if self.batch_finished_cams==[2,9,5,11]:
                    self.batch_finished_cams = []
                print('batch tracking speed in cam {}: {:2f} Hz'
                      .format(self.cam, self.scan_intrvl/(time.time()-start_time)))
                start_time = time.time()
            num_frame += 1


def SCT_main(q, init_params, cam_index, _config, global_track_num, batch_processed_cams):
    # ***Args: gpu, cam_x_batch, model
    # ***Return: batch tracking results: keep active/inactive tracks for associating with next batch
    #os.environ["CUDA_VISIBLE_DEVICES"] = str(init_params['gpu_id'][gpu_index])
    torch.cuda.set_device(init_params['gpu_id'][cam_index])
    torch.set_num_threads(1)
    _log = logging
    print('process start with gpu {} cam {}'.format(init_params['gpu_id'][cam_index], cam_index))

    single_cam_tracker = SCT_Clasp( _config,
                                   _log,
                                   cam_index,
                                   init_params['output_dir'],
                                   num_class=init_params['num_class'],
                                   batch=init_params['batch_size'],
                                   gpu=init_params['gpu_id'][cam_index],
                                   min_trklt_size=init_params['min_trklt_size'],
                                   global_track_num=global_track_num,
                                   batch_finished_cams = batch_processed_cams
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
    # initialize parameters
    init_params = {}
    init_params['global_mcta_graph'] = {}
    init_params['mc_all_trklts_dict'] = {}
    init_params['assodist_observer'] = []
    init_params['single_cam_multi_asso'] = False
    init_params['isSCT'] = False
    init_params['update_dist_thr'] = True
    init_params['update_frechet_dist'] = False
    init_params['sequential_id'] = False

    init_params['gpu_id'] = {2:0,9:0,5:1,11:1}
    init_params['process_id'] = {0: 2, 1:9, 2: 5, 3: 11}
    init_params['num_class'] = 2
    # TODO: 7 gpus for 7 cameras simultaneously
    init_params['dataset'] = 'exp2training'
    init_params['cams'] = [2, 9, 5, 11]#, 13, 14]
    # TODO: design cam graph and use BFS to get the connected cams at each batch
    init_params['cam_graph'] = ((2,9),(2,5),(5,11))#{2: [5], 5:[11]}#,13], 11:[13], 13:[14]} #9, 5, 4], 4: [5], 9: [2], 5: [11, 13], 11: [13], 13: [14]}
    init_params['last_batch_ids'] = {}
    init_params['current_batch_ids'] = {}
    init_params['new_ids'] = {}
    init_params['prev_affinity'] = {}
    init_params['global_track_id'] = 0
    init_params['keep_raw_id'] = {}  # TODO: keep camera info as well: camera info is not needed since we define unique global tracks
    init_params['batch_size']= 40#120#40
    init_params['min_trklt_size'] = 15#30#15
    init_params['batch_start'] = 1#3600
    init_params['start_frame'] =1#3600
    init_params['batch_end'] = 4500#13800
    init_params['server_loc'] = 'local'  # remote
    init_params['img_path'] = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/10FPS'
    init_params['output_dir'] = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d6/tracking_wo_bnw/output/tracktor/online_SCT'
    #init_params['flags']['path'] = init_params['output_dir']
    init_params['global_full_track_file'] = init_params['output_dir'] + '/global_tracks_new_10FPS.csv'

    init_params['result_fields'] = {'frame': [], 'cam':[], 'timeoffset': [],
                                      'x1':[], 'y1':[], 'x2':[],'y2':[],
                                        'id': [], 'firstused':[]}
    init_params['keep_batch_hist'] = {}
    init_params['mcta_batch_finished'] = False



    if not osp.exists(init_params['output_dir']):
        os.makedirs(init_params['output_dir'])
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
    batch_processed_cams = manager.list()
    #set of gpus
    num_gpus = torch.cuda.device_count()
    #list of processes
    q=[0,1,2, 4]#,4,5]

    # for total ram usage
    tracemalloc.start()
    process = psutil.Process(os.getpid())
    print(process.pid)
    print(str(round(process.memory_info()[0] / (1024 * 1024))) + ' MB')
    tracemalloc.stop()
    #assign process to each gpu
    for pid, cam in init_params['process_id'].items():
        q[pid] = mp.Queue()
        # Pass GPU number through q
        p = mp.Process(target=SCT_main, args=(q[pid], init_params, cam, _config, global_track_num, batch_processed_cams))
        p.start()
    # video frames: process synchronized frames from 6 cameras
    fr = init_params['start_frame']#+1
    in_loop = True

    max_gpu = []
    init_params['process_id'] = {value: key for key, value in init_params['process_id'].items()}
    while fr<init_params['batch_end']:

        if (fr)%init_params['batch_size']==0:

            timeoffset = '{:.2f}'.format(init_params['batch_start'] / 30.0)
            mcta_batch = MCTA(all_params=init_params,
                              timeoffset=timeoffset)

            print('start batch at {}'.format(init_params['batch_start']))
            global_batch={}
            associated_cams = []
            cam_visited = []
            #to start camera pair search
            init_params['mcta_batch_finished'] = False
            #get batch tracks independently
            print('SCTs done for cameras: {}'.format(batch_processed_cams))
            while not init_params['mcta_batch_finished']:
                for (cp, ca) in init_params['cam_graph']:
                    # after getting content queue should be empty
                    #print('searching for ca {} and cp {}'.format(ca, cp))
                    # TODO: do all camera preprocessing at this step currently we are doing trackers preprocessing multiple times for a camera: 2 times for C2 for (2,9) and (2,5)
                    if ca in batch_processed_cams and ca not in cam_visited:
                        global_batch[ca] = q[init_params['process_id'][ca]].get()
                        cam_visited.append(ca)
                        print('cam {} ids: {}'.format(ca, global_batch[ca].keys()))
                    if cp in batch_processed_cams and cp not in cam_visited:
                        global_batch[cp] = q[init_params['process_id'][cp]].get()
                        cam_visited.append(cp)
                        print('cam {} ids: {}'.format(cp, global_batch[cp].keys()))
                    #test that pair is already associated
                    # call MCTA if correspondance is eligible in MCTA
                    # reuse the already assigned process for eligible cams or create new process for each pair
                    # 1. call a new/already_assigned process id for each (ca,cp)
                    # 2. get the processed batch results for each pair
                    # 3. combine the pair results and update the G and then go to the next batch
                    if (cp,ca) not in associated_cams and ca in global_batch and cp in global_batch:
                        init_params['global_mcta_graph'], \
                        init_params['mc_all_trklts_dict'],\
                        matches= mcta_batch.get_global_matches(global_batch, ca, cp)
                        print('found matches {}'.format(matches))
                        associated_cams.append((cp,ca))


                if len(associated_cams)==3:
                    global_track_results = mcta_batch.save_tracks()
                    init_params['mcta_batch_finished'] = True

            init_params['batch_start']+=init_params['batch_size']

            if fr>=4000:#10000:
                # dump batch results to csv file
                Dframe = pd.DataFrame(global_track_results)
                Dframe.to_csv(init_params['global_full_track_file'], mode='w',index=False)
        fr += 1
    p.join()
    p.terminate()