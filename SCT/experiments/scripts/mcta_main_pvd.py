import multiprocessing as mp
import os
import time
from os import path as osp
import sys
import csv
import glob
import  datetime
sys.path.insert(0,'/home/marquetteu/MCTA/tracking_wo_bnw/experiments/cfgs')
sys.path.insert(1,'/home/marquetteu/MCTA/tracking_wo_bnw/experiments/scripts/MCTA')
sys.path.insert(2,'/home/marquetteu/MCTA/tracking_wo_bnw/src')
sys.path.insert(3,'/home/marquetteu/MCTA/tracking_wo_bnw/src/tracktor')
#export PYTHONPATH=${PYTHONPATH}:/home/marquetteu/MCTA/tracking_wo_bnw/experiments/cfgs:
# /home/marquetteu/MCTA/tracking_wo_bnw/experiments/cfgs:
# /home/marquetteu/MCTA/tracking_wo_bnw/experiments/scripts/MCTA
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from PIL import Image
import motmetrics as mm
import argparse
mm.lap.default_solver = 'lap'

import torchvision
import yaml
from tqdm import tqdm
import sacred
from sacred import Experiment
#from tracktor.frcnn_fpn import FRCNN_FPN
from tracktor.frcnn_fpn_batches import FRCNN_FPN
from tracktor.config import get_output_dir
#from tracktor.datasets.factory import Datasets
#from tracktor.oracle_tracker import OracleTracker
#from tracktor.tracker import Tracker
from tracktor.tracker_reid import Tracker
from tracktor.reid.resnet import resnet50
from tracktor.utils import interpolate, plot_sequence, plot_scanned_track, filter_tracklets, delete_all, get_mot_accum, \
    evaluate_mot_accums
import pdb
import logging
from read_config import Configuration
from MCTA.MCTA_Online_PVD import MCTA
import json
import requests
import zipfile
import io
import psutil
import tracemalloc
from numba import njit, prange
import numba
# TODO:
# 1. replace data loader with clasp batch image reader or online image reader
# 2.
class SCT_Clasp(object):
    def  __init__(self, _config, _log,
                 cam, output_dir, num_class=2,
                 batch=20, gpu=0, min_trklt_size=10,
                    global_track_num=None, global_track_start_time=None,
                  reid_rads_patience=None, init_flags=None):
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
        self.flags_mu = init_flags
        self.reid_rads_patience = reid_rads_patience

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
    def get_batch_frames(timeoffset, img_path, myurl):
        # Prepare all the inputs
        # Examples
        dataset = 'exp2training'
        cameralist = '2,5,9,11,13,14'
        duration = 4#1.2
        duration = "{0:.2f}".format(float(duration))

        inputParams = {"dataset": dataset, "cameralist": cameralist, "timeoffset": timeoffset, "duration": duration,
                       "filesize": "1920x1080"}
        jsoninputParams = json.dumps(inputParams)
        jsonParams = {"APIParams": jsoninputParams}
        response = requests.get(url=myurl, params=jsonParams)
        zf = zipfile.ZipFile(io.BytesIO(response.content))
        zf.extractall(img_path)  # replace with the actual output path

    @staticmethod
    def read_wrapper_flags(init_params):
        with open(osp.join(init_params['flags_mu']['path'],'Flags_Wrapper.yaml')) as file:
            server_flags = yaml.load(file, Loader=yaml.FullLoader)
            if server_flags is not None:
                if server_flags['Frames_Ready_MU'] is None:
                    init_params['flags_mu']['Frames_Ready_MU'] = 'TRUE'
                if server_flags['Frames_Ready_MU'] == 'TRUE':
                    init_params['flags_mu']['Frames_Ready_MU'] = 'TRUE'
                else:
                    init_params['flags_mu']['Frames_Ready_MU'] = 'FALSE'

                if server_flags['No_More_Frames'] is None:
                    init_params['flags_mu']['No_More_Frames'] = 'FALSE'
                if server_flags['No_More_Frames'] == 'FALSE':
                   init_params['flags_mu']['No_More_Frames'] = 'FALSE'
                if server_flags['No_More_Frames'] == 'TRUE':
                   init_params['flags_mu']['No_More_Frames'] = 'TRUE'
            else:
                init_params['flags_mu']['Frames_Ready_MU'] = 'FALSE'
                init_params['flags_mu']['No_More_Frames'] = 'FALSE'
        return init_params

    @staticmethod
    def read_rpi_flags(init_params):
        with open(osp.join(init_params['flags_mu']['path'],'Flags_RPI.yaml')) as file:
            server_flags = yaml.load(file, Loader=yaml.FullLoader)
            if server_flags is not None:
                if server_flags['Batch_Processed'] == 'TRUE':
                    init_params['flags_mu']['Batch_Processed_RPI'] = 'TRUE'
                else:
                    init_params['flags_mu']['Batch_Processed_RPI'] = 'FALSE'

            else:
                init_params['flags_mu']['Batch_Processed_RPI'] = 'FALSE'

        return init_params

    @staticmethod
    def write_mu_flags(init_params):
        #with open(osp.join(init_params['flags_mu']['path'],'Flags_MU.yaml')) as file:
            #server_flags = yaml.load(file, Loader=yaml.FullLoader)
        if init_params['flags_mu']['People_Processed']=='TRUE':
            flag_value = {'People_Processed':'TRUE'}
        else:
            flag_value = {'People_Processed':'FALSE'}
        with open(osp.join(init_params['flags_mu']['path'],'Flags_MU.yaml'), 'w') as flags_file:
            yaml.dump(flag_value, flags_file)

    @staticmethod
    #only for self experiment
    def write_wrapper_flags(init_params):
        with open(osp.join(init_params['flags_mu']['path'],'Flags_Wrapper.yaml')) as file:
            server_flags = yaml.load(file, Loader=yaml.FullLoader)
            server_flags['Frames_Ready_MU'] = 0
            server_flags['Frames_Ready_MU'] = 'TRUE'
        with open(osp.join(init_params['flags_mu']['path'],'Flags_Wrapper.yaml'), 'w') as flags_file:
            yaml.dump(server_flags, flags_file)

    @staticmethod
    def clean_dir(data_dir, fmt='.jpg'):
        framelist = [f for f in os.listdir(data_dir) if f.endswith(fmt)]
        for f in framelist:
            os.remove(os.path.join(data_dir, f))

    def init_det_model(self):
        # object detection
        self._log.info("Initializing object detector.")

        self.obj_detect = FRCNN_FPN(num_classes=self.num_class)
        if self.cam in [360]:
            self.obj_detect.load_state_dict(torch.load(self._config['tracktor']['obj_detect_model_pvd_iter9'],
                                                   map_location=lambda storage, loc: storage))
        elif self.cam in [330]:
            self.obj_detect.load_state_dict(torch.load(self._config['tracktor']['obj_detect_model_pvd_iter8'],
                                                   map_location=lambda storage, loc: storage))
        else:
            self.obj_detect.load_state_dict(torch.load(self._config['tracktor']['obj_detect_model_pvd_iter6'],
                                                   map_location=lambda storage, loc: storage))
        self.obj_detect.eval()
        self.obj_detect.cuda()#device=self.gpu
        return self.obj_detect

    def init_reid_model(self):
        self.reid_network = resnet50(pretrained=False, **self.reid['cnn'])
        self.reid_network.load_state_dict(torch.load(osp.join(os.getcwd().split('experiments')[0], self.tracktor['reid_weights_pvd']),
                                                     map_location=lambda storage, loc: storage))
        self.reid_network.eval()
        self.reid_network.cuda()#device=self.gpu
        return self.reid_network

    def init_tracktor(self):
        tracker = Tracker(self.init_det_model(), self.init_reid_model(), self.tracktor['tracker'],
                          self.global_track_num,
                          self.global_track_start_time,
                          self.reid_rads_patience,
                          start_frame=self.flags_mu['start_frame'],
                          class_id=1,
                          isSCT_ReID=self.flags_mu['isSCT_ReID'])
        return tracker

    def get_img_blob(self, path_i, dets=[]):
        """Return the ith image converted to blob"""
        img = Image.open(path_i).convert("RGB")
        #print('img shape {}'.format(img.size))
        self.flags_mu['img_HW'][0] = img.size[1]
        self.flags_mu['img_HW'][1] = img.size[0]
        #img = img.resize((540,960))
        img = self.transforms(img)

        sample = {}
        sample['img'] = torch.reshape(img, (1, 3, int(self.flags_mu['img_HW'][0]),
                                            int(self.flags_mu['img_HW'][1])))

        #sample['dets'] = torch.tensor([det[:4] for det in dets])
        sample['img_path'] = path_i
        sample['gt'] = {}
        sample['vis'] = {}

        return sample

    def init_online_dataloader(self, dataset, img_path, fr_num, data_loc, timeoffset = '0.00'):
        '''

        :param dataset: 'exp2training'
        :param img_path:
        :param frIndex:
        :return: return a batch of frames to SCT_clasp
        '''
        if data_loc == 'PVD':
            path_i = os.path.join(img_path, '{}_{:04d}.jpg'.format(self.cam, fr_num))
            if not osp.exists(path_i):
                return None
        return self.get_img_blob(path_i)


    def test_SCT(self, tracktor=None, start_frame=None, dataset=None, img_path=None, server='local', queue_batch=None, vis=False):
        num_frame = start_frame#+1#batch_start#batch_start
        print_stats = True

        self.flags_mu['throughput']['scts_start_time'] = time.time()
        tracemalloc.start()
        import cProfile
        import pstats
        pr = cProfile.Profile()

        read_time = []
        tracking_time = []
        det_time = []
        batch_frames = []
        frame_indexs = []
        steps_det = [5, 5, 5, 5]
        step_ind = 0
        while self.flags_mu['flags_mu']['No_More_Frames']=='FALSE':
            #start_time = time.time()
            pr.enable()
            sortby = 'cumulative'

            # self or integration
            if self.flags_mu['integration?']:
                self.flags_mu = SCT_Clasp.read_wrapper_flags(self.flags_mu)
                #end of video
                if self.flags_mu['flags_mu']['No_More_Frames']=='TRUE':
                    break
            else:
                self.flags_mu['flags_mu']['Frames_Ready_MU']='TRUE' #wrapper_flags will toggle this flag

            #get image blob
            if self.flags_mu['flags_mu']['Frames_Ready_MU']=='TRUE':
                time_offset = '{:.2f}'.format((num_frame)/10.0)#-1
                read_start = time.time()
                frame = self.init_online_dataloader(dataset, img_path, num_frame, data_loc=server)
                read_time.append(time.time()-read_start)
                if self.flags_mu['start_SCTs']['on'] and frame is not None:
                    print('[1-2] parallel SCTS start time: {}'.format(datetime.datetime.now()))#.strftime("%Y-%m-%d %H:%M:%S:%s")
                    self.flags_mu['start_SCTs']['on'] = False

            else:
                frame = None
            #wait until current frame is found
            if frame is not None:
                print_stats = True

            if frame is None:
                # to skip the wait time
                self.flags_mu['throughput']['scts_start_time'] = time.time()
                if print_stats:
                    print('cam {} is waiting for next batch'.format(self.cam))
                    print_stats = False
                continue
            else:
                batch_frames.append(frame['img'])
                frame_indexs.append(num_frame)

            if len(batch_frames)==steps_det[step_ind]:
                #batch detections in 2/3 steps
                #compute backbone features
                with torch.no_grad():

                    det_start = time.time()
                    images = torch.cat(tuple(batch_frames))
                    # get raw detections
                    detections, features = self.obj_detect.detect_batches(images)
                    #print('det time {:.2f} sec for #frames: {}'.format((time.time() - det_start), steps_det[step_ind] ))
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

            #at scan interval return all tracking history in camera
            if num_frame%self.scan_intrvl==0:
                #for gpu memory usage

                process = psutil.Process(os.getpid())
                #print(process.pid)
                self.flags_mu['throughput']['GPU_mem_minmax'].append(process.memory_info()[0])
                #tracemalloc.stop()

                batch_results = tracktor.get_results()

                batch_results = filter_tracklets(num_frame , batch_results, start_frame=start_frame,
                                                 batch_length=self.scan_intrvl, min_trklt_size=self.min_trklt_size,
                                                 reid_patience=self.reid_patience)

                queue_batch.put(batch_results)
                self.flags_mu['throughput']['scts_counter'].append(time.time() - self.flags_mu['throughput']['scts_start_time'])
                self.flags_mu['throughput']['scts_start_time'] = time.time()
                #for cpu memory usage
                current, peak = tracemalloc.get_traced_memory()
                #print(f"Current memory usage at cam {self.cam} SCT batch finish is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
                self.flags_mu['throughput']['SCTs_mem_minmax'].append(peak)
                #tracemalloc.start()
                pr.disable()
                ##if  self.flags_mu['print_cProfile']:
                    #print('SCT Profile for cam {}..............................'.format(self.cam))
                    #ps = pstats.Stats(pr, stream=sys.stdout).sort_stats(sortby).print_stats(20)
                if False:
                    print('----------------------------------------------Batch {} Timing Metrics ------------------------------------------'.format(num_frame // self.scan_intrvl))
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
                    print('-------------------------------------------------------------------------------------------------------------')
                step_ind = 0
            num_frame += 1

    def get_current_batch_results(self):
        # getframes from current batch results before publishhing to integration
        with open(self.flags_mu['output_dir']+'/log_batch_mu_current.csv') as batch_people:
            reader_people=csv.reader(batch_people)
            # Sample || (0)Index, (1)Frame, (2)Cam, (3)timeoffset, (4)x1, y1, x2, y2, (8)id, (9)firstused, (10)tu
            fids_results = []
            headers=next(reader_people)
            for line in reader_people:
                fids_results.append(int(line[1]))
        return fids_results

    def get_current_batch_frames(self):
        batch_frames = []
        imglist = glob.glob(self.flags_mu['img_path'] + '/300_*')
        imglist.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        for im in imglist:
            batch_frames.append(int(os.path.basename(im).split('.jpg')[0][-4:]))
        return batch_frames

    def is_frames_results_sync(self):
        #check sync. failures due to multiprocessing queue
        fids_results = self.get_current_batch_results()
        batch_frames   = self.get_current_batch_frames()
        return np.unique(fids_results)[0] in batch_frames[-self.scan_intrvl:] \
               and np.unique(fids_results)[-1] in batch_frames[-self.scan_intrvl:]

def SCT_main(q, init_params, gpu_index, _config, global_track_num,
             global_track_start_time, reid_rads_patience):
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
                                   reid_rads_patience = reid_rads_patience,
                                   init_flags = init_params
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
#@numba.jit(nopython=True, parallel=True)
def get_cameras_batch(q, global_batch, init_params):
    for i in range(len(q)): # range(num_gpus)
        global_batch[init_params['cams'][i]] = q[i].get()
    return q, global_batch

def parse_args():
    """Parse in command line arguments"""
    parser = argparse.ArgumentParser(description='Process clasp2 video for MCTA')
    parser.add_argument(
        "--start_time", type=int,
        default=0,help='starting time during experiment')

    parser.add_argument(
        "--frame_rate", type=int,
        default=10,help='video frame rate')
    #--cuda_list 0 1 2 3 4 5
    parser.add_argument(
        "--cuda_list", nargs="*",type=int,
        default=[0,1,2,3,4,4],help='list of 6 cuda ids')

    parser.add_argument(
        "--img_h",type=int,
        default=800,help='list of image height and width')

    parser.add_argument(
        "--integration",type=bool,
        default=False,help='integration or self test')

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    from read_meta_data import *
    # initialize parameters
    args = parse_args()
    init_params = {}

    #for throughput metrics
    init_params['print_mcta_stats'] = False
    init_params['save_gloabl_tracks'] = False
    init_params['print_mem_stats'] =True
    init_params['print_cProfile']= False
    init_params['throughput'] = {}
    init_params['throughput']['mcta_counter'] = []
    init_params['throughput']['mcta_wait_counter'] = []
    init_params['throughput']['sct_differ'] = []

    init_params['throughput']['full_mcta_counter'] = []
    init_params['throughput']['MCTA_mem_minmax'] = []

    init_params['single_cam_multi_asso'] = False
    init_params['global_mcta_graph'] = {}
    init_params['update_frechet_dist'] = False
    init_params['multi_match'] = False
    init_params['keep_batch_hist'] = {}
    init_params['meta_data'] = read_metadata_pvd('/home/marquetteu/MCTA/Training_PAX_Metadata.xlsx')
    init_params['use_metadata'] = True
    init_params['mapCLASPlabel'] = True


    init_params['num_class'] = 2
    # TODO: 7 gpus for 7 cameras simultaneously
    init_params['dataset'] = 'PVD'
    init_params['last_batch_ids'] = {}
    init_params['current_batch_ids'] = {}
    init_params['new_ids'] = {}
    init_params['prev_affinity'] = {}
    init_params['global_track_id'] = 50 ## 100 when use metadata
    init_params['keep_raw_id'] = {}  # TODO: keep camera info as well: camera info is not needed since we define unique global tracks
    init_params['keep_tu_id'] = {}

    init_params['keep_matched_pair'] = []
    init_params['lane3_ids'] = []
    init_params['reid_rads_patience'] = {}
    init_params['keep_id_pvd'] = {}

    init_params['isSCT_ReID'] = 1
    init_params['only_lane3_ids'] =1

    if args.img_h==800:
        init_params['img_HW'] = [800.0, 1280.0]
        init_params['frame_rate'] = args.frame_rate
        init_params['batch_size']=20
        init_params['min_trklt_size'] = 6
        init_params['batch_end'] =6000
        init_params['server_loc'] = 'PVD'
        init_params['only_SCTs'] = 0
    else:
        print('use --img_h=800')



    init_params['batch_start'] = args.start_time*init_params['frame_rate'] + 1 #1
    init_params['start_frame'] = args.start_time*init_params['frame_rate'] + 1 #1

    init_params['flags_mu'] = {}
    init_params['flags_mu']['global_batch_counter'] = 0
    # #
    init_params['flags_mu']['No_More_Frames'] = 'FALSE'
    init_params['flags_mu']['Batch_Processed_RPI'] = 'FALSE'
    init_params['flags_mu']['People_Processed'] = 'FALSE'

    cudas = json.loads(str(args.cuda_list))
    init_params['integration?'] = args.integration
    if init_params['integration?']:

        init_params['gpu_id'] = [cuda for cuda in cudas]#[0,1, 2, 3, 4, 5]
        init_params['cams'] = [300, 360, 340, 440, 361, 330]
        assert len(init_params['gpu_id'])==len(init_params['cams']), 'assign an individual cuda id for camera {}'.format(init_params['cams'])
        init_params['cam_graph'] = {440: [361, 300], 340:[300], 361:[360], 360:[330]}
        init_params['flags_mu']['path'] = '/data/ALERT-SHARE/alert-api-wrapper-data/runtime-files'
        init_params['img_path'] = '/data/ALERT-SHARE/alert-api-wrapper-data'
        init_params['server_url'] = 'http://127.0.0.1:5000/frames'
        init_params['output_dir'] ='/data/ALERT-SHARE/alert-api-wrapper-data/runtime-files/mu'
        init_params['vis'] = False
        init_params['vis_rate'] = 1
        init_params['isSCT'] = False
        init_params['save_global_tracks'] = False
        SCT_Clasp.write_mu_flags(init_params=init_params)
    else:
        init_params['gpu_id'] = [cuda for cuda in cudas] #[0,1, 2, 3, 4,4]
        init_params['cams'] = [300, 360, 340, 440, 361, 330]
        assert len(init_params['gpu_id'])==len(init_params['cams']), 'assign an individual cuda id for camera {}'.format(init_params['cams'])
        init_params['cam_graph'] = {440: [361, 300], 340:[300], 361:[360], 360:[330]}
        init_params['flags_mu']['path'] = '/data/ALERT-SHARE/alert-api-wrapper-data/runtime-files'
        init_params['img_path'] = '/data/ALERT-SHARE/PVD_8-06_Data/Training/All_Frames_NonStaggered'
        init_params['output_dir'] ='/home/marquetteu/MCTA/tracking_wo_bnw/output/tracktor/oline_MCT'
        init_params['isSCT'] =False
        init_params['vis'] = True
        init_params['vis_rate'] =1
        init_params['save_global_tracks'] = False
        init_params['global_track_results'] = {'frame': [], 'cam':[], 'timeoffset': [],
                                      'x1':[], 'y1':[], 'x2':[],'y2':[],
                                        'id': [], 'firstused':[]}
        init_params['global_full_track_file'] = init_params['output_dir'] + '/global_tracks_10FPS.csv'
    for cam in init_params['cams']:
        init_params['keep_id_pvd'][cam] = {}#deque()
    if not osp.exists(init_params['output_dir']):
        os.makedirs(init_params['output_dir'])

    #initialize cPROFILE TO GET TIME INFORMATION FOR EACH FUNCTION
    #cProfile
    import cProfile
    import pstats
    pr = cProfile.Profile()

    #get config
    configuration = []
    with open(osp.join(os.getcwd().split('scripts')[0],'cfgs/tracktor.yaml')) as file:
        # The FullLoader parameter handles the conversion from YAML
        # scalar values to Python the dictionary format
        _config = yaml.load(file, Loader=yaml.FullLoader)
        with open(osp.join(os.getcwd().split('experiments')[0],_config['tracktor']['reid_config'])) as file1:
            _config.update(yaml.load(file1, Loader=yaml.FullLoader))

    #get flags file from server
    init_params = SCT_Clasp.read_wrapper_flags(init_params)
    init_params['flags_mu']['No_More_Frames'] = 'FALSE' #read from server

    # start multiprocess for multi-camera
    #https://pytorch.org/docs/stable/multiprocessing.html
    mp.set_start_method('spawn')

    # global variable for sequential global tracks
    manager = mp.Manager()
    init_params['start_SCTs'] = manager.dict()
    init_params['start_SCTs']['on'] = True
    init_params['throughput']['scts_counter'] = manager.list()
    init_params['throughput']['SCTs_mem_minmax'] = manager.list()
    init_params['throughput']['GPU_mem_minmax'] = manager.list()
    reid_rads_patience = init_params['reid_rads_patience']

    global_track_num = manager.list()
    global_track_start_time = manager.dict()

    reid_rads_patience['rads'] = manager.dict()
    reid_rads_patience['patience'] = manager.dict()

    #set of gpus
    num_gpus = torch.cuda.device_count()

    #list of processes
    q=list(range(len(init_params['cams'])))

    #assign process to each gpu


    for i in range(len(q)):
        q[i] = mp.Queue()
        # Pass GPU number through q
        p = mp.Process(target=SCT_main, args=(q[i], init_params, i, _config, global_track_num,
                                              global_track_start_time, reid_rads_patience))
        p.start()

    # video frames: process synchronized frames from 6 cameras
    ##cams_ready = 0
    fr = init_params['start_frame']+0#1#
    init_params['throughput']['full_mcta_start_time'] = time.time()

    while init_params['flags_mu']['No_More_Frames']=='FALSE':
        pr.enable()
        if (fr)%init_params['batch_size']==0:
            #Collect 6 SCTs Results
            check_list = [not q[i].empty() for i in range(len(q))]
            if all(check_list):
                global_batch={}

                timeoffset = '{:.2f}'.format(init_params['batch_start']/10.0)
                tracemalloc.start()
                read_queue = time.time()

                for i in range(len(q)): # range(num_gpus)
                    global_batch[init_params['cams'][i]] = q[i].get()

                print('read SCTS queue:{}'.format(time.time() - read_queue))
                #MCTA Only
                init_params['throughput']['mcta_start_time'] = time.time()

                #if not init_params['start_SCTs']['on']:
                print('[3] parallel SCTs end time: {}'.format(datetime.datetime.now())) #.strftime("%Y-%m-%d %H:%M:%S.%s")
                init_params['start_SCTs']['on'] = True
                print('[4] MCTA start time: {}'.format(datetime.datetime.now())) #

                matches, batch_results, init_params['global_track_id'] = MCTA(mc_track_segments=global_batch,
                                              all_params=init_params,
                                              global_track_start_time=global_track_start_time,
                                              timeoffset=timeoffset).get_global_matches()

                print('[5] MCTA end time: {}'.format(datetime.datetime.now())) #.strftime("%Y-%m-%d %H:%M:%S:%s")
                pr.disable()
                sortby = 'cumulative'
                if init_params['print_cProfile']:
                    print('MCTA Profile..............................')
                    ps = pstats.Stats(pr, stream=sys.stdout).sort_stats(sortby).print_stats(20)
                #trace cpu memory
                if init_params['print_mem_stats']:
                    current, peak = tracemalloc.get_traced_memory()
                    init_params['throughput']['MCTA_mem_minmax'].append(peak)
                    ##print(f"Current memory usage at MCTA batch finish is {current / 10 ** 6}MB; Peak was {peak / 10 ** 6}MB")
                    #print(f"MCTA batch: peak min {min(init_params['throughput']['MCTA_mem_minmax'])/10**6}MB; "
                         # f"peak max {max(init_params['throughput']['MCTA_mem_minmax'])/10**6}MB")
                    tracemalloc.stop()
                    init_params['throughput']['mcta_counter'].append(time.time() - init_params['throughput']['mcta_start_time'])
                    print('MCTA: Avg Batch {} sec'.format(np.mean(init_params['throughput']['mcta_counter'])))
                    print(f"MCTA Batch Time: min {min(init_params['throughput']['mcta_counter'])} sec and max {max(init_params['throughput']['mcta_counter'])}")
                    #print('MC Internal Wait: Avg: {}, Max: {}'.format(np.mean(init_params['throughput']['mcta_wait_counter']), max(init_params['throughput']['mcta_wait_counter'])))

                print('Batch: {}, fr: {} found matches: {}'.format(fr//init_params['batch_size'], fr, matches))
                #print('process batch at {:.2f} FPS'.format(fr / (time.time() - start_time)))
                init_params['batch_start']+=init_params['batch_size']
                #--------------------------------------------------------Flags communication during integration---------------------------------------------

                #read rpi flags
                print_stats = True
                #comment it for integration
                #init_params['flags_mu']['Batch_Processed_RPI'] = 'TRUE'
                if init_params['integration?']:
                    #check the current batch results and current batch frames are sync
                    while True:
                        if SCT_Clasp.is_frames_results_sync():
                            #pdb.set_trace()
                            init_params['flags_mu']['People_Processed'] = 'TRUE'
                            SCT_Clasp.write_mu_flags(init_params=init_params)
                            init_params = SCT_Clasp.read_rpi_flags(init_params=init_params)

                        if print_stats:
                            print('[6] Wait start time: {}'.format(datetime.datetime.now())) #.strftime("%Y-%m-%d %H:%M:%S:%s")
                            print('in loop for rpi batch process with flag value {}'.format(init_params['flags_mu']['Batch_Processed_RPI']))
                            print('results start {}, end {}'.format(np.unique(fids_results)[0],np.unique(fids_results)[-1]))
                            print('frames in current batch {}'.format(batch_frames[-20:]))

                        print_stats = False
                        #if init_params['flags_mu']['Batch_Processed_RPI'] in not None:
                        if init_params['flags_mu']['Batch_Processed_RPI']=='TRUE' and init_params['flags_mu']['People_Processed'] == 'TRUE':
                            init_params['flags_mu']['People_Processed'] = 'FALSE'
                            SCT_Clasp.write_mu_flags(init_params=init_params)
                            print('[7] Wait end time: {}'.format(datetime.datetime.now())) #.strftime("%Y-%m-%d %H:%M:%S:%s")
                            break

                if init_params['print_mem_stats']:
                    init_params['throughput']['full_mcta_counter'].append(time.time() - init_params['throughput']['full_mcta_start_time'])
                    #print('MCT e2e Avg: {} sec'.format(np.mean(init_params['throughput']['full_mcta_counter'])))
                    #print(f"batch completion time: min {min(init_params['throughput']['full_mcta_counter'])} sec and max {max(init_params['throughput']['full_mcta_counter'])}")
                    init_params['throughput']['full_mcta_start_time'] = time.time()

                    print('SCT Time Avg: {}, max: {}, min: {}'.format(np.mean(init_params['throughput']['scts_counter'])
                          ,max(init_params['throughput']['scts_counter']), min(init_params['throughput']['scts_counter'])))
                    init_params['throughput']['sct_differ'].append(max(init_params['throughput']['scts_counter'][-6::])- min(init_params['throughput']['scts_counter'][-6::]))
                    print('SCTs relative wait: {} sec, max: {}'.format(np.mean(init_params['throughput']['sct_differ']), max(init_params['throughput']['sct_differ'])))

                    #print(f"CPU Memory SCT: peak average {np.mean(init_params['throughput']['SCTs_mem_minmax'])/10**6}MB; "
                            #f"peak max {max(init_params['throughput']['SCTs_mem_minmax'])/10**6}MB")

                   # print(f"GPU Memory SCT: peak average {np.mean(init_params['throughput']['GPU_mem_minmax'])/10**6}MB; "
                         # f"peak max {max(init_params['throughput']['GPU_mem_minmax'])/10**6}MB")
                    print('******** Average MU Batch Time: {} secs ********'.format(
                        np.mean(init_params['throughput']['mcta_counter'])+
                        np.mean(init_params['throughput']['scts_counter'])+
                        np.mean(init_params['throughput']['sct_differ'])))
                    print('######################### Overall: {} secs #############################'.format(np.mean(init_params['throughput']['full_mcta_counter'])))
                    print('                                                                                             ')

                #save global tracks for MOT evaluation
                if init_params['save_global_tracks']:
                    init_params['global_track_results']['frame'] = init_params['global_track_results']['frame']+batch_results['frame']
                    init_params['global_track_results']['cam'] = init_params['global_track_results']['cam']+batch_results['cam']
                    init_params['global_track_results']['timeoffset'] = init_params['global_track_results']['timeoffset']+batch_results['timeoffset']
                    init_params['global_track_results']['x1'] = init_params['global_track_results']['x1']+batch_results['x1']
                    init_params['global_track_results']['y1'] = init_params['global_track_results']['y1']+batch_results['y1']
                    init_params['global_track_results']['x2'] = init_params['global_track_results']['x2']+batch_results['x2']
                    init_params['global_track_results']['y2'] = init_params['global_track_results']['y2']+batch_results['y2']
                    init_params['global_track_results']['id'] = init_params['global_track_results']['id']+batch_results['id']
                    init_params['global_track_results']['firstused'] = init_params['global_track_results']['firstused']+batch_results['firstused']
                    if init_params['save_gloabl_tracks'] and fr>=4000:#10000:
                        import pandas as pd
                        # dump batch results to csv file
                        Dframe = pd.DataFrame(init_params['global_track_results'])
                        Dframe.to_csv(init_params['global_full_track_file'], mode='w',index=False)

            else:
                if init_params['integration?']:
                    init_params = SCT_Clasp.read_wrapper_flags(init_params)
                continue

        fr += 1

    p.join()
    p.terminate()
