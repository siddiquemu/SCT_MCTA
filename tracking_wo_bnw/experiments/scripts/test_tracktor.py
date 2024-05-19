import os
import time
from os import path as osp

import numpy as np
import torch
from torch.utils.data import DataLoader

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
ex = Experiment()

ex.add_config('experiments/cfgs/tracktor.yaml')

# hacky workaround to load the corresponding configs and not having to hardcode paths here
ex.add_config(ex.configurations[0]._conf['tracktor']['reid_config'])
ex.add_named_config('oracle', 'experiments/cfgs/oracle_tracktor.yaml')


@ex.automain #@ex.automain
def main(tracktor, reid, _config, _log, _run):
    sacred.commands.print_config(_run)

    # set all seeds
    torch.manual_seed(tracktor['seed'])
    torch.cuda.manual_seed(tracktor['seed'])
    np.random.seed(tracktor['seed'])
    torch.backends.cudnn.deterministic = True

    output_dir = osp.join(get_output_dir(tracktor['module_name']), tracktor['name'])
    #output_dir = '/home/siddique/tracktor_output/'
    sacred_config = osp.join(output_dir, 'sacred_config.yaml')

    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    with open(sacred_config, 'w') as outfile:
        yaml.dump(_config, outfile, default_flow_style=False)

    ##########################
    # Initialize the modules #
    ##########################

    # object detection
    _log.info("Initializing object detector.")

    obj_detect = FRCNN_FPN(num_classes=2)
    obj_detect.load_state_dict(torch.load(_config['tracktor']['obj_detect_model'],
                               map_location=lambda storage, loc: storage))

    obj_detect.eval()
    obj_detect.cuda()

    # reid
    reid_network = resnet50(pretrained=False, **reid['cnn'])
    reid_network.load_state_dict(torch.load(tracktor['reid_weights'],
                                 map_location=lambda storage, loc: storage))
    reid_network.eval()
    reid_network.cuda()

    # tracktor
    if 'oracle' in tracktor:
        tracker = OracleTracker(obj_detect, reid_network, tracktor['tracker'], tracktor['oracle'])
    else:
        tracker = Tracker(obj_detect, reid_network, tracktor['tracker'])

    time_total = 0
    num_frames = 0
    scan_intrvl = 120
    mot_accums = []
    dataset = Datasets(tracktor['dataset'])


    for seq in dataset:
        tracker.reset()

        start = time.time()

        _log.info(f"Tracking: {seq}")

        # init output img dir
        online_track_dir = osp.join(output_dir, 'scanned_intrvl', str(seq))
        delete_all(demo_path=online_track_dir)
        if not osp.exists(online_track_dir):
            os.makedirs(online_track_dir)
        num_frame = 1
        pdb.set_trace()
        data_loader = DataLoader(seq, batch_size=1, shuffle=False)
        for i, frame in enumerate(tqdm(data_loader)):
            if len(seq) * tracktor['frame_split'][0] <= i <= len(seq) * tracktor['frame_split'][1]:
                with torch.no_grad():
                    tracker.step(frame)
                    #pdb.set_trace()
                    if num_frame%scan_intrvl==0:
                        # Scanned tracks result should contain the followings;
                        # 1. pos: [x1,y1,x2,y2]
                        # 2. motion: [v_cx,v_cy]
                        # 3. appearance: [128D_app_descriptor]
                        # 4. identity: ID
                        # 5. Frame index: fr
                        # 6. score: track score is optional
                        tracks_scan = tracker.get_results()
                        #keep confirmed track and filter tracks with size<scan_intrvl-reid_patience
                        tracks_scan = filter_tracklets(num_frame, tracks_scan, min_len=scan_intrvl, reid_patience=10)
                        #TODO: visualize frame-wise tracking results
                        #tracks: self.results[t.id][self.im_index] = np.concatenate([t.pos[0].cpu().numpy(), np.array([t.score])])
                        #overlay tracks on image:
                        plot_scanned_track(num_frame, tracks_scan, frame['img_path'][0], online_track_dir)
                num_frames += 1
                num_frame+=1
        #pdb.set_trace()
        results = tracker.get_results()

        time_total += time.time() - start

        _log.info(f"Tracks found: {len(results)}")
        _log.info(f"Runtime for {seq}: {time.time() - start :.2f} s.")

        if tracktor['interpolate']:
            results = interpolate(results)

        if seq.no_gt:
            _log.info(f"No GT data for evaluation available.")
        else:
            mot_accums.append(get_mot_accum(results, seq))

        _log.info(f"Writing predictions to: {output_dir}")
        seq.write_results_clasp(results, output_dir)

        if tracktor['write_images']:
            plot_sequence(results, seq, osp.join(output_dir, tracktor['dataset'], str(seq)))

    _log.info(f"Tracking runtime for all sequences (without evaluation or image writing): "
              f"{time_total:.2f} s for {num_frames} frames ({num_frames / time_total:.2f} Hz)")
    if mot_accums:
        evaluate_mot_accums(mot_accums, [str(s) for s in dataset if not s.no_gt], generate_overall=True)
