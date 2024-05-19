import numpy as np
import os
import glob
from evaluate_3D_tracks import *
from project_3D_wildtrack import *
import copy
import pandas as pd

def get_world_point(gts, dets, c=None):
    # get camera calibration params
    cam_projections = Projection(gt_path=None, out_path=None, images_path=None)
    cam_params = {}
    cam_params['C{}'.format(c)] = {}
    cam_params['C{}'.format(c)]['A'], \
    cam_params['C{}'.format(c)]['rvec'], \
    cam_params['C{}'.format(c)]['tvec'], \
    cam_params['C{}'.format(c)]['dist_coeff'] = cam_projections.cam_params_getter(cam=c,
                                                                                  isDistorted=False)
    for i, det in enumerate(dets):
        dets[i, 2] = dets[i, 2] + dets[i, 4] / 2.
        dets[i, 3] = dets[i, 3] + dets[i, 5]
        dets[i, 2:4] = cam_projections.project3D(copy.deepcopy(dets[i, 2:4]), cam_params, cam=c)

    for gt in gts:
        gt[2] = gt[2] + gt[4] / 2.
        gt[3] = gt[3] + gt[5]
        gt[2:4] = cam_projections.project3D(copy.deepcopy(gt[2:4]), cam_params, cam=c)
    return gts, dets

def get_dirs(database):
    if database=='wildtrack':
        dataset = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/multicamera_wildtrack/' \
                  'wildtrack/Wildtrack_dataset/'
        gt_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/' \
                  'tracking_wo_bnw/data/wild-track/train_gt'
        # tracktor
        result_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/' \
                      'tracking_wo_bnw/output/tracktor_wildtrack/SCT'
        # resnet mht
        result_path = dataset + 'results-track-resnet-mht'

    if database=='clasp1':
        dataset = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/multicamera_wildtrack/' \
                  'wildtrack/Wildtrack_dataset/'
        gt_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/' \
                  'tracking_wo_bnw/data/wild-track/train_gt'
        # tracktor
        result_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/' \
                      'tracking_wo_bnw/output/tracktor_wildtrack/SCT'
        # resnet mht
        result_path = dataset + 'results-track-resnet-mht'

    return dataset, gt_path, result_path

if __name__ == '__main__':
    isMCTA = 0
    eval_plane = '3D'
    database = 'wildatrack'

    dataset, gt_path, result_path = get_dirs(database)

    #mcta results
    out_dir = None
    #read gt and results for cameras
    gt_folders = glob.glob(gt_path + '/*')
    gt_folders.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
    #SCT
    if not isMCTA:
        tr_files = glob.glob(result_path + '/*.txt')
        tr_files.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
    else:
        #MCTA
        #bbs = pd.read_csv(dataset+'results-track/global_tracks/2D_tracks_offline_mht_hausdorff.csv', index_col=False)
        bbs = pd.read_csv(dataset + 'results-track/global_tracks/2D_tracks_offline_tracktor_frechet_75.csv', index_col=False)
        #online
        #bbs = pd.read_csv('/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/output/tracktor_wildtrack/global_vis/global_track_wildtrack.csv', index_col=False)

        all_tracks = np.transpose(np.vstack(
            (bbs['frame'].array,
             bbs['id'].array,
             bbs['x'].array,
             bbs['y'].array,
             bbs['w'].array,
             bbs['h'].array,
             bbs['cam'].array)))
    for i in range(7):
        if i in [0,1,2,3,4,5,6]:
            gt = np.genfromtxt(gt_folders[i]+'/gt/gt.txt', delimiter=',')
            if not isMCTA:
                tr = np.genfromtxt(tr_files[i], delimiter=',')
            else:
                tr = all_tracks[all_tracks[:,6]==i+1]
            #TODO: SCT evaluation
            #python -m motmetrics.apps.eval_motchallenge
            # /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/wild-track/train_gt
            # /media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/multicamera_wildtrack/wildtrack/Wildtrack_dataset/results-track/C5C1
            results = {}
            mot_evaluation = EvaluateMOT(gt_path=gt_path,
                                         result_path=result_path,
                                         vis_path=out_dir,
                                         vis=False,
                                         proj_plane=eval_plane,
                                         radious=100,
                                         iou_max=0.5,
                                         isMCTA=isMCTA,
                                         max_gt_frame=1985,
                                         min_gt_frame=1800)


            #gt[:,2:4] = gt[:,2:4]/[gt[:,2].max(), gt[:,3].max()]
            #tr[:, 2:4] = tr[:, 2:4] / [tr[:, 2].max(), tr[:, 3].max()]
            if eval_plane=='2D':
                results['gt'] = gt
                results['tracks'] = tr
            if eval_plane=='3D':
                gt, tr = get_world_point(gt, tr, c=i + 1)
                results['gt'] = gt
                results['tracks'] = tr
            assert len(tr)>0
            #do mot accumulation
            #TODO: SCT evaluation
            #python -m motmetrics.apps.eval_motchallenge /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/wild-track/train_gt   /media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/multicamera_wildtrack/wildtrack/Wildtrack_dataset/results-track/C5C1
            # evaluate mot accumations
            name = 'C{}'.format(i+1)
            mot_evaluation.evaluate_mot_accums(results, name, generate_overall=True)