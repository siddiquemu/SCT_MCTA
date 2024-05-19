import os
import glob

import copy
import pdb

import motmetrics as mm
import numpy as np
import pandas as pd
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class EvaluateMOT(object):
    def __init__(self, gt_path=None, result_path=None, eval_dict=None,
                 vis_path=None, vis=None, proj_plane=None,
                 radious=None, iou_max=None, isMCTA=None, fr_factor=None,
                 max_gt_frame=None, min_gt_frame=None, auto_id=False, t_offset=None):
        self.gt_path = gt_path
        self.result_path = result_path
        self.mot_accum = mm.MOTAccumulator(auto_id=auto_id, max_switch_time=240)
        self.vis = vis
        self.colors = []
        self.out_dir = vis_path
        self.proj_plane = proj_plane
        self.radious = radious
        self.iou_max = iou_max
        self.max_gt_frame = max_gt_frame
        self.min_gt_frame = min_gt_frame
        self.isMCTA = isMCTA
        self.fr_factor = fr_factor
        self.offset=t_offset
        self.auto_id = auto_id
        self.eval_table = eval_dict

    @staticmethod
    def expand_from_temporal_list(box_all=None):
        if box_all is not None:
            box_list = [b for b in box_all if len(b) > 0]
            box_all = np.concatenate(box_list)
        return box_all

    @staticmethod
    def init_3d_figure():
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        _size = [1440, 480]
        _origin = [-300, -90, 0]
        _offset = 2.5
        ax.set_xlim3d(-310 - 50, 910 + 50)
        ax.set_ylim3d(-922 - 50, 2610 + 50)
        ax.set_zlim3d(0, 1)
        return ax

    def sct_track_eval_table(self, summary=None, eval_type='sct_overall'):
        assert 'Methods' in self.eval_table, 'Method key should be initialize first to get the eval table'
        for data_name, mot_name in mm.io.motchallenge_metric_names.items():
            #We can skip any measures based on the table requirements
            if mot_name not in self.eval_table:
                self.eval_table[mot_name]  = []
            if mot_name=='MOTP':
                self.eval_table[mot_name].append(np.round(1-summary[data_name].values[:-1].mean(), 3))
            elif mot_name in ['IDF1', 'IDP', 'IDR', 'MOTA', 'Rcll', 'Prcn']:
                self.eval_table[mot_name].append(np.round(summary[data_name].values[:-1].mean(), 3))
            else:
                self.eval_table[mot_name].append(np.round(summary[-1::][data_name].values[0], 3))
            #print(data_name, mot_name)


    def compare_gt_track(self, gt, track, fr=None):
        #ax = self.init_3d_figure()
        # ax = init_2d_figure()
        for bb in gt:
            x = bb[2]
            y = bb[3]
            z = 0.03
            id = int(bb[1])
            # color = ImageColor.getcolor(colors[int(id)], "RGB")
            # color = webcolors.rgb_to_name(color,  spec='html4')
            # ax.scatter(x, y, z, alpha=0.8, cmap=colors(int(id)), s=5)
            ax.scatter(x, y, z, alpha=0.5, c='red', s=3)  # self.colors[-1]
            # pdb.set_trace()
            # ax.scatter(x, y, alpha=0.8, c=colors[id], s=5)
        for bb in track:
            x = bb[2]
            y = bb[3]
            z = 0
            id = int(bb[1])
            # color = ImageColor.getcolor(colors[int(id)], "RGB")
            # color = webcolors.rgb_to_name(color,  spec='html4')
            # ax.scatter(x, y, z, alpha=0.8, cmap=colors(int(id)), s=5)
            ax.scatter(x, y, z, alpha=0.8, c=self.colors[id], s=6)

        plt.title('mc-tracking in 3D world for frame {}'.format(fr))
        plt.savefig(self.out_dir + '/{:06d}.png'.format(int(fr)), dpi=300)
        plt.close()

    def read_file(self, resultorgt='gt'):
        if resultorgt == 'gt':
            gt_world = pd.read_csv(self.gt_path, index_col=False)
            bbs = gt_world.values
        else:
            bbs = pd.read_csv(self.result_path, index_col=False)
            # Cx Y2 frame h id w
            # bbs = bbs.values
            bbs = np.transpose(np.vstack((bbs['frame'].array, bbs['id'].array, bbs['Cx'].array, bbs['Y2'].array,
                                          bbs['w'].array, bbs['h'].array)))
        return bbs

    def get_mot_accums(self, results, names):
        accums = []
        for all_gt_boxes, all_track_boxes, name in zip(results['gt'], results['tracks'], names):
            accum = mm.MOTAccumulator(auto_id=self.auto_id)#, max_switch_time=240
            print('accumulating: {} ...'.format(name))
            self.colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                           for i in range(int(max(all_track_boxes[:, 1]) + 2))]
            # all_track_boxes = track_boxes[track_boxes[:, 0] <= all_gt_boxes[:, 0].max()]

            # update accumulator frame-by-frame
            for fr in np.unique(all_gt_boxes[:, 0]):
                # print('Frame {}'.format(fr))
                gt_boxes = all_gt_boxes[all_gt_boxes[:, 0] == fr]
                
                #TODO: modify based on the clasp1 ann
                tr_fr = fr * self.fr_factor[name]
                
                track_boxes = all_track_boxes[all_track_boxes[:, 0] == tr_fr-self.offset[name]]
                #print(track_boxes)
                # if len(track_boxes)>0:
                #     # pdb.set_trace()
                #     print('gt frame: {}, track frame: {}'.format(fr, tr_fr))

                    
                    # pdb.set_trace()
                    # assert len(track_boxes)>0, 'no track found at {}'.format(tr_fr)
                
                if self.vis:
                    self.compare_gt_track(gt_boxes, track_boxes, fr=fr)

                gts = []
                gt_ids = []
                tracks = []
                track_ids = []

                for id in np.unique(gt_boxes[:, 1]):
                    # print('get all gt dets for id {}'.format(id))
                    gts.append(gt_boxes[gt_boxes[:, 1] == id])
                    gt_ids.append(gt_boxes[:, 1][gt_boxes[:, 1] == id].astype('int'))

                # print('total gt ids: {}'.format(len(gts)))
                # gts = np.stack(gts, axis=0)
                gts = self.expand_from_temporal_list(box_all=gts)
                # x1, y1, x2, y2 --> x1, y1, width, height
                gts = np.stack((gts[:, 2],
                                gts[:, 3],
                                gts[:, 4],
                                gts[:, 5]),
                               axis=1)
                if len(track_boxes) > 0:
                    for id in np.unique(track_boxes[:, 1]):
                        # print('get all track dets for id {}'.format(id))
                        tracks.append(track_boxes[track_boxes[:, 1] == id])
                        track_ids.append(track_boxes[:, 1][track_boxes[:, 1] == id].astype('int'))

                # print('total track ids: {}'.format(len(tracks)))
                # x1, y1, x2, y2 --> x1, y1, width, height
                # tracks = np.stack(tracks, axis=0)
                if len(tracks) > 0:
                    tracks = self.expand_from_temporal_list(box_all=tracks)
                    tracks = np.stack((tracks[:, 2],
                                       tracks[:, 3],
                                       tracks[:, 4],
                                       tracks[:, 5]),
                                      axis=1)
                distance = mm.distances.iou_matrix(gts, tracks, max_iou=self.iou_max)

                gt_ids = self.expand_from_temporal_list(box_all=gt_ids)
                if len(tracks) > 0:
                    track_ids = self.expand_from_temporal_list(box_all=track_ids)
                # print(distance.shape)
                '''
                def update(self, oids, hids, dists, frameid=None):
                """Updates the accumulator with frame specific objects/detections.
    
                This method generates events based on the following algorithm [1]:
                1. Try to carry forward already established tracks. If any paired object / hypothesis
                from previous timestamps are still visible in the current frame, create a 'MATCH' 
                event between them.
                2. For the remaining constellations minimize the total object / hypothesis distance
                error (Kuhn-Munkres algorithm). If a correspondence made contradicts a previous
                match create a 'SWITCH' else a 'MATCH' event.
                3. Create 'MISS' events for all remaining unassigned objects.
                4. Create 'FP' events for all remaining unassigned hypotheses.  frameid = 
                '''
                print(list(gt_ids),
                    list(track_ids),
                    distance)
                accum.update(
                    list(gt_ids),
                    list(track_ids),
                    distance)
            accums.append(accum)
        return accums

    def evaluate_mot_accums(self, results, names, generate_overall=False):
        mh = mm.metrics.create()
        accums = self.get_mot_accums(results, names)
        summary = mh.compute_many(
            accums,
            metrics=mm.metrics.motchallenge_metrics,
            names=names,
            generate_overall=generate_overall
        )
        
        #combine method overall
        self.sct_track_eval_table(summary=summary)

        str_summary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names)
        print(str_summary)
        return self.eval_table


def printLatexArray(arr):
    dims = arr.shape

    alignment = 'c' * dims[0]

    if type(arr[0, 0]) is np.int64:
        specifier = '{0:3d}'
    elif type(arr[0, 0]) is np.float64:
        specifier = '{2:2f}'
    elif type(arr[0, 0]) is np.complex128:
        specifier = '({0.real:.2f} + {0.imag:.2f+}i)'
    else:
        print('invalid specifier')
        return

    print('\\left[\\begin{array}' + '{' + alignment + '}')

    for (i, r) in enumerate(arr):
        for (j, x) in enumerate(r):
            if j < dims[1] - 1:
                terminator = ' & '
            elif i < dims[0] - 1:
                terminator = ' \\\\ \n'
            else:
                terminator = '\n'
            print(specifier.format(x), end=terminator)

    print('\\end{array}\\right]')

def filter_tracks(tracks, min_size = 10):
    filtered_tracks = []
    for id in np.unique(tracks[:,1]):
        track_id = tracks[tracks[:,1]==id]
        if len(track_id)>min_size:
            filtered_tracks.append(track_id)
    return np.concatenate(filtered_tracks)

def get_dirs(database, storage, model_name, isMTA=0):

    if database == 'clasp1':
        #full set
        gt_path = storage + 'tracking_wo_bnw/data/CLASP1/train_gt'
        #20% test set: monocular
        #gt_path = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/ourdataset/PB_GT'
        if isMTA:
            result_path = storage + 'tracking_wo_bnw/output/{}/MTA/{}'.format(database, model_name)
        else:
            result_path = storage + 'tracking_wo_bnw/output/{}/{}'.format(database, model_name)

    if database == 'clasp2':
        # 20% test set: random set unable to generate mot metrics
        # Full annotation set is used to evaluate Sl and SSL based SCT
        # gt_path = storage + 'tracking_wo_bnw/data/CLASP1/train_gt'
        gt_path = storage + 'tracking_wo_bnw/data/CLASP/train_gt_all/PB_gt'
        # tracktor
        if isMTA:
            result_path = storage + 'tracking_wo_bnw/output/{}/MTA/{}'.format(database, model_name)
        else:
            result_path = storage + 'tracking_wo_bnw/output/{}/{}'.format(database, model_name)


    return  gt_path, result_path

def get_params(database):
    if database=='clasp1':
        meta_map = {'exp9A': 'A', 'exp6a': 'B', 'exp7a': 'C', 'exp10a': 'D', 'exp5a': 'E'}
        fr_rate_factor = {'A': 30, 'B': 15, 'C': 15, 'D': 15, 'E': 3}
        offset = {'A': 15, 'B': 30, 'C': 30, 'D': 30, 'E': 4}
        categories = {1: 'person', 2: 'bag'}
        # categories = {0: 'person', 1: 'bag'}
    if database=='clasp2':
        meta_map = {'G': 'G', 'H':'H', 'I': 'I'}
        # use ft frame index directly instead of frame rate conversion
        #here the GT frames are subset of all video frames and maintain same fr index
        fr_rate_factor = {'G': 1, 'H': 1, 'I': 1}
        #no offset
        offset = {'G': 0, 'H': 0, 'I': 0}
        categories = {1: 'person', 2: 'bag'}
        # categories = {0: 'person', 1: 'bag'}
    return meta_map, fr_rate_factor, offset, categories

def get_data(database, exp_name, d_name, class_id, categories, isMTA=0):
    if database=='clasp1':
        gt_folders = glob.glob(gt_path +
                               '/{}_*'.format(d_name))
        gt_folders.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
        #MTA
        if isMTA:
            tr_files = glob.glob(result_path +
                                 '/{}/{}_C*'.format(categories[class_id], exp_name.lower()))
            pdb.set_trace()
        else:# SCT
            tr_files = glob.glob(result_path +
                                 '/{}/{}_C*.txt'.format(exp_name.lower(), class_id))
        tr_files.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))

    if database=='clasp2':
        gt_folders = glob.glob(gt_path + '/{}_*'.format(d_name))
        gt_folders.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
        #MTA
        if isMTA:
            tr_files = glob.glob(result_path +
                                 '/{}/{}_C*'.format(categories[class_id], exp_name))
        else:# SCT
            tr_files = glob.glob(result_path +
                                 '/{}/{}_C*.txt'.format(exp_name, class_id))
        tr_files.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
    return gt_folders, tr_files

def get_gt_stats(gt, seq, database):
    gt_person = gt[gt[:,-2]==1]
    gt_bag = gt[gt[:, -2] == 2]
    print('              ')
    print('######## {} SCT GT Stats: {} ########'.format(database.upper(), seq))
    print('#Unique Identity Person: {}, Bag: {}'.format(len(np.unique(gt_person[:,1])),
                                                        len(np.unique(gt_bag[:,1]))))
    print('#Annotated Frames: {}'.format(len(np.unique(gt[:, 0]))))
    print('#Annotations: {}, Person: {}, Bag: {}'.format(len(gt), len(gt_person), len(gt_bag)))
    print('              ')

def apply_ROI_costraints(gt_file, database, gt, tr, class_id=1):
    if database == 'clasp2' and class_id == 1:
        if gt_file.split('/')[-1] in ['G_11', 'H_11', 'I_11']:
            cxy = tr[:, 2:4] + tr[:, 4:6] / 2
            tr = tr[cxy[:, 1] > 450]
        if gt_file.split('/')[-1] in ['G_9']:  # filter person like appearance on the belt
            cxy = tr[:, 2:4] + tr[:, 4:6] / 2
            tr = tr[cxy[:, 1] > 600]
        if gt_file.split('/')[-1] in ['G_9', 'H_9', 'I_9']:  # filter person like appearance on the belt
            ymax = 1010
            cxy = tr[:, 2:4] + tr[:, 4:6] / 2
            tr = tr[cxy[:, 1] < ymax]
            cxy = cxy[cxy[:, 1] < ymax]
            tr = tr[cxy[:, 0] > 130]

            gcxy = gt[:, 2:4] + gt[:, 4:6] / 2
            gt = gt[gcxy[:, 1] < ymax]
            gcxy = gcxy[gcxy[:, 1] < ymax]
            gt = gt[gcxy[:, 0] > 130]

    if database == 'clasp2' and class_id == 2:
        if gt_file.split('/')[-1] in ['G_11', 'H_11', 'I_11']:
            cxy = tr[:, 2:4] + tr[:, 4:6] / 2
            tr = tr[cxy[:, 1] < 650]

            gcxy = gt[:, 2:4] + gt[:, 4:6] / 2
            gt = gt[gcxy[:, 1] < 650]

        if gt_file.split('/')[-1] in ['G_9', 'H_9', 'I_9']:  # filter person like appearance on the belt
            cxy = tr[:, 2:4] + tr[:, 4:6] / 2
            tr = tr[cxy[:, 1] < 550]

            gcxy = gt[:, 2:4] + gt[:, 4:6] / 2
            gt = gt[gcxy[:, 1] < 550]

    return gt, tr

if __name__ == '__main__':
    #install pandas 0.25.3
    storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62/'
    min_size = {'clasp1':1, 'clasp2':1} #{'clasp1':30, 'clasp2':60}
    print_gt_stats = 0
    isMCTA = 1 #unused
    isMTA =  0 #SCA
    database = 'clasp1'
    model_names = ['tracktor_tracks'] #['SSL_alpha_tracks', 'SSL_tracks', 'SL_tracks', 'Base_tracks']
    gt_type = 'SCT'
    class_id = 1
    eval_dict = {}
    eval_dict['Methods'] = []
    min_track_size = min_size[database]
    #Run all models evaluation jointly
    for model_name in model_names:
        eval_dict['Methods'].append(model_name)
        meta_map, fr_rate_factor, offset, categories = get_params(database)
        gt_path, result_path = get_dirs(database, storage, model_name, isMTA=isMTA)

        names = []
        results = {'gt':[], 'tracks':[]}
        FPS = {}
        OFFSET = {}

        for exp_name, d_name in meta_map.items():
            # read gt and results for cameras
            gt_folders, tr_files = get_data(database, exp_name, d_name, class_id, categories, isMTA=isMTA)
            print(gt_folders, tr_files)
            for gt_file, tr_file in zip(gt_folders, tr_files):
                if database=='clasp1':
                    gt = np.genfromtxt(gt_file + '/gt/gt.txt', delimiter=',')
                if database=='clasp2':
                    gt = np.genfromtxt(gt_file + '/gt_sct/gt.txt', delimiter=',')

                print(gt_file, tr_file)
                tr = np.genfromtxt(tr_file, delimiter=',')
                #apply ROI constraint filtering in clasp2 C11
                # if class_id==1:
                #     gt, tr = apply_ROI_costraints(gt_file, database, gt, tr, class_id=1)


                print('gt: {}, tracks: {}'.format(gt_file.split('/')[-1], tr_file.split('/')[-1]))
                if print_gt_stats:
                    get_gt_stats(gt, gt_file.split('/')[-1], database)
                # TODO: SCT evaluation
                # python -m motmetrics.apps.eval_motchallenge
                # /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/CLASP1/train_gt
                # /media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/multicamera_wildtrack/wildtrack/Wildtrack_dataset/results-track/C5C1
                results['gt'].append(gt[gt[:,-2]==class_id])
                tr = filter_tracks(tr, min_size=min_track_size)
                print(f'total gt boxes {len(gt)} for class {class_id}')
                print('total track boxes: {}'.format(tr.shape))
                results['tracks'].append(tr)
                
                assert len(tr) > 0
                # do mot accumulation
                # TODO: SCT evaluation
                # python -m motmetrics.apps.eval_motchallenge /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/wild-track/train_gt   /media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/multicamera_wildtrack/wildtrack/Wildtrack_dataset/results-track/C5C1
                # evaluate mot accumations
                name = categories[class_id]+ '_'+ gt_file.split('/')[-1]
                names.append(name)
                FPS[name] = fr_rate_factor[d_name]
                OFFSET[name] = offset[d_name]

    mot_evaluation = EvaluateMOT(gt_path=gt_path,
                                    result_path=result_path,
                                    eval_dict=eval_dict,
                                    vis_path=None,
                                    vis=False,
                                    radious=100,
                                    iou_max=0.8,
                                    isMCTA=isMCTA,
                                    auto_id=True,
                                    fr_factor=FPS,
                                    t_offset=OFFSET)
    eval_dict = mot_evaluation.evaluate_mot_accums(results, names, generate_overall=True)
    #print table or convert to latex format
    for metric, value in eval_dict.items():
        print(metric, value)
#print latex array
#arr = np.array(list(eval_dict.values()))
#printLatexArray(arr)
