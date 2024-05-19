import os
import glob

import copy
import motmetrics as mm
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt


class EvaluateMOT(object):
    def __init__(self, gt_path=None, result_path=None,
                 vis_path=None, vis=None, proj_plane=None,
                 radious=None, iou_max=None, isMCTA=None, fr_factor=None,
                 max_gt_frame=None, min_gt_frame=None, auto_id=False, t_offset=None):
        self.gt_path = gt_path
        self.result_path = result_path
        self.mot_accum = mm.MOTAccumulator(auto_id=auto_id) #, max_switch_time=240
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

    @staticmethod
    def expand_from_temporal_list(box_all=None):
        if box_all is not None:
            box_list = [b for b in box_all if len(b) > 0]
            box_all = np.concatenate(box_list)
        return box_all

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

    def get_mot_accum(self, results, names):
        accums = []
        for all_gt_boxes, all_track_boxes, name in zip(results['gt'], results['tracks'], names):
            accum = mm.MOTAccumulator(auto_id=self.auto_id, max_switch_time=240)
            print('accumulating: {} ...'.format(name))
            self.colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                           for i in range(int(max(all_track_boxes[:, 1]) + 2))]
            # all_track_boxes = track_boxes[track_boxes[:, 0] <= all_gt_boxes[:, 0].max()]

            # update accumulator frame-by-frame
            for fr in np.unique(all_gt_boxes[:, 0]):
                # print('Frame {}'.format(fr))
                gt_boxes = all_gt_boxes[all_gt_boxes[:, 0] == fr]
                # TODO: modify based on the clasp1 ann
                tr_fr = fr * self.fr_factor[name]
                # print('gt frame: {}, track frame: {}'.format(fr, tr_fr))
                track_boxes = all_track_boxes[all_track_boxes[:, 0] == tr_fr - self.offset[name]]
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

                accum.update(
                    list(gt_ids),
                    list(track_ids),
                    distance)
            accums.append(accum)
        return accums

    def evaluate_mot_accums(self, results, names, generate_overall=False):
        mh = mm.metrics.create()
        accums = self.get_mot_accum(results, names)
        summary = mh.compute_many(
            accums,
            metrics=mm.metrics.motchallenge_metrics,
            names=names,
            generate_overall=generate_overall
        )

        str_summary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names)
        print(str_summary)


def get_dirs(database, storage):

    if database == 'MNIST':
        #full set
        #gt_path = storage + 'tracking_wo_bnw/data/CLASP1/train_gt'
        #20% test set: monocular
        gt_path = storage + 'tracking-by-animation/DHAE_feature/3Object_test_org/Results/tracking_results/evaluation' #/MNIST_MOT_bmvc
        # tracktor
        result_path = storage + 'tracking-by-animation/DHAE_feature/3Object_test_org/Results/tracking_results'
        # resnet mht
        #result_path = storage + 'tracking_wo_bnw/output/mrcnn/supervised_clasp1'
    if database == 'SPRITE':
        #20% test set: random set unable to generate mot metrics
        # Full annotation set is used to evaluate Sl and SSL based SCT
        #gt_path = storage + 'tracking_wo_bnw/data/CLASP1/train_gt'
        gt_path = storage + 'tracking_wo_bnw/data/CLASP/train_gt_all/PB_gt'
        # tracktor
        result_path = storage + 'tracking_wo_bnw/output/clasp2/SSL_tracks'
        # resnet mht
        #result_path = storage + 'tracking_wo_bnw/output/mrcnn/supervised_clasp1'
    return  gt_path, result_path

def get_data(database, gt_path, result_path):
    if database=='MNIST':
        gt_folders = glob.glob(gt_path + '/*')
        gt_folders.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
        # SCT
        tr_files = glob.glob(result_path + '/*.txt')
        tr_files.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
    if database=='SPRITE':
        gt_folders = glob.glob(gt_path + '/{}_*'.format(d_name))
        gt_folders.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
        # SCT
        tr_files = glob.glob(result_path + '/{}/eval/{}*.txt'.format(exp_name.lower(), class_id))
        tr_files.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
    return gt_folders, tr_files



if __name__ == '__main__':
    import pandas as pd
    storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/'
    isMCTA = 0
    database = 'MNIST' #'clasp1'
    eval_plane = '2D'
    gt_path, result_path = get_dirs(database, storage)
    gt_folders, tr_files = get_data(database, gt_path, result_path)

    print(gt_folders)
    print(tr_files)

    names = []
    results = {'gt':[], 'tracks':[]}
    fr_factor = {}
    offset = {}

    for gt_file, tr_file in zip(gt_folders, tr_files):
        gt = np.genfromtxt(gt_file + '/gt/gt.txt', delimiter=',')
        trs = []
        with open(tr_file) as f:
            lines = f.readlines()
            for line in lines:
                trs.append([float(item) for item in line.split('\n')[0].split(' ')])
        tr = np.array(trs)

        print('gt: {}'.format(gt.shape))
        print('tr: {}'.format(tr.shape))

        # python -m motmetrics.apps.eval_motchallenge
        # /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/CLASP1/train_gt
        # /media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/multicamera_wildtrack/wildtrack/Wildtrack_dataset/results-track/C5C1

        results['gt'].append(gt)
        results['tracks'].append(tr)
        names.append(gt_file.split('/')[-1])
        fr_factor[gt_file.split('/')[-1]] = 1
        offset[gt_file.split('/')[-1]] = 0

        assert len(tr) > 0
        # do mot accumulation
        # TODO: SCT evaluation
        # python -m motmetrics.apps.eval_motchallenge /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/wild-track/train_gt   /media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/multicamera_wildtrack/wildtrack/Wildtrack_dataset/results-track/C5C1
        # evaluate mot accumations
    mot_evaluation = EvaluateMOT(gt_path=gt_path,
                                 result_path=result_path,
                                 vis_path=None,
                                 vis=False,
                                 radious=100,
                                 iou_max=0.5,
                                 isMCTA=isMCTA,
                                 auto_id=True,
                                 fr_factor=fr_factor,
                                 t_offset=offset)
    mot_evaluation.evaluate_mot_accums(results, names, generate_overall=True)
    # to generate overall: run eval_2D_clasp1_overall.py