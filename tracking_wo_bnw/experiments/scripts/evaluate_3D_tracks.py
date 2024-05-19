import motmetrics as mm
import numpy as np
import pandas as pd
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

class EvaluateMOT(object):
    def __init__(self, gt_path=None, result_path=None,
                 vis_path=None, vis=None, proj_plane=None,
                 radious=None, iou_max=None, isMCTA=None,
                 max_gt_frame=None, min_gt_frame=None, auto_id=False):
        self.gt_path = gt_path
        self.result_path = result_path
        self.mot_accum = mm.MOTAccumulator(auto_id=auto_id, max_switch_time=240)
        self.vis=vis
        self.colors = []
        self.out_dir = vis_path
        self.proj_plane=proj_plane
        self.radious = radious
        self.iou_max = iou_max
        self.max_gt_frame = max_gt_frame
        self.min_gt_frame = min_gt_frame
        self.isMCTA = isMCTA

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

    def compare_gt_track(self, gt, track, fr=None):
        ax = self.init_3d_figure()
        # ax = init_2d_figure()
        for bb in gt:
            x = bb[2]
            y = bb[3]
            z = 0.03
            id = int(bb[1])
            # color = ImageColor.getcolor(colors[int(id)], "RGB")
            # color = webcolors.rgb_to_name(color,  spec='html4')
            # ax.scatter(x, y, z, alpha=0.8, cmap=colors(int(id)), s=5)
            ax.scatter(x, y, z, alpha=0.5, c='red', s=3) #self.colors[-1]
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

    def read_file(self, resultorgt = 'gt'):
        if resultorgt == 'gt':
            gt_world = pd.read_csv(self.gt_path, index_col=False)
            bbs = gt_world.values
        else:
            bbs = pd.read_csv(self.result_path,index_col=False)
            #Cx Y2 frame h id w
            #bbs = bbs.values
            bbs = np.transpose(np.vstack((bbs['frame'].array, bbs['id'].array, bbs['Cx'].array, bbs['Y2'].array, bbs['w'].array, bbs['h'].array)))
        return bbs


    def get_mot_accum(self, results):
        gt_boxes = results['gt']
        gt_boxes = gt_boxes[self.min_gt_frame<=gt_boxes[:,0]]
        all_gt_boxes = gt_boxes[gt_boxes[:, 0] <= self.max_gt_frame]
        #gt_boxes =  np.array(gt_boxes)
        all_track_boxes = results['tracks']
        self.colors = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                  for i in range(int(max(all_track_boxes[:, 1])+2))]
        #all_track_boxes = track_boxes[track_boxes[:, 0] <= all_gt_boxes[:, 0].max()]

        # update accumulator frame-by-frame
        for i, fr in enumerate(np.unique(all_gt_boxes[:,0])):
            #print('Frame {}'.format(fr))
            gt_boxes = all_gt_boxes[all_gt_boxes[:,0]==fr]
            if self.proj_plane=='3D':
                if self.isMCTA:
                    tr_fr=fr
                else:
                    tr_fr =  (fr//5 - 1)*15 + 1 #(fr//5)*15 + 1
                track_boxes = all_track_boxes[all_track_boxes[:, 0] == tr_fr]

            if self.proj_plane=='2D':
                if self.isMCTA:
                    tr_fr=fr
                else:
                    tr_fr = (fr//5 - 1)*15 + 1# i*15 + 1
                track_boxes = all_track_boxes[all_track_boxes[:, 0] == tr_fr]
                #assert len(track_boxes)>0, 'no track found at {}'.format(tr_fr)

            if self.vis:
                self.compare_gt_track(gt_boxes, track_boxes, fr=fr)

            gts = []
            gt_ids = []
            tracks = []
            track_ids = []

            for id in np.unique(gt_boxes[:,1]):
                #print('get all gt dets for id {}'.format(id))
                gts.append(gt_boxes[gt_boxes[:,1]==id])
                gt_ids.append(gt_boxes[:,1][gt_boxes[:,1]==id].astype('int'))

            #print('total gt ids: {}'.format(len(gts)))
            #gts = np.stack(gts, axis=0)
            gts = self.expand_from_temporal_list(box_all=gts)
            # x1, y1, x2, y2 --> x1, y1, width, height
            if self.proj_plane=='3D':
                gts = np.stack((gts[:, 2],
                                gts[:, 3]),
                                axis=1)
            if self.proj_plane=='2D':
                gts = np.stack((gts[:, 2],
                                gts[:, 3],
                                gts[:, 4],
                                gts[:, 5]),
                               axis=1)

            for id in np.unique(track_boxes[:, 1]):
                # print('get all track dets for id {}'.format(id))
                tracks.append(track_boxes[track_boxes[:, 1] == id])
                track_ids.append(track_boxes[:, 1][track_boxes[:, 1] == id].astype('int'))

            #print('total track ids: {}'.format(len(tracks)))
            # x1, y1, x2, y2 --> x1, y1, width, height
            if self.proj_plane=='3D':
                # tracks = np.stack(tracks, axis=0)

                tracks = self.expand_from_temporal_list(box_all=tracks)

                tracks = np.stack((tracks[:, 2],
                                   tracks[:, 3]),
                                   axis=1)
                distance = mm.distances.norm2squared_matrix(gts, tracks, max_d2=self.radious)
            if self.proj_plane=='2D':
                # tracks = np.stack(tracks, axis=0)
                if len(tracks)>0:
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
            #print(distance.shape)
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

            self.mot_accum.update(
                    list(gt_ids),
                    list(track_ids),
                    distance)
        return self.mot_accum



    def evaluate_mot_accums(self, results, name, generate_overall=False):
        mh = mm.metrics.create()
        self.mot_accum = self.get_mot_accum(results)
        summary = mh.compute(
            self.mot_accum,
            metrics=mm.metrics.motchallenge_metrics,
            name=name)

        str_summary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names )
        print(str_summary)

if __name__ == '__main__':
    database = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/multicamera_wildtrack/wildtrack/Wildtrack_dataset'
    gt_path = database + '/gt_world.csv'
    #3D_tracks.csv
    result_path = database + '/results-track/global_tracks/3D_tracks_offline_mht_hausdorff.csv'
    out_dir = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/output/tracktor_wildtrack/evaluation_global'
    #read gt and results for a camera
    results = {}
    mot_evaluation = EvaluateMOT(gt_path=gt_path,
                                 result_path=result_path,
                                 vis_path=out_dir,
                                 vis=False,
                                 proj_plane='3D',
                                 radious=100,
                                 iou_max=0.5,
                                 max_gt_frame=1955,
                                 min_gt_frame=840)
    results['gt'] = mot_evaluation.read_file(resultorgt='gt')
    results['tracks'] = mot_evaluation.read_file(resultorgt='tracks')
    #do mot accumulation
    #TODO: SCT evaluation
    #python -m motmetrics.apps.eval_motchallenge /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/wild-track/train_gt   /media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/multicamera_wildtrack/wildtrack/Wildtrack_dataset/results-track/C5C1
    # evaluate mot accumations
    name = 'ours'
    mot_evaluation.evaluate_mot_accums(results, name, generate_overall=False)