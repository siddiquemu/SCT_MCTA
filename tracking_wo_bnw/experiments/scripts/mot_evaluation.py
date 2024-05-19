import motmetrics as mm
import numpy as np
import pandas as pd

class EvaluateMOT(object):
    def __init__(self, gt_path=None, result_path=None):
        self.gt_path = gt_path
        self.result_path = result_path
        self.mot_accum = mm.MOTAccumulator(auto_id=True)

    @staticmethod
    def expand_from_temporal_list(box_all=None):
        if box_all is not None:
            box_list = [b for b in box_all if len(b) > 0]
            box_all = np.concatenate(box_list)
        return box_all

    def read_file(self, cam=None, resultorgt = 'gt'):
        bbs = []
        if resultorgt == 'gt':
            with open(self.gt_path, "r") as f:
                for line in f:
                    line = line.strip()
                    fields = line.split(",")
                    if float(fields[0])%100==0:
                        bbs.append([float(fields[0]),float(fields[1]), float(fields[2]), float(fields[3]), float(fields[4]), float(fields[5])])
        else:
            bbs = pd.read_csv(self.result_path,index_col=False)
            bbs = bbs.values
            #bbs = bbs[bbs[:,0]>=2100]
            bbs = bbs[bbs[:,1]==cam]
        return bbs


    def get_mot_accum(self, results):
        gt_boxes = results['gt']
        gt_boxes =  np.array(gt_boxes)
        track_boxes = results['tracks']
        gts = []
        gt_ids = []
        tracks = []
        track_ids = []

        for id in np.unique(gt_boxes[:,1]):
            print('get all gt dets for id {}'.format(id))
            gts.append(gt_boxes[gt_boxes[:,1]==id])
            gt_ids.append(gt_boxes[:,1][gt_boxes[:,1]==id].astype('int'))

        #gts = np.stack(gts, axis=0)
        gts = self.expand_from_temporal_list(box_all=gts)
        # x1, y1, x2, y2 --> x1, y1, width, height
        gts = np.stack((gts[:, 2],
                        gts[:, 3],
                        gts[:, 4],
                        gts[:, 5]),
                        axis=1)

        for id in np.unique(track_boxes[:,7]):
            print('get all track dets for id {}'.format(id))
            tracks.append(track_boxes[track_boxes[:,7]==id])
            track_ids.append(track_boxes[:,7][track_boxes[:,7]==id].astype('int'))

        #tracks = np.stack(tracks, axis=0)
        tracks = self.expand_from_temporal_list(box_all=tracks)
        # x1, y1, x2, y2 --> x1, y1, width, height

        tracks = np.stack((tracks[:, 3],
                           tracks[:, 4],
                           tracks[:, 5] - tracks[:, 3],
                           tracks[:, 6] - tracks[:, 4]),
                           axis=1)


        distance = mm.distances.iou_matrix(gts, tracks, max_iou=0.5)
        gt_ids = self.expand_from_temporal_list(box_all=gt_ids)
        track_ids = self.expand_from_temporal_list(box_all=track_ids)
        print(distance[distance>0])
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
            name=name )

        str_summary = mm.io.render_summary(
            summary,
            formatters=mh.formatters,
            namemap=mm.io.motchallenge_metric_names )
        print(str_summary)

if __name__ == '__main__':
    gt_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d6/tracking_wo_bnw/data/CLASP/train_gt/cam02exp2.mp4/gt/gt.txt'
    result_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d6/tracking_wo_bnw/output/tracktor/online_SCT/global_tracks_C5C2.csv'
    #read gt and results for a camera
    results = {}
    mot_evaluation = EvaluateMOT(gt_path=gt_path,result_path=result_path)
    results['gt'] = mot_evaluation.read_file(cam=2, resultorgt='gt')
    results['tracks'] = mot_evaluation.read_file(cam=2, resultorgt='tracks')
    #do mot accumulation
    # evaluate mot accumations
    name = 'C2'
    mot_evaluation.evaluate_mot_accums(results, name, generate_overall=False)