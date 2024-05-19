import os
import glob

import copy

import cv2
import motmetrics as mm
import numpy as np
import pandas as pd
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt


class EvaluateMOT(object):
    def __init__(self, gt_path=None, result_path=None, eval_dict=None,
                 vis_path=None, img_path=None, vis=None, proj_plane=None,
                 radious=None, iou_max=None, isMCTA=None, fr_factor=None,
                 max_gt_frame=None, min_gt_frame=None, auto_id=False, t_offset=None, gt_folder_map=None):
        self.gt_path = gt_path
        self.result_path = result_path
        self.auto_id = auto_id
        self.mot_accum = mm.MOTAccumulator(auto_id=self.auto_id) #, max_switch_time=240
        self.vis = vis
        self.colors = []

        self.proj_plane = proj_plane
        self.radious = radious
        self.iou_max = iou_max
        self.max_gt_frame = max_gt_frame
        self.min_gt_frame = min_gt_frame
        self.isMCTA = isMCTA
        self.fr_factor = fr_factor
        self.offset=t_offset
        self.img_path = img_path
        self.vis_path = vis_path
        self.gt_folder_map=gt_folder_map
        self.eval_table = eval_dict

    @staticmethod
    def delete_all(demo_path, fmt='png'):
        import glob
        filelist = glob.glob(os.path.join(demo_path, '*.' + fmt))
        if len(filelist) > 0:
            for f in filelist:
                os.remove(f)
    @staticmethod
    def expand_from_temporal_list(box_all=None):
        if box_all is not None:
            box_list = [b for b in box_all if len(b) > 0]
            box_all = np.concatenate(box_list)
        return box_all

    @staticmethod
    def write_box(img, bb, color = (255, 0, 0), isgt=1):
        xmin = bb[2]
        ymin = bb[3]
        w = bb[4]
        h = bb[5]
        xmax = xmin + w
        ymax = ymin + h
        display_txt = '{}'.format(int(bb[1]))
        img = cv2.rectangle(img, (np.int(xmin), np.int(ymin)), (np.int(xmax), np.int(ymax)), color, 5)
        if isgt:
            cv2.putText(img, display_txt, (int(xmin+w/2), int(ymin+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
        else:
            cv2.putText(img, display_txt, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
        return img

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

    def sct_track_eval_table(self, summary=None, eval_type='mcta_overall'):
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

    def compare_gt_track3D(self, gt, track, fr=None):
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

    def compare_gt_track2D(self, gt, track, fr=None, result_name=None):
        out_dir = os.path.join(self.vis_path, result_name)
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        assert os.path.exists(out_dir), 'vis path {} not exist'.format(out_dir)
        if self.gt_folder_map[result_name] in ['G_9', 'G_11']:
            self.img = cv2.imread(
                os.path.join(self.img_path, self.gt_folder_map[result_name], 'img1/{:05d}.jpg'.format(int(fr))))
        else:
            self.img = cv2.imread(
                os.path.join(self.img_path, self.gt_folder_map[result_name], 'img1/{:06d}.png'.format(int(fr))))


        #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("image", 640, 480)
        for bb in gt:
            self.img = self.write_box(self.img, bb, color=(0,255,0), isgt=1)
        for bb in track:
            self.img = self.write_box(self.img, bb, color=(0, 0, 255), isgt=0)

        cv2.imwrite(os.path.join(out_dir, '{:06d}.png'.format(int(fr))), self.img)
        #cv2.waitKey(30)
        #cv2.destroyAllWindows()

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
            accum = mm.MOTAccumulator(auto_id=self.auto_id) #, max_switch_time=240
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
                    self.compare_gt_track2D(gt_boxes, track_boxes, fr=fr, result_name=name)

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

def filter_tracks(tracks, min_size = 10):
    filtered_tracks = []
    for id in np.unique(tracks[:,1]):
        track_id = tracks[tracks[:,1]==id]
        if len(track_id)>min_size:
            filtered_tracks.append(track_id)
    return np.concatenate(filtered_tracks)

def get_dirs(database, storage, isMCTA, det_model, dist_metric):

    if database == 'clasp2':
        #full set
        gt_path = storage + 'tracking_wo_bnw/data/CLASP/train_gt_all/PB_gt'
        gt_img_path = storage + 'tracking_wo_bnw/data/CLASP/train_gt_all/PB_gt'
        #20% test set: monocular - 20 % test set only works for detections. tracking: use full set
        #TODO: apply MC GT instead of SCT GT
        #gt_path = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/ourdataset/P_GT_MC'
        # tracktor
        if isMCTA:
            result_path = storage + 'tracking_wo_bnw/output/{}/MCTA/{}/{}'.format(database, det_model, dist_metric)
        else:
            result_path = storage + 'tracking_wo_bnw/output/{}/MCTA/{}/SCT_MC_GT'.format(database, det_model)
            #result_path = storage + 'tracking_wo_bnw/output/{}/{}'.format(database, det_model+'_tracks')

        vis_path = storage + 'tracking_wo_bnw/output/{}/MCTA/{}/vis'.format(database, det_model)
        # resnet mht
        #result_path = storage + 'tracking_wo_bnw/output/mrcnn/supervised_clasp1'
    return  gt_path, result_path, vis_path, gt_img_path

def apply_partial_constraints(gt, tr, database, gt_file):
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
    return gt, tr

def get_gt_stats(gt, seq):
    print('              ')
    print('######## CLASP2 MCTA Person GT Stats: {} ########'.format(seq))
    print('Unique Identity: {}'.format(len(np.unique(gt[:,1]))))
    print('Annotated Frames: {}'.format(len(np.unique(gt[:, 0]))))
    print('Annotations: {}'.format(len(gt)))
    print('              ')

if __name__ == '__main__':
    storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d62/'
    print_gt_stats = 0
    isMCTA = 1
    eval_plane = '2D'
    database = 'clasp2'
    det_model = 'SSL_alpha'
    metric = {1: 'hausdorff', 2: 'frechet'}
    dist_metric = metric[2]
    class_id = 1
    vis=0
    cams=[9, 11]
    #to report eval
    eval_dict = {}
    eval_dict['Methods'] = []
    eval_dict['Methods'].append(det_model)
    #for SCT
    min_track_size = 0

    meta_map = {'G': 'G', 'H': 'H', 'I': 'I'}
    if isMCTA:
        #<!> use gt frame index directly in track results instead of frame rate conversion
        #[!} here the GT frames are subset of all video frames and maintain same fr index
        fr_rate_factor = {'G': 1, 'H': 1, 'I': 1}
        # no offset
        offset = {'G': 0, 'H': 0, 'I': 0}
    else:
        fr_rate_factor = {'G': 1, 'H': 1, 'I': 1}
        # no offset
        offset = {'G': 0, 'H': 0, 'I': 0}

    categories = {1:'person', 2:'bag'}
    #categories = {0: 'person', 1: 'bag'}
    gt_path, result_path, vis_path, gt_img_path = get_dirs(database, storage, isMCTA,
                                                           det_model, dist_metric)

    names = []
    results = {'gt': [], 'tracks': []}
    FPS = {}
    OFFSET = {}
    gt_folder_map = {}

    for exp_name, d_name in meta_map.items():
        # read MC gt and results for cameras
        gt_folders = glob.glob(gt_path + '/{}_*'.format(d_name))
        gt_folders.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
        if isMCTA:
            tr_files = glob.glob(result_path + '/{}*.txt'.format(exp_name))
        else:
            # SCT
            print('evaluate SCT on MC annotations')
            tr_files = glob.glob(result_path + '/{}*.txt'.format(exp_name))
            #tr_files = glob.glob(result_path + '/{}/eval/{}*.txt'.format(exp_name.lower(), class_id))

        tr_files.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
        ind_gt = 0
        for gt_file, tr_file in zip(gt_folders, tr_files):
            print(gt_file, tr_file)
            gt = np.genfromtxt(gt_file + '/person_label_{}{}_mc_partial_correct.txt'.format(d_name, cams[ind_gt]), delimiter=',')
            tr = np.genfromtxt(tr_file, delimiter=',')
            gt, tr = apply_partial_constraints(gt, tr, database, gt_file)
            if print_gt_stats:
                get_gt_stats(gt, '{}_{}'.format(d_name, cams[ind_gt]))
            # TODO: SCT evaluation
            # python -m motmetrics.apps.eval_motchallenge
            # /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/wild-track/train_gt
            # /media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/multicamera_wildtrack/wildtrack/Wildtrack_dataset/results-track/C5C1
            results['gt'].append(gt)
            if not isMCTA:
                tr = filter_tracks(tr, min_size=min_track_size)
            results['tracks'].append(tr)

            assert len(tr) > 0
            if isMCTA:
                name = categories[class_id] + '_{}_{}'.format(d_name, cams[ind_gt]) + '_MCTA'
            else:
                name = categories[class_id] + '_{}_{}'.format(d_name, cams[ind_gt]) + '_SCT'

            FPS[name] = fr_rate_factor[d_name]
            OFFSET[name] = offset[d_name]

            print('running {} ...'.format(name))
            names.append(name)
            gt_folder_map[name] = '{}_{}'.format(d_name, cams[ind_gt])
            ind_gt += 1

    mot_evaluation = EvaluateMOT(gt_path=gt_path,
                                 result_path=result_path,
                                 eval_dict=eval_dict,
                                 vis_path=vis_path,
                                 img_path=gt_img_path,
                                 vis=vis,
                                 proj_plane=eval_plane,
                                 radious=100,
                                 iou_max=0.5,
                                 isMCTA=isMCTA,
                                 auto_id=True,
                                 fr_factor=FPS,
                                 t_offset=OFFSET,
                                 gt_folder_map=gt_folder_map)
    eval_dict = mot_evaluation.evaluate_mot_accums(results, names, generate_overall=True)
    # print table or convert to latex format
    for metric, value in eval_dict.items():
        print(metric, value)