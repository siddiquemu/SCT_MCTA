from __future__ import division
import numpy as np
from scipy.spatial.distance import directed_hausdorff, cosine, mahalanobis
from scipy.optimize import linear_sum_assignment
import cv2
import sys
import os
import glob
import copy
import pdb
import matplotlib.pyplot as plt
from MCTA.tracklet_formation import form_tracklets
from MCTA.Tracker_Merge_SingleCam import *
from read_meta_data import *
import collections
import pandas as pd
import copy
from MCTA.utils import *
from MCTA.discrete_frechet import frechetdist
from mc_projections import Projection
import time
import collections
from PIL import ImageColor
np.set_printoptions(threshold=sys.maxsize)

__version__ = 0.1

class global_tracker(object):
    def __init__(self, fr, id=None, global_tracker=None, first_used=0):
        self.global_tracker_file = global_tracker
        self.fr = fr
        self.id = id
        self.first_used = first_used

    def write_csv(self,data):
        data.to_csv(self.global_tracker_file, mode='a', index=False)

    def update_state(self, bb):
        camera = str(bb[-1])
        event_type = 'LOC'
        type = 'PAX'
        pax_id = 'P'+str(self.id)
        self.global_tracker_file.writelines(
            event_type + ' ' + 'type: ' + type + ' ' + 'camera-num: ' + camera + ' ' + 'frame: ' + str(self.fr) + ' '
            'time-offset: ' + '{:.2f}'.format(self.fr / 30.0) + ' ' + 'BB: ' + str(int(bb[2])) + ', ' + str(int(bb[3])) + ', '
            + str(int(bb[2] + bb[4])) + ', ' + str(int(bb[3] + bb[5])) + ' ' + 'ID: ' + pax_id + ' ' + 'PAX-ID: ' + pax_id
            + ' ' + 'first-used: ' + self.first_used + ' ' + 'partial-complete: ' + 'description: ' + '\n')

    def update_state_csv(self, bb):
        camera = str(bb[-1])
        pax_id = self.id
        # prepare data frame:

        Dframe = pd.DataFrame({'frame': int(self.fr), 'camera-num': camera,
                               'time-offset': float('{:.2f}'.format(self.fr / 30.0)),
                               'x1':int(bb[2]), 'y1':int(bb[3]), 'x2':int(bb[2]+bb[4]),
                               'y2':int(bb[3]+bb[5]), 'id': pax_id, 'first-used':self.first_used
                               })
        global_tracker.write_csv(Dframe)

class MCTA(object):
    def __init__(self, mc_track_segments=None, all_params=None, global_track_start_time=None, timeoffset=None, min_trklt_size=60):
        #mc_track_segments should have the keys as all cameras: accessible for parwise association
        #default fixed global variable
        self.out_dir = all_params['output_dir']
        self.img_path = all_params['img_path']
        self. projection_class = Projection(gt_path=None, out_path=self.out_dir, images_path=self.img_path)
        self.cam_list = all_params['cams']
        self.cam_net = all_params['cam_graph']
        self.BFS_cam_graph=self.cam_net
        self.min_trklt_size = all_params['min_trklt_size']
        self.isSCA = False
        self.vis = all_params['save_imgs']
        self.eval_plane = '2D'
        self.new_ids = all_params['new_ids']
        self.motion_model = False
        self.appearance_model = False
        self.dist_correct = False
        self.print_stats = all_params['print_stats']
        self.sequential_id = all_params['sequential_id']
        self.batch_start=all_params['batch_start']
        self.batch_size=all_params['batch_size']
        self.global_track_results = all_params['result_fields']
        self.global_track_saver = all_params['global_full_track_file']
        self.img_HW = all_params['img_HW']
        self.isSCT = all_params['isSCT']
        self.vis_rate = all_params['vis_rate']
        self.save_global_tracks = all_params['save_global_tracks']
        self.colors = all_params['colors']
        self.dist_metric = all_params['dist_metric']
        self.debug = all_params['print_stats']
        self.frame_2fps = all_params['start_gt_frame']

        # updated for each batch
        #update locally
        # one camera correspondence with all other camera: Cij=Cji, Cii=0
        #if self.eval_plane=='2D':
           # self.global_track_results3D = {'frame': [], 'id': [], 'Cx': [], 'Y2': [], 'w': [], 'h': []}
        #else:
            #self.global_track_results2D = {'frame': [], 'id': [], 'x': [], 'y': [], 'w': [], 'h': [], 'cam': []}

        self. MCparams = {}
        # to concatenate camera tracklets for joint affinity
        self.combined_tracks3D = []
        self.combined_tracks2D = []
        self.cam_track_idx = {keys: 0 for keys in range(0, len(self.cam_list))}

        self.MC_tracklets = {}
        self.cam_visited = []
        self.index_factor = {}
        self.cam_trklts_size = {}
        self.cam_trklts_id = []
        self.mc_all_trklts_dict = {}
        self.mc_all_trklts = []
        self.mc_batch_ids = [] #used to verify already associated tracks should associate again or not
        self.combined_tracks_proj = []
        self.combined_tracks_org = []

        #update globally
        self.mc_track_segments = mc_track_segments
        self.global_track_start_time = global_track_start_time
        self.G_mc = all_params['global_mcta_graph']
        self.single_cam_multi_asso = all_params['single_cam_multi_asso']
        self.global_track_id = all_params['global_track_id']
        self.keep_raw_id = all_params['keep_raw_id']
        self.store_prev_dist = all_params['prev_affinity']



    def _dfs(self, graph, node, visited):
        if node not in visited:
            visited.append(node)
            for n in graph[node]:
                self._dfs(graph, n, visited)
        return visited

    @staticmethod
    def linear_sum_assignment_with_inf(cost_matrix):
        cost_matrix = np.asarray(cost_matrix)
        min_inf = np.isneginf(cost_matrix).any()
        max_inf = np.isposinf(cost_matrix).any()
        if min_inf and max_inf:
            raise ValueError("matrix contains both inf and -inf")
        if min_inf or max_inf:
            values = cost_matrix[~np.isinf(cost_matrix)]
            if values.size == 0:
                cost_matrix = np.full(cost_matrix.shape, 1000)  # workaround for the cast of no finite costs
            else:
                m = values.min()
                M = values.max()
                n = min(cost_matrix.shape)
                # strictly positive constant even when added
                # to elements of the cost matrix
                positive = n * (M - m + np.abs(M) + np.abs(m) + 1)
                if max_inf:
                    place_holder = (M + (n - 1) * (M - m)) + positive
                if min_inf:
                    place_holder = (m + (n - 1) * (m - M)) - positive
                cost_matrix[np.isinf(cost_matrix)] = place_holder
        return linear_sum_assignment(cost_matrix)

    @staticmethod
    def expand_from_temporal_list(box_all=None, mask_all=None):
        if box_all is not None:
            box_list = [b for b in box_all if len(b) > 0]
            box_all = np.concatenate(box_list)
        if mask_all is not None:
            mask_list = [m for m in mask_all if len(m) > 0]
            masks_all = np.concatenate(mask_list)
        else:
            masks_all = []
        return box_all, masks_all

    def vis_box(self, img, bb, id, colors):
        color = ImageColor.getcolor(colors[int(id)], "RGB")
        cv2.rectangle(img, (int(bb[2]), int(bb[3])),
                      (int(bb[2] + bb[4]), int(bb[3] + bb[5])),
                      color, 3, cv2.LINE_AA)
        cv2.putText(img, '{}'.format(id),
                    (int(bb[2] + bb[4] / 2), int(bb[3] + bb[5] / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, color, 2, cv2.LINE_AA)
        return img

    def read_cams(self, fr, imgs_fr={}):
        for cam in self.cam_list:
            imgs_fr[cam] = cv2.imread('{}/C{}/{:08d}.png'.format(self.img_path, cam, fr))
        return imgs_fr

    def cams_grid(self, imgs_fr, fr):
        if len(self.cam_list)==7:
            blankImg = np.zeros(shape=[imgs_fr[1].shape[0], imgs_fr[1].shape[1], 3], dtype=np.uint8)
            imgC1C5 = cv2.vconcat([imgs_fr[1], imgs_fr[5]])
            imgC2C6 = cv2.vconcat([imgs_fr[2], imgs_fr[6]])
            imgC3C7 = cv2.vconcat([imgs_fr[3], imgs_fr[7]])
            imgC4CB = cv2.vconcat([imgs_fr[4], blankImg])
            final_img = cv2.hconcat([imgC1C5, imgC2C6, imgC3C7, imgC4CB])
            W,H=3840, 1080
        else:
            imgC1C4 = cv2.vconcat([imgs_fr[1], imgs_fr[4]])
            imgC2C3 = cv2.vconcat([imgs_fr[2], imgs_fr[3]])
            final_img = cv2.hconcat([imgC1C4, imgC2C3])
            W,H = 1920,1080
        final_img = cv2.resize(final_img, (W, H), interpolation=cv2.INTER_AREA)  # 1.5

        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        #pdb.set_trace()
        cv2.imwrite(os.path.join(self.out_dir, '{:08d}.png'.format(fr)), final_img)

    def get_sequential_results_3D(self, label_map):
        """

        :param mc_global_trklts: list of all camera tracklets: [fr, id, x, y, w, h, cam]
        :param global_track_results: dictionary of combined mc tracks
        :param out_dir:
        :param fr_start:
        :param fr_end:
        :param colors:
        :return: save each camera tracks after updating the mc-correspondences
        """
        frame_2fps = 0
        for fr in range(self.batch_start, self.batch_start + self.batch_size):
            # 1:5, 16:10, 31:15
            if (fr - 1) % 15 == 0:
                frame_2fps += 5
                # ax = init_3d_figure()
                # ax = init_2d_figure()
                if self.vis:
                    imgs_fr = self.read_cams(fr, imgs_fr={})
                # pdb.set_trace()
                # pdb.set_trace()
                track_count = 0
                for id, track in self.mc_all_trklts_dict.items():
                        for bb in track:
                            if bb[0] == fr:
                                if label_map[id] not in self.keep_raw_id:
                                    first_used = 1
                                    if self.sequential_id:
                                        self.global_track_id += 1
                                        self.keep_raw_id[label_map[c]] = self.global_track_id
                                    else:
                                        self.keep_raw_id[label_map[c]] = label_map[c]
                                x = bb[2]
                                y = bb[3]
                                # color = ImageColor.getcolor(colors[int(id)], "RGB")
                                # color = webcolors.rgb_to_name(color,  spec='html4')
                                # ax.scatter(x, y, z, alpha=0.8, cmap=colors(int(id)), s=5)
                                # ax.scatter(x, y, z, alpha=0.8, c=colors[id], s=5)
                                track_count += 1
                                # pdb.set_trace()
                                # ax.scatter(x, y, alpha=0.8, c=colors[id], s=5)
                                self.global_track_results['frame'].append(int(frame_2fps))
                                self.global_track_results['id'].append(id)
                                self.global_track_results['x'].append(x)
                                self.global_track_results['y'].append(y)
                                self.global_track_results['w'].append(bb[4])
                                self.global_track_results['h'].append(bb[5])
                                self.global_track_results['cam'].append(bb[6])
                                if self.vis:
                                    imgs_fr[int(bb[6])] = self.vis_box(imgs_fr[int(bb[6])], bb,
                                                                  self.keep_raw_id[label_map[c]],
                                                                  self.colors)

                # plt.title('mc-tracking in 3D world for frame {} total tracks {}'.format(fr, track_count))
                # plt.savefig(out_dir + '/{:06d}.png'.format(int(fr)), dpi=300)
                # plt.close()
                if self.vis:
                    self.cams_grid(imgs_fr, fr)

        #Dframe = pd.DataFrame(global_track_results)
        #Dframe.to_csv(out_dir + '/2D_tracks.csv', index=False)


    def get_sequential_results_2D(self, label_map):
        """
        :param mc_global_trklts: list of all camera tracklets: [fr, id, x, y, w, h, cam]
        :param global_track_results: dictionary of combined mc tracks
        :param out_dir:
        :param fr_start:
        :param fr_end:
        :param colors:
        :return: save each camera tracks after updating the mc-correspondences
        """
        for fr in range(self.batch_start, self.batch_start + self.batch_size):
            if self.debug:
                print('frame {} batch {} to {}'.format(fr, self.batch_start, self.batch_start + self.batch_size-1))
            # 1:5, 16:10, 31:15
            if (fr-1) % 15 == 0:
                self.frame_2fps += 5
                # ax = init_3d_figure()
                # ax = init_2d_figure()
                if self.vis:
                    imgs_fr = self.read_cams(fr, imgs_fr={})
                # pdb.set_trace()
                # pdb.set_trace()
                track_count = 0

                for id, bb in self.mc_all_trklts_dict.items():
                        for i in range(len(bb)):
                            if self.batch_start <= bb[i, 0] <= fr:
                                if bb[i, 0] == fr:
                                    if label_map[id] not in self.keep_raw_id:
                                        if self.sequential_id:
                                            self.global_track_id += 1
                                            self.keep_raw_id[label_map[id]] = self.global_track_id
                                        else:
                                            self.keep_raw_id[label_map[id]] = label_map[id]
                                    x = bb[i, 2]
                                    y = bb[i, 3]
                                    # color = ImageColor.getcolor(colors[int(id)], "RGB")
                                    # color = webcolors.rgb_to_name(color,  spec='html4')
                                    # ax.scatter(x, y, z, alpha=0.8, cmap=colors(int(id)), s=5)
                                    # ax.scatter(x, y, z, alpha=0.8, c=colors[id], s=5)
                                    track_count += 1
                                    # pdb.set_trace()
                                    # ax.scatter(x, y, alpha=0.8, c=colors[id], s=5)
                                    self.global_track_results['frame'].append(int(self.frame_2fps))
                                    self.global_track_results['id'].append(self.keep_raw_id[label_map[id]])
                                    self.global_track_results['x'].append(x)
                                    self.global_track_results['y'].append(y)
                                    self.global_track_results['w'].append(bb[i, 4])
                                    self.global_track_results['h'].append(bb[i, 5])
                                    self.global_track_results['cam'].append(bb[i, 6])
                                    if self.vis:
                                        imgs_fr[int(bb[i, 6])] = self.vis_box(imgs_fr[int(bb[i, 6])], bb[i,:],
                                                                      int(self.keep_raw_id[label_map[id]]),
                                                                      self.colors)

                # plt.title('mc-tracking in 3D world for frame {} total tracks {}'.format(fr, track_count))
                # plt.savefig(out_dir + '/{:06d}.png'.format(int(fr)), dpi=300)
                # plt.close()
                if self.vis:
                    self.cams_grid(imgs_fr, fr)

        #Dframe = pd.DataFrame(global_track_results)
        #Dframe.to_csv(out_dir + '/2D_tracks.csv', index=False)


    def project_tracklets(self, in_tracklets, params, cam=None):
        out_tracklets = []
        active_trklt = []

        for i, trklt in enumerate(in_tracklets):
            for bb in trklt:
                # _, bbt1 = projection_class.projectCentroid(bb[1:3], params, ca=ca, cp=cp)
                # _, cxy2 = projection_class.projectCentroid(bb[1:3] + bb[3:5] / [2.0,1.0], params, ca=ca, cp=cp)
                # 3D world point
                cxy2 = self.projection_class.project3D(bb[2:4] + bb[4:6] / [2.0, 1.0], params, cam=cam)
                if self.motion_model:
                    vxvy = self.projection_class.project3D(bb[6:8], params, cam=cam)
                    #vxvy = self.projection_class.projectCentroid(bb[5:7], params, ca=ca, cp=cp)
                    bbt2 = cxy2 - bbt1  # projected w/2,h/2
                    bb[1:7] = np.concatenate([cxy2, bbt2, vxvy])  # projected: [cx,cy,w/2,h/2,vx,vy]
                else:
                    # bbt2 = cxy2 - bbt1  # projected w/2,h/2
                    # bb[1:5] = np.concatenate([cxy2, bbt2])  # projected: [cx,y2,w/2,h]
                    bb[2:4] = cxy2
            # Delete tracklets that don't have any detection visible in thecommon ground plane
            #if -310 - 50 < max(trklt[:, 2]) and min(trklt[:, 2]) <= 910 + 50 and -922 - 50 < max(trklt[:, 3]) and min(
                    #trklt[:, 3]) <= 2607 + 50:
            out_tracklets.append(trklt)
            active_trklt.append(i)
            #else:
                #print('tracks found outside the global grid')
        return out_tracklets, active_trklt

    def get_hungarian_matches_combined(self, max_dist=None, dist_metric='frechet'):
        """
        :param tracks_ci:
        :param tracks_cj:
        :param max_dist:
        :param edges:
        :param dist_metric: type of distance metric between two 3D polynomial (Z=0) trajectory
        :return: updated mc correspondence graph edges globally for each batch
        """
        # compute concatenated affinity and then the connection edges by applying cam net graph
        # cam_net = {1:[2,3,4,5,6,7], 2:[3,4,5,6,7], 3: [4,5,6,7], 4: [5,6,7], 5:[6,7], 6:[7]}
        tracks_c = self.combined_tracks3D
        batch_matches = {}
        for cp, ca_list in self.cam_net.items():
            print('Start correspondence for cam {}: {}'.format(cp, ca_list))
            for ca in ca_list:

                # to get pair correspondence
                print('....{}-{}'.format(cp, ca))
                look_back_window = np.arange(self.batch_start - 10 * self.batch_size,
                                             self.batch_start + self.batch_size)
                cost = np.ones((len(tracks_c), len(tracks_c))) * np.inf
                for i, track_ci in enumerate(tracks_c):
                    for j, track_cj in enumerate(tracks_c):
                        #cam index
                        ci = track_ci[0, -1]
                        cj = track_cj[0, -1]
                        # get actual id
                        id_i = track_ci[0, 1]
                        id_j = track_cj[0, 1]
                        # since Aij=Aji and Aii=0 (diagonal) and Ci!=Cj
                        if i in self.MCparams[cp]['track_idx'] and j in self.MCparams[ca]['track_idx']:
                            # apply temporal relation:
                            common_t_stamps = np.array(list(set(list(track_ci[:, 0])).intersection(list(track_cj[:, 0]))))
                            if len(common_t_stamps) >= self.min_trklt_size // 2 \
                                    and (self.global_track_start_time[id_i] in look_back_window
                                         or self.global_track_start_time[id_j] in look_back_window):

                                assert len(common_t_stamps) > 0, 'Non-overlaped T_p, T_a can not be associated'
                                mask1 = np.isin(track_ci[:, 0], common_t_stamps)
                                pos3D_ci = track_ci[mask1]

                                mask2 = np.isin(track_cj[:, 0], common_t_stamps)
                                pos3D_cj = track_cj[mask2]

                                pos3D_ci = copy.deepcopy(pos3D_ci[:, 2:4])  # / [910., 2610.]
                                pos3D_cj = copy.deepcopy(pos3D_cj[:, 2:4])  # / [910., 2610.]
                                # compute frechet distance between trajectory
                                # to solve the maximum recursion limit
                                scan_intvl = 120
                                st_time = time.time()
                                if len(pos3D_ci) > scan_intvl:
                                    if dist_metric == 'frechet':
                                        sys.setrecursionlimit(len(pos3D_ci) * len(pos3D_cj))
                                        d_h = frechetdist.frdist(pos3D_ci[-scan_intvl:], pos3D_cj[-scan_intvl:])
                                        # d_h = linear_frechet(pos3D_ci[:scan_intvl], pos3D_cj[:scan_intvl])
                                    if dist_metric == 'hausdorff':
                                        d_h = max(directed_hausdorff(pos3D_ci, pos3D_cj)[0],
                                                  directed_hausdorff(pos3D_cj, pos3D_ci)[0])
                                else:
                                    if dist_metric == 'frechet':
                                        sys.setrecursionlimit(len(pos3D_ci) * len(pos3D_cj) + scan_intvl)
                                        d_h = frechetdist.frdist(pos3D_ci, pos3D_cj)
                                        # d_h = linear_frechet(pos3D_ci, pos3D_cj)
                                    if dist_metric == 'hausdorff':
                                        d_h = max(directed_hausdorff(pos3D_ci, pos3D_cj)[0],
                                                  directed_hausdorff(pos3D_cj, pos3D_ci)[0])
                                # print('Processing tracklets {} and {} : cost {}, time {} sec'.format(i, j, d_h, time.time()-st_time))
                                if d_h <= max_dist:
                                    cost[i, j] = d_h
                                    # cost[j, i] = d_h

                # DEBUG
                # import matplotlib.pyplot as plt1
                # hist = np.histogram(cost[cost < 10000], bins=50, range=(0, 2))
                # plt1.hist(cost[cost < 10000],bins=50,range=(0,max_dist+2))
                # plt1.savefig('/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/multicamera_wildtrack/wildtrack/Wildtrack_dataset/cost_hist.png',dpi=300)
                # plt1.close()
                # pdb.set_trace()
                row_ind, col_ind = linear_sum_assignment_with_inf(cost)
                total_matches = 0
                for r, c in zip(row_ind, col_ind):
                    if cost[r, c] <= max_dist:
                        id_r = tracks_c[r][0, 1]
                        id_c = tracks_c[c][0, 1]
                        total_matches += 1
                        batch_matches[id_r] = id_c
                        batch_matches[id_c] = id_r
                print('total matches found in {}-{}: {}'.format(cp, ca, total_matches))
                #update global graph using new global ids in a batch
                self.merge_edges(batch_matches, cp=cp, ca=ca)

    def get_projected_tracks(self, params):
        trackers = self.mc_track_segments[params['cam']]
        if len(trackers.keys())!=0:
            #[frame, float(i), x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, float(self.cam)]
            tracklets, totalTrack, ids = form_tracklets(trackers, cam=params['cam'], tracker_min_size=self.min_trklt_size,
                                                       t_thr=30, d_thr=50, motion=self.motion_model,
                                                       single_cam_association=self.isSCA).get_trklts()
        params['2D_tracks'] = copy.deepcopy(tracklets)
        params['batch_ids'] = ids
        trackletsProj, active_trklt = self.project_tracklets(tracklets, params, cam=params['cam'])
        print('total track {} in cam {}'.format(len(trackletsProj), params['cam']))
        params['3D_tracks'] = trackletsProj
        return params

    def merge_edges(self, mt, cp=None, ca=None):
        # G_mc: empty graph which is updated iteratively for each camera pair association
        # k: global track id from primary camera
        # l: global track id from auxiliary camera
        # mt: association between
        for k, l in mt.items():
            # initialize empty association list when tracklet first appear in camera system
            # both ids are new
            if k not in self.G_mc and l not in self.G_mc:
                self.G_mc[k] = []
                self.G_mc[l] = []

                self.G_mc[k].append(l)
                self.G_mc[l].append(k)
                continue
            # l is new
            if k in self.G_mc and l not in self.G_mc:  # check the previous association of k present in current batch ids
                # check k has the previously associated l in current batch ids
                # TODO: check batch_ids only in current batch: current_batch_ids
                # TODO: before updating association: verify duplicity
                if len(set(self.G_mc[k]).intersection(self.MCparams[ca]['batch_ids']+self.MCparams[cp]['batch_ids'])) == 0:  # and k not in cam_batch_ids[ca]:
                    self.G_mc[l] = []
                    # k should not already in [ca current batch ids]
                    self.G_mc[l].append(k)
                    self.G_mc[k].append(l)
                    continue
            # k: primary is new
            if l in self.G_mc and k not in self.G_mc:  # check the previous association of l present in current batch ids
                # check l has the previously associated k in current batch ids
                # TODO: before updating association: verify duplicity
                if len(set(self.G_mc[l]).intersection(self.MCparams[cp]['batch_ids']+self.MCparams[ca]['batch_ids'])) == 0:  # and l not in cam_batch_ids[cp]:
                    self.G_mc[k] = []
                    # l should not already in [cp current batch ids]
                    self.G_mc[k].append(l)
                    self.G_mc[l].append(k)
                    continue
            # both ids are already in Gmc:

            # if k in G_mc and l in G_mc:
            # if len(set(G_mc[k]).intersection(batch_ids))==0 and len(set(G_mc[l]).intersection(batch_ids))==0:
            # if l not in cam_batch_ids[cp]:
            # G_mc[k].append(l)
            # if k not in cam_batch_ids[ca]:
            # G_mc[l].append(k)

    def get_tracks_dict(self):
        # G_mc: global association graph based on global track id
        # mc_all_trklts_dict: dictionary of all mc global tracks use for reporting and visualizing the results
        # prepare traklets info for mc system

        self.combined_tracks2D, _ = self.expand_from_temporal_list(box_all=self.combined_tracks2D, mask_all=None)
        for tr in self.combined_tracks2D:
                #only get tracks when visible in current batch
                if set(list(tr[:, 0])).intersection(range(self.batch_start,
                                                          self.batch_start + self.batch_size)):
                    self.mc_all_trklts_dict[tr[0,1]] = tr
                    self.cam_trklts_id.append(tr[0,1])

    def update_global_batch_labels(self):
        self.get_tracks_dict()
        # apply DFS on mc system graph
        label_map = {keys: [] for keys in self.mc_all_trklts_dict.keys()}
        # each track identity is unique
        for label_tp, tklt in self.mc_all_trklts_dict.items():
            if self.isSCT:
                label_map[label_tp]=label_tp
            else:
                if label_tp in self.G_mc.keys():
                    solnPath = self._dfs(self.G_mc, label_tp, [])
                    #min id (used) or min time

                    label_map[label_tp] = min(solnPath)
                else:
                    label_map[label_tp] = label_tp
        return label_map

    def dafault_params(self, cam=None):
        # cam_sync_start = {1:10, 2:9, 3:16, 4:18, 5:7, 6:7, 7:7}
        param = {}
        param['cam'] = cam
        # 3D projection parameters
        Cparam = param['C{}'.format(param['cam'])] = {}
        Cparam['A'], \
        Cparam['rvec'], \
        Cparam['tvec'], \
        Cparam['dist_coeff'] = self.projection_class.cam_params_getter(cam=param['cam'],
                                                                    isDistorted=self.dist_correct)
        param['d_th'] = 100

        param['ImgsPath'] = os.path.join(self.img_path, 'C{}'.format(cam))
        param['outPath'] = os.path.join(self.out_dir, 'global_tracks')
        if not os.path.exists(param['outPath']):
            os.makedirs(param['outPath'])
        return param

    def get_global_matches(self):
        for c in self.cam_list:
            #get camera params
            self.MCparams[c] = self.dafault_params(cam=c)
            #get 3D projection of all tracks in a batch
            self.MCparams[c] = self.get_projected_tracks(self.MCparams[c])

            # accumulate camera tracklets for joint affinity
            assert len(self.MCparams[c]['3D_tracks'])==len(self.MCparams[c]['2D_tracks'])
            self.combined_tracks3D.append(self.MCparams[c]['3D_tracks'])
            self.combined_tracks2D.append(self.MCparams[c]['2D_tracks'])
            # keep track the tracklet index in affinity concatenation
            self.MCparams[c]['track_idx'] = self.cam_track_idx[c - 1] + np.array(range(len(self.combined_tracks3D[c - 1])))
            self.cam_track_idx[c] = self.cam_track_idx[c - 1] + len(self.combined_tracks3D[c - 1])

        self.combined_tracks3D, _ = self.expand_from_temporal_list(box_all=self.combined_tracks3D, mask_all=None)
        if self.debug:
            for i in self.cam_list:
                print('track_idx {}'.format(self.MCparams[i]['track_idx']))
                print('cam {}'.format(self.combined_tracks3D[self.MCparams[i]['track_idx']][0][-1][-1]))
        # TODO: optimum threshold
        # hausdorf: 75, frechet: 150
        # apply assignment optimization on multicameratracks affinity and get global correspondence graph
        self.get_hungarian_matches_combined(max_dist=100, dist_metric=self.dist_metric)
        label_map = self.update_global_batch_labels()

        if self.eval_plane == '2D':
            self.get_sequential_results_2D(label_map)
        else:
            self.get_sequential_results_3D(label_map)
        return self.global_track_results


