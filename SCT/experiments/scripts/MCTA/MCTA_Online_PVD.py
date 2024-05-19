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
from MCTA.frechet_dist import frechetdist_new, frechetdist
import time
import collections
import random
random.seed(1234)
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
        self.mc_track_segments=mc_track_segments
        self.global_track_start_time = global_track_start_time
        self.BFS_cam_graph=all_params['cam_graph']
        self.min_trklt_size = all_params['min_trklt_size']
        self.isSCA = False
        self.vis_pair = True
        self.new_ids = all_params['new_ids']
        self.motion_model = False
        self.appearance_model = False
        self.dist_correct = False
        self.print_stats = all_params['print_mcta_stats']
        self.batch_start=all_params['batch_start']
        self.batch_size=all_params['batch_size']
        self.timeoffset = timeoffset

        self.global_track_saver = all_params['output_dir'] + '/log_batch_mu_current.csv'
        self.global_track_results = {'frame': [], 'cam':[], 'timeoffset': [],
                                      'x1':[], 'y1':[], 'x2':[],'y2':[],
                                        'id': [], 'firstused':[], 'tu': []}


        self.keep_tu_id = all_params['keep_tu_id']
        self.global_out_path = all_params['output_dir']
        self.img_path = all_params['img_path']
        self.img_HW = all_params['img_HW']
        self.global_track_id = all_params['global_track_id']

        self.keep_raw_id = all_params['keep_raw_id'] #TODO: keep camera info as well
        self.isSCT = all_params['isSCT']
        self.multi_match = all_params['multi_match']
        self.vis = all_params['vis']
        self.vis_rate = all_params['vis_rate']
        self.meta_data = all_params['meta_data']
        self.clasp_label_map = all_params['mapCLASPlabel']
        self.use_metadata = all_params['use_metadata']
        self.save_global_tracks = all_params['save_global_tracks']
        self.server_loc = all_params['server_loc']
        self.only_lane3_ids = all_params['only_lane3_ids']
        self.keep_id_pvd = all_params['keep_id_pvd']
        self.lane3_ids = all_params['lane3_ids']

        #update locally
        self. MCparams = {}
        self.MC_tracklets = {}
        self.cam_visited = []
        self.index_factor = {}
        self.cam_trklts_size = {}
        self.cam_trklts_id = {}
        self.mc_all_trklts_dict = {}
        self.mc_all_trklts = []
        self.mc_batch_ids = [] #used to verify already associated tracks should associate again or not
        #update globally
        self.G_mc = all_params['global_mcta_graph']
        #self.single_cam_multi_asso = all_params['single_cam_multi_asso']
        self.store_prev_dist = all_params['prev_affinity']
        self.keep_batch_hist = all_params['keep_batch_hist']
        self.integration = all_params['integration?']




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


    def vis_box(self, img, bb, label, label_tu=None):
        cv2.rectangle(img, (int(bb[2]), int(bb[3])),
                      (int(bb[2] + bb[4]), int(bb[3] + bb[5])),
                      (255, 0, 0), 4, cv2.LINE_AA)

        if label_tu:
            cv2.putText(img, '{}'.format(label),
                    (int(bb[2] + bb[4] / 2), int(bb[3] + bb[5] / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255),3, cv2.LINE_AA)
            cv2.putText(img, '{}'.format(label_tu),
                    (int(bb[2] + bb[4] / 2), int(bb[3] + bb[5] / 2)+80),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255),2, cv2.LINE_AA)
        else:
            cv2.putText(img, '{}'.format(label),
                    (int(bb[2] + bb[4] / 2), int(bb[3] + bb[5] / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255), 3, cv2.LINE_AA)
        return img


    def vis_projection(self, fr, img, cam2cam):
        # get projection from auxiliary
        indx = self.MCparams[cam2cam]['trkltFamilySizeP']
        for tl, bb in enumerate(self.MCparams[cam2cam]['PAproj'][indx:]):
            # TODO: Set the colors of the rectangles
            if 'activeTrkltIndexsA' in self.MCparams[cam2cam]:
                if indx+tl in self.MCparams[cam2cam]['activeTrkltIndexsA']:
                    for i in range(1, len(bb)):
                        if self.batch_start <= bb[i, 0] <= fr:
                            if bb[i, 0] == fr:
                                #bb[i, 3] = bb[i, 3] + bb[i, 5]
                                # undistorted projected centers
                                img = cv2.circle(img, (int(bb[i, 2]), int(bb[i, 3])), 10, (255, 0, 255), -10)

        # get undistorted primary centers
        for bb in self.MCparams[cam2cam]['PAproj'][:indx]:
            # TODO: Set the colors of the rectangles
            for i in range(1, len(bb)):
                if self.batch_start <= bb[i, 0] <= fr:
                    if bb[i, 0] == fr:
                        # undistorted primary centers
                        # Pid box: [fr, id, cx,cy,w,h]
                        cxy_undist = np.copy(bb[i, 2:4])  # + bb[i, 4:6] / 2.0
                        if self.dist_correct:
                            dist_coeff, A = camera_intrinsics(cam=self.MCparams[cam2cam]['Pid'],
                                                          img_HW=self.img_HW)
                            #bb[i, 3] = bb[i, 3]+bb[i, 5]/2.
                            cxy_undist = self._undistorted_coords(cxy_undist.reshape(1, 1, 2), dist_coeff, A)
                        else:
                            cxy_undist = cxy_undist.reshape(1, 2)
                        img = cv2.circle(img,  (int(cxy_undist[0, 0]), int(cxy_undist[0, 1])), 10, (0, 255, 255), -10)
        return img

    def save_batch_results(self, fr,img,label_map,save_imgs, cam=2):
        for c in self.cam_trklts_id[cam]:
            bb = self.mc_all_trklts_dict[c]
            # To show the association with the projections (tracklets2)
            for i in range(len(bb)):
                first_used = 0
                if self.batch_start <= bb[i, 0] <= fr:
                    if bb[i, 0] == fr:
                        # global_track_counter: labelCam2[c]- tracklet index mapped to raw id
                        # TODO: create buffer for each camera to store global track id
                        # TODO: each camera should not allow repeated identity in a batch results
                        if label_map[c] not in self.keep_raw_id:
                            first_used = 1
                            self.global_track_id += 1
                            self.keep_raw_id[label_map[c]] = 'P{}'.format(self.global_track_id)
                            self.keep_tu_id[label_map[c]] = 0
                        if self.clasp_label_map:
                            # map global track id to CLASP id
                            if bb[i, 0] in self.meta_data['Frame'] and bb[i,-1] in [300]:#self.meta_data['cam'] and bb[i,-1] not in [360, 330]:
                                matched_label, matched_tu = match_id_pvd(self.meta_data, fr,
                                                         [bb[i,2],bb[i,3],bb[i,2]+bb[i,4],bb[i,3]+bb[i,5]],
                                                         cam=bb[i,-1])
                                if matched_label is not None:
                                    self.keep_raw_id[label_map[c]] = matched_label
                                    if  matched_tu is not None:
                                        self.keep_tu_id[label_map[c]] = matched_tu
                        # update global tracks
                        #print('sequential_id {}'.format(self.global_track_id))
                        #global_tracker(fr, id=self.keep_raw_id[label_map[c]],
                                       #global_tracker=self.global_track_saver,
                                       #first_used=first_used).update_state(bb[i])
                        #if (bb[i,-1] in [300] and 1<=fr<=4079) or (bb[i,-1] in [330] and 1883<=fr<=5726) or (bb[i,-1] in [340] and 961<=fr<=4535)\
                            #or (bb[i,-1] in [360] and 1721<=fr<=5702) or (bb[i,-1] in [440] and 1<=fr<=4179):
                        self.global_track_results['frame'].append(int(fr))
                        self.global_track_results['cam'].append(int(bb[i,-1]))
                        self.global_track_results['timeoffset'].append(float('{:.2f}'.format(fr /10.0)))
                        self.global_track_results['x1'].append(int(bb[i,2]))
                        self.global_track_results['y1'].append(int(bb[i,3]))
                        self.global_track_results['x2'].append(int(bb[i,2]+bb[i,4]))
                        self.global_track_results['y2'].append(int(bb[i,3]+bb[i,5]))
                        self.global_track_results['id'].append(self.keep_raw_id[label_map[c]])
                        self.global_track_results['firstused'].append(first_used)
                        self.global_track_results['tu'].append(self.keep_tu_id[label_map[c]])

                        # self.keep_raw_id[label_map[c]] in self.meta_data['ID'] and
                        if save_imgs and self.keep_raw_id[label_map[c]] in self.meta_data['Lane3']:# and self.keep_tu_id[label_map[c]] in ['TU1']:
                            #if self.keep_raw_id[label_map[c]] in ['P9', 'P11', 'P15']:
                            img = self.vis_box(img, bb[i], self.keep_raw_id[label_map[c]], self.keep_tu_id[label_map[c]])

        return img

    def get_global_tracks_MC(self,label_map,
                             vis_rate=30,
                             save_imgs=0,
                             frame_rate=10
                             ):
        # BGR
        # magenta = (255,0,255)
        # yellow = (0,255,255)
        # green = (0,255,0)
        # blue = (255,0,0)
        if save_imgs:
            cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            cv2.resizeWindow("image", 640*2, 480)
        for fr in range(self.batch_start, self.batch_start+self.batch_size):
            #print('batch start {}, batch end {}'.format(self.batch_start, self.batch_start + self.batch_size - 1))
            if (fr) % vis_rate == 0:
                #try: #to avoid crash due to missing frames
                timeoffset = '{:.2f}'.format(fr/10.0)
                if save_imgs:
                    if self.server_loc == 'PVD':
                        #self.img_path : train_gt folder
                        imgC300 = cv2.imread('{}/{}_{:04d}.jpg'.format(self.img_path, 300, fr))
                        imgC340 = cv2.imread('{}/{}_{:04d}.jpg'.format(self.img_path, 340, fr))
                        imgC440 = cv2.imread('{}/{}_{:04d}.jpg'.format(self.img_path, 440, fr))
                        imgC361 = cv2.imread('{}/{}_{:04d}.jpg'.format(self.img_path, 361, fr))
                        imgC360 = cv2.imread('{}/{}_{:04d}.jpg'.format(self.img_path, 360, fr))
                        imgC330 = cv2.imread('{}/{}_{:04d}.jpg'.format(self.img_path, 330, fr))
                else:
                    imgC340=imgC330=imgC300 = img340= imgC440 = imgC361 = imgC360 = imgC2 = imgC5 = imgC9 = imgC11 = imgC1=imgC3 =imgC4=imgC10=None

                if self.MCparams['C300C440']['trkltFamilySizeP'] > 0:
                    imgC440 = self.save_batch_results(fr, imgC440, label_map, save_imgs, cam=440)

                if self.MCparams['C300C340']['trkltFamilySizeP'] > 0:
                    imgC340 = self.save_batch_results(fr, imgC340, label_map, save_imgs, cam=340)

                if self.MCparams['C300C340']['trkltFamilySizeA'] > 0:
                    imgC300 = self.save_batch_results(fr, imgC300, label_map, save_imgs, cam=300)

                if self.MCparams['C361C440']['trkltFamilySizeA'] > 0:
                    imgC361 = self.save_batch_results(fr, imgC361, label_map, save_imgs, cam=361)

                if self.MCparams['C360C361']['trkltFamilySizeA'] > 0:
                    imgC360 = self.save_batch_results(fr, imgC360, label_map, save_imgs, cam=360)

                if self.MCparams['C330C360']['trkltFamilySizeA'] > 0:
                    imgC330 = self.save_batch_results(fr, imgC330, label_map, save_imgs, cam=330)


                if save_imgs:
                    #blankImg = np.zeros(shape=[int(self.img_HW[0]), int(self.img_HW[1]), 3], dtype=np.uint8)

                    C300440= cv2.vconcat([imgC300, imgC440])
                    C340361= cv2.vconcat([imgC340, imgC361])# img: c4, img2: c2
                    C360330 = cv2.vconcat([imgC360, imgC330])
                    final_img = cv2.hconcat([C360330, C340361, C300440])
                    #print(final_img.shape)
                    final_img = cv2.resize(final_img, (int(1.5 * self.img_HW[1]), int(1*self.img_HW[0])), interpolation=cv2.INTER_AREA)#1.5

                    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(self.global_out_path + '/{:06d}.png'.format(int(fr)), final_img)
                    cv2.imshow("image", final_img)
                    cv2.waitKey(5)
                    #dump batch results to csv file
        Dframe = pd.DataFrame(self.global_track_results)
        Dframe.to_csv(self.global_track_saver,mode='w')
        #if save_imgs:
            #cv2.destroyAllWindows()

    def _undistorted_coords(self, trklt, dist_coeff, A):
        # use a copy of traclet centroids to convert into undistorted format
        # ***A: Camera Intrinsic Matrix
        # https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html
        # https: // www.mathworks.com / help / vision / ref / estimatecameraparameters.html
        # new camMatrix
        im_shape = (int(self.img_HW[1]), int(self.img_HW[0]))
        newcameramtx, roi = cv2.getOptimalNewCameraMatrix(A, dist_coeff, im_shape, 1, im_shape)
        trklt = cv2.undistortPoints(trklt.reshape(trklt.shape[0], 1, 2), A, dist_coeff, 0, newcameramtx)
        return trklt.reshape(trklt.shape[0], 2)

    def _applyTransform(self, source_corners, H):
        dest_corners = np.empty(2)
        w = H[2][0] * source_corners[0] + H[2][1] * source_corners[1] + H[2][2] * 1
        dest_corners[0] = (H[0][0] * source_corners[0] + H[0][1] * source_corners[1] + H[0][2] * 1) / w
        dest_corners[1] = (H[1][0] * source_corners[0] + H[1][1] * source_corners[1] + H[1][2] * 1) / w
        return dest_corners

    # ---------------ok siddique
    # TODO: apply undistorted coords on H or undistort coords after applying H: here H is based distorted image
    # try undistorted image to compute homography
    def project_tracklets(self, in_tracklets, params, ca=9, cp=2):
        out_tracklets = []
        active_trklt = []
        H = params['H']
        if H is not None: # H is None for C5C3
            Hv = copy.deepcopy(H)
            Hv[0, 2], Hv[1, 2] = 0, 0
        # [fr,cx,cy,x,y,vx,vy]<<< projected coordinates: center is used
        for i, trklt in enumerate(in_tracklets):
            if H is not None:
                if self.dist_correct:
                    dist_coeff, A = camera_intrinsics(ca, self.img_HW)
                    #generate undistorted centroids
                    xy = copy.deepcopy(trklt[:, 2:4])
                    xy = self._undistorted_coords(xy, dist_coeff, A)
                    trklt[:, 2:4] = xy
                    if self.motion_model:
                        #generate undistorted motion
                        vxvy = copy.deepcopy(trklt[:, 5:7])
                        vxvy = self._undistorted_coords(vxvy, dist_coeff, A)
                        trklt[:, 5:7] =vxvy
                for bb in trklt:
                    bbt1 = self._applyTransform(bb[2:4], H)
                    if ca in [330]:
                        cxcy = self._applyTransform(bb[2:4] + bb[4:6] / [2.0, 2.0], H)
                    else:
                        cxcy = self._applyTransform(bb[2:4] + bb[4:6] / [2.0, 1.0], H)
                    if self.motion_model:
                        vxvy = self._applyTransform(bb[5:7], Hv)
                        bbt2 = cxcy - bbt1  # projected w/2,h/2
                        bb[1:7] = np.concatenate([cxcy, bbt2, vxvy])  # projected: [cx,cy,w/2,h/2,vx,vy]
                    else:
                        bbt2 = cxcy - bbt1  # projected w/2,h/2
                        bb[2:6] = np.concatenate([cxcy, bbt2])  # projected: [cx,cy,w/2,h/2]
                # Delete tracklets that don't have any detection visible in the second camera
                # TODO: There must be a smarter and faster way to do that
                # cy-h/2 > 0 and w/2>0
                if ca in [300] and cp in [440]:
                    #cy>100 and cx>400
                    trklt = trklt[trklt[:,2]>640]
                    trklt = trklt[trklt[:, 3] > 250]
                    if len(trklt)>0:
                        out_tracklets.append(trklt)
                        active_trklt.append(i)

                elif ca in [300] and cp in [340]: #to 340, 440
                    trklt = trklt[trklt[:,2]>270]
                    trklt = trklt[trklt[:,2]<1000]

                    #trklt = trklt[trklt[:, 3] > 250]
                    if len(trklt)>0:
                        out_tracklets.append(trklt)
                        active_trklt.append(i)

                elif ca in [361] and cp in [440]:
                    #3:ymax
                    trklt = trklt[trklt[:, 3] > 350]

                    if len(trklt)>0:
                        out_tracklets.append(trklt)
                        active_trklt.append(i)

                elif ca in [340] and cp in [440]:
                    trklt = trklt[trklt[:,2]>640]
                    #trklt = trklt[trklt[:, 3] > 250]
                    if len(trklt)>0:
                        out_tracklets.append(trklt)
                        active_trklt.append(i)
                else:
                    if max(trklt[:, 3] - trklt[:, 5]) > 0 and max(trklt[:, 4]) > 0:
                        out_tracklets.append(trklt)
                        active_trklt.append(i)
            else:
                #tracklets should be sorted
                #keep start ids in 360
                #Already do this step in form_tracklet
                if ca in [360] and cp in [361]:
                    if (trklt[0,3]+trklt[0,5]/2) < 280 and 540<(trklt[0,2]+trklt[0,4]/2) < 720:
                        print('Person {} found at {}'.format(trklt[0,1],
                                                             self.global_track_start_time[trklt[0,1]]))
                        out_tracklets.append(trklt)
                        active_trklt.append(i)

                if ca in [360] and cp in [340]:
                    if (trklt[0,3]+trklt[0,5]/2) < 200 and 870<(trklt[0,2]+trklt[0,4]/2):
                        print('Person {} found at {}'.format(trklt[0,1],
                                                             self.global_track_start_time[trklt[0,1]]))
                        out_tracklets.append(trklt)
                        active_trklt.append(i)
        return out_tracklets, active_trklt, params

    def associate_tracklets_DFS_MC(self, params, cam2cam, max_dist=None):
        # ***i : master camera tracklets index: time should be synced to this master camera
        # ***j : projected/auxiliary camera tracklet index
        # ***np.array(active_trklt): acyive_trklt_indexes from slave cameras
        # ***len(cam1Tracklets): total tracklets in master camera (C2)
        # ------------------------------------------------------------------------------------
        if self.print_stats:
            print('Total tracklets in a camera pairs {}'.format(len(params['PAproj'])))
        look_back_window = np.arange(self.batch_start-10*self.batch_size, self.batch_start+self.batch_size)
        cost = np.ones((len(params['PAproj']), len(params['PAproj']))) * np.inf
        tilde_t_pa = np.ones((len(params['PAproj']), len(params['PAproj'])), dtype='float') * np.inf
        i = 0
        matches = []
        d_h = None
        for tracklet1 in params['PAproj']:
            j = 0
            for tracklet2 in params['PAproj']:
                # condition to check primary-i and auxiliary-j tracklets
                if i != j and i < params['trkltFamilySizeP'] and j >= params['trkltFamilySizeP'] \
                        and j in params['activeTrkltIndexsA']:
                    # get actual id
                    id_i = params['PAproj'][i][0, 1]
                    id_j = params['PAproj'][j][0, 1]

                    # TODO: Assess whether the timestamp should be considered in the distance computation as well
                     # TODO: Is it possible to include tracklets starting time to refine the tracklets sets?
                    #to make cost computation faster:  and (id_i not in self.G_mc or id_j not in self.G_mc) \
                    common_t_stamps = np.array(list(set(list(tracklet2[:, 0])).intersection(list(tracklet1[:, 0]))))
                    #if id_i in [10, 21] or id_j in [10, 21]:
                        #print(common_t_stamps)

                    if len(common_t_stamps)>=self.min_trklt_size//2:
                        #For PVD: 400 keep unuse the track start time constraint
                        #\and (self.global_track_start_time[id_i] in look_back_window
                             #or self.global_track_start_time[id_j] in look_back_window)

                        assert len(common_t_stamps) >= self.min_trklt_size//2, 'Non-overlaped T_p, T_a can not be associated'

                        #TODO: If id_i and id_j already associated, do not compute cost and association again...
                        # TODO: use G_mc to find already associated pair and then continue: keep infinite cost
                        if self.motion_model:
                            featIndx = [0, 2, 3, 6, 7]
                        else:
                            featIndx = [0, 2, 3, 4, 5] # 4,5- width and height are currently not used

                        mask1 = np.isin(tracklet1[:, 0], common_t_stamps)
                        final_tracklet1 = tracklet1[mask1][:, featIndx]
                        #if cam2cam == 'C5C2' and max(final_tracklet1[:, 2])>360:
                            #continue
                        if self.dist_correct:
                            dist_coeff, A = camera_intrinsics(cam=params['Pid'],
                                                              img_HW=self.img_HW)  # 2:c5 or 9:c11
                            # Pid box: [fr, id, cx,cy,w,h]
                            xy = copy.deepcopy(final_tracklet1[:, 1:3])
                            xy = self._undistorted_coords(xy, dist_coeff, A)
                            final_tracklet1[:, 1:3] = xy

                            #get undistorted motion
                            if self.motion_model:
                                vxvy = copy.deepcopy(final_tracklet1[:, 3:5])
                                vxvy = self._undistorted_coords(vxvy, dist_coeff, A)
                                final_tracklet1[:, 3:5] = vxvy
                        # compute the location from velocity and time difference
                        #final_tracklet1[:, 2] = final_tracklet1[:, 2]+final_tracklet1[:, 4]/2.#cy+h/2

                        final_tracklet1[:, 1:3] = final_tracklet1[:, 1:3] / [float(self.img_HW[1]), float(self.img_HW[0])]

                        mask2 = np.isin(tracklet2[:, 0], common_t_stamps)
                        final_tracklet2 = tracklet2[mask2][:, featIndx]

                        #final_tracklet2[:, 2] = final_tracklet2[:, 2] + final_tracklet1[:, 4]  # cy+h/2

                        final_tracklet2[:, 1:3] = final_tracklet2[:, 1:3] / [float(self.img_HW[1]), float(self.img_HW[0])]

                        if  not self.motion_model:
                            st_time = time.time()
                            if len(final_tracklet1[:, 1:3]) > 3*self.batch_size:
                                #sys.setrecursionlimit(len(final_tracklet1[:, 1:3]) * len(final_tracklet2[:, 1:3]))
                                #d_h_center = frechetdist.frdist(final_tracklet1[:2*self.batch_size, 1:3],final_tracklet2[:2*self.batch_size, 1:3])
                                d_h = frechetdist_new.frdist(final_tracklet1[:2 * self.batch_size, 1:3],
                                                                final_tracklet2[:2 * self.batch_size, 1:3],
                                                                compute_type='iterative')
                                #d_h_center = directed_hausdorff(final_tracklet1[:, 1:3], final_tracklet2[:, 1:3])[0]
                            else:
                                #sys.setrecursionlimit(len(final_tracklet1[:, 1:3]) * len(final_tracklet2[:, 1:3]))

                                #d_h_center = frechetdist.frdist(final_tracklet1[:, 1:3],final_tracklet2[:, 1:3])
                                d_h = frechetdist_new.frdist(final_tracklet1[:, 1:3],
                                                                final_tracklet2[:, 1:3],
                                                                compute_type='iterative')
                                #d_h_center = directed_hausdorff(final_tracklet1[:, 1:3], final_tracklet2[:, 1:3])[0]
                            #print('cost between {} and {}: {}'.format(id_i, id_j, d_h))
                            if self.print_stats:
                                print('Frechet distance computation time {} sec'.format(time.time()-st_time))

                            if self.print_stats:
                                print('Processing global tracklets {} and {} without motion for cost {}'.format(tracklet1[:,1][0], tracklet2[:,1][0], d_h))

                        if d_h is not None:
                            if d_h <= max_dist:
                                cost[i, j] = d_h
                                tilde_t_pa[i, j] = min(common_t_stamps)
                j = j + 1
            i = i + 1

        if self.multi_match:
            loop_id = 0
            total_matches = 0
            while len(cost[cost <= max_dist]) > 0:
                row_ind, col_ind = linear_sum_assignment_with_inf(cost)
                for r, c in zip(row_ind, col_ind):
                    if cost[r, c] <= max_dist:
                        total_matches += 1
                        id_r = params['PAproj'][r][0, 1]
                        id_c = params['PAproj'][c][0, 1]
                        # return raw id instead of matched tracklet index
                        matches.append((cost[r, c], tilde_t_pa[r, c], id_r, id_c))
                        # keep dist for next frame
                        self.store_prev_dist[id_r] = {id_c: cost[r, c]}
                    cost[r, c] = np.inf
                loop_id += 1
                print('multi-match literation {}'.format(loop_id))

        else:
            row_ind, col_ind = linear_sum_assignment_with_inf(cost)
            #Mincosts = cost[row_ind, col_ind]
            #TODO: verify that each row find column with min distance
            #issue: row find column for second minima
            for r, c in zip(row_ind, col_ind):
                if cost[r,c] <= max_dist:
                    id_r = params['PAproj'][r][0, 1]
                    id_c = params['PAproj'][c][0, 1]
                    # return raw id instead of matched tracklet index
                    matches.append((cost[r,c], tilde_t_pa[r, c], id_r, id_c))
                    #keep dist for next frame
                    self.store_prev_dist[id_r] = {id_c:cost[r,c]}
        return matches, tilde_t_pa

    def associate_queue_MC(self, params, max_dist=40):
        tilde_t_pa = -1
        matches = []
        if self.keep_id_pvd[params['Aid']] and self.keep_id_pvd[params['Pid']]:
            cost = np.ones((len(self.keep_id_pvd[params['Aid']].keys()), len(self.keep_id_pvd[params['Pid']].keys()))) * np.inf
            i = 0
            map_idi = {}
            map_idj = {}
            for id_i, tracklet1 in self.keep_id_pvd[params['Aid']].items():
                j = 0
                for id_j, tracklet2 in self.keep_id_pvd[params['Pid']].items():
                    map_idi[i] = id_i
                    map_idj[j] = id_j
                    # condition to check temporal constraints: 0<t_i - t_j = thr_bs<30
                    if -10<tracklet1['start'] - tracklet2['end']<max_dist:
                        d_h = tracklet1['start'] - tracklet2['end']
                        #print('cost between {} and {}: {}'.format(id_i, id_j, d_h))
                        if d_h < max_dist:
                            cost[i, j] = d_h
                    j = j + 1
                i = i + 1

            if self.multi_match:
                loop_id = 0
                total_matches = 0
                while len(cost[cost < max_dist]) > 0:
                    row_ind, col_ind = linear_sum_assignment_with_inf(cost)
                    for r, c in zip(row_ind, col_ind):
                        if cost[r, c] < max_dist:
                            total_matches += 1
                            id_r = params['PAproj'][r][0, 1]
                            id_c = params['PAproj'][c][0, 1]
                            # return raw id instead of matched tracklet index
                            matches.append((cost[r, c], tilde_t_pa[r, c], id_r, id_c))
                            # keep dist for next frame
                            self.store_prev_dist[id_r] = {id_c: cost[r, c]}
                        cost[r, c] = np.inf
                    loop_id += 1
                    print('multi-match literation {}'.format(loop_id))

            else:
                row_ind, col_ind = linear_sum_assignment_with_inf(cost)
                # Mincosts = cost[row_ind, col_ind]
                # TODO: verify that each row find column with min distance
                # issue: row find column for second minima
                for r, c in zip(row_ind, col_ind):
                    if cost[r, c] < max_dist:
                        # return raw id instead of matched tracklet index
                        print('found AIT association: {} and {}'.format(map_idi[r], map_idj[c]))
                        matches.append((cost[r, c], -1, map_idi[r], map_idj[c]))
                        del self.keep_id_pvd[params['Aid']][map_idi[r]] #360
                        # keep auxiliary until track exit the AIT area

                        del self.keep_id_pvd[params['Pid']][map_idj[c]]

        return matches, tilde_t_pa

    def get_cam2cam_matches(self, params, cam2cam):
        #init matches to return empty match
        mt = []
        trackersP = self.mc_track_segments[params['Pid']]
        trackersA = self.mc_track_segments[params['Aid']]
        # collect primary camera: P:C2 features
        #note: keys might be available when created once but values may be none: initiate batch dictionary for every batch
        #TODO: make sure that all keys or track has its values (availability in look back frames)
        if len(trackersP.keys())!=0:
            tracklets, totalTrack, ids,\
            self.keep_id_pvd, self.lane3_ids  = form_tracklets(trackersP, cam=params['Pid'],
                                                        tracker_min_size=self.min_trklt_size,
                                                        t_thr=30, d_thr=50, motion=self.motion_model,
                                                        pvd_id_ait=self.keep_id_pvd, cam2cam=cam2cam,
                                                        global_track_start=self.global_track_start_time,
                                                        single_cam_association=self.isSCA, lane3_ids=self.lane3_ids).get_trklts()
            # list C2 tracklets
            params['pri_batch_ids'] = ids
            if params['Pid'] in [360] and params['Aid'] in [330]:
                trackletsP = convert_centroids(tracklets)
            else:
                trackletsP = convert_bottom_center(tracklets)
        else:
            trackletsP = []
            params['pri_batch_ids'] = []
        # A:C9:------------------------------------------------------------------
        # Collect auxiliary camera features
        if len(trackersA.keys())!=0:
            tracklets, totalTrack, ids,\
            self.keep_id_pvd, self.lane3_ids = form_tracklets(trackersA, cam=params['Aid'],
                                                        tracker_min_size=self.min_trklt_size,
                                                        t_thr=30, d_thr=50, motion=self.motion_model,
                                                        pvd_id_ait=self.keep_id_pvd, cam2cam=cam2cam,
                                                        global_track_start=self.global_track_start_time,
                                                        single_cam_association=self.isSCA, lane3_ids=self.lane3_ids).get_trklts()
            # keep both original and projected tracklets for auxiliary
            params['aux_batch_ids'] = ids
            tracklet_2 = copy.deepcopy(tracklets)
            # Project: center>(cx,cy), top-left>(x,y), velocity>(vx,vy) [fr,cx,cy,x,y,vx,vy]
            # active_trklt: indexes of tracklets whose projected onto the destination (primary) image boundary
            trackletsA, active_trklt, params = self.project_tracklets(tracklets, params, ca=params['Aid'], cp=params['Pid'])
        else:
            tracklet_2 = []
            trackletsA = []
            active_trklt=[]
            params['aux_batch_ids'] = []
        if len(trackletsP)!=0 or len(tracklet_2)!=0:
            # form tracklets family set: [tracklets['C2'].org, tracklets['C9'].projected]
            params['PAproj'] = np.array(trackletsP+tracklet_2)
            assert len(params['PAproj'])==len(trackletsP)+len(tracklet_2)
            params['trkltFamilySizeP'] = len(trackletsP)
            params['trkltFamilySizeA'] = len(tracklet_2)
            # TODO: need to verify for 9to2, 5to11
            # use raw tracklets [fr,id,x,y,w,h] for visualization, metrics evaluation, reporting results
            if params['Pid'] in [360] and params['Aid'] in [330]:
                params['PAorg'] = np.array(centroids2xywh(copy.deepcopy(trackletsP))+ tracklet_2)
            else:
                params['PAorg'] = np.array(botcenter2xywh(copy.deepcopy(trackletsP)) + tracklet_2)

            # Apply MCTA when both camera has tracks and aux projection in pri image boundary
            #print(' number of eligible tracklet for association {}'.format(len(active_trklt)))
            if len(trackletsP)!=0 and len(trackletsA)!=0 \
                    and len(active_trklt) > 0 and cam2cam not in ['C360C361', 'C360C340']:
                # since all the projected tracklets might not in the primary image boundary
                #trackletsA = np.array(trackletsA)
                #trackletsA = np.resize(trackletsA, (trackletsA.shape[0],))
                params['activeTrkltIndexsA'] = list(len(trackletsP) + np.array(active_trklt))
                #params['PAproj'][params['activeTrkltIndexsA']] = trackletsA
                #Auxiliary and primary tracks might have different size so need to handle numpy object and float array
                if len(params['PAproj'].shape)>1 and not self.integration:
                    params['PAproj'] = list(params['PAproj'])
                    for i, ind_PA in enumerate(params['activeTrkltIndexsA']):
                        params['PAproj'][ind_PA] = trackletsA[i]
                else:
                    params['PAproj'][params['activeTrkltIndexsA']] = trackletsA

                mt, tCommonPA = self.associate_tracklets_DFS_MC(params,
                                                           cam2cam,
                                                           max_dist=params['d_th']
                                                            )

        else:
            params['trkltFamilySizeP'] = len(trackletsP)
            params['trkltFamilySizeA'] = len(tracklet_2)

        if cam2cam in ['C360C361', 'C360C340']:
            if self.keep_id_pvd[params['Aid']] and self.keep_id_pvd[params['Pid']]:
                mt, tCommonPA = self.associate_queue_MC(params, max_dist=params['d_th'])
        return mt, params

    def generate_global_graph(self,cacp, mt):
        # G_mc: global association graph based on global track id
        # mc_all_trklts_dict: dictionary of all mc global tracks use for reporting and visualizing the results
        # prepare traklets info for mc system
        cam2cam_params = self.MCparams[cacp]
        cp = cam2cam_params['Pid']
        ca = cam2cam_params['Aid']

        # get factors for individual camera position in G_mc when first appear in BFS
        print('ca {} cp {}'.format(ca, cp))
        cam2cam_params['batch_ids_list'] = []
        if cp not in self.cam_visited:
            self.cam_trklts_id[cp] = []#cam2cam_params['pri_batch_ids']
            if cam2cam_params['trkltFamilySizeP']>0:
                #mixed size list is created due to similarity of batch size in camera pair
                for tr in cam2cam_params['PAorg'][:cam2cam_params['trkltFamilySizeP']]:
                    #only get tracks when visible in current batch
                    if set(list(tr[:, 0])).intersection(range(self.batch_start,
                                                              self.batch_start + self.batch_size)):
                        self.mc_all_trklts_dict[tr[0,1]] = tr
                        cam2cam_params['batch_ids_list'].append(tr[0,1])
                        self.cam_trklts_id[cp].append(tr[0,1])
            self.cam_visited.append(cp)

        if ca not in self.cam_visited:
            self.cam_trklts_id[ca] = []#keep only the current batch id sor a single camera
            if cam2cam_params['trkltFamilySizeA']>0:
                #mixed size list is created due to similarity of batch size in camera pair
                for tr in cam2cam_params['PAorg'][cam2cam_params['trkltFamilySizeP']:]:
                    # only get tracks when visible in current batch
                    if set(list(tr[:, 0])).intersection(range(self.batch_start,
                                                              self.batch_start + self.batch_size)):
                        self.mc_all_trklts_dict[tr[0, 1]] = tr
                        cam2cam_params['batch_ids_list'].append(tr[0, 1])
                        self.cam_trklts_id[ca].append(tr[0, 1])
            self.cam_visited.append(ca)

        # update system graph from pair association (generated from Hungarian)
        if len(mt)>0: #TODO: G_mc only for associated tracks or for all in unique track list??
            #TODO: debug edge merging approach for online MCTA
            #TODO: verify that pairwise to multicamera index mapping keep gloabl track id similar for same track
            #generate cam2cam batch_ids_list to map camera pair graph to global graph
            #cam2cam_params['batch_ids_list'] = cam2cam_params['pri_batch_ids'] + cam2cam_params['aux_batch_ids']
            self.G_mc = merge_edges(mt, self.G_mc, batch_ids=cam2cam_params['batch_ids_list'],
                                    cam_batch_ids=self.cam_trklts_id, ca=ca,cp=cp)
        return self.G_mc, self.mc_all_trklts_dict

    def update_global_label_graph(self):
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
                    #enforce the C300 lane  ids
                    if self.server_loc=='PVD' and self.only_lane3_ids:
                        lane3_id = [l3id for l3id in np.unique(np.array(solnPath)) if l3id in self.lane3_ids]
                        if len(lane3_id)==1:
                            assert len(lane3_id)==1, 'found lane3 id: {}'.format(lane3_id)
                            label_map[label_tp] = lane3_id[0]
                        elif len(lane3_id)>1:
                            # TODO: best way to propagate lane3 ids is to keep meta associated raw ids
                            label_map[label_tp] = min(lane3_id)
                        else:
                            label_map[label_tp] = min(solnPath)
                    else:
                        label_map[label_tp] = min(solnPath)
                else:
                    label_map[label_tp] = label_tp
        return label_map

    def get_global_matches(self):
        #check new id in batch results
        #if self.new_ids:
        #get matches for new ids in camera pair for each current batch
        for cp in sorted(self.BFS_cam_graph.keys()):
            auxs = self.BFS_cam_graph[cp]
            for ca in auxs:
                cacp = 'C' + str(ca) + 'C' + str(cp)
                print(cacp)
                #self.MCparams[cacp] = dafault_params(cam2cam=cacp,img_path = self.img_path, img_HW=self.img_HW, dataset='clasp2')
                #self.MCparams[cacp] = default_params_pvd(cam2cam=cacp)
                self.MCparams[cacp] = default_params_pvd_new(cam2cam=cacp)
                mt, self.MCparams[cacp] = self.get_cam2cam_matches(self.MCparams[cacp],cacp)
                # update global label map graph
                #TODO: debug global graph formation using pairwise association
                if cacp=='C5C2' and self.batch_start==361:
                    print(self.G_mc)
                #update Gmc for each camera pair association
                self.G_mc, self.mc_all_trklts_dict = self.generate_global_graph(cacp, mt)

        label_map = self.update_global_label_graph() #use self.G_mc, self.mc_all_trklts
        self.MCparams['demo_path'] = self.global_out_path
        self.get_global_tracks_MC(label_map,
                             vis_rate=self.vis_rate,
                             save_imgs=self.vis
                             )
        #update the global graph if found matching or initialte new global tracks
        #update the global tracks and return
        #self.new_ids=[]
        return mt, self.global_track_results, self.global_track_id


