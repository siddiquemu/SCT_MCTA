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
import collections
import pandas as pd
import copy
from MCTA.utils import *
from MCTA.discrete_frechet import frechetdist
import time
import collections
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
    def __init__(self, all_params=None, timeoffset=None, min_trklt_size=60):
        #mc_track_segments should have the keys as all cameras: accessible for parwise association
        self.BFS_cam_graph=all_params['cam_graph']
        self.min_trklt_size = all_params['min_trklt_size']
        self.isSCA = False
        self. MCparams = {}
        self.MC_tracklets = {}
        self.vis_pair = True
        self.new_ids = all_params['new_ids']
        self.motion_model = False
        self.appearance_model = False
        self.dist_correct = True
        self.sequential_id = all_params['sequential_id']
        self.save_global_tracks = True
        self.batch_start=all_params['batch_start']
        self.batch_size=all_params['batch_size']
        self.global_track_results = all_params['result_fields']
        self.global_track_saver = all_params['global_full_track_file'] #open(all_params['output_dir'] + '/global_batch_tracks_c9c2.txt', 'w')
        self.global_out_path = all_params['output_dir']
        self.img_path = all_params['img_path']
        self.global_track_id = all_params['global_track_id']
        self.keep_raw_id = all_params['keep_raw_id'] #TODO: keep camera info as well
        self.isSCT = all_params['isSCT']
        self.update_dist = all_params['update_dist_thr']
        self.update_frechet_dist = all_params['update_frechet_dist']

        # updated for each batch
        self.cam_visited = []
        self.index_factor = {}
        self.cam_trklts_size = {}
        self.cam_trklts_id = {}
        self.mc_all_trklts_dict = {}
        self.G_mc = all_params['global_mcta_graph']
        self.single_cam_multi_asso = all_params['single_cam_multi_asso']
        self.mc_all_trklts = []
        self.timeoffset=timeoffset
        self.store_prev_dist = all_params['prev_affinity']
        self.keep_batch_hist = all_params['keep_batch_hist']



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


    def vis_box(self, img, bb, label):
        cv2.rectangle(img, (int(bb[2]), int(bb[3])),
                      (int(bb[2] + bb[4]), int(bb[3] + bb[5])),
                      (255, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(img, 'P{}'.format(int(label)),
                    (int(bb[2] + bb[4] / 2), int(bb[3] + bb[5] / 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 5, cv2.LINE_AA)
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
                        if label_map[c] not in self.keep_raw_id:
                            first_used = 1
                            if self.sequential_id:
                                self.global_track_id += 1
                                self.keep_raw_id[label_map[c]] = self.global_track_id
                            else:
                                self.keep_raw_id[label_map[c]] = label_map[c]
                        # update global tracks
                        #global_tracker(fr, id=self.keep_raw_id[label_map[c]],
                                       #global_tracker=self.global_track_saver,
                                       #first_used=first_used).update_state_csv(bb[i])
                        self.global_track_results['frame'].append(int(fr))
                        self.global_track_results['cam'].append(int(bb[i,-1]))
                        self.global_track_results['timeoffset'].append(float('{:.2f}'.format(fr / 30.0)))
                        self.global_track_results['x1'].append(int(bb[i,2]))
                        self.global_track_results['y1'].append(int(bb[i,3]))
                        self.global_track_results['x2'].append(int(bb[i,2]+bb[i,4]))
                        self.global_track_results['y2'].append(int(bb[i,3]+bb[i,5]))
                        self.global_track_results['id'].append(int(self.keep_raw_id[label_map[c]]))
                        self.global_track_results['firstused'].append(first_used)
                        if save_imgs:
                            img = self.vis_box(img, bb[i], self.keep_raw_id[label_map[c]])
        return img

    def get_global_tracks_MC(self,label_map,
                             vis_rate=30,
                             save_imgs=0,
                             ):
        # BGR
        # magenta = (255,0,255)
        # yellow = (0,255,255)
        # green = (0,255,0)
        # blue = (255,0,0)
        #if save_imgs:
            #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            #cv2.resizeWindow("image", 2*640, 2*480)
        for fr in range(self.batch_start, self.batch_start+self.batch_size):
            # 9A:9 fr-15, 11 fr-15
            if (fr) % vis_rate == 0:  # 15 for 6A,7A,.. 3 for 5A, 5B.. 30 for 9A
                #try: #to avoid crash due to missing frames
                timeoffset = '{:.2f}'.format(fr/30.0)
                if save_imgs:
                    imgC2 = cv2.imread('{}/{:06d}.png'.format(self.MCparams['C9C2']['PImgsPath'], fr + (15+15))) #46+44
                    imgC9 = cv2.imread('{}/{:06d}.png'.format(self.MCparams['C9C2']['AImgsPath'], fr + 0+15))
                    imgC5 = cv2.imread('{}/{:06d}.png'.format(self.MCparams['C5C2']['AImgsPath'], fr + (9+15))) #28+44
                    imgC11 = cv2.imread('{}/{:06d}.png'.format(self.MCparams['C11C5']['AImgsPath'],fr + (1+15))) #3+44
                    #imgC13 = cv2.imread('{}/{:06d}.png'.format(self.MCparams['C13C5']['AImgsPath'],fr + (-44+44)))  # 46
                    #imgC14 = cv2.imread('{}/{:06d}.png'.format(self.MCparams['C14C13']['AImgsPath'], fr + (-41+44)))

                    # visulaize box on C2: H9to2
                    if self.MCparams['C9C2']['trkltFamilySizeP'] > 0:
                       imgC2 = self.save_batch_results(fr,imgC2,label_map,save_imgs, cam=2)

                    # show bbox on ref: C9:
                    if self.MCparams['C9C2']['trkltFamilySizeA'] > 0:
                       imgC9 = self.save_batch_results(fr,imgC9,label_map,save_imgs, cam=9)

                    # show bbox on C5: use H5to11 instead of H5to2 since labels are already propagated to H5to11
                    if self.MCparams['C5C2']['trkltFamilySizeA'] > 0:
                       imgC5 = self.save_batch_results(fr,imgC5,label_map,save_imgs, cam=5)

                    # show bbox on C11
                    if self.MCparams['C11C5']['trkltFamilySizeA'] > 0:
                       imgC11 = self.save_batch_results(fr,imgC11,label_map,save_imgs, cam=11)

                    # show bbox on C13
                    #if self.MCparams['C13C5']['trkltFamilySizeA'] > 0:
                       #imgC13 =  self.save_batch_results(fr,imgC13,label_map,save_imgs, cam=13)

                    # show bbox on C14
                    #if self.MCparams['C14C13']['trkltFamilySizeA'] > 0:
                        #imgC14 = self.save_batch_results(fr,imgC14,label_map,save_imgs, cam=14)

                    if save_imgs:
                        # translate image
                        imgC2 = cv2.copyMakeBorder(imgC2, 0, 0, 0, 250, cv2.BORDER_CONSTANT, value=0)
                        imgC9 = cv2.copyMakeBorder(imgC9, 0, 0, 250, 0, cv2.BORDER_CONSTANT, value=0)
                        #imgC13 = cv2.copyMakeBorder(imgC13, 0, 0, 700, 0, cv2.BORDER_CONSTANT, value=0)
                        #imgC14 = cv2.copyMakeBorder(imgC14, 0, 0, 0, 700, cv2.BORDER_CONSTANT, value=0)

                        imgC9C2 = cv2.vconcat([imgC9, imgC2])
                        imgC11C5 = cv2.vconcat([imgC11, imgC5])
                        final_img = cv2.hconcat([imgC11C5, imgC9C2])  # img: c4, img2: c2

                        #blankImg = np.zeros(shape=[1080, 1920, 3], dtype=np.uint8)
                        #img13Blnk = cv2.vconcat([imgC13, blankImg])
                        #img1314 = cv2.vconcat([imgC14, imgC13])
                        #final_img = cv2.hconcat([img1314, final_img])

                        #final_img = cv2.resize(final_img, (int(1 * 1920), 1080), interpolation=cv2.INTER_AREA)#1.5

                        #cv2.imshow("image", imgC9C2)
                        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(self.MCparams['demo_path'] + '/{:06d}.jpg'.format(fr), final_img)
                    #if save_imgs:
                        #cv2.waitKey(10)
                #except:
                    #print("Frame {} not found".format(fr))
                    #continue
        #if save_imgs:
            #cv2.destroyAllWindows()
    def get_global_tracks(self,label_map,
                             vis_rate=30,
                             save_imgs=0,
                             cam2cam=None,
                             cp=None,
                             ca=None
                             ):
        # BGR
        # magenta = (255,0,255)
        # yellow = (0,255,255)
        # green = (0,255,0)
        # blue = (255,0,0)
        #if save_imgs:
            #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
            #cv2.resizeWindow("image", 2*640, 2*480)
        for fr in range(self.batch_start, self.batch_start+self.batch_size):
            # 9A:9 fr-15, 11 fr-15
            if (fr) % vis_rate == 0:  # 15 for 6A,7A,.. 3 for 5A, 5B.. 30 for 9A
                #try: #to avoid crash due to missing frames
                    timeoffset = '{:.2f}'.format(fr/30.0)
                    if save_imgs:
                        imgP = cv2.imread('{}/{:06d}.png'.format(self.MCparams[cam2cam]['PImgsPath'], fr + 7))#7  46
                        imgA = cv2.imread('{}/{:06d}.png'.format(self.MCparams[cam2cam]['AImgsPath'], fr+0))#0 28
                    else:
                        imgP=imgA=None
                    # visulaize box on C2: H9to2
                    if self.MCparams[cam2cam]['trkltFamilySizeP'] > 0:
                       imgP = self.save_batch_results(fr,imgP,label_map,save_imgs, cam=cp)

                    # show bbox on ref: C9:
                    if self.MCparams[cam2cam]['trkltFamilySizeA'] > 0:
                       imgA = self.save_batch_results(fr,imgA,label_map,save_imgs, cam=ca)

                    if save_imgs:
                        if cam2cam == 'C9C2':
                            final_img = cv2.vconcat([imgA, imgP])
                        if cam2cam == 'C5C2':
                            final_img = cv2.hconcat([imgA, imgP])  # img: c5, img2: c2
                        if cam2cam == 'C11C5':
                            final_img = cv2.vconcat([imgA, imgP])
                        if cam2cam == 'C13C5':
                            final_img = cv2.hconcat([imgA, imgP])  # img: c11, img2: c13
                        if cam2cam == 'C13C11':
                            img2 = cv2.copyMakeBorder(imgA, 0, 0, 320, 0, cv2.BORDER_CONSTANT, value=0)
                            img = cv2.copyMakeBorder(imgP, 0, 0, 0, 320, cv2.BORDER_CONSTANT, value=0)
                            final_img = cv2.hconcat([imgA, imgP])  # img: c11, img2: c13
                        if cam2cam == 'C14C13':
                            img = cv2.copyMakeBorder(imgP, 0, 0, 700, 0, cv2.BORDER_CONSTANT, value=0)
                            img2 = cv2.copyMakeBorder(imgA, 0, 0, 0, 700, cv2.BORDER_CONSTANT, value=0)
                            final_img = cv2.vconcat([imgA, imgP])  # img: c13, img2: c14

                        #cv2.imshow("image", final_img)
                        # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(self.MCparams['demo_path'] + '/{:06d}.jpg'.format(fr), final_img)
                    #if save_imgs:
                        #cv2.waitKey(10)
                #except:
                    #print("Frame {} not found".format(fr))
                    #continue
        #if save_imgs:
            #cv2.destroyAllWindows()

    def _undistorted_coords(self, trklt, dist_coeff, A, im_shape = (1920, 1080)):
        # use a copy of traclet centroids to convert into undistorted format
        # ***A: Camera Intrinsic Matrix
        # https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html
        # https: // www.mathworks.com / help / vision / ref / estimatecameraparameters.html
        # new camMatrix
          # img.shape[:2]
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
    def project_tracklets(self, in_tracklets, H, cam=9):
        out_tracklets = []
        active_trklt = []
        Hv = copy.deepcopy(H)
        Hv[0, 2], Hv[1, 2] = 0, 0
        # [fr,cx,cy,x,y,vx,vy]<<< projected coordinates: center is used
        for i, trklt in enumerate(in_tracklets):
            if self.dist_correct:
                dist_coeff, A = camera_intrinsics(cam)
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
                cxcy = self._applyTransform(bb[2:4] + bb[4:6] / 2.0, H)
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
            if max(trklt[:, 3] - trklt[:, 5]) > 0 and max(trklt[:, 4]) > 0:
                out_tracklets.append(trklt)
                active_trklt.append(i)
        return out_tracklets, active_trklt

    def associate_tracklets_DFS_MC(self, params, cam2cam, max_dist=400):
        # ***i : master camera tracklets index: time should be synced to this master camera
        # ***j : projected/auxiliary camera tracklet index
        # ***np.array(active_trklt): acyive_trklt_indexes from slave cameras
        # ***len(cam1Tracklets): total tracklets in master camera (C2)
        # ------------------------------------------------------------------------------------
        print('Total tracklets in a camera pairs {}'.format(len(params['PAproj'])))
        cost = np.ones((len(params['PAproj']), len(params['PAproj']))) * np.inf
        tilde_t_pa = np.ones((len(params['PAproj']), len(params['PAproj'])), dtype='float') * np.inf
        i = 0
        matches = []
        d_h = None
        for tracklet1 in params['PAproj']:
            j = 0
            for tracklet2 in params['PAproj']:
                # condition to check multicamera trackets have any overlap
                if i != j and i < params['trkltFamilySizeP'] and j >= params['trkltFamilySizeP'] \
                        and j in params['activeTrkltIndexsA']:

                    # TODO: There must be a more compact way of doing this (find the overlapped region)
                    # TODO: Assess whether the timestamp should be considered in the distance computation as well
                    common_t_stamps = np.array(list(set(list(tracklet2[:, 0])).intersection(list(tracklet1[:, 0]))))
                    if len(common_t_stamps)>self.min_trklt_size//2:
                        assert len(common_t_stamps) > self.min_trklt_size//2, 'Non-overlaped T_p, T_a can not be associated'
                        #get actual id
                        id_i = params['PAproj'][i][0, 1]
                        id_j = params['PAproj'][j][0, 1]
                        if self.motion_model:
                            featIndx = [0, 2, 3, 6, 7]
                        else:
                            featIndx = [0, 2, 3, 4, 5] # 4,5- width and height are currently not used

                        mask1 = np.isin(tracklet1[:, 0], common_t_stamps)
                        final_tracklet1 = tracklet1[mask1][:, featIndx]

                        if self.dist_correct:
                            dist_coeff, A = camera_intrinsics(cam=params['Pid'])  # 2:c5 or 9:c11
                            xy = copy.deepcopy(final_tracklet1[:, 1:3])
                            xy = self._undistorted_coords(xy, dist_coeff, A)
                            final_tracklet1[:, 1:3] = xy

                            #get undistorted motion
                            if self.motion_model:
                                vxvy = copy.deepcopy(final_tracklet1[:, 3:5])
                                vxvy = self._undistorted_coords(vxvy, dist_coeff, A)
                                final_tracklet1[:, 3:5] = vxvy
                        # compute the location from velocity and time difference
                        final_tracklet1[:, 1:3] = final_tracklet1[:, 1:3] / [1920.0, 1080.0]

                        mask2 = np.isin(tracklet2[:, 0], common_t_stamps)
                        final_tracklet2 = tracklet2[mask2][:, featIndx]
                        final_tracklet2[:, 1:3] = final_tracklet2[:, 1:3] / [1920.0, 1080.0]

                        if  not self.motion_model:
                            scan_intrvl = 240  # C5C11:240
                            st_time = time.time()
                            if len(final_tracklet1[:, 1:3]) > scan_intrvl:
                                sys.setrecursionlimit(len(final_tracklet1[:, 1:3]) * len(final_tracklet2[:, 1:3]))
                                d_h_center = frechetdist.frdist(final_tracklet1[:scan_intrvl, 1:3],final_tracklet2[:scan_intrvl, 1:3])
                                #d_h_center = directed_hausdorff(final_tracklet1[:, 1:3], final_tracklet2[:, 1:3])[0]
                            else:
                                sys.setrecursionlimit(len(final_tracklet1[:, 1:3]) * len(final_tracklet2[:, 1:3]))
                                d_h_center = frechetdist.frdist(final_tracklet1[:, 1:3],final_tracklet2[:, 1:3])
                                #d_h_center = directed_hausdorff(final_tracklet1[:, 1:3], final_tracklet2[:, 1:3])[0]
                            print('Frechet distance computation time {} sec'.format(time.time()-st_time))
                            #TODO: update d_frechet for each batch to find the nminima
                            if self.update_frechet_dist:
                                #verify that the pair is already associated in previous batches matching
                                if id_i in self.store_prev_dist and id_j in self.store_prev_dist[id_i]:
                                    d_h_center_prev = self.store_prev_dist[id_i][id_j]
                                    if d_h_center_prev<d_h_center:
                                        d_h=d_h_center_prev
                                    else:
                                        d_h=d_h_center
                                else:
                                    d_h=d_h_center
                            else:
                                d_h=d_h_center
                            print('Processing global tracklets {} and {} without motion for cost {}'.format(tracklet1[:,1][0], tracklet2[:,1][0], d_h))

                        if self.motion_model and self.appearance_model:
                            # use recursive method: maximum recursion limit needs to be updated
                            # TODO: How to select the subset of tracklet to compute frechet distance faster??
                            scan_intrvl = 240  # C5C11:240
                            if len(final_tracklet1[:, 1:3]) > scan_intrvl:
                                sys.setrecursionlimit(len(final_tracklet1[:, 1:3]) * len(final_tracklet2[:, 1:3]))
                                d_h_center = frechetdist.frdist(final_tracklet1[:scan_intrvl, 1:3],
                                                                final_tracklet2[:scan_intrvl, 1:3])
                                # d_h_app = similaritymeasures.frechet_dist(final_tracklet1[:scan_intrvl, 5::], final_tracklet2[:scan_intrvl, 5::])
                                # d_h_motion = frechetdist.frdist(final_tracklet1[:scan_intrvl, 3:5], final_tracklet2[:scan_intrvl, 3:5])
                            else:
                                sys.setrecursionlimit(len(final_tracklet1[:, 1:3]) * len(final_tracklet2[:, 1:3]))
                                d_h_center = frechetdist.frdist(final_tracklet1[:, 1:3],
                                                                final_tracklet2[:, 1:3])
                                # d_h_app = similaritymeasures.frechet_dist(final_tracklet1[:, 5::], final_tracklet2[:, 5::])
                                # d_h_motion = frechetdist.frdist(final_tracklet1[:, 3:5], final_tracklet2[:, 3:5])
                        if d_h is not None:
                            if d_h <= max_dist:
                                cost[i, j] = d_h
                                tilde_t_pa[i, j] = min(common_t_stamps)
                j = j + 1
            i = i + 1
        # DEBUG
        if self.update_dist:
            #if params['Pid']==2 and params['Aid']==9:
            import matplotlib.pyplot as plt1
            hist = np.histogram(cost[cost < 10000],bins=50,range=(0,2))
            self.keep_batch_hist[self.timeoffset] = hist[0]
            #max_dist = hist[1][np.argmax(hist[0])]
            plt1.hist(cost[cost < 10000],bins=50,range=(0,2))
            plt.savefig(self.global_out_path+'/process_hist/cam{}to{}_{}.png'.format(params['Aid'],params['Pid'],self.timeoffset),dpi=300)
            plt.close()

        if self.single_cam_multi_asso:
            while len(cost[cost <= max_dist]) > 0:
                row_ind, col_ind = linear_sum_assignment_with_inf(cost)
                Mincosts = cost[row_ind, col_ind]
                idx = np.where(Mincosts <= max_dist)[0]

                for i in idx:
                    #return raw id instead of matched tracklet index
                    matches.append((Mincosts[i], tilde_t_pa[row_ind[i], col_ind[i]],
                                    params['PAproj'][row_ind[i]][0,1],params['PAproj'][col_ind[i]][0,1]))  #  row_ind[i], col_ind[i])

                    # t_p:row_ind[i] fixed> for matched t_a:col_ind[i] search over t_as in T_a to get any overlap
                    indx_t_a = np.where(cost[row_ind[i], :] <= max_dist)[0]
                    if len(indx_t_a) > 0:
                        T_a = params['PAproj'][indx_t_a]
                        isIntersect = [len(set(params['PAproj'][col_ind[i]][:, 0]).intersection(t_a[:, 0])) > 10
                                       for t_a in T_a]
                        cost[row_ind[i], indx_t_a[isIntersect]] = np.inf

                    # t_a:col_ind[i] fixed> for matched t_p:row_ind[i] search over t_ps in T_p to get any overlap
                    indx_t_p = np.where(cost[:, col_ind[i]] <= max_dist)[0]
                    if len(indx_t_p) > 0:
                        T_p = params['PAproj'][indx_t_p]
                        isIntersect = [len(set(params['PAproj'][row_ind[i]][:, 0]).intersection(t_p[:, 0])) > 10
                                       for t_p in T_p]
                        cost[indx_t_p[isIntersect], col_ind[i]] = np.inf
        else:
            if self.batch_start==361:
                print(self.batch_start)
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


    def get_cam2cam_matches(self, mc_track_segments, params, cam2cam):
        #init matches to return empty match
        mt = []
        trackersP = mc_track_segments[params['Pid']]
        trackersA = mc_track_segments[params['Aid']]
        # collect primary camera: P:C2 features
        #note: keys might be available when created once but values may be none: initiate batch dictionary for every batch
        #TODO: make sure that all keys or track has its values (availability in look back frames)
        if len(trackersP.keys())!=0:
            tracklets, totalTrack, ids = form_tracklets(trackersP, cam=params['Pid'], tracker_min_size=self.min_trklt_size,
                                                       t_thr=30, d_thr=50, motion=self.motion_model,
                                                       single_cam_association=self.isSCA).get_trklts()
            # list C2 tracklets
            params['pri_batch_ids'] = ids
            trackletsP = convert_centroids(tracklets)
        else:
            trackletsP = []
            params['pri_batch_ids'] = []
        # A:C9:------------------------------------------------------------------
        # Collect auxiliary camera features
        if len(trackersA.keys())!=0:
            tracklets, totalTrack, ids = form_tracklets(trackersA, cam=params['Aid'], tracker_min_size=self.min_trklt_size,
                                                       t_thr=30, d_thr=50, motion=self.motion_model,
                                                       single_cam_association=self.isSCA).get_trklts()
            # keep both original and projected tracklets for auxiliary
            params['aux_batch_ids'] = ids
            tracklet_2 = copy.deepcopy(tracklets)
            # Project: center>(cx,cy), top-left>(x,y), velocity>(vx,vy) [fr,cx,cy,x,y,vx,vy]
            # active_trklt: indexes of tracklets whose projected onto the destination (primary) image boundary
            trackletsA, active_trklt = self.project_tracklets(tracklets, params['H'], cam=params['Aid'])
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
            params['PAorg'] = np.array(centroids2xywh(trackletsP)+ tracklet_2)

            # Apply MCTA when both camera has tracks and aux projection in pri image boundary
            print(' number of eligible tracklet for association {}'.format(len(active_trklt)))
            if len(trackletsP)!=0 and len(trackletsA)!=0 and len(active_trklt) > 0:
                # since all the projected tracklets might not in the primary image boundary
                #trackletsA = np.array(trackletsA)
                #trackletsA = np.resize(trackletsA, (trackletsA.shape[0],))
                params['activeTrkltIndexsA'] = list(len(trackletsP) + np.array(active_trklt))
                params['PAproj'][params['activeTrkltIndexsA']] = trackletsA
                mt, tCommonPA = self.associate_tracklets_DFS_MC(params,
                                                           cam2cam,
                                                           max_dist=params['d_th']
                                                               )
        else:
            params['trkltFamilySizeP'] = len(trackletsP)
            params['trkltFamilySizeA'] = len(tracklet_2)
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
            self.cam_trklts_id[ca] = []#cam2cam_params['aux_batch_ids']
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
            self.G_mc = merge_edges(mt, self.G_mc, batch_ids=cam2cam_params['batch_ids_list'])
        return self.G_mc, self.mc_all_trklts_dict

    def update_global_label_graph(self):
        # apply DFS on mc system graph
        label_map = {keys: [] for keys in self.mc_all_trklts_dict.keys()}
        # TODO: each track identity should be unique
        for label_tp, tklt in self.mc_all_trklts_dict.items():
            if self.isSCT:
                label_map[label_tp]=label_tp
            else:
                if label_tp in self.G_mc.keys():
                    solnPath = self._dfs(self.G_mc, label_tp, [])
                    #TODO:make sure min id or min time based id is useful
                    label_map[label_tp] = min(solnPath)
                else:
                    label_map[label_tp] = label_tp
        return label_map

    def get_global_matches(self, mc_track_segments, ca, cp):
        cacp = 'C' + str(ca) + 'C' + str(cp)
        self.MCparams[cacp] = dafault_params(cam2cam=cacp,img_path = self.img_path)
        mt, self.MCparams[cacp] = self.get_cam2cam_matches(mc_track_segments, self.MCparams[cacp],cacp)
        # update global label map graph
        #TODO: debug global graph formation using pairwise association
        self.G_mc, self.mc_all_trklts_dict = self.generate_global_graph(cacp, mt)
        return self.G_mc, self.mc_all_trklts_dict, mt

    def save_tracks(self):
        label_map = self.update_global_label_graph() #use self.G_mc, self.mc_all_trklts
        if self.save_global_tracks:
            self.MCparams['demo_path'] = self.global_out_path
            self.get_global_tracks_MC(label_map,
                                 vis_rate=10,
                                 save_imgs=1
                                 )
        return self.global_track_results


