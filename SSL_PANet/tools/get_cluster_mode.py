################# Refining Detection set from Multiple Sample image through Mask-RCNN ##############
#
#
#
#
#
####### Prepared by - Abubakar Siddique, COVISS Lab, MU, USA #######################################
from __future__ import division
import cv2
import imutils
#import matplotlib.pyplot as plt
import os
import sys
import skimage.io
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
import mean_shift as MS
from scipy.spatial import distance

class Cluster_Mode(object):
    def __init__(self, detPB, frame, angle_set, img, out_path, save_dets_dict, vis, nms=None,
                 save_modes=False, cluster_scores_thr=[0,0], feature_type='mask', feat_dim=4,
                 save_scores=None, global_frame=None, dataset='clasp1', bw_type='estimated', ann_dir=None,
                 im_name=None, verbose=False):
        self.detPB = detPB
        self.verbose=verbose
        if self. verbose:
            print('total det_remapped: {}'.format(detPB.shape))
        self.det_at_t_p = detPB[detPB[:, 7] == 1, :] #person class
        self.det_at_t_b = detPB[detPB[:, 7] > 1, :]  #backpack, handbag, suitcase classes
        self.det_indexsp = np.where(detPB[:, 7] == 1)[0]
        self.det_indexsb = np.where(detPB[:, 7] > 1)[0]

        self.color_blue = (255, 0, 0)  # blue
        #self.color_b = (255, 0, 255) # magenta
        self.color_remap = [(0, 255, 0), (255, 0, 255)]
        self.frame = frame
        self.img = img
        self.n_angles = len(angle_set)
        self.out_path = out_path
        self.det_for_mht = save_dets_dict['dets_aug']
        self.vis=vis
        self.im_name = im_name
        self.nms=nms
        self.feature_type = feature_type
        self.cluster_score_thrp = cluster_scores_thr[0]
        self.cluster_score_thrb = cluster_scores_thr[1]
        self.final_cluster_indexs = []
        self.save_modes = save_modes
        self.save_scores = save_scores
        self.global_frame = global_frame
        self.dataset = dataset
        self.modes_pb = []
        self.ann_dir = ann_dir
        self.bw_type = bw_type
        self.XYWH = [self.img.shape[1], self.img.shape[0], self.img.shape[1], self.img.shape[0]]
        self.feat_index = [0,1,6,7]
        self.feat_dim = feat_dim


    @staticmethod
    def normalize_feature(X, img):
        #4D feature: [cx, cy, w, h]
        #TODO: w, h normalized by max value in frame or by img width, height
        #[img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
        X_scaled = X[:, [0, 1, 6, 7]] / [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
        return X_scaled

    def show_active_cluster(self, cluster, class_id=1):
        """

        :param cluster: active cluster ready for learning
        :param class_id: target class
        :return: None
        """
        class_id = cluster[0][7]
        if class_id == 1:
            c = self.color_remap[0]
        else:
            c = self.color_remap[1]
        for bb in cluster:
            self.img = cv2.rectangle(self.img, (int(bb[2]), int(bb[3])), (int(bb[2] + bb[4]), int(bb[3] + bb[5])), c, 2)

    # prepare feature for using in Mean-Shift Clustering
    def detection_feature(self, det_Set):
        box_center_feature = []
        # remap all bbox to original image from multiple inference
        box_center = []
        assert det_Set.shape[1]==9
        for detBox in det_Set:
            # [0,3,6,12,348,351,354,357]
            #print(angle)
            x = detBox[2]
            y = detBox[3]
            w = detBox[4]
            h = detBox[5]
            fr = detBox[0]
            score = detBox[6]
            classID = detBox[7]
            angle = detBox[8]
            if self.feature_type=='mask':
                CXbox = x + w / 2.
                CYbox = y + h / 2.
            else:
                CXbox = x
                CYbox = y
            #if (w <= 1920 and h <= 1080):
            box_center.append([CXbox, CYbox,fr,detBox[1], x, y, w, h, score, classID, angle])
                # all remapped detection on original image
                #if self.vis:
                    #self.img = cv2.rectangle(self.img, (int(x), int(y)), (int(x + w), int(y + h)), self.color_remap, 2)
                    #if classID==1: c = self.color_remap[0]
                    #else: c = self.color_remap[1]
                    #self.img = cv2.rectangle(self.img, (int(x), int(y)), (int(x + w), int(y + h)), c, 2)
                    #self.img = cv2.circle(self.img, (int(x), int(y)), 5, c, -2)
        return np.array(box_center)

    def frame_detection_clustering(self, feature_norm, box_features, det_ind=None,
                                   score_thr=0, color=None):
        # set of bbox centers
        feature = feature_norm[:,:self.feat_dim]  # norm feature (0-1)
        det_frame = box_features # contain all info [CXbox,CYbox,fr,instance_index, x, y, w, h, score,classID, angle]
        #feature = det_frame[:, [0, 1, 6, 7]] / [max(det_frame[:, [0]])[0], max(det_frame[:, [1]])[0],
                                                #max(det_frame[:, [6]])[0], max(det_frame[:, [7]])[0]]

        assert len(feature)==len(det_frame)
        # to use multivariate gaussian kernel
        # Make sure the dimensions of 'data' and the kernel match
        # '''
        # '''

        # nothing mention about mean??
        if self.bw_type=='estimated':
            x_kernel = np.var(feature[:, 0], axis=0)
            x_kernel = float("{0:.5f}".format(x_kernel))
            y_kernel = np.var(feature[:, 1], axis=0)
            y_kernel = float("{0:.5f}".format(y_kernel))

            w_kernel = np.var(feature[:, 2], axis=0)
            w_kernel = float("{0:.5f}".format(w_kernel))
            h_kernel = np.var(feature[:, 3], axis=0)
            h_kernel = float("{0:.5f}".format(h_kernel))

            # covariance matrix should not be the singular matrix: positive definite, symmetrical, Invertible
            if (x_kernel == 0):
                x_kernel = x_kernel + 0.0000000001
            if (y_kernel == 0):
                y_kernel = y_kernel + 0.0000000001

            if (w_kernel == 0):
                w_kernel = w_kernel + 0.0000000001
            if (h_kernel == 0):
                h_kernel = h_kernel + 0.00000000001

            mean_shifter = MS.MeanShift(kernel='multivariate_gaussian')
            mean_shift_result = mean_shifter.cluster(feature, kernel_bandwidth=[x_kernel, y_kernel, w_kernel, h_kernel])#w_kernel, h_kernel
            shifted_points = mean_shift_result.shifted_points
            labels = mean_shift_result.cluster_ids

        # from mpl_toolkits import mplot3d
        # import matplotlib.pyplot as plt
        # from scipy.stats import multivariate_normal
        # x = np.linspace(-1, 1, 100)
        # y = np.linspace(0, 2, 100)
        # X, Y = np.meshgrid(x, y)
        # pos = np.dstack((X, Y))
        # mu = np.mean(np.transpose(feature[:,0:2]),axis=1)
        # cov = np.cov(np.transpose(feature[:,0:2]))
        # rv = multivariate_normal(mu, cov)
        # Z = rv.pdf(pos)
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.plot_surface(X, Y, Z)
        # fig.savefig('/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d6/tracking_wo_bnw/mv_panet_det.png', dpi=200)

        if self.bw_type=='fixed':
            ms = MeanShift(bandwidth=0.02, bin_seeding=True, cluster_all=True)  # ,0.25
            # Mean-Shift Model Fitting
            ms.fit(feature)
            labels = ms.labels_
            #print('Labels:', labels)
            cluster_centers = ms.cluster_centers_

        #labels = labels[np.where(labels >= 0)]
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        if self.verbose:
            print("number of estimated clusters before filtering : %d" % n_clusters_)

        unique, counts = np.unique(labels, return_counts=True)
        refined_frame = []
        for j in labels_unique:  # cluster wise loop
            if j!=-1:
                cluster_Q = det_frame[labels == j, :]
                # TODO: Why the first shifted point is selected as centroids??? must correct this idea......
                #cluster_center = shifted_points[labels == j, :][0, 0]  # why the first
                if self.bw_type=='estimated':
                    cluster_center = np.mean(shifted_points[labels == j, :], axis=0)
                if self.bw_type=='fixed':
                    cluster_center = cluster_centers[j]
                if self.nms:
                    cluster_prob_score = np.sum(cluster_Q[:, 8]) / self.n_angles
                else:
                    cluster_prob_score = np.sum(cluster_Q[:, 8]) / np.sum(det_frame[:,8])
                # Apply threshold
                #print('Cluster Score: >>>>> ', cluster_prob_score)
                # search cluster for angle=0
                score = np.around(cluster_Q[:, 8], decimals=2)  # cluster detection score rounded upto two decimal points
                refined_det = cluster_Q[score == score.max()]  # single or multiple detections might be the representative members of a cluster

                if (len(refined_det) > 1):  # case for multiple representative
                    dist = []
                    # box feature
                    mode_points = refined_det[:, self.feat_index[:feature.shape[1]]] / self.XYWH[:feature.shape[1]] #[self.img.shape[1], self.img.shape[0]]

                    for i in range(len(refined_det)):
                        dist = np.append(dist, distance.euclidean(cluster_center, mode_points[i]))
                    refined_det = refined_det[dist.argmin(), :]  # select closest one from multiple representative
                    refined_det[8] = cluster_prob_score  # * refined_det[7]  # det score weighted by cluster probability
                else:
                    refined_det = refined_det[:, :][0]
                    refined_det[8] = cluster_prob_score  # * refined_det[7] # det score weighted by cluster probability
                if self.save_scores:
                    self.save_scores['clusters']['cluster_score'].append(cluster_prob_score)
                    self.save_scores['clusters']['class_id'].append(refined_det[9])
                    self.save_scores['clusters']['frame'].append(self.global_frame)
                # refine cluster based on cluster size
                if self.verbose:
                    print('cluster {}, score {}'.format(j, cluster_prob_score))
                if cluster_prob_score>= score_thr:
                    self.final_cluster_indexs.append(det_ind[labels==j])

                    if self.vis:
                        self.show_active_cluster(cluster_Q[:,2:11])

                    refined_frame.append(refined_det[2:11])
                    score_cluster = np.round(refined_det[8],2)
                    if score_cluster>1:
                        score_cluster = 1.00
                    score = '%.2f' % score_cluster
                    #show cluster modes
                    if self.vis:
                        self.img = cv2.rectangle(self.img,  (int(refined_det[4]-5+refined_det[6]//2 ), int(refined_det[5]-50+refined_det[7]//2-5)),
                                                 (int(refined_det[4]-5+145+refined_det[6]//2 ), int(refined_det[5]-50+60+refined_det[7]//2-5)), (255, 255, 255), -1)
                        cv2.putText(self.img, str(score), (int(refined_det[4]+refined_det[6]//2 ), int(refined_det[5]+refined_det[7]//2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 5)
                        self.img = cv2.rectangle(self.img, (int(refined_det[4]), int(refined_det[5])), (int(refined_det[4] + refined_det[6]), int(refined_det[5] + refined_det[7])), color, 5)

        assert len(labels)==len(det_frame)
        return np.array(refined_frame)
    #currently unused
    # def frame_detection_clustering_det(self, feature_norm, box_features,  color=None):
    #     # set of bbox centers
    #     feature = feature_norm  # norm feature (0-1)
    #     det_frame = box_features # contain all info [CXbox,CYbox,fr,instance_index, x, y, w, h, score,classID, angle]
    #     #feature = det_frame[:, [0, 1, 6, 7]] / [max(det_frame[:, [0]])[0], max(det_frame[:, [1]])[0],
    #                                             #max(det_frame[:, [6]])[0], max(det_frame[:, [7]])[0]]
    #
    #     assert len(feature)==len(det_frame)
    #     # to use multivariate gaussian kernel
    #     # Make sure the dimensions of 'data' and the kernel match
    #     # '''
    #     # '''
    #     x_kernel = np.var(feature[:, 0], axis=0)
    #     x_kernel = float("{0:.5f}".format(x_kernel))
    #     y_kernel = np.var(feature[:, 1], axis=0)
    #     y_kernel = float("{0:.5f}".format(y_kernel))
    #
    #     w_kernel = np.var(feature[:, 2], axis=0)
    #     w_kernel = float("{0:.5f}".format(w_kernel))
    #     h_kernel = np.var(feature[:, 3], axis=0)
    #     h_kernel = float("{0:.5f}".format(h_kernel))
    #
    #     # covariance matrix should not be the singular matrix: positive definite, symmetrical, Invertible
    #     if (x_kernel == 0):
    #         x_kernel = x_kernel + 0.0000000001
    #     if (y_kernel == 0):
    #         y_kernel = y_kernel + 0.0000000001
    #
    #     if (w_kernel == 0):
    #         w_kernel = w_kernel + 0.0000000001
    #     if (h_kernel == 0):
    #         h_kernel = h_kernel + 0.00000000001
    #
    #     # nothing mention about mean??
    #
    #     mean_shifter = ms.MeanShift(kernel='multivariate_gaussian')
    #     mean_shift_result = mean_shifter.cluster(feature, kernel_bandwidth=[x_kernel, y_kernel, w_kernel, h_kernel])#w_kernel, h_kernel
    #     shifted_points = mean_shift_result.shifted_points
    #     labels = mean_shift_result.cluster_ids
    #     '''
    #     from mpl_toolkits import mplot3d
    #     import matplotlib.pyplot as plt
    #     from scipy.stats import multivariate_normal
    #     x = np.linspace(-1, 1, 100)
    #     y = np.linspace(0, 2, 100)
    #     X, Y = np.meshgrid(x, y)
    #     pos = np.dstack((X, Y))
    #     mu = np.mean(np.transpose(feature[:,0:2]),axis=1)
    #     cov = np.cov(np.transpose(feature[:,0:2]))
    #     rv = multivariate_normal(mu, cov)
    #     Z = rv.pdf(pos)
    #     fig = plt.figure()
    #     ax = fig.add_subplot(111, projection='3d')
    #     ax.plot_surface(X, Y, Z)
    #     fig.savefig('/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d6/tracking_wo_bnw/mv_panet_det.png', dpi=200)
    #     '''
    #     #ms = MeanShift(bandwidth=0.05, bin_seeding=True, cluster_all=True)  # ,0.25
    #     # Mean-Shift Model Fitting
    #    # ms.fit(feature)
    #     #labels = ms.labels_
    #     # print('Labels:', labels)
    #     #cluster_centers = ms.cluster_centers_
    #
    #     #labels = labels[np.where(labels >= 0)]
    #     labels_unique = np.unique(labels)
    #     n_clusters_ = len(labels_unique)
    #     print("number of estimated clusters : %d" % n_clusters_)
    #
    #     unique, counts = np.unique(labels, return_counts=True)
    #     refined_frame = []
    #     for j in labels_unique:  # cluster wise loop
    #         if j!=-1:
    #             cluster_Q = det_frame[labels == j, :]
    #             # TODO: Why the first shifted point is selected as centroids??? must correct this idea......
    #             #cluster_center = shifted_points[labels == j, :][0, 0]  # why the first
    #             cluster_center = np.mean(shifted_points[labels == j, :], axis=0)
    #             #cluster_center = cluster_centers[j]
    #             if self.nms:
    #                 cluster_prob_score = np.sum(cluster_Q[:, 8]) / self.n_angles
    #             else:
    #                 cluster_prob_score = np.sum(cluster_Q[:, 8]) / np.sum(det_frame[:,8])
    #             # Apply threshold
    #             #print('Cluster Score: >>>>> ', cluster_prob_score)
    #             # search cluster for angle=0
    #             Q_angle_0 = cluster_Q[cluster_Q[:,-1]==0,:]
    #             if len(Q_angle_0)==0:
    #                 score = np.around(cluster_Q[:, 8], decimals=2)  # cluster detection score rounded upto two decimal points
    #                 refined_det = cluster_Q[score == score.max()]  # single or multiple detections might be the representative members of a cluster
    #                 XYWH = [self.img.shape[1], self.img.shape[0], self.img.shape[1], self.img.shape[0]]
    #                 if (len(refined_det) > 1):  # case for multiple representative
    #                     dist = []
    #                     # box feature
    #                     mode_points = refined_det[:, [0, 1, 6, 7]] / XYWH  # [1920,1080,1920,1080]
    #                     #mode_points = refined_det[:, [0, 1]] / XYWH[0:2]
    #                     for i in range(len(refined_det)):
    #                         dist = np.append(dist, distance.euclidean(cluster_center, mode_points[i]))
    #                     refined_det = refined_det[dist.argmin(), :]  # select closest one from multiple representative
    #                     refined_det[8] = cluster_prob_score  # * refined_det[7]  # det score weighted by cluster probability
    #                 else:
    #                     refined_det = refined_det[:, :][0]
    #                     refined_det[8] = cluster_prob_score  # * refined_det[7] # det score weighted by cluster probability
    #             elif len(Q_angle_0)==1:
    #                 refined_det=Q_angle_0[0]
    #                 refined_det[8] = cluster_prob_score
    #             elif len(Q_angle_0)>1:
    #                 refined_det=Q_angle_0[np.argmax(Q_angle_0[:,8]),:]
    #                 refined_det[8] = cluster_prob_score
    #             # refine cluster based on cluster size
    #             if counts[j] >= 1 and cluster_prob_score>=0:
    #                 refined_frame.append(refined_det[2:11])
    #                 score = '%.3f' % refined_det[8]
    #                 if self.vis:
    #                     #self.img = cv2.circle(self.img, (int(refined_det[0]), int(refined_det[1])), 5, color, -10)
    #                     cv2.putText(self.img, str(score), (int(refined_det[4] ), int(refined_det[5])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)
    #                     self.img = cv2.rectangle(self.img, (int(refined_det[4]), int(refined_det[5])), (int(refined_det[4] + refined_det[6]), int(refined_det[5] + refined_det[7])), color, 4)
    #
    #     return np.array(refined_frame)

    def predict_modes(self, det_set, det_ind, mode_color, score_thr):
        # remapped box feature : [CXbox, CYbox,fr,detBox[1], x, y, w, h, score, classID, angle]
        box_center_remapped = self.detection_feature(det_set)
        #normalize 4D [cx,cy,w,h] feature
        if len(box_center_remapped) >= 2:
            feature_norm = self.normalize_feature(box_center_remapped,self.img)
            # apply mean-shift clustering
            cleaned_detection = self.frame_detection_clustering(feature_norm, box_center_remapped,
                                                                det_ind=det_ind,
                                                                score_thr=score_thr,
                                                                color=mode_color)
        else:
            cleaned_detection = []
        return cleaned_detection


    def get_modes(self):
        cleaned_detection_p = self.predict_modes(det_set=self.det_at_t_p,
                                                 det_ind=self.det_indexsp,
                                                 mode_color=self.color_blue,
                                                 score_thr=self.cluster_score_thrp)
        self.modes_pb.append(cleaned_detection_p)
        if len(cleaned_detection_p)>0 and self.save_modes:
            if self.save_modes:
                for det in cleaned_detection_p:
                    self.det_for_mht.writelines(
                        str(det[0]) + ',' + str(det[1]) + ',' + str(det[2]) + ',' + str(det[3]) + ',' + str(
                            det[4]) + ',' + str(det[5]) + ',' + str(det[6]) + ',' + str(det[7]) + ',' + str(det[8]) + '\n')

        cleaned_detection_b = self.predict_modes(det_set=self.det_at_t_b,
                                                    det_ind=self.det_indexsb,
                                                    mode_color=self.color_blue,
                                                    score_thr=self.cluster_score_thrb)
        self.modes_pb.append(cleaned_detection_b)
        if len(cleaned_detection_b)>0 and self.save_modes:
            for det in cleaned_detection_b:
                self.det_for_mht.writelines(
                    str(det[0]) + ',' + str(det[1]) + ',' + str(det[2]) + ',' + str(det[3]) + ',' + str(
                        det[4]) + ',' + str(det[5]) + ',' + str(det[6]) + ',' + str(det[7]) + ',' + str(det[8]) + '\n')
        if self.vis:
            output_name = os.path.basename(self.im_name)
            if self.ann_dir:
                self.img = cv2.hconcat(
                    [self.img, cv2.imread(os.path.join(self.ann_dir, 'rgbMasksSplit', os.path.basename(self.im_name).split('.')[0]+'.png'))])
            cv2.imwrite(os.path.join(self.out_path, output_name), self.img)

        if self.final_cluster_indexs:
            det_indexs = np.concatenate([cluster for cluster in self.final_cluster_indexs])
            cluster_modes = np.concatenate([cluster for cluster in self.modes_pb if len(cluster) > 0])
        else:
            det_indexs = []
            cluster_modes = []
        return det_indexs, self.save_scores, cluster_modes
