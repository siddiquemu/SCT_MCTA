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
import matplotlib.pyplot as plt
import os
import sys
#import skimage.io
import numpy as np
from sklearn.cluster import MeanShift, estimate_bandwidth
#from sklearn.datasets.samples_generator import make_blobs
import MCD.mean_shift as ms
from scipy.spatial import distance

class Cluster_Mode(object):
    def __init__(self, detPB=None, frame=None, angle_set=[0], img=None, out_path=None,
                 save_file=None, num_class=None, vis=False, skip_nms=None):
        self.detPB =  detPB.cpu().numpy()
        self.num_class = num_class
        print('total det_remapped: {}'.format(detPB.shape))

        if self.num_class>1:
            self.det_at_t_p = detPB[detPB[:, 7] == 1, :]  # person class
            self.det_at_t_b = detPB[detPB[:, 7] > 1, :]  #backpack, handbag, suitcase classes
        else:
            self.det_at_t_p = self.detPB
        self.color_p = (5, 0, 255)  # blue
        self.color_b = (255, 0, 255) # magenta
        self.color_remap = (0, 255, 0)
        self.frame = frame
        self.vis = vis
        if self.vis:
            self.im_name = img
            self.img = cv2.imread(self.im_name)
        self.n_angles = len(angle_set)
        self.out_path = out_path
        self.file = save_file
        self.skip_nms=skip_nms

    @staticmethod
    def normalize_feature(X, img):
        #4D feature: [cx, cy, w, h]
        #TODO: w, h normalized by max value in frame or by img width, height
        #[img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
        X_scaled = X[:, [0, 1, 6, 7]] / [img.shape[1], img.shape[0], img.shape[1], img.shape[0]]
        return X_scaled

    # prepare feature for using in Mean-Shift Clustering
    def detection_feature(self, det_Set):
        box_center_feature = []
        # rotation angles at which detection founds
        #angle_set = np.int16(det_Set[:, 8])
        # remap all bbox to original image from multiple inference
        box_center = []
        for i, detBox in enumerate(det_Set):
            # [0,3,6,12,348,351,354,357]
            #print(angle)
            x = detBox[0]
            y = detBox[1]
            w = detBox[2]-detBox[0]
            h = detBox[3]-detBox[1]
            score = detBox[4]
            #classID = detBox[7]
            #angle = detBox[8]

            CXbox = x #+ w / 2.
            CYbox = y #+ h / 2.
            if (w < 1000 and h < 1000):
                box_center.append([CXbox, CYbox,self.frame, i, x, y, w, h, score])
                # all remapped detection on original image
                if self.vis:
                    self.img = cv2.rectangle(self.img, (int(x), int(y)), (int(x + w), int(y + h)), self.color_remap, 2)
        return np.array(box_center)

    def frame_detection_clustering(self, feature_norm, box_features, color=None):
        # set of bbox centers
        feature = feature_norm  # norm feature (0-1)
        det_frame = box_features # contain all info [CXbox,CYbox,fr,instance_index, x, y, w, h, score,classID, angle]
        #feature = det_frame[:, [0, 1, 6, 7]] / [max(det_frame[:, [0]])[0], max(det_frame[:, [1]])[0],
                                                #max(det_frame[:, [6]])[0], max(det_frame[:, [7]])[0]]

        assert len(feature)==len(det_frame)
        # to use multivariate gaussian kernel
        # Make sure the dimensions of 'data' and the kernel match
        # '''
        # '''
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

        # nothing mention about mean??

        #mean_shifter = ms.MeanShift(kernel='multivariate_gaussian')
        #mean_shift_result = mean_shifter.cluster(feature[:, 0:4], kernel_bandwidth=[x_kernel, y_kernel,w_kernel, h_kernel])#w_kernel, h_kernel
        #shifted_points = mean_shift_result.shifted_points
        #labels = mean_shift_result.cluster_ids
        '''
        from mpl_toolkits import mplot3d
        import matplotlib.pyplot as plt
        from scipy.stats import multivariate_normal
        x = np.linspace(-1, 1, 100)
        y = np.linspace(0, 2, 100)
        X, Y = np.meshgrid(x, y)
        pos = np.dstack((X, Y))
        mu = np.mean(np.transpose(feature[:,0:2]),axis=1)
        cov = np.cov(np.transpose(feature[:,0:2]))
        rv = multivariate_normal(mu, cov)
        Z = rv.pdf(pos)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(X, Y, Z)
        fig.savefig('/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d6/tracking_wo_bnw/mv_panet_det.png', dpi=200)
        '''

        ms = MeanShift(bandwidth=0.08, bin_seeding=True, cluster_all=True)  # ,
        # Mean-Shift Model Fitting
        ms.fit(feature)
        labels = ms.labels_
        # print('Labels:', labels)
        cluster_centers = ms.cluster_centers_



        #labels = labels[np.where(labels >= 0)]
        labels_unique = np.unique(labels)
        n_clusters_ = len(labels_unique)
        print("number of estimated clusters : %d" % n_clusters_)

        unique, counts = np.unique(labels, return_counts=True)
        refined_frame = []
        for j in labels_unique:  # cluster wise loop
            if j!=-1:
                cluster_Q = det_frame[labels == j, :]
                # TODO: Why the first shifted point is selected as centroids??? must correct this idea......
                #cluster_center = np.mean(shifted_points[labels == j, :], axis=0) #shifted_points[labels == j, :][0, 0]  # why the first
                cluster_center = cluster_centers[j]
                if self.skip_nms:
                    cluster_prob_score = np.sum(cluster_Q[:, 8]) / np.sum(det_frame[:, 8])
                else:
                    cluster_prob_score = np.sum(cluster_Q[:, 8]) / self.n_angles
                # Apply threshold
                #print('Cluster Score: >>>>> ', cluster_prob_score)
                # Apply NMS instead of hard thresholding
                score = np.around(cluster_Q[:, 8], decimals=2)  # cluster detection score rounded upto two decimal points
                refined_det = cluster_Q[score == score.max()]  # single or multiple detections might be the representative members of a cluster
                XYWH = [self.img.shape[1], self.img.shape[0], self.img.shape[1], self.img.shape[0]]
                if (len(refined_det) > 1):  # case for multiple representative
                    dist = []
                    # box feature
                    mode_points = refined_det[:, [0, 1, 6, 7]] / XYWH  # [1920,1080,1920,1080]
                    for i in range(len(refined_det)):
                        dist = np.append(dist, distance.euclidean(cluster_center, mode_points[i]))
                    refined_det = refined_det[dist.argmin(), :]  # select closest one from multiple representative
                    refined_det[8] = cluster_prob_score  # * refined_det[7]  # det score weighted by cluster probability
                else:
                    refined_det = refined_det[:, :][0]
                    refined_det[8] = cluster_prob_score  # * refined_det[7] # det score weighted by cluster probability
                # refine cluster based on cluster size
                if counts[j] >= 1 and cluster_prob_score>=0:
                    refined_frame.append(refined_det[2::])
                    score = '%.3f' % refined_det[8]
                    if self.vis:
                        cv2.putText(self.img, str(score), (int(refined_det[4] ), int(refined_det[5])), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 4)
                        self.img = cv2.rectangle(self.img, (int(refined_det[4]), int(refined_det[5])), (int(refined_det[4] + refined_det[6]), int(refined_det[5] + refined_det[7])), color, 4)

        return np.array(refined_frame)

    def predict_modes(self, det_set, mode_color):
        # remapped box feature : [CXbox, CYbox,fr,detBox[1], x, y, w, h, score, classID, angle]
        box_center_remapped = self.detection_feature(det_set)
        #normalize 4D [cx,cy,w,h] feature
        if len(box_center_remapped) >= 4:
            feature_norm = self.normalize_feature(box_center_remapped, self.img)
            # apply mean-shift clustering
            cleaned_detection = self.frame_detection_clustering(feature_norm, box_center_remapped, color=mode_color)
        else:
            cleaned_detection = []
        return cleaned_detection

    def get_modes(self):
        if self.num_class==1:
            cleaned_detection_p = self.predict_modes(det_set=self.det_at_t_p, mode_color=self.color_p)
            cleaned_detection_b = []
        else:
            cleaned_detection_p = self.predict_modes(det_set=self.det_at_t_p, mode_color=self.color_p)
            cleaned_detection_b = self.predict_modes(det_set=self.det_at_t_b, mode_color=self.color_b)

        if self.vis:
            output_name = os.path.basename(self.im_name)
            cv2.imwrite(os.path.join(self.out_path, output_name), self.img)

        if len(cleaned_detection_p)>0:
            if self.file:
                for det in cleaned_detection_p:
                    self.file.writerow(det)

        if len(cleaned_detection_b)>0:
            if self.file:
                for det in cleaned_detection_b:
                    self.file.writerow(det)
        return cleaned_detection_p, cleaned_detection_b

