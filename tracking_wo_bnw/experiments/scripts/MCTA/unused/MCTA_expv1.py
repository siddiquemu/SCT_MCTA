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
from tracklet_formation import form_tracklets
from Tracker_Merge_SingleCam import *
import collections
np.set_printoptions(threshold=sys.maxsize)

__version__ = 0.5


def camera_intrinsics(cam):
    # currently used camera: 2,4,5,9,11,13
    # A: 3*3 camera matrix
    # dist_coeff: distortion coefficient k1, k2
    if cam == 9:
        A = np.array([[1217.6, 0.0, 972.3],
                      [0, 1217.8, 550.9],
                      [0.0, 0.0, 1.0]])
        dist_coeff = np.array([-0.3235, 0.0887, 0, 0])  # why 4 or 5 coefficients are used in opencv????
    if cam == 11:  # assume that 9 and 11 have similar distortion
        A = np.array([[1217.6, 0.0, 972.3],
                      [0, 1217.8, 550.9],
                      [0.0, 0.0, 1.0]])
        dist_coeff = np.array([-0.3235, 0.0887, 0, 0])  # why 4 or 5 coefficients are used in opencv????
    if cam == 13:  # assume that 9 and 11 have similar distortion
        A = np.array([[1217.6, 0.0, 972.3],
                      [0, 1217.8, 550.9],
                      [0.0, 0.0, 1.0]])
        dist_coeff = np.array([-0.3235, 0.0887, 0, 0])  # why 4 or 5 coefficients are used in opencv????

    if cam == 14:  # assume that 9 and 11 have similar distortion
        A = np.array([[1217.6, 0.0, 972.3],
                      [0, 1217.8, 550.9],
                      [0.0, 0.0, 1.0]])
        dist_coeff = np.array([-0.3235, 0.0887, 0, 0])  # why 4 or 5 coefficients are used in opencv????

    if cam == 2:
        A = np.array([[1216.5, 0.0, 989.0],
                      [0, 1214.5, 595.0],
                      [0.0, 0.0, 1.0]])
        dist_coeff = np.array([-0.3655, 0.1296, 0, 0])  # why 4 or 5 coefficients are used in opencv????
    if cam == 5:  # assume that c2 and c5 have similar distortion
        A = np.array([[1216.5, 0.0, 989.0],
                      [0, 1214.5, 595.0],
                      [0.0, 0.0, 1.0]])
        dist_coeff = np.array([-0.3655, 0.1296, 0, 0])  # why 4 or 5 coefficients are used in opencv????
    if cam == 4:  # assume that c2 and c5 have similar distortion
        A = np.array([[1216.5, 0.0, 989.0],
                      [0, 1214.5, 595.0],
                      [0.0, 0.0, 1.0]])
        dist_coeff = np.array([-0.3655, 0.1296, 0, 0])  # why 4 or 5 coefficients are used in opencv????

    return dist_coeff, A


def undistorted_coords(trklt, dist_coeff, A):
    # use a copy of traclet centroids to convert into undistorted format
    # ***A: Camera Intrinsic Matrix
    # https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html
    # https: // www.mathworks.com / help / vision / ref / estimatecameraparameters.html
    # new camMatrix
    im_shape = (1920, 1080)  # img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(A, dist_coeff, im_shape, 1, im_shape)
    trklt = cv2.undistortPoints(trklt.reshape(trklt.shape[0], 1, 2), A, dist_coeff, 0, newcameramtx)
    return trklt.reshape(trklt.shape[0], 2)


def filter_tracklets(trackers, min_len=60):
    # Remove tracklets with <60 detections
    return [tl for tl in trackers if len(tl) >= min_len]


def get_tracklets(trackers):
    # when it is needed to sort trackelts detections in terms of time-stamps
    tracklets = []
    for tid in range(len(trackers)):
        # tracker_i = trackers.get(str(tid + 1))
        tracker_i = trackers[tid]
        tracker_i = np.array(list(tracker_i))
        tracker_i = np.array(sorted(tracker_i, key=lambda x: x[0]))
        tracklets.append(tracker_i)
    return tracklets


def convert_centroids(tracklet):
    # [fr,x,y,w,h] = [fr,x+w/2.y+h/2,w,h]
    for tl in tracklet:
        for bb in tl:
            bb[1] = bb[1] + bb[3] / 2.0
            bb[2] = bb[2] + bb[4] / 2.0
    return tracklet


def centroids2xywh(tracklet):
    for tl in tracklet:
        for bb in tl:
            bb[1] = bb[1] - bb[3] / 2.0
            bb[2] = bb[2] - bb[4] / 2.0
    return tracklet


def applyTransform(source_corners, H):
    dest_corners = np.empty(2)
    w = H[2][0] * source_corners[0] + H[2][1] * source_corners[1] + H[2][2] * 1
    dest_corners[0] = (H[0][0] * source_corners[0] + H[0][1] * source_corners[1] + H[0][2] * 1) / w
    dest_corners[1] = (H[1][0] * source_corners[0] + H[1][1] * source_corners[1] + H[1][2] * 1) / w
    return dest_corners


# ---------------ok siddique
# TODO: apply undistorted coords on H or undistort coords after applying H: here H is based distorted image
# try undistorted image to compute homography
def project_tracklets(in_tracklets, H, isDistorted=False, cam=9):
    out_tracklets = []
    active_trklt = []
    Hv = np.copy(H)
    Hv[0, 2], Hv[1, 2] = 0, 0
    # [fr,cx,cy,x,y,vx,vy]<<< projected coordinates: center is used
    for i, trklt in enumerate(in_tracklets):
        if isDistorted:
            dist_coeff, A = camera_intrinsics(cam)
            xy = np.copy(trklt[:, 1:3])
            xy = undistorted_coords(xy, dist_coeff, A)
            trklt[:, 1:3] = xy
        for bb in trklt:
            bbt1 = applyTransform(bb[1:3], H)
            cxcy = applyTransform(bb[1:3] + bb[3:5] / 2.0, H)
            vxvy = applyTransform(bb[5:7], Hv)
            bbt2 = cxcy - bbt1  # projected w/2,h/2
            bb[1:7] = np.concatenate([cxcy, bbt2, vxvy])  # projected: [cx,cy,w/2,h/2,vx,vy]
        # Delete tracklets that don't have any detection visible in the second camera
        # TODO: There must be a smarter and faster way to do that
        # cy+h/2 > 0 and w/2>0
        if max(trklt[:, 2] - trklt[:, 4]) > 0 and max(trklt[:, 3]) > 0:
            out_tracklets.append(trklt)
            active_trklt.append(i)
    return out_tracklets, active_trklt


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


def expand_from_temporal_list(box_all=None, mask_30=None):
    if box_all is not None:
        box_list = [b for b in box_all if len(b) > 0]
        box_all = np.concatenate(box_list)
    if mask_30 is not None:
        mask_list = [m for m in mask_30 if len(m) > 0]
        masks_30 = np.concatenate(mask_list)
    else:
        masks_30 = []
    return box_all, masks_30


def augment_feature_with_velocity(trackletAux, trackletPri):
    trackletAuxDelayed = np.insert(trackletAux, 0, np.array([trackletAux[0, 0] - 1, 0, 0]), axis=0)
    trackletPriDelayed = np.insert(trackletPri, 0, np.array([trackletPri[0, 0] - 1, 0, 0]), axis=0)

    # [delta.Cx/delta.t,delta.Cy/delta.t]
    # Due to the smaller overlap between camera pairs, Vy changes are dissimilar
    trackletAuxVelocity = (trackletAux[:, 1:3] - trackletAuxDelayed[:, 1:3][:-1, :]) / np.transpose(
        [trackletAux[:, 0] - trackletAuxDelayed[:, 0][:-1],
         trackletAux[:, 0] - trackletAuxDelayed[:, 0][:-1]])
    trackletPriVelocity = (trackletPri[:, 1:3] - trackletPriDelayed[:, 1:3][:-1, :]) / np.transpose(
        [trackletPri[:, 0] - trackletPriDelayed[:, 0][:-1],
         trackletPri[:, 0] - trackletPriDelayed[:, 0][:-1]])
    # size check
    assert trackletAuxVelocity.shape[0] == trackletAux.shape[
        0], 'number of observation in velocity vector and centroids vector shuld same' \
            ' but found {},{} respectively'.format(trackletAuxVelocity.shape[0], trackletAux.shape[0])
    # append to form [fr,Cx,Cy,dot.Cx,dot.Cy]
    trackletAuxAug = np.append(trackletAux, trackletAuxVelocity, axis=1)
    trackletPriAug = np.append(trackletPri, trackletPriVelocity, axis=1)
    return trackletAuxAug[:, 0:5], trackletPriAug[:, 0:5]


def mahalanobis21(final_tracklet1, final_tracklet2):
    diff = final_tracklet1[:, 1:6] - final_tracklet2[:, 1:6]
    X12 = np.stack((final_tracklet1[:, 1:6], final_tracklet2[:, 1:6]), axis=1)
    iv = np.linalg.inv(np.cov(X12.reshape(X12.shape[0], 8)))
    MD = np.linalg.norm(np.sqrt(np.diag(np.dot(np.dot(diff.T, iv), diff))))
    return MD


def refine_cost_row(cost, tilde_t_pa, common_t_stamps, indP):
    # refine the cost for each primary with all auxiliary
    i = indP
    if min(common_t_stamps) in tilde_t_pa[i, :]:
        if len(cost[i, :][tilde_t_pa[i, :] == min(common_t_stamps)]) > 1:
            # cost[:, j][tilde_t_pa[:, j]==min(common_t_stamps)] = np.inf
            MultiAssoCost = cost[i, :][tilde_t_pa[i, :] == min(common_t_stamps)]
            multiAssoInd = np.where(tilde_t_pa[i, :] == min(common_t_stamps))[0]
            # for hausdorff distance, minimum cost is not always the required solution.???: worst case- +ve pair cost> -ve pair
            cost[i, :][multiAssoInd[MultiAssoCost > min(MultiAssoCost)]] = np.inf
            tilde_t_pa[i, :][multiAssoInd[MultiAssoCost > min(MultiAssoCost)]] = np.inf
            # tilde_t_pa[i, j] = min(common_t_stamps)
    return cost, tilde_t_pa


def refine_cost_column(cost, tilde_t_pa, common_t_stamps, indA):
    # refine the cost for each primary with all auxiliary
    j = indA
    if min(common_t_stamps) in tilde_t_pa[:, j]:
        if len(cost[i, :][tilde_t_pa[:, j] == min(common_t_stamps)]) > 1:
            # cost[:, j][tilde_t_pa[:, j]==min(common_t_stamps)] = np.inf
            MultiAssoCost = cost[:, j][tilde_t_pa[:, j] == min(common_t_stamps)]
            multiAssoInd = np.where(tilde_t_pa[:, j] == min(common_t_stamps))[0]
            # for hausdorff distance, minimum cost is not always the required solution.???: worst case- +ve pair cost> -ve pair
            cost[:, j][multiAssoInd[MultiAssoCost > min(MultiAssoCost)]] = np.inf
            tilde_t_pa[:, j][multiAssoInd[MultiAssoCost > min(MultiAssoCost)]] = np.inf
            # tilde_t_pa[i, j] = min(common_t_stamps)
    return cost, tilde_t_pa


def associate_tracklets_DFS(cam1, cam2to1_tracklets, cam1Tracklets, cam2Tracklets,
                            active_trklt, isDistorted=False, max_dist=400):
    # cam2to1_tracklets = [cam1,cam2]
    print('Total tracklets in a camera pairs {}'.format(len(cam2to1_tracklets)))
    # cost = np.empty((len(cam2to1_tracklets), len(cam2to1_tracklets)))
    cost = np.ones((len(cam2to1_tracklets), len(cam2to1_tracklets))) * np.inf
    tilde_t_pa = np.ones((len(cam2to1_tracklets), len(cam2to1_tracklets)), dtype='float') * np.inf
    # it = np.nditer(cost, op_flags=['readwrite'])
    # it_cosine = np.nditer(cosine_cost, op_flags=['readwrite'])
    i = 0
    matches = []
    for tracklet1 in cam2to1_tracklets:
        j = 0
        for tracklet2 in cam2to1_tracklets:
            print('Processing tracklets {} and {}'.format(i, j))
            # condition to check multicamera trackets have any overlap
            if i != j and i < len(cam1Tracklets) and j >= len(cam1Tracklets) and j in len(cam1Tracklets) + np.array(
                    active_trklt):
                if tracklet1[-1, 0] < tracklet2[0, 0] or tracklet1[0, 0] > tracklet2[
                    -1, 0]:  # cam1_end<cam2_start or cam1_start>cam2_end
                    # it[0] = np.inf
                    cost[i, j] = np.inf
                # Search for the overlapping portions of multicamera tracklets
                else:
                    print 'Trackets are overlapped'
                    # TODO: There must be a more compact way of doing this (find the overlapped region)
                    # TODO: Assess whether the timestamp should be considered in the distance computation as well
                    common_t_stamps = np.array(list(set(list(tracklet2[:, 0])).intersection(list(tracklet1[:, 0]))))
                    assert len(common_t_stamps) > 0, 'Non-overlaped T_p, T_a can not be associated'
                    # tilde_t_pa[j, i] = min(common_t_stamps)
                    # 2D or 4D feature: [fr,cx,cy,w/2,h/2,5-vx,6-vy]
                    # augment feature (4D) with velocity (aux,primary): [Cx,Cy,dot.Cx,dot.Cy]
                    # tracklet2,tracklet1 = augment_feature_with_velocity(tracklet2[:,0:3], tracklet1[:,0:3])
                    featIndx = [0, 1, 2, 5, 6]  # ,5,6]
                    mask1 = np.isin(tracklet1[:, 0], common_t_stamps)
                    final_tracklet1 = tracklet1[mask1][:, featIndx]
                    if isDistorted:
                        dist_coeff, A = camera_intrinsics(cam=cam1)  # 2:c5 or 9:c11
                        xy = np.copy(final_tracklet1[:, 1:3])
                        xy = undistorted_coords(xy, dist_coeff, A)
                        final_tracklet1[:, 1:3] = xy
                    # compute the location from velocity and time difference
                    final_tracklet1[:, 1:5] = final_tracklet1[:, 1:5] / [1920.0, 1080.0, 50.0,
                                                                         50.0]  # max(abs(final_tracklet1[:,3]))

                    mask2 = np.isin(tracklet2[:, 0], common_t_stamps)
                    final_tracklet2 = tracklet2[mask2][:, featIndx]
                    final_tracklet2[:, 1:5] = final_tracklet2[:, 1:5] / [1920.0, 1080.0, 50.0, 50.0]

                    d_h = directed_hausdorff(final_tracklet1[:, 1:3], final_tracklet2[:, 1:3])[0]
                    dOverlap = 0.3  # final_tracklet1.shape[0]/float(max(tracklet1.shape[0],tracklet2.shape[0]))
                    # d_h = (1-dOverlap)*directed_hausdorff(final_tracklet1[:,1:3], final_tracklet2[:,1:3])[0]  \
                    # + (dOverlap)*cosine(final_tracklet2[:,3:5].flatten(), final_tracklet1[:,3:5].flatten())
                    # d_cos = 1-cosine(final_tracklet2.flatten(), final_tracklet1.flatten())
                    # d_mah = mahalanobis21(final_tracklet1, final_tracklet2)
                    if d_h <= max_dist:
                        # it[0] = d_h
                        cost[i, j] = d_h
                        tilde_t_pa[i, j] = min(common_t_stamps)
                        # tracklet orientation distance
                        # print 'Cosine distance between {} in cam9 and {} in cam2 is {}'.format(j,i,d_cos)
            else:
                cost[i, j] = np.inf
                # it[0] = np.inf
            # it.iternext()
            # it_cosine.iternext()
            j = j + 1
        # refine the cost for each primary with all auxiliary
        i = i + 1
    # DEBUG
    # import matplotlib.pyplot as plt
    # hist = np.histogram(cost[cost < 10000])
    # plt.hist(cost[cost < 10000])
    iteration = 0
    while len(cost[cost <= max_dist]) > 0:
        row_ind, col_ind = linear_sum_assignment_with_inf(cost)
        Mincosts = cost[row_ind, col_ind]
        idx = np.where(Mincosts <= max_dist)[0]
        for i in idx:
            matches.append((Mincosts[i], tilde_t_pa[row_ind[i], col_ind[i]], row_ind[i],
                            col_ind[i]))  # tilde_t_pa[row_ind[i],col_ind[i]]  cam2_tracklets[col_ind[i]][0,0]
            # t_p fixed> search over t_a to get any overlapping with T_a
            indx_t_a = np.where(cost[row_ind[i], :] <= max_dist)[0]
            if len(indx_t_a) > 0:
                T_a = cam2to1_tracklets[indx_t_a]
                isIntersect = [len(set(cam2to1_tracklets[col_ind[i]][:, 0]).intersection(t_a[:, 0])) > 0 for t_a in T_a]
                cost[row_ind[i], indx_t_a[isIntersect]] = np.inf

            indx_t_p = np.where(cost[:, col_ind[i]] <= max_dist)[0]
            if len(indx_t_p) > 0:
                T_p = cam2to1_tracklets[indx_t_p]
                isIntersect = [len(set(cam2to1_tracklets[row_ind[i]][:, 0]).intersection(t_p[:, 0])) > 0 for t_p in T_p]
                cost[indx_t_p[isIntersect], col_ind[i]] = np.inf

    return matches, tilde_t_pa


def associate_tracklets_DFS_MC(params, isDistorted=False, max_dist=400):
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
    for tracklet1 in params['PAproj']:
        j = 0
        for tracklet2 in params['PAproj']:
            # condition to check multicamera trackets have any overlap
            if i != j and i < params['trkltFamilySizeP'] and j >= params['trkltFamilySizeP'] \
                    and j in params['activeTrkltIndexsA']:
                # cam1_end<cam2_start or cam1_start>cam2_end
                if tracklet1[-1, 0] < tracklet2[0, 0] or tracklet1[0, 0] > tracklet2[-1, 0]:

                    cost[i, j] = np.inf
                # Search for the overlapping portions of multicamera tracklets
                else:
                    # TODO: There must be a more compact way of doing this (find the overlapped region)
                    # TODO: Assess whether the timestamp should be considered in the distance computation as well
                    common_t_stamps = np.array(list(set(list(tracklet2[:, 0])).intersection(list(tracklet1[:, 0]))))
                    assert len(common_t_stamps) > 0, 'Non-overlaped T_p, T_a can not be associated'
                    # tilde_t_pa[j, i] = min(common_t_stamps)
                    # 2D or 4D feature: [fr,cx,cy,w/2,h/2,5-vx,6-vy]
                    # augment feature (4D) with velocity (aux,primary): [Cx,Cy,dot.Cx,dot.Cy]
                    # tracklet2,tracklet1 = augment_feature_with_velocity(tracklet2[:,0:3], tracklet1[:,0:3])
                    featIndx = [0, 1, 2, 5, 6]  # ,5,6]
                    mask1 = np.isin(tracklet1[:, 0], common_t_stamps)
                    final_tracklet1 = tracklet1[mask1][:, featIndx]
                    if isDistorted:
                        dist_coeff, A = camera_intrinsics(cam=params['Pid'])  # 2:c5 or 9:c11
                        xy = np.copy(final_tracklet1[:, 1:3])
                        xy = undistorted_coords(xy, dist_coeff, A)
                        final_tracklet1[:, 1:3] = xy
                    # compute the location from velocity and time difference
                    final_tracklet1[:, 1:5] = final_tracklet1[:, 1:5] / [1920.0, 1080.0, 50.0,
                                                                         50.0]  # max(abs(final_tracklet1[:,3]))

                    mask2 = np.isin(tracklet2[:, 0], common_t_stamps)
                    final_tracklet2 = tracklet2[mask2][:, featIndx]
                    final_tracklet2[:, 1:5] = final_tracklet2[:, 1:5] / [1920.0, 1080.0, 50.0, 50.0]

                    d_h = directed_hausdorff(final_tracklet1[:, 1:3], final_tracklet2[:, 1:3])[0]
                    print('Processing tracklets {} and {} with cost {}'.format(i, j, d_h))
                    dOverlap = 0.3  # final_tracklet1.shape[0]/float(max(tracklet1.shape[0],tracklet2.shape[0]))
                    # d_h = (1-dOverlap)*directed_hausdorff(final_tracklet1[:,1:3], final_tracklet2[:,1:3])[0]  \
                    # + (dOverlap)*cosine(final_tracklet2[:,3:5].flatten(), final_tracklet1[:,3:5].flatten())
                    # d_cos = 1-cosine(final_tracklet2.flatten(), final_tracklet1.flatten())
                    # d_mah = mahalanobis21(final_tracklet1, final_tracklet2)
                    if d_h <= max_dist:
                        cost[i, j] = d_h
                        tilde_t_pa[i, j] = min(common_t_stamps)
            else:
                cost[i, j] = np.inf
            j = j + 1
        i = i + 1
    # DEBUG
    # import matplotlib.pyplot as plt
    # hist = np.histogram(cost[cost < 10000])
    # plt.hist(cost[cost < 10000])
    while len(cost[cost <= max_dist]) > 0:
        row_ind, col_ind = linear_sum_assignment_with_inf(cost)
        Mincosts = cost[row_ind, col_ind]
        idx = np.where(Mincosts <= max_dist)[0]
        for i in idx:
            matches.append((Mincosts[i], tilde_t_pa[row_ind[i], col_ind[i]], row_ind[i],
                            col_ind[i]))  # tilde_t_pa[row_ind[i],col_ind[i]]  cam2_tracklets[col_ind[i]][0,0]

            # t_p fixed> search over t_a to get any overlapping with T_a
            indx_t_a = np.where(cost[row_ind[i], :] <= max_dist)[0]
            if len(indx_t_a) > 0:
                T_a = params['PAproj'][indx_t_a]
                isIntersect = [len(set(params['PAproj'][col_ind[i]][:, 0]).intersection(t_a[:, 0])) > 0
                               for t_a in T_a]
                cost[row_ind[i], indx_t_a[isIntersect]] = np.inf

            # t_p fixed> search over t_a to get any overlapping with T_a
            indx_t_p = np.where(cost[:, col_ind[i]] <= max_dist)[0]
            if len(indx_t_p) > 0:
                T_p = params['PAproj'][indx_t_p]
                isIntersect = [len(set(params['PAproj'][row_ind[i]][:, 0]).intersection(t_p[:, 0])) > 0
                               for t_p in T_p]
                cost[indx_t_p[isIntersect], col_ind[i]] = np.inf

    return matches, tilde_t_pa


def associate_tracklets(cam2to1_tracklets, cam1_tracklets, cam2_tracklets, max_dist=400):
    cost = np.empty((len(cam1_tracklets), len(cam2_tracklets)))
    # cosine_cost = np.empty((len(cam1_tracklets), len(cam2_tracklets)))
    tilde_t_pa = np.ones((len(cam1_tracklets), len(cam2_tracklets)), dtype='float') * np.inf
    it = np.nditer(cost, op_flags=['readwrite'])
    # it_cosine = np.nditer(cosine_cost, op_flags=['readwrite'])
    i = 0
    matches = []
    for tracklet1 in cam1_tracklets:
        j = 0
        for tracklet2 in cam2_tracklets:
            print('Processing tracklets {} and {}'.format(i, j))
            # Quick check if tracklets overlap
            # TODO: This condition (and the loop below) could be relaxed to allow a maximum offset between two
            #       tracklets. That could be useful to apply the same procedure for same camera association
            # TODO: need to consider the trackelts overlap in  both direction: cam2 to cam1 or cam1 to cam2..... tracking in world coordinate
            # condition to check trackets have any overlap
            if tracklet1[-1, 0] < tracklet2[0, 0] or tracklet1[0, 0] > tracklet2[
                -1, 0]:  # cam1_end<cam2_start or cam1_start>cam2_end
                it[0] = np.inf
            # Search for the overlapping portions of both tracklets
            else:
                print 'Trackets are overlapped'
                # TODO: There must be a more compact way of doing this (find the overlapped region)
                '''
                s = [0, 0]
                e = [len(tracklet1) - 1, len(tracklet2) - 1]

                while s[0] < len(tracklet1) and tracklet1[s[0]][0] < tracklet2[s[1]][0]:
                    s[0] = s[0] + 1
                while s[1] < len(tracklet2) and tracklet2[s[1]][0] < tracklet1[s[0]][0]:
                    s[1] = s[1] + 1
                while e[0] > 0 and tracklet1[e[0]][0] > tracklet2[e[1]][0]:
                    e[0] = e[0] - 1
                while e[1] >0 and tracklet2[e[1]][0] > tracklet1[e[0]][0] and e[1] >= 0:
                    e[1] = e[1] - 1
                # This paper uses Hausdorff distances: https://www.ijcai.org/Proceedings/16/Papers/479.pdf
                #it[0] = directed_hausdorff(tracklet1[s[0]:e[0], 1:], tracklet2[s[1]:e[1], 1:])[0]
                it[0] = directed_hausdorff(tracklet1[s[0]:e[0]], tracklet2[s[1]:e[1]])[0]
                '''
                # TODO: Assess whether the timestamp should be considered in the distance computation as well
                common_t_stamps = np.array(list(set(list(tracklet2[:, 0])).intersection(list(tracklet1[:, 0]))))
                tilde_t_pa[i, j] = min(common_t_stamps)
                mask1 = np.isin(tracklet1[:, 0], common_t_stamps)
                final_tracklet1 = tracklet1[mask1]

                mask2 = np.isin(tracklet2[:, 0], common_t_stamps)
                final_tracklet2 = tracklet2[mask2]
                d_h = directed_hausdorff(final_tracklet1[:, 1:3], final_tracklet2[:, 1:3])[0]
                d_cos = cosine(final_tracklet2[:, 1:3].flatten(), final_tracklet1[:, 1:3].flatten())
                if d_h < max_dist:
                    it[0] = d_h
                    # it_cosine[0] = cosine(final_tracklet2[:,1:3].flatten(), final_tracklet1[:,1:3].flatten())
                    # tracklet orientation distance
                    print 'Cosine distance between {} in cam9 and {} in cam2 is {}'.format(j, i, d_cos)
                else:
                    it[0] = np.inf
            it.iternext()
            # it_cosine.iternext()
            j = j + 1
        i = i + 1
    # DEBUG
    # hist = np.histogram(cost[cost < 10000])
    # plt.hist(cost[cost < 10000])
    while len(cost[cost < max_dist]) > 0:
        row_ind, col_ind = linear_sum_assignment_with_inf(cost)
        mcost = cost[row_ind, col_ind]
        idx = np.where(mcost < max_dist)[0]
        for i in idx:
            matches.append((mcost[i], tilde_t_pa[row_ind[i], col_ind[i]], row_ind[i],
                            col_ind[i]))  # tilde_t_pa[row_ind[i],col_ind[i]]  cam2_tracklets[col_ind[i]][0,0]
            # for multiple association restrict currently associated pairs from further association
        cost[row_ind, col_ind] = np.inf
    return matches


def plot_trajectories(label_map, fr_start, fr_end, cam2to1_tracklets, tracklet_cam9, tracklets,
                      fr_offsetP, fr_offsetA, folder, folder2, out_path, cam2cam, cams, vis_rate=30,
                      isDistorted=False, metrics=False, save_imgs=False, motion=False):
    # TODO: Figure out why I need the reshape here
    labelCam1 = dict(list(label_map.items())[:len(tracklets)])
    labelCam2 = dict(list(label_map.items())[len(tracklets):])
    color = np.column_stack([[255.0 * np.random.rand(1, len(cam2to1_tracklets))],
                             [255.0 * np.random.rand(1, len(cam2to1_tracklets))],
                             [255.0 * np.random.rand(1, len(cam2to1_tracklets))]]).reshape(3, len(cam2to1_tracklets))
    # BGR
    # magenta = (255,0,255)
    # yellow = (0,255,255)
    # green = (0,255,0)
    # blue = (255,0,0)
    # plt.imshow(np.ones((1080, 1920, 3)))
    # fr_offset = 48#11to13
    # fr_offset = 29  # 9to2, cam2 lag = 29
    if save_imgs:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 640, 480)
    for fr in range(fr_start, fr_end):
        # 9A:9 fr-15, 11 fr-15
        if (fr) % vis_rate == 0:  # 15 for 6A,7A,.. 3 for 5A, 5B.. 30 for 9A
            try:
                img = cv2.imread('{}{:06d}.png'.format(folder, fr + fr_offsetP))  # target camera 2,11 - folder:
                # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img2 = cv2.imread('{}{:06d}.png'.format(folder2,
                                                        fr + fr_offsetA))  # source camera 9,5 - folder2: + fr_offset for 9to2, + fr_offset for 5to11
                # visualize the projections on primary camera (11,2)
                c = len(tracklets)
                for bb in cam2to1_tracklets[len(tracklets):]:  # show projection of auxiliary camera
                    # TODO: Set the colors of the rectangles
                    for i in range(1, len(bb)):
                        if fr_start <= bb[i, 0] <= fr:
                            # image = cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength = 0.5)
                            if save_imgs and motion and bb[i, 0] % 5 == 0:
                                # 2d velocity plot
                                xx = int(bb[i, 1] + bb[i, 5] * 2 * (bb[i, 0] - bb[i - 1, 0]))  # xx=x0+vxt
                                yy = int(bb[i, 2] + bb[i, 6] * 2 * (bb[i, 0] - bb[i - 1, 0]))
                                cv2.arrowedLine(img, (int(bb[i, 1]), int(bb[i, 2])),
                                                (xx, yy),
                                                (255, 0, 255), thickness=4, tipLength=1)
                                # line between centroids
                                # cv2.arrowedLine(img, (int(bb[i - 1, 1]), int(bb[i - 1, 2])), (int(bb[i, 1]), int(bb[i, 2])),
                                # magenta, thickness=5, tipLength = 1)#lineType=8

                            # TODO: Think about how to show this only at the last detection in the current frame
                            if bb[i, 0] == fr:  # and len(label_map_aux[c])>0
                                cv2.putText(img, '{}'.format(labelCam2[c]), (int(bb[i - 1, 1]), int(bb[i - 1, 2])),
                                            cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 5, cv2.LINE_AA)
                    c = c + 1
                c = 0
                # visulaize the identity handover in primary camera
                for i_p, bb in enumerate(cam2to1_tracklets[:len(tracklets)]):  # primary camera
                    # To show the association with the projections (tracklets2)

                    for i in range(1, len(bb)):
                        if fr_start <= bb[i, 0] <= fr:
                            if save_imgs and motion and bb[i, 0] % 5 == 0:
                                # 2d velocity plot, bb[i,1]
                                xx = int(
                                    (bb[i, 1] + bb[i, 3] / 2.0) + bb[i, 5] * 2 * (bb[i, 0] - bb[i - 1, 0]))  # xx=x0+vxt
                                yy = int((bb[i, 2] + bb[i, 4] / 2.0) + bb[i, 6] * 2 * (bb[i, 0] - bb[i - 1, 0]))
                                cv2.arrowedLine(img, (int(bb[i, 1] + bb[i, 3] / 2.0), int(bb[i, 2] + bb[i, 4] / 2.0)),
                                                (xx, yy), (0, 255, 255),
                                                thickness=4, tipLength=1)
                                # line between centroids
                                # cv2.arrowedLine(img,
                                # (int(bb[i - 1, 1] + bb[i - 1, 3] / 2.0), int(bb[i - 1, 2] + bb[i - 1, 4] / 2.0)),
                                # (int(bb[i, 1] + bb[i, 3] / 2.0), int(bb[i, 2] + bb[i, 4] / 2.0)),
                                # (0, 255, 255), thickness=5, tipLength = 1)#lineType=8
                            if metrics:
                                if bb[i, 0] == fr:  # 9A:9 fr-15
                                    # if bb[i, 0]>=5415:
                                    # label_map[c]=0
                                    # Adjust frame delay and frame rate to find matching with GT
                                    # 6A:9,11>bb[i, 0]+30, 5A:9,11>bb[i, 0]+30, 7A:9>bb[i, 0],11>bb[i, 0]+30, 10A: 9,11>bb[i, 0]+30,9A: 9,11>bb[i, 0]+15,frdelay=15

                                    cv2.rectangle(img, (int(bb[i, 1]), int(bb[i, 2])),
                                                  (int(bb[i, 1] + bb[i, 3]), int(bb[i, 2] + bb[i, 4])),
                                                  (255, 0, 0), 5, cv2.LINE_AA)
                                    # show identity at undistorted centroid
                                    if isDistorted:
                                        dist_coeff, A = camera_intrinsics(cam=2)
                                        cxy_undist = np.copy(bb[i, 1:3] + bb[i, 3:5] / 2.0)
                                        cxy_undist = undistorted_coords(cxy_undist.reshape(1, 1, 2), dist_coeff, A)
                                    else:
                                        cxy_undist = np.copy(bb[i, 1:3] + bb[i, 3:5] / 2.0).reshape(1, 2)
                                    cv2.putText(img, 'P{}'.format(labelCam1[c]),
                                                (int(cxy_undist[0, 0]), int(cxy_undist[0, 1])),
                                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5, cv2.LINE_AA)
                                    if cam2cam == 'H5to11':
                                        track_for_mot.writelines(
                                            str((bb[i, 0] + 30) // 15) + ',' + str(labelCam1[c] + 1) + ',' + str(
                                                bb[i, 1]) + ',' + str(
                                                bb[i, 2]) + ',' + str(bb[i, 3]) + ',' + str(bb[i, 4]) + ',' +
                                            '-1' + ',' + '-1' + ',' + '-1' + ',' + '-1' + '\n')

                            else:
                                if bb[i, 0] == fr and label_map[c] == -1:
                                    cv2.putText(img, '{}'.format(c), (int(bb[i - 1, 1]), int(bb[i - 1, 2])),
                                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5, cv2.LINE_AA)
                                if bb[i, 0] == fr and label_map[c] != -1:
                                    cv2.putText(img, '{}'.format(label_map[c]), (int(bb[i - 1, 1]), int(bb[i - 1, 2])),
                                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)

                    c = c + 1
                # visualize the actual target identity in auxiliary camera (9,5)
                c = len(tracklets)
                for bb in tracklet_cam9[len(tracklets):]:  # source camera : yellow
                    # TODO: Set the colors of the rectangles
                    for i in range(1, len(bb)):
                        if fr_start <= bb[i, 0] <= fr:
                            # image = cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength = 0.5)
                            if save_imgs and motion and bb[i, 0] % 5 == 0:
                                # 2d velocity plot
                                xx = int(
                                    (bb[i, 1] + bb[i, 3] / 2.0) + bb[i, 5] * 2 * (bb[i, 0] - bb[i - 1, 0]))  # xx=x0+vxt
                                yy = int((bb[i, 2] + bb[i, 4] / 2.0) + bb[i, 6] * 2 * (bb[i, 0] - bb[i - 1, 0]))
                                cv2.arrowedLine(img2, (int(bb[i, 1] + bb[i, 3] / 2.0), int(bb[i, 2] + bb[i, 4] / 2.0)),
                                                (xx, yy), (0, 255, 0),
                                                thickness=4, tipLength=1)
                                # line between centroids
                                # cv2.arrowedLine(img2,
                                # (int(bb[i - 1, 1] + bb[i - 1, 3] / 2.0), int(bb[i - 1, 2] + bb[i - 1, 4] / 2.0)),
                                # (int(bb[i, 1] + bb[i, 3] / 2.0), int(bb[i, 2] + bb[i, 4] / 2.0)),
                                # (0, 255, 0), thickness=5, tipLength = 1)#lineType=8
                            if metrics:
                                # TODO: Think about how to show this only at the last detection in the current frame
                                if bb[i, 0] == fr:
                                    cv2.putText(img2, '{}'.format(labelCam2[c]),
                                                (int(bb[i, 1] + bb[i, 3] / 2), int(bb[i, 2] + bb[i, 4] / 2)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 2555, 0), 5, cv2.LINE_AA)
                                    cv2.rectangle(img2, (int(bb[i, 1]), int(bb[i, 2])),
                                                  (int(bb[i, 1] + bb[i, 3]), int(bb[i, 2] + bb[i, 4])),
                                                  (255, 0, 0), 5, cv2.LINE_AA)
                                    if cam2cam == 'H9to2' and metrics:
                                        track_for_mot.writelines(
                                            str((bb[i, 0] + 30) // 15) + ',' + str(labelCam2[c] + 1) + ',' + str(
                                                bb[i, 1]) + ',' + str(
                                                bb[i, 2]) + ',' + str(bb[i, 3]) + ',' + str(bb[i, 4]) + ',' +
                                            '-1' + ',' + '-1' + ',' + '-1' + ',' + '-1' + '\n')
                    c = c + 1
                if cam2cam == 'H9to2':
                    final_img = cv2.vconcat([img2, img])
                if cam2cam == 'H2to5':
                    final_img = cv2.hconcat([img, img2])  # img: c5, img2: c2
                if cam2cam == 'H2to4':
                    final_img = cv2.hconcat([img, img2])  # img: c4, img2: c2
                else:
                    final_img = cv2.vconcat([img2, img])
                cv2.imshow("image", final_img)
                if save_imgs:
                    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(out_path + '/{:06d}.png'.format(fr), final_img)
                cv2.waitKey(30)

            except:
                print("Frame {} not found".format(fr))
                continue
    track_for_mot.close()


def match_tl(det1, det2, t_thr=150, d_thr=200):
    # Associate pairs of tracklets with maximum overlap between the last and first detections
    # using the Hungarian algorithm

    new_track = []
    prev_track = []
    matches = []

    for tr in det1:
        prev_track.append(tr[-1])
    for tr in det2:
        new_track.append(tr[0])

    # Compute the cost matrix IoU between each pair of last and first detections
    cost = np.full((len(prev_track), len(new_track)), np.inf)
    i = 0
    for prev_det in prev_track:
        j = 0
        for new_det in new_track:
            # This is the only difference from the single-camera function
            # TODO: Make both a common function that take this test as a parameter
            delta_t = new_det[0] - prev_det[0]
            dist = np.linalg.norm(new_det[1:5] - prev_det[1:5])
            if 0 < delta_t < t_thr and dist < d_thr:
                cost[i, j] = dist
            j = j + 1
        i = i + 1

    row_ind, col_ind = linear_sum_assignment_with_inf(cost)

    # Find the maximum IoU for each pair
    for i in row_ind:
        if cost[i, col_ind[i]] < 50:  # 1
            matches.append((cost[i, col_ind[i]], i, col_ind[i]))

    return matches


def merge_tracklets(filtered_tracker, unassigned_tracklets):
    # Iterate a few times to join pairs of tracklets that correspond to multiple
    # fragments of the same track
    # TODO: Find a more elegant solution -- Possibly using DFS
    del_tl = []

    for merge_steps in range(0, 5):

        mt = match_tl(filtered_tracker, unassigned_tracklets)

        for (c, k, l) in mt:
            filtered_tracker[k] = np.vstack((filtered_tracker[k], unassigned_tracklets[l]))
            if l not in del_tl:
                del_tl.append(l)

    del_tl.sort(reverse=True)
    # delete associated tracklets to discard the duplicity
    for l in del_tl:
        del (filtered_tracker[l])
        del (unassigned_tracklets[l])

    return filtered_tracker, unassigned_tracklets


def dfs(graph, node, visited):
    if node not in visited:
        visited.append(node)
        for n in graph[node]:
            dfs(graph, n, visited)
    return visited


def computeEdges(vertices, matchList):
    edges = {keys: [] for keys in vertices}
    for i, vi in enumerate(vertices):
        for j, vj in enumerate(vertices):
            if i != j:
                if (vi, vj) in [('0', '6'), ('0', '8'), ('2', '7'), ('6', '2'), ('7', '3')]:
                    edges[vi].append(vj)
    return edges


def refine_association(mt):
    del_tl = []
    tStampsMin = [elmntList[1] for elmntList in mt]
    relatedCost = [elmntList[0] for elmntList in mt]
    tStampsMin = np.array(tStampsMin)
    relatedCost = np.array(relatedCost)
    unq, cnt = np.unique(tStampsMin, return_counts=True)
    multiTrStampsMin = unq[cnt > 1]
    for multiTr in multiTrStampsMin:
        multiTrCost = relatedCost[tStampsMin == multiTr]
        delInd = relatedCost > min(multiTrCost)
        if l not in del_tl:
            del_tl.append(l)
    del mt[del_t]
    return mt


def sort_tracklets(tracker):
    # when it is needed to sort trackelts detections in terms of time-stamps
    tracker = np.array(list(tracker))
    tracker = np.array(sorted(tracker, key=lambda x: x[0]))
    return tracker


def delete_all(demo_path, fmt='png'):
    filelist = glob.glob(os.path.join(demo_path, '*.' + fmt))
    if len(filelist) > 0:
        for f in filelist:
            os.remove(f)


def merge_update_tracklets(cam2to1_tracklets, tracklets1, label_map, tStampStart):
    # **cam2to1_tracklets: list of camera pair tracklets - [tracklets['C2'].org, tracklets['C9'].org]
    # **tracklets1: list of org. version of primarytracklets
    # **label_map: DFS solution for all tracklets association in the list - 1(C2): [1,56, 58, 46](C9)
    # **tStampStart: starting time of all the mapped tracklets - 1(C2): [100, 500, 8000, 900]
    # **k: indexes of primary camera whose will be updated
    # **l: indexes of auxiliary camera whose will be deleted
    # try edges instead of label_map
    # -------------------------------------------------------------------------------------------------------
    del_tl = []
    updTracklets = []
    updLabels = []
    updtStamp = []
    for k in label_map.keys()[0:len(tracklets1)]:  # only update primary tracklets
        if k in del_tl:
            continue
        label_tp = label_map[k][0]
        if len(label_tp) >= 2:  # association happened
            for l in label_tp[1::]:  # first item is the key in DFS solution
                # k = 1, l= 56, 58, 46
                cam2to1_tracklets[k] = np.vstack((cam2to1_tracklets[k], cam2to1_tracklets[l]))
                if l not in del_tl:
                    del_tl.append(l)
        if len(label_tp) == 1:
            continue

    del_tl.sort(reverse=True)
    # delete associated tracklets from auxiliary
    for tl in label_map.keys():
        if tl not in del_tl:
            updTracklets.append(sort_tracklets(cam2to1_tracklets[tl]))
            updLabels.append(label_map[tl])
            updtStamp.append(tStampStart[tl])
    return updTracklets, updLabels, updtStamp


def plot_trajectories_MC(labelsC9,
                         labelsC2,
                         labelsC5,
                         labelsC11,
                         labelsC13,
                         fr_start,
                         fr_end,
                         MCparams,
                         cam2cam,
                         cams,
                         vis_rate=30,
                         isDistorted=True,
                         metrics=eval,
                         save_imgs=0,
                         motion=0):
    # TODO: Figure out why I need the reshape here
    # TODO: Is it necessary to split the trackers identity? yes
    assert len(MCparams['H9to2']['PAorg']) == len(labelsC9) + len(labelsC2)
    assert len(MCparams['H2to5']['PAorg']) ==  len(labelsC5) + len(labelsC2)
    assert len(MCparams['H5to11']['PAorg']) == len(labelsC5) + len(labelsC11)
    assert len(MCparams['H11to13']['PAorg']) == len(labelsC13) + len(labelsC11)
    # BGR
    # magenta = (255,0,255)
    # yellow = (0,255,255)
    # green = (0,255,0)
    # blue = (255,0,0)
    # plt.imshow(np.ones((1080, 1920, 3)))
    # fr_offset = 48#11to13
    # fr_offset = 29  # 9to2, cam2 lag = 29
    if os.path.exists(MC_trklt_C9C2['outPath']):
        delete_all(MC_trklt_C9C2['outPath'])
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 640, 480)
    for fr in range(fr_start, fr_end):
        # 9A:9 fr-15, 11 fr-15
        if (fr) % vis_rate == 0:  # 15 for 6A,7A,.. 3 for 5A, 5B.. 30 for 9A
            #try:
            # Master camera (C9) fr need to be synced:
            imgC2 = cv2.imread('{}{:06d}.png'.format(MCparams[cam2cam['C9C2']]['PImgsPath'],
                                                     fr + (46)))
            # source camera 9,5 - folder2: + fr_offset for 9to2, + fr_offset for 5to11
            imgC9 = cv2.imread('{}{:06d}.png'.format(MCparams[cam2cam['C9C2']]['AImgsPath'],
                                                     fr + 0))
            imgC5 = cv2.imread('{}{:06d}.png'.format(MCparams[cam2cam['C2C5']]['AImgsPath'],
                                                     fr + (26)))
            imgC11 = cv2.imread('{}{:06d}.png'.format(MCparams[cam2cam['C5C11']]['PImgsPath'],
                                                      fr + (2)))
            imgC13 = cv2.imread('{}{:06d}.png'.format(MCparams[cam2cam['C11C13']]['AImgsPath'],
                                                    fr + (-44)))

            c = 0
            # visulaize box on C2: H9to2
            for i_p, bb in enumerate(MCparams['H9to2']['PAorg'][:len(labelsC2)]):
                # To show the association with the projections (tracklets2)
                for i in range(1, len(bb)):
                    if fr_start <= bb[i, 0] <= fr :
                        if save_imgs and motion and bb[i, 0] % 5 == 0:
                            # 2d velocity plot, bb[i,1]
                            xx = int(
                                (bb[i, 1] + bb[i, 3] / 2.0) + bb[i, 5] * 2 * (bb[i, 0] - bb[i - 1, 0]))  # xx=x0+vxt
                            yy = int((bb[i, 2] + bb[i, 4] / 2.0) + bb[i, 6] * 2 * (bb[i, 0] - bb[i - 1, 0]))
                            cv2.arrowedLine(imgC2, (int(bb[i, 1] + bb[i, 3] / 2.0), int(bb[i, 2] + bb[i, 4] / 2.0)),
                                            (xx, yy), (0, 255, 255),
                                            thickness=4, tipLength=1)
                            # line between centroids
                            # cv2.arrowedLine(img,
                            # (int(bb[i - 1, 1] + bb[i - 1, 3] / 2.0), int(bb[i - 1, 2] + bb[i - 1, 4] / 2.0)),
                            # (int(bb[i, 1] + bb[i, 3] / 2.0), int(bb[i, 2] + bb[i, 4] / 2.0)),
                            # (0, 255, 255), thickness=5, tipLength = 1)#lineType=8
                        if metrics:
                            if bb[i, 0] == fr and labelsC2[c] in [7,72,12]:

                                # show box
                                cv2.rectangle(imgC2, (int(bb[i, 1]), int(bb[i, 2])),
                                              (int(bb[i, 1] + bb[i, 3]), int(bb[i, 2] + bb[i, 4])),
                                              (255, 0, 0), 5, cv2.LINE_AA)
                                # show identity at undistorted centroid
                                if isDistorted:
                                    dist_coeff, A = camera_intrinsics(cam=cams['C2'])
                                    cxy_undist = np.copy(bb[i, 1:3] + bb[i, 3:5] / 2.0)
                                    cxy_undist = undistorted_coords(cxy_undist.reshape(1, 1, 2), dist_coeff, A)
                                else:
                                    cxy_undist = np.copy(bb[i, 1:3] + bb[i, 3:5] / 2.0).reshape(1, 2)
                                # show track id: undistorted centers
                                cv2.putText(imgC2, 'P{}'.format(labelsC2[c]),
                                        (int(cxy_undist[0, 0]), int(cxy_undist[0, 1])),
                                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5, cv2.LINE_AA)
                c = c + 1

            # show bbox on C9: visualize the actual target identity in auxiliary camera (9,5)
            c = len(labelsC2)
            # tracks9 = tracks[C2:C2+C9]
            for bb in MCparams['H9to2']['PAorg'][len(labelsC2):]:
                # TODO: Set the colors of the rectangles
                for i in range(1, len(bb)):
                    if fr_start <= bb[i, 0] <= fr :
                        # image = cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength = 0.5)
                        if save_imgs and motion and bb[i, 0] % 5 == 0:
                            # 2d velocity plot
                            xx = int(
                                (bb[i, 1] + bb[i, 3] / 2.0) + bb[i, 5] * 2 * (bb[i, 0] - bb[i - 1, 0]))  # xx=x0+vxt
                            yy = int((bb[i, 2] + bb[i, 4] / 2.0) + bb[i, 6] * 2 * (bb[i, 0] - bb[i - 1, 0]))
                            cv2.arrowedLine(imgC9, (int(bb[i, 1] + bb[i, 3] / 2.0), int(bb[i, 2] + bb[i, 4] / 2.0)),
                                            (xx, yy), (0, 255, 0),
                                            thickness=4, tipLength=1)
                            # line between centroids
                            # cv2.arrowedLine(img2,
                            # (int(bb[i - 1, 1] + bb[i - 1, 3] / 2.0), int(bb[i - 1, 2] + bb[i - 1, 4] / 2.0)),
                            # (int(bb[i, 1] + bb[i, 3] / 2.0), int(bb[i, 2] + bb[i, 4] / 2.0)),
                            # (0, 255, 0), thickness=5, tipLength = 1)#lineType=8
                        if metrics:
                            # TODO: Think about how to show this only at the last detection in the current frame
                            if bb[i, 0] == fr and labelsC9[c] in [7,72,12]:#and labelsC9[c]==15:
                                cv2.putText(imgC9, 'P{}'.format(labelsC9[c]),
                                            (int(bb[i, 1] + bb[i, 3] / 2), int(bb[i, 2] + bb[i, 4] / 2)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 2555, 0), 5, cv2.LINE_AA)
                                cv2.rectangle(imgC9, (int(bb[i, 1]), int(bb[i, 2])),
                                              (int(bb[i, 1] + bb[i, 3]), int(bb[i, 2] + bb[i, 4])),
                                              (255, 0, 0), 5, cv2.LINE_AA)
                c = c + 1
            # show bbox on C5: use H5to11 instead of H5to2 since labels are already propagated to H5to11
            c = len(labelsC2)
            for bb in MCparams['H2to5']['PAorg'][len(labelsC2):]:
                # TODO: Set the colors of the rectangles
                for i in range(1, len(bb)):
                    if fr_start <= bb[i, 0] <= fr+26 :
                        # image = cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength = 0.5)
                        if save_imgs and motion and bb[i, 0] % 5 == 0:
                            # 2d velocity plot
                            xx = int(
                                (bb[i, 1] + bb[i, 3] / 2.0) + bb[i, 5] * 2 * (bb[i, 0] - bb[i - 1, 0]))  # xx=x0+vxt
                            yy = int((bb[i, 2] + bb[i, 4] / 2.0) + bb[i, 6] * 2 * (bb[i, 0] - bb[i - 1, 0]))
                            cv2.arrowedLine(imgC5, (int(bb[i, 1] + bb[i, 3] / 2.0), int(bb[i, 2] + bb[i, 4] / 2.0)),
                                            (xx, yy), (0, 255, 0),
                                            thickness=4, tipLength=1)
                            # line between centroids
                            # cv2.arrowedLine(img2,
                            # (int(bb[i - 1, 1] + bb[i - 1, 3] / 2.0), int(bb[i - 1, 2] + bb[i - 1, 4] / 2.0)),
                            # (int(bb[i, 1] + bb[i, 3] / 2.0), int(bb[i, 2] + bb[i, 4] / 2.0)),
                            # (0, 255, 0), thickness=5, tipLength = 1)#lineType=8
                        if metrics:
                            # TODO: Think about how to show this only at the last detection in the current frame
                            if bb[i, 0] == fr+26 and labelsC5[c] in [7,72,12]:#and labelsC5[c]==15:
                                cv2.putText(imgC5, 'P{}'.format(labelsC5[c]),
                                            (int(bb[i, 1] + bb[i, 3] / 2), int(bb[i, 2] + bb[i, 4] / 2)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 2555, 0), 5, cv2.LINE_AA)
                                cv2.rectangle(imgC5, (int(bb[i, 1]), int(bb[i, 2])),
                                              (int(bb[i, 1] + bb[i, 3]), int(bb[i, 2] + bb[i, 4])),
                                              (255, 0, 0), 5, cv2.LINE_AA)
                c = c + 1

            # show bbox on C11
            c = 0
            for bb in MCparams['H5to11']['PAorg'][:len(labelsC11)]:
                # TODO: Set the colors of the rectangles
                for i in range(1, len(bb)):
                    if fr_start <= bb[i, 0] <= fr+2 :
                        # image = cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength = 0.5)
                        if save_imgs and motion and bb[i, 0] % 5 == 0:
                            # 2d velocity plot
                            xx = int(
                                (bb[i, 1] + bb[i, 3] / 2.0) + bb[i, 5] * 2 * (bb[i, 0] - bb[i - 1, 0]))  # xx=x0+vxt
                            yy = int((bb[i, 2] + bb[i, 4] / 2.0) + bb[i, 6] * 2 * (bb[i, 0] - bb[i - 1, 0]))
                            cv2.arrowedLine(imgC11,
                                            (int(bb[i, 1] + bb[i, 3] / 2.0), int(bb[i, 2] + bb[i, 4] / 2.0)),
                                            (xx, yy), (0, 255, 0),
                                            thickness=4, tipLength=1)
                            # line between centroids
                            # cv2.arrowedLine(img2,
                            # (int(bb[i - 1, 1] + bb[i - 1, 3] / 2.0), int(bb[i - 1, 2] + bb[i - 1, 4] / 2.0)),
                            # (int(bb[i, 1] + bb[i, 3] / 2.0), int(bb[i, 2] + bb[i, 4] / 2.0)),
                            # (0, 255, 0), thickness=5, tipLength = 1)#lineType=8
                        if metrics:
                            # TODO: Think about how to show this only at the last detection in the current frame
                            if bb[i, 0] == fr+2 and labelsC11[c] in [7,72,12]:#and labelsC11[c]==15:
                                cv2.rectangle(imgC11, (int(bb[i, 1]), int(bb[i, 2])),
                                              (int(bb[i, 1] + bb[i, 3]), int(bb[i, 2] + bb[i, 4])),
                                              (255, 0, 0), 5, cv2.LINE_AA)
                                if isDistorted:
                                    dist_coeff, A = camera_intrinsics(cam=cams['C11'])
                                    cxy_undist = np.copy(bb[i, 1:3] + bb[i, 3:5] / 2.0)
                                    cxy_undist = undistorted_coords(cxy_undist.reshape(1, 1, 2), dist_coeff, A)
                                else:
                                    cxy_undist = np.copy(bb[i, 1:3] + bb[i, 3:5] / 2.0).reshape(1, 2)
                                # show track id: undistorted centers
                                cv2.putText(imgC11, 'P{}'.format(labelsC11[c]),
                                            (int(cxy_undist[0, 0]), int(cxy_undist[0, 1])),
                                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
                c = c + 1
            # show bbox on C13
            c = len(labelsC11)
            for bb in MCparams['H11to13']['PAorg'][len(labelsC11):]:
                # TODO: Set the colors of the rectangles
                for i in range(1, len(bb)):
                    if fr_start <= bb[i, 0] <= fr-44:
                        # image = cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength = 0.5)
                        if save_imgs and motion and bb[i, 0] % 5 == 0:
                            # 2d velocity plot
                            xx = int(
                                (bb[i, 1] + bb[i, 3] / 2.0) + bb[i, 5] * 2 * (bb[i, 0] - bb[i - 1, 0]))  # xx=x0+vxt
                            yy = int((bb[i, 2] + bb[i, 4] / 2.0) + bb[i, 6] * 2 * (bb[i, 0] - bb[i - 1, 0]))
                            cv2.arrowedLine(imgC13,
                                            (int(bb[i, 1] + bb[i, 3] / 2.0), int(bb[i, 2] + bb[i, 4] / 2.0)),
                                            (xx, yy), (0, 255, 0),
                                            thickness=4, tipLength=1)
                            # line between centroids
                            # cv2.arrowedLine(img2,
                            # (int(bb[i - 1, 1] + bb[i - 1, 3] / 2.0), int(bb[i - 1, 2] + bb[i - 1, 4] / 2.0)),
                            # (int(bb[i, 1] + bb[i, 3] / 2.0), int(bb[i, 2] + bb[i, 4] / 2.0)),
                            # (0, 255, 0), thickness=5, tipLength = 1)#lineType=8
                        if metrics:
                            # TODO: Think about how to show this only at the last detection in the current frame
                            if bb[i, 0] == fr-44 and labelsC13[c] in [7,72,12]:#and labelsC13[c]==15:
                                cv2.rectangle(imgC13, (int(bb[i, 1]), int(bb[i, 2])),
                                              (int(bb[i, 1] + bb[i, 3]), int(bb[i, 2] + bb[i, 4])),
                                              (255, 0, 0), 5, cv2.LINE_AA)
                                if isDistorted:
                                    dist_coeff, A = camera_intrinsics(cam=cams['C13'])
                                    cxy_undist = np.copy(bb[i, 1:3] + bb[i, 3:5] / 2.0)
                                    cxy_undist = undistorted_coords(cxy_undist.reshape(1, 1, 2), dist_coeff, A)
                                else:
                                    cxy_undist = np.copy(bb[i, 1:3] + bb[i, 3:5] / 2.0).reshape(1, 2)
                                # show track id: undistorted centers
                                cv2.putText(imgC13, 'P{}'.format(labelsC13[c]),
                                            (int(cxy_undist[0, 0]), int(cxy_undist[0, 1])),
                                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
                c = c + 1

            imgC2 = cv2.copyMakeBorder(imgC2, 0, 0, 0, 250, cv2.BORDER_CONSTANT, value=0)
            imgC9 = cv2.copyMakeBorder(imgC9, 0, 0, 250, 0, cv2.BORDER_CONSTANT, value=0)

            imgC9C2 = cv2.vconcat([imgC9, imgC2])
            imgC11C5 = cv2.vconcat([imgC11, imgC5])
            final_img = cv2.hconcat([imgC11C5, imgC9C2])  # img: c4, img2: c2
            blankImg = np.zeros(shape=[1080, 1920, 3], dtype=np.uint8)
            img13Blnk = cv2.vconcat([imgC13, blankImg])
            final_img = cv2.hconcat([img13Blnk, final_img])
            final_img = cv2.resize(final_img, (int(1.5*1920), 1080), interpolation=cv2.INTER_AREA)

            cv2.imshow("image", final_img)
            if save_imgs:
                # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                cv2.imwrite(MCparams[cam2cam['C9C2']]['outPath'] + '/{:06d}.png'.format(fr), final_img)
            cv2.waitKey(10)
            #except:
                #print("Frame {} not found".format(fr))
                #continue



def plot_trajectories_pairs(labelCam2,
                            labelCam1,
                            fr_start,
                            fr_end,
                            MC_trklt_C9C2,
                            cam2cam,
                            cams,
                            vis_rate=30,
                            isDistorted=0,
                            metrics=eval,
                            save_imgs=0,
                            motion=0
                            ):
    assert len(MC_trklt_C9C2['PAproj']) == len(MC_trklt_C9C2['PAorg']) == MC_trklt_C9C2['trkltFamilySizeP'] + \
                                                                          MC_trklt_C9C2['trkltFamilySizeA']
    # labelCam1 = dict(list(label_map_update.items())[:MC_trklt_C9C2['trkltFamilySizeP']])
    # labelCam2 = dict(list(label_map_update.items())[MC_trklt_C9C2['trkltFamilySizeP']:])
    # BGR
    # magenta = (255,0,255)
    # yellow = (0,255,255)
    # green = (0,255,0)
    # blue = (255,0,0)
    if os.path.exists(MC_trklt_C9C2['outPath']):
        delete_all(MC_trklt_C9C2['outPath'])
    if not os.path.exists(MC_trklt_C9C2['outPath']):
        os.makedirs(MC_trklt_C9C2['outPath'])
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 640, 480)
    for fr in range(fr_start, fr_end):
        if (fr) % vis_rate == 0:
            # try:
            img = cv2.imread('{}{:06d}.png'.format(MC_trklt_C9C2['PImgsPath'], fr + MC_trklt_C9C2['fr_offsetPVis']))
            img2 = cv2.imread('{}{:06d}.png'.format(MC_trklt_C9C2['AImgsPath'], fr + MC_trklt_C9C2['fr_offsetAVis']))
            # visualize the projections on primary camera (11,2)
            c = MC_trklt_C9C2['trkltFamilySizeP']
            # get projection from auxiliary
            for bb in MC_trklt_C9C2['PAproj'][
                      MC_trklt_C9C2['trkltFamilySizeP']:]:  # show projection of auxiliary camera
                # TODO: Set the colors of the rectangles
                for i in range(1, len(bb)):
                    if fr_start <= bb[i, 0] <= fr:
                        if save_imgs and motion and bb[i, 0] % 5 == 0:
                            # 2d velocity plot
                            xx = int(bb[i, 1] + bb[i, 5] * 2 * (bb[i, 0] - bb[i - 1, 0]))  # xx=x0+vxt
                            yy = int(bb[i, 2] + bb[i, 6] * 2 * (bb[i, 0] - bb[i - 1, 0]))
                            cv2.arrowedLine(img, (int(bb[i, 1]), int(bb[i, 2])),
                                            (xx, yy),
                                            (255, 0, 255), thickness=4, tipLength=1)
                            # line between centroids
                            # cv2.arrowedLine(img, (int(bb[i - 1, 1]), int(bb[i - 1, 2])), (int(bb[i, 1]), int(bb[i, 2])),
                            # magenta, thickness=5, tipLength = 1)#lineType=8

                        # TODO: Think about how to show this only at the last detection in the current frame
                        if bb[i, 0] == fr:  # and len(label_map_aux[c])>0
                            cv2.putText(img, '{}'.format(labelCam2[c]), (int(bb[i - 1, 1]), int(bb[i - 1, 2])),
                                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 255), 5, cv2.LINE_AA)
                c = c + 1
            c = 0
            # visulaize the identity handover in primary camera
            # get detection from primary
            for i_p, bb in enumerate(MC_trklt_C9C2['PAorg'][:MC_trklt_C9C2['trkltFamilySizeP']]):  # primary camera
                # To show the association with the projections (tracklets2)
                for i in range(1, len(bb)):
                    if fr_start <= bb[i, 0] <= fr:
                        if save_imgs and motion and bb[i, 0] % 5 == 0:
                            # 2d velocity plot, bb[i,1]
                            xx = int(
                                (bb[i, 1] + bb[i, 3] / 2.0) + bb[i, 5] * 2 * (bb[i, 0] - bb[i - 1, 0]))  # xx=x0+vxt
                            yy = int((bb[i, 2] + bb[i, 4] / 2.0) + bb[i, 6] * 2 * (bb[i, 0] - bb[i - 1, 0]))
                            cv2.arrowedLine(img, (int(bb[i, 1] + bb[i, 3] / 2.0), int(bb[i, 2] + bb[i, 4] / 2.0)),
                                            (xx, yy), (0, 255, 255),
                                            thickness=4, tipLength=1)
                            # line between centroids
                            # cv2.arrowedLine(img,
                            # (int(bb[i - 1, 1] + bb[i - 1, 3] / 2.0), int(bb[i - 1, 2] + bb[i - 1, 4] / 2.0)),
                            # (int(bb[i, 1] + bb[i, 3] / 2.0), int(bb[i, 2] + bb[i, 4] / 2.0)),
                            # (0, 255, 255), thickness=5, tipLength = 1)#lineTyp
                        if bb[i, 0] == fr:
                            cv2.rectangle(img, (int(bb[i, 1]), int(bb[i, 2])),
                                          (int(bb[i, 1] + bb[i, 3]), int(bb[i, 2] + bb[i, 4])),
                                          (255, 0, 0), 5, cv2.LINE_AA)
                            # show identity at undistorted centroid
                            if isDistorted:
                                dist_coeff, A = camera_intrinsics(cam=MC_trklt_C9C2['Pid'])
                                cxy_undist = np.copy(bb[i, 1:3] + bb[i, 3:5] / 2.0)
                                cxy_undist = undistorted_coords(cxy_undist.reshape(1, 1, 2), dist_coeff, A)
                            else:
                                cxy_undist = np.copy(bb[i, 1:3] + bb[i, 3:5] / 2.0).reshape(1, 2)
                            cv2.putText(img, 'P{}'.format(labelCam1[c]),
                                        (int(cxy_undist[0, 0]), int(cxy_undist[0, 1])),
                                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5, cv2.LINE_AA)
                c = c + 1
            # visualize the actual target identity in auxiliary camera (9,5)
            c = MC_trklt_C9C2['trkltFamilySizeP']
            for bb in MC_trklt_C9C2['PAorg'][MC_trklt_C9C2['trkltFamilySizeP']:]:  # source camera : yellow
                # TODO: Set the colors of the rectangles
                for i in range(1, len(bb)):
                    if fr_start <= bb[i, 0] <= fr:
                        # image = cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength = 0.5)
                        if motion and bb[i, 0] % 5 == 0:
                            # 2d velocity plot
                            xx = int(
                                (bb[i, 1] + bb[i, 3] / 2.0) + bb[i, 5] * 2 * (bb[i, 0] - bb[i - 1, 0]))  # xx=x0+vxt
                            yy = int((bb[i, 2] + bb[i, 4] / 2.0) + bb[i, 6] * 2 * (bb[i, 0] - bb[i - 1, 0]))
                            cv2.arrowedLine(img2, (int(bb[i, 1] + bb[i, 3] / 2.0), int(bb[i, 2] + bb[i, 4] / 2.0)),
                                            (xx, yy), (0, 255, 0),
                                            thickness=4, tipLength=1)
                            # line between centroids
                            # cv2.arrowedLine(img2,
                            # (int(bb[i - 1, 1] + bb[i - 1, 3] / 2.0), int(bb[i - 1, 2] + bb[i - 1, 4] / 2.0)),
                            # (int(bb[i, 1] + bb[i, 3] / 2.0), int(bb[i, 2] + bb[i, 4] / 2.0)),
                            # (0, 255, 0), thickness=5, tipLength = 1)#lineType=8
                        # TODO: Think about how to show this only at the last detection in the current frame
                        if bb[i, 0] == fr:
                            cv2.putText(img2, '{}'.format(labelCam2[c]),
                                        (int(bb[i, 1] + bb[i, 3] / 2), int(bb[i, 2] + bb[i, 4] / 2)),
                                        cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 2555, 0), 5, cv2.LINE_AA)
                            cv2.rectangle(img2, (int(bb[i, 1]), int(bb[i, 2])),
                                          (int(bb[i, 1] + bb[i, 3]), int(bb[i, 2] + bb[i, 4])),
                                          (255, 0, 0), 5, cv2.LINE_AA)
                c = c + 1
            if cam2cam == 'H9to2':
                final_img = cv2.vconcat([img2, img])
            if cam2cam == 'H2to5':
                final_img = cv2.hconcat([img2, img])  # img: c5, img2: c2
            if cam2cam == 'H5to11':
                final_img = cv2.vconcat([img, img2])
            if cam2cam == 'H11to13':
                final_img = cv2.hconcat([img2, img])  # img: c11, img2: c13

            cv2.imshow("image", final_img)
            if save_imgs:
                # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                cv2.imwrite(MC_trklt_C9C2['outPath'] + '/{:06d}.png'.format(fr), final_img)
            cv2.waitKey(30)
            # except:
            # print("Frame {} not found".format(fr))
            # continue
    cv2.destroyAllWindows()


def dafault_params(cam2cam, experiment, cams, mhtTrackers):
    params = {}
    if cam2cam == 'H9to2':
        # C2: primary, C9: auxiliary
        param = {}
        param['Pid'] = 2
        param['Aid'] = 9
        param['tOffsetPstart'] = -46  # 2 will start at 40 wrt 9 is 1: exp1-test: 46, exp1-train: 40
        param['tOffsetAstart'] = 0  # 9
        # similarity based homography: TODO: compute homography using undistorted image
        # H9to2
        param['H'] = [[0.9812, 0.1426, 0.00],
                      [-0.1426, 0.9812, 0.00],
                      [463.84, -782.41, 1.00]]
        param['fr_offsetAVis'] = 0  # cam9 delay
        param['fr_offsetPVis'] = 46  ## cam2 delay
        param['H'] = np.transpose(param['H'])
        param['d_th'] = 0.3  # 0.17#0.17#0.45#200 # multi-task cost: 0.45, only loc: 0.17, loc+velocity 4d hd: 0.4
        param['primary'] = '/home/siddique/Desktop/NEU_Data/May_Demo_Trackers/exp2-train/cam02exp2.mp4testTracksv2.txt'
        # mhtTrackers + 'cam{:02d}{}PersonMASK_30FPSrot.txt'.format(cams['C2'], experiment)#cam2: test_aug: v1
        param['auxiliary'] = '/home/siddique/Desktop/NEU_Data/May_Demo_Trackers/exp2-train/cam09exp2.mp4testTracks.txt'
        # mhtTrackers + 'cam{:02d}{}PersonMASK_30FPSrot.txt'.format(cams['C9'], experiment)#cam9-test_aug:v2exp1
        param['finalTrackFile'] = 'cam09exp2_MCTA.txt'
        # param['finalTrackFileC2'] = 'cam02exp2_MCTA.txt'
        param['PImgsPath'] = benchmark + experiment + '/imgs/cam{:02d}'.format(cams[
                                                                               'C2']) + experiment + '.mp4/'  # benchmark + 'test_data/exp1'+ '/imgs/cam{:02d}'.format(cams['C2']) + 'exp1' + '.mp4/'
        # benchmark + experiment + '/imgs/cam{:02d}'.format(cams['C2']) + tracklet_exp + '.mp4/'
        param['AImgsPath'] = benchmark + experiment + '/imgs/cam{:02d}'.format(cams[
                                                                               'C9']) + experiment + '.mp4/'  # benchmark + 'test_data/exp1'+ '/imgs/cam{:02d}'.format(cams['C9']) + 'exp1' + '.mp4/'
        # benchmark + experiment + '/imgs/cam{:02d}'.format(cams['C9']) + tracklet_exp + '.mp4/'
        param['outPath'] = benchmark + experiment + '/Results/MCTA/vis/cam{}to{}/'.format(cams['C9'], cams['C2'])

    if cam2cam == 'H2to5':
        param = {}
        param['Pid'] = 2
        param['Aid'] = 5
        # C5: aux, C2: pri
        param['tOffsetPstart'] = -21  # 2 will start at 40 wrt 9 is 1: exp1-test: 46, exp1-train: 40, exp2-train:21
        param['tOffsetAstart'] = 0  # 5
        # similarity based homography: TODO: compute homography using undistorted image
        # H2to5
        param['H'] = [[1.2, -0.2, 0.0],
                      [0.2, 1.2, 0.0],
                      [1422.4, 46.8, 1.0]]
        param['fr_offsetAVis'] = 0  # cam9 delay
        param['fr_offsetPVis'] = 21  ## cam2 delay
        param['H'] = np.transpose(param['H'])
        param['H'] = np.linalg.inv(param['H'])
        param['d_th'] = 0.25  # 0.17#0.17#0.45#200 # multi-task cost: 0.45, only loc: 0.17, loc+velocity 4d hd: 0.4
        param['primary'] = '/home/siddique/Desktop/NEU_Data/May_Demo_Trackers/exp2-train/cam02exp2.mp4testTracksv2.txt'
        # mhtTrackers + 'cam{:02d}{}PersonMASK_30FPSrot.txt'.format(cams['C2'], experiment)#cam2: test_aug: v1
        param['auxiliary'] = '/home/siddique/Desktop/NEU_Data/May_Demo_Trackers/exp2-train/cam05exp2.mp4testTracks.txt'
        # mhtTrackers + 'cam{:02d}{}PersonMASK_30FPSrot.txt'.format(cams['C9'], experiment)#cam9-test_aug:v2exp1
        param['finalTrackFile'] = 'cam05exp2_MCTA.txt'
        # param['finalTrackFileC2'] = 'cam02exp2_MCTA.txt'
        param['PImgsPath'] = benchmark + experiment + '/imgs/cam{:02d}'.format(cams[
                                                                               'C2']) + experiment + '.mp4/'  # benchmark + 'test_data/exp1'+ '/imgs/cam{:02d}'.format(cams['C2']) + 'exp1' + '.mp4/'
        # benchmark + experiment + '/imgs/cam{:02d}'.format(cams['C2']) + tracklet_exp + '.mp4/'
        param['AImgsPath'] = benchmark + experiment + '/imgs/cam{:02d}'.format(cams[
                                                                               'C5']) + experiment + '.mp4/'  # benchmark + 'test_data/exp1'+ '/imgs/cam{:02d}'.format(cams['C9']) + 'exp1' + '.mp4/'
        # benchmark + experiment + '/imgs/cam{:02d}'.format(cams['C9']) + tracklet_exp + '.mp4/'
        param['outPath'] = benchmark + experiment + '/Results/MCTA/vis/cam{}to{}/'.format(cams['C5'], cams['C2'])

    if cam2cam == 'H5to11':
        param = {}
        param['Pid'] = 11
        param['Aid'] = 5
        # C11: pri, C5: aux
        param['tOffsetPstart'] = 0  # 11 delayed 5 by 23: 11>1 == 5>24
        param['tOffsetAstart'] = -23  #
        # similarity based homography: TODO: compute homography using undistorted image
        # H5to11
        '''
                param['H'] = [[0.9704, 0.1027, 0.00],
                      [-0.1027, 0.9704, 0.00],
                      [37.211, 514.028, 1.00]]
            0.9453   -0.0154         0
    0.0154    0.9453         0
   23.8510  508.6742    1.0000'''
        param['H'] = [[0.9453, -0.0154, 0.00],
                      [0.0154, 0.9453, 0.00],
                      [23.8510, 508.6742, 1.00]]

        param['fr_offsetPVis'] = 0
        param['fr_offsetAVis'] = 23  ## T_c2 - T_c11 = offset
        param['H'] = np.transpose(param['H'])
        # param['H'] = np.linalg.inv(param['H'])
        param['d_th'] = 0.25  # 0.17#0.17#0.45#200 # multi-task cost: 0.45, only loc: 0.17, loc+velocity 4d hd: 0.4
        param['primary'] = '/home/siddique/Desktop/NEU_Data/May_Demo_Trackers/exp2-train/cam11exp2.mp4testTracks.txt'
        # mhtTrackers + 'cam{:02d}{}PersonMASK_30FPSrot.txt'.format(cams['C2'], experiment)#cam2: test_aug: v1
        param['auxiliary'] = '/home/siddique/Desktop/NEU_Data/May_Demo_Trackers/exp2-train/cam05exp2.mp4testTracks.txt'
        # mhtTrackers + 'cam{:02d}{}PersonMASK_30FPSrot.txt'.format(cams['C9'], experiment)#cam9-test_aug:v2exp1
        param['finalTrackFile'] = 'cam11exp2_MCTA.txt'
        # param['finalTrackFileC2'] = 'cam02exp2_MCTA.txt'
        param['PImgsPath'] = benchmark + experiment + '/imgs/cam{:02d}'.format(cams[
                                                                               'C11']) + experiment + '.mp4/'  # benchmark + 'test_data/exp1'+ '/imgs/cam{:02d}'.format(cams['C2']) + 'exp1' + '.mp4/'
        # benchmark + experiment + '/imgs/cam{:02d}'.format(cams['C2']) + tracklet_exp + '.mp4/'
        param['AImgsPath'] = benchmark + experiment + '/imgs/cam{:02d}'.format(cams[
                                                                               'C5']) + experiment + '.mp4/'  # benchmark + 'test_data/exp1'+ '/imgs/cam{:02d}'.format(cams['C9']) + 'exp1' + '.mp4/'
        # benchmark + experiment + '/imgs/cam{:02d}'.format(cams['C9']) + tracklet_exp + '.mp4/'
        param['outPath'] = benchmark + experiment + '/Results/MCTA/vis/cam{}to{}/'.format(cams['C5'], cams['C11'])

    if cam2cam == 'H11to13':
        param = {}
        param['Pid'] = 11
        param['Aid'] = 13
        # C11: pri, C5: aux
        param['tOffsetPstart'] = -48  # 11 delayed 5 by 23: 11>1 == 5>24
        param['tOffsetAstart'] = 0  #
        # similarity based homography: TODO: compute homography using undistorted image
        # H13to11

        '''
   1.0e+03 *

    0.0011   -0.0000         0
    0.0000    0.0011         0
   -1.0191    0.0383    0.0010
        '''

        param['H'] = [[1.1, -0.000, 0.0],
                      [-0.000, 1.1, 0.0],
                      [-1400.1, 10.3, 1.0]]

        param['fr_offsetPVis'] = 48
        param['fr_offsetAVis'] = 0  ## T_c2 - T_c11 = offset
        param['H'] = np.transpose(param['H'])
        # param['H'] = np.linalg.inv(param['H'])
        param['d_th'] = 0.8  # 0.17#0.17#0.45#200 # multi-task cost: 0.45, only loc: 0.17, loc+velocity 4d hd: 0.4
        param['primary'] = '/home/siddique/Desktop/NEU_Data/May_Demo_Trackers/exp2-train/cam11exp2.mp4testTracks.txt'
        # mhtTrackers + 'cam{:02d}{}PersonMASK_30FPSrot.txt'.format(cams['C2'], experiment)#cam2: test_aug: v1
        param['auxiliary'] = '/home/siddique/Desktop/NEU_Data/May_Demo_Trackers/exp2-train/cam13exp2.mp4testTracks.txt'
        # mhtTrackers + 'cam{:02d}{}PersonMASK_30FPSrot.txt'.format(cams['C9'], experiment)#cam9-test_aug:v2exp1
        param['finalTrackFile'] = 'cam13exp2_MCTA.txt'
        # param['finalTrackFileC2'] = 'cam02exp2_MCTA.txt'
        param['PImgsPath'] = benchmark + experiment + '/imgs/cam{:02d}'.format(cams[
                                                                               'C11']) + experiment + '.mp4/'  # benchmark + 'test_data/exp1'+ '/imgs/cam{:02d}'.format(cams['C2']) + 'exp1' + '.mp4/'
        # benchmark + experiment + '/imgs/cam{:02d}'.format(cams['C2']) + tracklet_exp + '.mp4/'
        param['AImgsPath'] = benchmark + experiment + '/imgs/cam{:02d}'.format(cams[
                                                                               'C13']) + experiment + '.mp4/'  # benchmark + 'test_data/exp1'+ '/imgs/cam{:02d}'.format(cams['C9']) + 'exp1' + '.mp4/'
        # benchmark + experiment + '/imgs/cam{:02d}'.format(cams['C9']) + tracklet_exp + '.mp4/'
        param['outPath'] = benchmark + experiment + '/Results/MCTA/vis/cam{}to{}/'.format(cams['C13'], cams['C11'])

    if cam2cam == 'H13to14':
        param = {}
        param['Pid'] = 14
        param['Aid'] = 13
        # C5: primary, C2: auxiliary
        param['tOffsetPstart'] = -4  # 13 delayed 14 by 5: 13>1 == 14>5
        param['tOffsetAstart'] = 0  #
        # similarity based homography: TODO: compute homography using undistorted image
        # H13to14
        param['H'] = [[1.1163, 0.0542, 0],
                      [-0.0542, 1.1163, 0],
                      [589.7222, -87.6723, 1.0000]]

        param['fr_offsetPVis'] = 4
        param['fr_offsetAVis'] = 0  ## T_c2 - T_c11 = offset
        param['H'] = np.transpose(param['H'])
        # param['H'] = np.linalg.inv(param['H'])
        param['d_th'] = 0.3
        param['primary'] = '/home/siddique/Desktop/NEU_Data/May_Demo_Trackers/exp2-train/cam14exp2.mp4testTracks.txt'
        # mhtTrackers + 'cam{:02d}{}PersonMASK_30FPSrot.txt'.format(cams['C2'], experiment)#cam2: test_aug: v1
        param['auxiliary'] = '/home/siddique/Desktop/NEU_Data/May_Demo_Trackers/exp2-train/cam13exp2.mp4testTracks.txt'
        # mhtTrackers + 'cam{:02d}{}PersonMASK_30FPSrot.txt'.format(cams['C9'], experiment)#cam9-test_aug:v2exp1
        param['finalTrackFile'] = 'cam14exp2_MCTA.txt'
        # param['finalTrackFileC2'] = 'cam02exp2_MCTA.txt'
        param['PImgsPath'] = benchmark + 'exp2' + '/imgs/cam{:02d}'.format(cams[
                                                                               'C14']) + 'exp2' + '.mp4/'  # benchmark + 'test_data/exp1'+ '/imgs/cam{:02d}'.format(cams['C2']) + 'exp1' + '.mp4/'
        # benchmark + experiment + '/imgs/cam{:02d}'.format(cams['C2']) + tracklet_exp + '.mp4/'
        param['AImgsPath'] = benchmark + 'exp2' + '/imgs/cam{:02d}'.format(cams[
                                                                               'C13']) + 'exp2' + '.mp4/'  # benchmark + 'test_data/exp1'+ '/imgs/cam{:02d}'.format(cams['C9']) + 'exp1' + '.mp4/'
        # benchmark + experiment + '/imgs/cam{:02d}'.format(cams['C9']) + tracklet_exp + '.mp4/'
        param['outPath'] = benchmark + experiment + '/Results/MCTA/vis/cam{}to{}/'.format(cams['C13'], cams['C14'])

    return param


def assignID2matches(params, mt, isseqlID):
    # form edges from matches
    params['adjacencyPA'] = np.zeros((len(params['PAproj']), len(params['PAproj'])), dtype='int')
    edges = {keys: [] for keys in range(len(params['PAproj']))}
    for (c, tStampComnMin, k, l) in mt:
        edges[k].append(l)
        edges[l].append(k)
        print 'k {} l {}'.format(k, l)
    # get tracklet start information: if necessary
    label_map = {keys: [] for keys in range(len(params['PAproj']))}
    tStampStart = {keys: [] for keys in range(len(params['PAproj']))}
    for label_tp, tklt in enumerate(params['PAproj']):
        solnPath = dfs(edges, label_tp, [])
        label_map[label_tp].append(solnPath)
        # TODO: create an adjacency matrix to see which node (in P) is connected to other nodes (in A)
        for tStampInd in solnPath:
            tStamp = params['PAproj'][tStampInd][:, 0].min()
            tStampStart[label_tp].append(tStamp)
            # add incidence for each node
            # DFS solns have all connected nodes for each node including loop (self connection)
            params['adjacencyPA'][label_tp, tStampInd] = 1

    # to select track birth ID
    params['tStart'] = tStampStart
    # form sequential IDs
    # assign labels based track birth (minimum time stamp)
    label_map_update = {keys: [] for keys in range(len(params['PAproj']))}
    for k in label_map.keys():
        label_tp = label_map[k][0]
        if len(label_tp) >= 2:  # association happened
            # select first appeared tracklets in solnPath to initialize id for remaining
            label_map_update[k] = np.array(label_tp)[tStampStart[k] == min(tStampStart[k])][0]
        if len(label_tp) == 1:
            label_map_update[k] = label_tp[0]
    if isseqlID:
        print 'making identity sequential.....'
        isMapped = []
        seqID = 0
        seq_label = sorted((min(value), key) for (key, value) in tStampStart.items())
        for (fr, id) in seq_label:
            if id not in isMapped:
                seqID += 1
                for l_t in label_map[id][0]:
                    label_map_update[l_t] = seqID
                    isMapped.append(l_t)
                    # else:
                    # label_map_update = label_map

    return label_map_update, params


def get_cam2cam_matchedTrklt(params, cam2cam, isDistorted=True, isseqID=False):
    # collect primary camera: P:C2 features
    trackers = np.loadtxt(params['primary'], dtype='float32', delimiter=',')
    trklt_size = 360
    trackers[:, 0] = trackers[:, 0] + params['tOffsetPstart']
    fr_startP = min(trackers[:, 0][trackers[:, 0] > 0])
    fr_endP = trackers[-1, 0]
    isSCA = 0
    print 'Frame start and end {}, {}'.format(fr_startP, fr_endP)
    # use [cx,cy] to compute distance and d_th = 10 for single camera association
    # Apply SCTA and Filtering
    mht_tracklets, totalTrack = form_tracklets(trackers, fr_startP, fr_endP, tracker_min_size=trklt_size,
                                               t_thr=30, d_thr=20, single_cam_association=isSCA).tracker_mht()
    # list C2 tracklets
    trackletsP = convert_centroids(mht_tracklets)

    # A:C9:------------------------------------------------------------------
    # Collect auxiliary camera features
    trackers = np.loadtxt(params['auxiliary'], dtype='float32', delimiter=',')
    trklt_size = 60  # cam9: 30
    trackers[:, 0] = trackers[:, 0] + params['tOffsetAstart']
    fr_startA = min(trackers[:, 0][trackers[:, 0] > 0])
    fr_endA = trackers[-1, 0]
    isSCA = 0

    print 'Frame start and end {}, {}'.format(trackers[0, 0], trackers[-1, 0])
    # use [cx,cy] to compute distance and d_th = 10 for single camera association
    # Apply SCTA and Filtering
    mht_tracklets, totalTrack = form_tracklets(trackers, fr_startA, fr_endA, tracker_min_size=trklt_size,
                                               t_thr=30, d_thr=20, single_cam_association=isSCA).tracker_mht()
    # keep both original and projected tracklets for auxiliary
    tracklet_2 = copy.deepcopy(mht_tracklets)
    # tracklet_2 = convert_centroids(tracklet_2)
    # Project: center>(cx,cy), top-left>(x,y), velocity>(vx,vy) [fr,cx,cy,x,y,vx,vy]
    # active_trklt: indexes of tracklets whose projected onto the destination (primary) image boundary
    trackletsA, active_trklt = project_tracklets(mht_tracklets, params['H'],
                                                 isDistorted, cam=params['Aid'])

    # form tracklets family set: [tracklets['C2'].org, tracklets['C9'].projected]
    params['PAproj'] = np.append(trackletsP, tracklet_2)
    params['activeTrkltIndexsA'] = list(len(trackletsP) + np.array(active_trklt))
    params['trkltFamilySizeP'] = len(trackletsP)
    params['trkltFamilySizeA'] = len(tracklet_2)
    # TODO: need to verify for 9to2, 5to11
    # since all the projected tracklets might not in the primary image boundary
    params['PAproj'][params['activeTrkltIndexsA']] = trackletsA
    # use raw tracklets for visualization and metrics evaluation
    params['PAorg'] = np.append(centroids2xywh(trackletsP), tracklet_2)
    # Apply MCTA
    mt, tCommonPA = associate_tracklets_DFS_MC(params,
                                               isDistorted,
                                               max_dist=params['d_th']
                                               )
    label_map_update, params = assignID2matches(params, mt, isseqID)

    return label_map_update, params, min(fr_startP, fr_startA), max(fr_endP, fr_endA)


def split_labelMap(label_map, MC_trklt):
    # to keep the order of the trklt index sequential
    labelPrimary = collections.OrderedDict()
    labelAuxiliary = collections.OrderedDict()
    for key in sorted(label_map.keys()):
        if key < MC_trklt['trkltFamilySizeP']:
            labelPrimary[key] = label_map[key]
        else:
            labelAuxiliary[key] = label_map[key]

    return labelAuxiliary, labelPrimary


def update_aux_ids(aux_labels=None, pri_labels=None, pri_labels_prev=None):
    # *** labelsC2 : already associated map of primary (in previously 9to2 association)
    # *** labelsC5 : current association map of auxiliary
    # *** labelsC2C5 : current association map of priimary
    #     #labelsC5_update = {keys: [] for keys in range(len(labelsC5.keys()))}
    # -----------------------------------------------------------------------------------
    # tStart rules are already applied
    # update labelsC5 based on labelsC2 in H9to2
    max_ID_prev = max(pri_labels_prev.values())
    assert len(pri_labels.keys()) == len(pri_labels_prev.keys())
    for k in aux_labels.keys():
        # TODO: consider different approach if minimum tStamp based id assignment does not works
        candIDkey = aux_labels[k]  # candKey is the values of aux but key for primary
        # check that candIDkey is within primary camera id keys
        if candIDkey in pri_labels_prev.keys():
            aux_labels[k] = pri_labels_prev[candIDkey]
        else:
            max_ID_prev+=1 #initialize new id in auxiliary camera
            aux_labels[k] = max_ID_prev
    # update labelsC2 based on labelsC25 in H5to2
    return aux_labels


def update_pri_ids(pri_labels=None, aux_labels=None, aux_labels_prev=None):
    # *** labelsC2 : already associated map of primary (in previously 9to2 association)
    # *** labelsC5 : current association map of auxiliary
    # *** labelsC2C5 : current association map of priimary
    #     #labelsC5_update = {keys: [] for keys in range(len(labelsC5.keys()))}
    # -----------------------------------------------------------------------------------
    # tStart rules are already applied
    # update labelsC5 based on labelsC2 in H9to2
    max_ID_prev = max(aux_labels_prev.values())
    assert len(aux_labels.keys()) == len(aux_labels_prev.keys())
    for k in pri_labels.keys():
        # TODO: consider different approach if minimum tStamp based id assignment does not works
        targetIDkey = pri_labels[k]  # candKey is the values of aux but key for primary
        # check that candIDkey is within primary camera id keys
        if targetIDkey in aux_labels_prev.keys():
            pri_labels[k] = aux_labels_prev[targetIDkey]
        else:
            max_ID_prev+=1 #initialize new id in auxiliary camera
            pri_labels[k] = max_ID_prev
    # update labelsC2 based on labelsC25 in H5to2
    return pri_labels

def reform_aux_keys(curr_aux=None, prev_aux=None):
    #diff_factor =  prev_aux.keys()[0] - curr_aux.keys()[0]
    new_key = curr_aux.keys()[0]
    # TODO: diff_factor may be +ve or -ve
    reformed_prev_aux = collections.OrderedDict()
    for k in  prev_aux.keys():
        reformed_prev_aux[new_key] = prev_aux[k]
        new_key += 1
    assert reformed_prev_aux.keys()[0]==curr_aux.keys()[0] and reformed_prev_aux.keys()[-1]==curr_aux.keys()[-1]
    return reformed_prev_aux


if __name__ == "__main__":
    dataset = 'clasp2'
    benchmark = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/'
    mhtTrackers = '/home/siddique/Desktop/NEU_Data/TrackResult/'
    DFS = 1
    DistCorrect = True
    eval = True
    tracker = 'mht'

    if dataset == 'clasp2':
        experiment = 'exp2'
        tracklet_exp = 'exp1'
        raw_mht = 'det' + 'exp1'
    cams = {}
    MCparams = {}
    cam2cam = {}

    cams['C2'] = 2  # t_p
    cams['C9'] = 9  # t_a
    cams['C5'] = 5  # H5to2
    cams['C11'] = 11  # H5to11
    cams['C13'] = 13  # H13to11
    cams['C14'] = 14  # H13to14

    cam2cam['C9C2'] = 'H9to2'
    cam2cam['C2C5'] = 'H2to5'
    cam2cam['C5C11'] = 'H5to11'
    cam2cam['C11C13'] = 'H11to13'
    cam2cam['C13C14'] = 'H13to14'

    folder = []
    MC_tracklets = {}
    MC_labelMap = []
    cam2cam_vis = 1
    #
    # C9C2:------------------------------------------------------------------
    MCparams[cam2cam['C9C2']] = dafault_params(cam2cam['C9C2'], experiment, cams, mhtTrackers)
    label_map_update, MC_trklt_C9C2, fr_start, fr_end = get_cam2cam_matchedTrklt(MCparams[cam2cam['C9C2']],
                                                                                 cam2cam,
                                                                                 isDistorted=DistCorrect,
                                                                                 isseqID=True)
    assert len(MC_trklt_C9C2['PAorg']) == len(label_map_update)
    labelsC9, labelsC2 = split_labelMap(label_map_update, MC_trklt_C9C2)


    # call plot function to visualize the association correct
    if not cam2cam_vis:
        fr_start = 2100
        fr_end = 6000
        plot_trajectories_pairs(labelsC9,
                                labelsC2,
                                fr_start,
                                fr_end,
                                MC_trklt_C9C2,
                                cam2cam['C9C2'],
                                cams,
                                vis_rate=30,
                                isDistorted=DistCorrect,
                                metrics=eval,
                                save_imgs=0,
                                motion=0
                                )
        pdb.set_trace()
    ##C2C5:------------------------------------------------------------------------------------------
    # initialize camera pair parameters

    MCparams[cam2cam['C2C5']] = dafault_params(cam2cam['C2C5'], experiment, cams, mhtTrackers)
    label_map_update, MC_trklt_C5C2, fr_start, fr_end = get_cam2cam_matchedTrklt(MCparams[cam2cam['C2C5']], cam2cam,
                                                                                 isDistorted=DistCorrect, isseqID=False)
    assert len(MC_trklt_C5C2['PAorg']) == len(label_map_update)
    labelsC5, labelsC25 = split_labelMap(label_map_update, MC_trklt_C5C2) # aux, pri = split()
    labelsC5 = update_aux_ids(aux_labels=labelsC5, pri_labels=labelsC25, pri_labels_prev=labelsC2)
    # TODO: need to update the primary??
    if not cam2cam_vis:
        fr_start = 2100
        fr_end = 6000
        plot_trajectories_pairs(labelsC5,
                                labelsC2,
                                fr_start,
                                fr_end,
                                MC_trklt_C5C2,
                                cam2cam['C2C5'],
                                cams,
                                vis_rate=10,
                                isDistorted=DistCorrect,
                                metrics=eval,
                                save_imgs=0,
                                motion=0
                                )
        pdb.set_trace()
    ##C5C11:------------------------------------------------------------------------------------------
    # initialize camera pair parameters
    MCparams[cam2cam['C5C11']] = dafault_params(cam2cam['C5C11'], experiment, cams, mhtTrackers)
    label_map_update, MC_trklt_C5C11, fr_start, fr_end = get_cam2cam_matchedTrklt(MCparams[cam2cam['C5C11']], cam2cam,
                                                                                  isDistorted=DistCorrect,
                                                                                  isseqID=False)
    # since primary change, make sure that prev and vur aux labels starting from same index
    assert len(MC_trklt_C5C11['PAorg']) == len(label_map_update)
    labelsC511, labelsC11 = split_labelMap(label_map_update, MC_trklt_C5C11)  # aux, pri = ()
    # assert labelsC5.keys()[0]==labelsC511.keys()[0],'primary change, so current and previous auxiliary labelmap starting index should be equal but' \
    # ' found IndStartAprev {} and IndStartAcur {}'.format(labelsC5.keys()[0],labelsC511.keys()[0])
    # update primary cam ids
    # TODO: before updating primary (C11), make sure prev. auxiliary (C52) labels has same set of tracklet indexes (keys()) as in curr, aux (C511)
    # NOTE: reformed aux label dic only used for propagating labels to primary
    reformed_prev_labelsC5 = reform_aux_keys(curr_aux=labelsC511, prev_aux=labelsC5)

    # previous aux: labelsC5.keys(), current aux: labelsC511.keys()
    assert len(reformed_prev_labelsC5.keys())==len(labelsC511.keys())
    assert reformed_prev_labelsC5.keys()[0]==labelsC511.keys()[0],'primary change, so current and previous auxiliary labelmap starting index should be equal but' \
    'found IndStartAprev {} and IndStartAcur {}'.format(reformed_prev_labelsC5.keys()[0],labelsC511.keys()[0])
    labelsC11 = update_pri_ids(pri_labels=labelsC11, aux_labels=labelsC511,aux_labels_prev=reformed_prev_labelsC5)
    fr_start = 1
    fr_end = 13400
    if not cam2cam_vis:
        fr_start = 3000
        fr_end = 6000
        plot_trajectories_pairs(reformed_prev_labelsC5,
                                labelsC11,
                                fr_start,
                                fr_end,
                                MC_trklt_C5C11,
                                cam2cam['C5C11'],
                                cams,
                                vis_rate=30,
                                isDistorted=DistCorrect,
                                metrics=eval,
                                save_imgs=1,
                                motion=0
                                )
        pdb.set_trace()
    ##C11C13:------------------------------------------------------------------------------------------
    # initialize camera pair parameters
    MCparams[cam2cam['C11C13']] = dafault_params(cam2cam['C11C13'], experiment, cams, mhtTrackers)
    label_map_update, MC_trklt_C13C11, fr_start, fr_end = get_cam2cam_matchedTrklt(MCparams[cam2cam['C11C13']], cam2cam,
                                                                                   isDistorted=DistCorrect,
                                                                                   isseqID=False)
    # since primary change, make sure that prev and vur aux labels starting from same index
    assert len(MC_trklt_C13C11['PAorg']) == len(label_map_update)
    labelsC13, labelsC1113 = split_labelMap(label_map_update, MC_trklt_C13C11)  # aux, pri = ()
    assert labelsC11.keys()[0]==labelsC1113.keys()[0],'primary change, so current and previous auxiliary labelmap starting index should be equal but' \
    ' found IndStartAprev {} and IndStartAcur {}'.format(labelsC13.keys()[0],labelsC1113.keys()[0])
    # update aux cam ids
    labelsC13 = update_aux_ids(aux_labels=labelsC13, pri_labels=labelsC1113, pri_labels_prev=labelsC11)
    if not cam2cam_vis:
        fr_start, fr_end = 3500, 6000
        plot_trajectories_pairs(labelsC13,
                                labelsC11,
                                fr_start,
                                fr_end,
                                MC_trklt_C13C11,
                                cam2cam['C11C13'],
                                cams,
                                vis_rate=30,
                                isDistorted=DistCorrect,
                                metrics=eval,
                                save_imgs=0,
                                motion=0
                                )

        pdb.set_trace()
    ##C3C14:------------------------------------------------------------------------------------------
    # initialize camera pair parameters
    '''
    MCparams[cam2cam['C13C14']] = dafault_params(cam2cam['C13C14'], experiment, cams, mhtTrackers)
    label_map_update, MC_trklt_C13C14, fr_start, fr_end = get_cam2cam_matchedTrklt(MCparams[cam2cam['C13C14']], cam2cam,
                                                                                   isDistorted=DistCorrect,
                                                                                   isseqID=False)
    # since primary change, make sure that prev and vur aux labels starting from same index
    labelsC1314, labelsC14 = split_labelMap(label_map_update, MC_trklt_C13C14)  # aux, pri = ()
    # assert labelsC5.keys()[0]==labelsC511.keys()[0],'primary change, so current and previous auxiliary labelmap starting index should be equal but' \
    # ' found IndStartAprev {} and IndStartAcur {}'.format(labelsC5.keys()[0],labelsC511.keys()[0])
    # update primary cam ids
    # labelsC11 = update_pri_ids(pri_labels=labelsC11, aux_labels=labelsC511,aux_labels_prev=labelsC5)
    fr_start = 4000
    fr_end = 13400
    plot_trajectories_pairs(labelsC1314,
                            labelsC14,
                            fr_start,
                            fr_end,
                            MC_trklt_C13C14,
                            cam2cam,
                            cams,
                            vis_rate=30,
                            isDistorted=DistCorrect,
                            metrics=eval,
                            save_imgs=1,
                            motion=0
                            )
    '''
    if not os.path.exists(MCparams[cam2cam['C9C2']]['outPath']):
        os.makedirs(MCparams[cam2cam['C9C2']]['outPath'])

    if eval:
        # MC_tracklets['C2C9C5org'] = centroids2xywh(MC_tracklets['C2C9C5org'])
        track_for_mot = open(MCparams[cam2cam['C9C2']]['outPath'] + MCparams[cam2cam['C9C2']]['finalTrackFile'], 'w')
    # start and end frames are based on primary camera
    fr_start, fr_end = 500,3000
    plot_trajectories_MC(labelsC9,
                         labelsC2,
                         labelsC5,
                         labelsC11,
                         labelsC13,
                         fr_start,
                         fr_end,
                         MCparams,
                         cam2cam,
                         cams,
                         vis_rate=10,
                         isDistorted=DistCorrect,
                         metrics=eval,
                         save_imgs=1,
                         motion=0
                         )


