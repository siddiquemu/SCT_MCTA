from __future__ import division
import numpy as np
from scipy.spatial.distance import directed_hausdorff, cosine, mahalanobis
from hausdorff_utils import *
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
import copy
import similaritymeasures
from discrete_frechet import frechetdist
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
        A = np.array([[1315.7, 0.0, 989.6],
                       [0.0, 1313.0, 623.4],
                       [0.0, 0.0, 1.0]])
        dist_coeff = np.array([-0.415, 0.17,0.0, 0.0])  # why 4 or 5 coefficients are used in opencv????
    if cam == 5:  # assume that c2 and c5 have similar distortion
        A = np.array([[1315.7, 0.0, 989.6],
                       [0.0, 1313.0, 623.4],
                       [0.0, 0.0, 1.0]])
        dist_coeff = np.array([-0.415, 0.17, 0.0, 0.0])  # why 4 or 5 coefficients are used in opencv????
    if cam == 4:  # assume that c2 and c5 have similar distortion
        A = np.array([[1315.7, 0.0, 989.6],
                       [0.0, 1313.0, 623.4],
                       [0.0, 0.0, 1.0]])
        dist_coeff = np.array([-0.415, 0.17, 0.0, 0.0]) # why 4 or 5 coefficients are used in opencv????

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

def xywh2x1y1x2y2(tracklet):
    for tl in tracklet:
        for bb in tl:
            bb[3] = bb[1] + bb[3]
            bb[4] = bb[2] + bb[4]
    return tracklet


def applyTransform(source_corners, H):
    dest_corners = np.empty(2)
    w = H[2][0] * source_corners[0] + H[2][1] * source_corners[1] + H[2][2] * 1
    dest_corners[0] = (H[0][0] * source_corners[0] + H[0][1] * source_corners[1] + H[0][2] * 1) / w
    dest_corners[1] = (H[1][0] * source_corners[0] + H[1][1] * source_corners[1] + H[1][2] * 1) / w
    return dest_corners


# ---------------ok siddique
#homography is computed on undistorted image
# TODO: improve distortion coefficient
def project_tracklets(in_tracklets, H, motion=False, isDistorted=False, cam=9):
    # **in_tracklets: [fr, x, y, w, h, vx, vy, app-128]
    out_tracklets = []
    active_trklt = []
    Hv = copy.deepcopy(H)
    Hv[0, 2], Hv[1, 2] = 0, 0
    # [fr,cx,cy,x,y,vx,vy]<<< projected coordinates: center is used
    for i, trklt in enumerate(in_tracklets):
        # TODO: is it necessary to do any transformation for appearance: no
        # since center and motion transform in-place, we can keep an return app wo transformation.
        if isDistorted:
            dist_coeff, A = camera_intrinsics(cam)
            #generate undistorted top-left and bottom-right
            x1y1 = copy.deepcopy(trklt[:, 1:3])
            #x2y2 = copy.deepcopy(trklt[:, 1:3]+trklt[:, 3:5])

            x1y1 = undistorted_coords(x1y1, dist_coeff, A)
            trklt[:, 1:3] = x1y1
            # TODO: undistorted bottom-right and then compute undistorted width-height: later do both task in-place
            #x2y2 = undistorted_coords(x2y2, dist_coeff, A)
            #TODO: verify that w>0 and h>0
            #trklt[:, 3:5] = x2y2 - x1y1
            if motion:
                #generate undistorted motion
                vxvy = copy.deepcopy(trklt[:, 5:7])
                vxvy = undistorted_coords(vxvy, dist_coeff, A)
                trklt[:, 5:7] =vxvy
        for bb in trklt:
            bbt1 = applyTransform(bb[1:3], H)
            cxcy = applyTransform(bb[1:3] + bb[3:5] / 2.0, H)
            if motion:
                vxvy = applyTransform(bb[5:7], Hv)
                bbt2 = cxcy - bbt1  # projected w/2,h/2
                bb[1:7] = np.concatenate([cxcy, bbt2, vxvy])  # projected: [cx,cy,w/2,h/2,vx,vy]
            else:
                bbt2 = cxcy - bbt1  # projected w/2,h/2
                bb[1:5] = np.concatenate([cxcy, bbt2])  # projected: [cx,cy,w/2,h/2]
        # Delete tracklets that don't have any detection visible in the second camera
        # TODO: There must be a smarter and faster way to do that
        # cy+h/2 > 0 and w/2>0
        # Instead of zero use some positive constant
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


def vis_overlapped_trklt(final_tracklet1,final_tracklet2,i,j,out_dir):
    from mpl_toolkits.mplot3d import Axes3D
    out_dir = out_dir+'/feature_comp/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    import matplotlib.pyplot as plt
    Dh_c = max(directed_hausdorff(final_tracklet1[:, 1:3], final_tracklet2[:, 1:3])[0],
             directed_hausdorff(final_tracklet2[:, 1:3], final_tracklet1[:, 1:3])[0])

    Dh_m = max(directed_hausdorff(final_tracklet1[:, 3:5], final_tracklet2[:, 3:5])[0],
             directed_hausdorff(final_tracklet2[:, 3:5], final_tracklet1[:, 3:5])[0])
    Dh_m_cos = cosine(final_tracklet2[:, 3:5].flatten(), final_tracklet1[:, 3:5].flatten())

    Dh_a = max(directed_hausdorff(final_tracklet1[:, 5:-1], final_tracklet2[:, 5:-1])[0],
             directed_hausdorff(final_tracklet2[:, 5:-1], final_tracklet1[:, 5:-1])[0])
    d_h = Dh_c+Dh_m+Dh_a
    data = (final_tracklet1[:,0:3], final_tracklet2[:,0:3])
    colors = ("red", "green")
    groups = ("primary {}".format(i), "auxiliary {}".format(j), 'time')
    # Create plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('t')
    for points, color, group in zip(data, colors, groups):
        x, y, z =points[:,1],points[:,2],points[:,0]
        ax.scatter(x, y, z, alpha=0.8, c=color, edgecolors='none', s=30, label=group)

    plt.title('d_h_c {:.2f},d_h_m {:.2f},d_h_m_cos {:.2f},d_h_a {:.2f},d_h {:.2f}'.format(Dh_c, Dh_m, Dh_m_cos, Dh_a, d_h))
    plt.legend(loc=3)
    plt.savefig(out_dir+'{}_{}.png'.format(i,j), dpi=300)
    plt.close()

def associate_tracklets_DFS_MC(params, cam2cam, isDistorted=False, motion=False, appearance=False, max_dist=400):
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
                #if tracklet1[-1, 0] < tracklet2[0, 0] or tracklet1[0, 0] > tracklet2[-1, 0]:

                    #cost[i, j] = np.inf
                # Search for the overlapping portions of multicamera tracklets
               # else:
                # TODO: There must be a more compact way of doing this (find the overlapped region)
                # TODO: Assess whether the timestamp should be considered in the distance computation as well
                common_t_stamps = np.array(list(set(list(tracklet2[:, 0])).intersection(list(tracklet1[:, 0]))))
                if len(common_t_stamps)>60:
                    assert len(common_t_stamps) > 60, 'Non-overlaped T_p, T_a can not be associated'
                    # tilde_t_pa[j, i] = min(common_t_stamps)
                    # 2D or 4D feature: [fr,cx,cy,w/2,h/2,5-vx,6-vy]
                    # augment feature (4D) with velocity (aux,primary): [Cx,Cy,dot.Cx,dot.Cy]
                    # tracklet2,tracklet1 = augment_feature_with_velocity(tracklet2[:,0:3], tracklet1[:,0:3])
                    if motion:
                        featIndx = [0, 1, 2, 5, 6]
                    if motion and appearance:
                        assert tracklet1.shape[1]==135
                        #skip width and height model
                        featIndx = [l for l in range(0, tracklet1.shape[1]) if l not in [3,4]]
                    else:
                        featIndx = [0, 1, 2, 3, 4] # 3,4- width and height are currently not used
                    #if len(common_t_stamps)>120:
                        #common_t_stamps = common_t_stamps[:120]

                    mask1 = np.isin(tracklet1[:, 0], common_t_stamps)
                    final_tracklet1 = tracklet1[mask1][:, featIndx]

                    if isDistorted:
                        dist_coeff, A = camera_intrinsics(cam=params['Pid'])  # 2:c5 or 9:c11
                        x1y1 = copy.deepcopy(final_tracklet1[:, 1:3])

                        x1y1 = undistorted_coords(x1y1, dist_coeff, A)
                        final_tracklet1[:, 1:3] = x1y1

                        #get undistorted motion
                        if motion:
                            vxvy = copy.deepcopy(final_tracklet1[:, 3:5])
                            vxvy = undistorted_coords(vxvy, dist_coeff, A)
                            final_tracklet1[:, 3:5] = vxvy
                    # normalize transformed location and motion
                    final_tracklet1[:, 1:5] = final_tracklet1[:, 1:5] / [1920.0, 1080.0, 50.0,
                                                                         50.0]  # max(abs(final_tracklet1[:,3]))

                    mask2 = np.isin(tracklet2[:, 0], common_t_stamps)
                    final_tracklet2 = tracklet2[mask2][:, featIndx]
                    final_tracklet2[:, 1:5] = final_tracklet2[:, 1:5] / [1920.0, 1080.0, 50.0, 50.0]
                    if  not motion:
                        d_h = max(directed_hausdorff(final_tracklet1[:, 1:3], final_tracklet2[:, 1:3])[0],
                        directed_hausdorff(final_tracklet2[:, 1:3], final_tracklet1[:, 1:3])[0])
                        print('Processing tracklets {} and {} without motion for cost {}'.format(i, j, d_h))
                    if  motion and appearance:
                        #pdb.set_trace()
                        #d_h_center = directed_hausdorff(final_tracklet1[:, 1:3], final_tracklet2[:, 1:3])[0]
                        #d_h_app = directed_hausdorff(final_tracklet1[:, 5::], final_tracklet2[:, 5::])[0]
                        #d_h_motion = directed_hausdorff(final_tracklet1[:, 3:5], final_tracklet2[:, 3:5])[0]
                        #Hausdorff
                        #d_h_center = hausdorff(final_tracklet1[:, 1:3], final_tracklet2[:, 1:3])
                        #d_h_app = hausdorff(final_tracklet1[:, 5::], final_tracklet2[:, 5::])
                        #d_h_motion = hausdorff(final_tracklet1[:, 3:5], final_tracklet2[:, 3:5])

                        #Hausdorff average
                        #d_h_center = hausdorff_avg(final_tracklet1[:, 1:3], final_tracklet2[:, 1:3])
                        #d_h_app = hausdorff_avg(final_tracklet1[:, 5::], final_tracklet2[:, 5::])
                        #d_h_motion = hausdorff_avg(final_tracklet1[:, 3:5], final_tracklet2[:, 3:5])

                        #Hausdorff weighted average
                        #d_h_center = hausdorff_wavg(final_tracklet1[:, 1:3], final_tracklet2[:, 1:3])
                        d_h_app = hausdorff_wavg(final_tracklet1[:, 5::], final_tracklet2[:, 5::])
                        d_h_motion = hausdorff_wavg(final_tracklet1[:, 3:5], final_tracklet2[:, 3:5])

                        #discrete_frechet
                        # larger computation time
                        #https://pypi.org/project/similaritymeasures/
                        #pdb.set_trace()
                        #pdb.set_trace()
                        #iterative method: 4-for loop O(N^4)
                        #d_h_center = similaritymeasures.frechet_dist(final_tracklet1[:, 1:3], final_tracklet2[:, 1:3])
                        #pdb.set_trace()
                        #use recursive method: maximum recursion limit needs to be updated
                        # TODO: How to select the subset of tracklet to compute frechet distance faster??
                        scan_intrvl = 240 #C5C11:240
                        if len(final_tracklet1[:, 1:3])>scan_intrvl:
                            sys.setrecursionlimit(len(final_tracklet1[:, 1:3])*len(final_tracklet2[:, 1:3]))
                            d_h_center = frechetdist.frdist(final_tracklet1[:scan_intrvl, 1:3], final_tracklet2[:scan_intrvl, 1:3])
                            #d_h_app = similaritymeasures.frechet_dist(final_tracklet1[:scan_intrvl, 5::], final_tracklet2[:scan_intrvl, 5::])
                            #d_h_motion = frechetdist.frdist(final_tracklet1[:scan_intrvl, 3:5], final_tracklet2[:scan_intrvl, 3:5])
                        else:
                            sys.setrecursionlimit(len(final_tracklet1[:, 1:3]) * len(final_tracklet2[:, 1:3]))
                            d_h_center = frechetdist.frdist(final_tracklet1[:, 1:3],
                                                            final_tracklet2[:, 1:3])
                            #d_h_app = similaritymeasures.frechet_dist(final_tracklet1[:, 5::], final_tracklet2[:, 5::])
                            #d_h_motion = frechetdist.frdist(final_tracklet1[:, 3:5], final_tracklet2[:, 3:5])
                        #d_h_motion = cosine(final_tracklet2[:, 3:5].flatten(), final_tracklet1[:, 3:5].flatten())


                        d_h = d_h_center #+d_h_motion + d_h_app
                        d_h_all = d_h_center +d_h_motion + d_h_app
                        print('Processing {} and {} with l_c {} l_m {} l_a {} and d_h {}'.format(i, j, d_h_center,d_h_motion,d_h_app, d_h_all))
                    else:
                        dOverlap = 0.3  # final_tracklet1.shape[0]/float(max(tracklet1.shape[0],tracklet2.shape[0]))
                        d_h = (1-dOverlap)*directed_hausdorff(final_tracklet1[:,1:3], final_tracklet2[:,1:3])[0]  \
                         + (dOverlap)*cosine(final_tracklet2[:,3:5].flatten(), final_tracklet1[:,3:5].flatten())
                        print('Processing tracklets {} and {} with motion max {} for cost {}'.format(i, j, max(final_tracklet1[:, 3:5]), d_h))
                        # d_cos = 1-cosine(final_tracklet2.flatten(), final_tracklet1.flatten())
                        # d_mah = mahalanobis21(final_tracklet1, final_tracklet2)
                        #if cam2cam in ['H9to2']:
                            #d_h = directed_hausdorff(final_tracklet1[:, 1:3], final_tracklet2[:, 1:3])[0]
                    if d_h <= max_dist:
                        cost[i, j] = d_h
                        tilde_t_pa[i, j] = min(common_t_stamps)
                    # vis-overlapped tracklets: 2D scatter plot
                    if cam2cam=='H10to11':# and i==11 and j==13:
                        vis_overlapped_trklt(final_tracklet1,final_tracklet2,i,j,params['outPath'])
            j = j + 1
        i = i + 1
    # DEBUG
    #import matplotlib.pyplot as plt
    #hist = np.histogram(cost[cost < 10000])
    #plt.hist(cost[cost < 10000])
    #pdb.set_trace()
    while len(cost[cost <= max_dist]) > 0:
        row_ind, col_ind = linear_sum_assignment_with_inf(cost)
        Mincosts = cost[row_ind, col_ind]
        idx = np.where(Mincosts <= max_dist)[0]
        #if cam2cam=='H5to11':
           # pdb.set_trace()
        for i in idx:
            matches.append((Mincosts[i], tilde_t_pa[row_ind[i], col_ind[i]], row_ind[i],
                            col_ind[i]))  # tilde_t_pa[row_ind[i],col_ind[i]]  cam2_tracklets[col_ind[i]][0,0]

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

    return matches, tilde_t_pa


def interpolate(tracks):
    interpolated = {}
    for i, track in tracks.items():
        interpolated[i] = {}
        frames = []
        x0 = []
        y0 = []
        x1 = []
        y1 = []

        for f, bb in track.items():
            frames.append(f)
            x0.append(bb[0])
            y0.append(bb[1])
            x1.append(bb[2])
            y1.append(bb[3])

        if len(frames) > 1:
            x0_inter = interp1d(frames, x0)
            y0_inter = interp1d(frames, y0)
            x1_inter = interp1d(frames, x1)
            y1_inter = interp1d(frames, y1)

            for f in range(min(frames), max(frames) + 1):
                bb = np.array([x0_inter(f), y0_inter(f), x1_inter(f), y1_inter(f)])
                interpolated[i][f] = bb
        else:
            interpolated[i][frames[0]] = np.array([x0[0], y0[0], x1[0], y1[0]])

    return interpolated

def interpolate_left_track(track):
    #interpolate left tracks for 30 frames to associate with immediately associated tracks
    #from scipy import interpolate

    #x = np.arange(0,10)
    #y = np.exp(-x/3.0)
    #f = interpolate.interp1d(x, y, fill_value='extrapolate')

    #print f(9)
    #print f(11)

    from scipy.interpolate import interp1d
    interpolated = []
    frames = []
    x0 = []
    y0 = []
    x1 = []
    y1 = []

    for bb in track:
        frames.append(bb[0])
        x0.append(bb[1])
        y0.append(bb[2])
        x1.append(bb[3])
        y1.append(bb[4])

    if len(frames) > 1:
        x0_inter = interp1d(frames, x0)
        y0_inter = interp1d(frames, y0)
        x1_inter = interp1d(frames, x1)
        y1_inter = interp1d(frames, y1)

        for f in range(int(min(frames)), int(max(frames)) + 1):
            pdb.set_trace()
            bb = np.array([x0_inter(f), y0_inter(f), x1_inter(f), y1_inter(f)])
            interpolated.append(bb)
    else:
        interpolated.append(np.array([x0[0], y0[0], x1[0], y1[0]]))

    return interpolated


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


def sort_tracklets(tracker):
    # when it is needed to sort trackelts detections in terms of time-stamps
    tracker = np.array(list(tracker))
    tracker = np.array(sorted(tracker, key=lambda x: x[0]))
    return tracker


def delete_all(demo_path, fmt='png'):
    filelist = glob.glob(os.path.join(demo_path, '*.' + fmt))
    if len(filelist) > 0:
        for f in filelist:
            if os.path.exists(f):
                os.remove(f)

def plot_trajectories_MC(label_map,
                         mc_all_trklts,
                         cam_trklts_size,
                         index_factor,
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
    # BGR
    # magenta = (255,0,255)
    # yellow = (0,255,255)
    # green = (0,255,0)
    # blue = (255,0,0)
    # plt.imshow(np.ones((1080, 1920, 3)))
    # fr_offset = 48#11to13
    # fr_offset = 29  # 9to2, cam2 lag = 29

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 640, 480)
    for fr in range(fr_start, fr_end):
        # 9A:9 fr-15, 11 fr-15
        if (fr) % vis_rate == 0:  # 15 for 6A,7A,.. 3 for 5A, 5B.. 30 for 9A
            # try:
            # Master camera (C9) fr need to be synced:
            imgC2 = cv2.imread('{}{:06d}.png'.format(MCparams[cam2cam['C9C2']]['PImgsPath'],
                                                     fr + (44)))
            # source camera 9,5 - folder2: + fr_offset for 9to2, + fr_offset for 5to11
            imgC9 = cv2.imread('{}{:06d}.png'.format(MCparams[cam2cam['C9C2']]['AImgsPath'],
                                                     fr + 0))
            imgC5 = cv2.imread('{}{:06d}.png'.format(MCparams[cam2cam['C5C2']]['AImgsPath'],
                                                     fr + (42)))
            imgC11 = cv2.imread('{}{:06d}.png'.format(MCparams[cam2cam['C11C5']]['AImgsPath'],
                                                      fr + (1)))
            imgC13 = cv2.imread('{}{:06d}.png'.format(MCparams[cam2cam['C13C5']]['AImgsPath'],
                                                      fr + (-97))) #46
            imgC14 = cv2.imread('{}{:06d}.png'.format(MCparams[cam2cam['C14C13']]['AImgsPath'],
                                                      fr + (-106)))

            c = index_factor[2]
            # visulaize box on C2: H9to2
            for i_p, bb in enumerate(mc_all_trklts[index_factor[2]:index_factor[2]+cam_trklts_size[2]]):
                # To show the association with the projections (tracklets2)
                for i in range(1, len(bb)):
                    if fr_start <= bb[i, 0]+44 <= fr+44:
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
                            if bb[i, 0]+44 == fr+44:# and labelsC2[c] in [27,18,14, 16, 22]:

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
                                cv2.putText(imgC2, 'P{}'.format(label_map[c]),
                                            (int(cxy_undist[0, 0]), int(cxy_undist[0, 1])),
                                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5, cv2.LINE_AA)
                c = c + 1

            # show bbox on C9: visualize the actual target identity in auxiliary camera (9,5)
            c = index_factor[9]
            # tracks9 = tracks[C2:C2+C9]
            for bb in mc_all_trklts[index_factor[9]:index_factor[9]+cam_trklts_size[9]]:
                # TODO: Set the colors of the rectangles
                for i in range(1, len(bb)):
                    if fr_start <= bb[i, 0] <= fr:
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
                            if bb[i, 0] == fr:# and labelsC9[c] in [27,18,14,16,22]:
                                cv2.putText(imgC9, 'P{}'.format(label_map[c]),
                                            (int(bb[i, 1] + bb[i, 3] / 2), int(bb[i, 2] + bb[i, 4] / 2)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 2555, 0), 5, cv2.LINE_AA)
                                cv2.rectangle(imgC9, (int(bb[i, 1]), int(bb[i, 2])),
                                              (int(bb[i, 1] + bb[i, 3]), int(bb[i, 2] + bb[i, 4])),
                                              (255, 0, 0), 5, cv2.LINE_AA)
                c = c + 1
            # show bbox on C5: use H5to11 instead of H5to2 since labels are already propagated to H5to11
            c = index_factor[5]
            for bb in mc_all_trklts[index_factor[5]:index_factor[5]+cam_trklts_size[5]]:
                # TODO: Set the colors of the rectangles
                for i in range(1, len(bb)):
                    if fr_start <= bb[i, 0] <= fr + 41:
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
                            if bb[i, 0] == fr + 41:# and labelsC5[c] in [27,18,14,16,22]:
                                cv2.putText(imgC5, 'P{}'.format(label_map[c]),
                                            (int(bb[i, 1] + bb[i, 3] / 2), int(bb[i, 2] + bb[i, 4] / 2)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 2555, 0), 5, cv2.LINE_AA)
                                cv2.rectangle(imgC5, (int(bb[i, 1]), int(bb[i, 2])),
                                              (int(bb[i, 1] + bb[i, 3]), int(bb[i, 2] + bb[i, 4])),
                                              (255, 0, 0), 5, cv2.LINE_AA)
                c = c + 1

            # show bbox on C11
            c = index_factor[11]
            for bb in mc_all_trklts[index_factor[11]:index_factor[11]+cam_trklts_size[11]]:
                # TODO: Set the colors of the rectangles
                for i in range(1, len(bb)):
                    if fr_start <= bb[i, 0] <= fr + 1:
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
                            if bb[i, 0] == fr +1:# and labelsC11[c] in [27,18,14,16,22] :
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
                                cv2.putText(imgC11, 'P{}'.format(label_map[c]),
                                            (int(cxy_undist[0, 0]), int(cxy_undist[0, 1])),
                                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
                c = c + 1
            # show bbox on C13
            c = index_factor[13]
            for bb in mc_all_trklts[index_factor[13]:index_factor[13]+cam_trklts_size[13]]:
                # TODO: Set the colors of the rectangles
                for i in range(1, len(bb)):
                    if fr_start <= bb[i, 0] <= fr - 97:
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
                            if bb[i, 0] == fr - 97: #and labelsC13[c] in [27,18,14,16,22]:
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
                                cv2.putText(imgC13, 'P{}'.format(label_map[c]),
                                            (int(cxy_undist[0, 0]), int(cxy_undist[0, 1])),
                                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
                c = c + 1

            # show bbox on C14
            c = index_factor[14]
            for bb in mc_all_trklts[index_factor[14]:index_factor[14]+cam_trklts_size[14]]:
                # TODO: Set the colors of the rectangles
                for i in range(1, len(bb)):
                    if fr_start <= bb[i, 0] <= fr - 106:
                        # image = cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength = 0.5)
                        if save_imgs and motion and bb[i, 0] % 5 == 0:
                            # 2d velocity plot
                            xx = int(
                                (bb[i, 1] + bb[i, 3] / 2.0) + bb[i, 5] * 2 * (bb[i, 0] - bb[i - 1, 0]))  # xx=x0+vxt
                            yy = int((bb[i, 2] + bb[i, 4] / 2.0) + bb[i, 6] * 2 * (bb[i, 0] - bb[i - 1, 0]))
                            cv2.arrowedLine(imgC14,
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
                            if bb[i, 0] == fr - 106: #and labelsC14[c] in [27,18,14,16,22]:
                                cv2.rectangle(imgC14, (int(bb[i, 1]), int(bb[i, 2])),
                                              (int(bb[i, 1] + bb[i, 3]), int(bb[i, 2] + bb[i, 4])),
                                              (255, 0, 0), 5, cv2.LINE_AA)
                                if isDistorted:
                                    dist_coeff, A = camera_intrinsics(cam=cams['C14'])
                                    cxy_undist = np.copy(bb[i, 1:3] + bb[i, 3:5] / 2.0)
                                    cxy_undist = undistorted_coords(cxy_undist.reshape(1, 1, 2), dist_coeff, A)
                                else:
                                    cxy_undist = np.copy(bb[i, 1:3] + bb[i, 3:5] / 2.0).reshape(1, 2)
                                # show track id: undistorted centers
                                cv2.putText(imgC14, 'P{}'.format(label_map[c]),
                                            (int(cxy_undist[0, 0]), int(cxy_undist[0, 1])),
                                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
                c = c + 1
            # translate image
            imgC2 = cv2.copyMakeBorder(imgC2, 0, 0, 0, 250, cv2.BORDER_CONSTANT, value=0)
            imgC9 = cv2.copyMakeBorder(imgC9, 0, 0, 250, 0, cv2.BORDER_CONSTANT, value=0)
            imgC13 = cv2.copyMakeBorder(imgC13, 0, 0, 700,0, cv2.BORDER_CONSTANT, value=0)
            imgC14 = cv2.copyMakeBorder(imgC14, 0, 0,0 , 700, cv2.BORDER_CONSTANT, value=0)

            imgC9C2 = cv2.vconcat([imgC9, imgC2])
            imgC11C5 = cv2.vconcat([imgC11, imgC5])
            final_img = cv2.hconcat([imgC11C5, imgC9C2])  # img: c4, img2: c2

            #blankImg = np.zeros(shape=[1080, 1920, 3], dtype=np.uint8)
           # img13Blnk = cv2.vconcat([imgC13, blankImg])
            img1314 = cv2.vconcat([imgC14, imgC13])
            final_img = cv2.hconcat([img1314, final_img])


            final_img = cv2.resize(final_img, (int(1.5 * 1920), 1080), interpolation=cv2.INTER_AREA)

            cv2.imshow("image", final_img)
            if save_imgs:
                # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                cv2.imwrite(MCparams['demo_path'] + '/{:06d}.png'.format(fr), final_img)
            cv2.waitKey(10)
            # except:
            # print("Frame {} not found".format(fr))
            # continue


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

    if not os.path.exists(MC_trklt_C9C2['outPath']):
        os.makedirs(MC_trklt_C9C2['outPath'])
    #delete_all(MC_trklt_C9C2['outPath'])
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("image", 640, 480)
    for fr in range(fr_start, fr_end):
        if (fr) % vis_rate == 0:
            try:
                img = cv2.imread('{}{:06d}.png'.format(MC_trklt_C9C2['PImgsPath'], fr + MC_trklt_C9C2['fr_offsetPVis']))
                img2 = cv2.imread('{}{:06d}.png'.format(MC_trklt_C9C2['AImgsPath'], fr + MC_trklt_C9C2['fr_offsetAVis']))
                print('Frame {} for {}'.format(fr, cam2cam))
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
                                cv2.putText(img, 'P{}'.format(labelCam2[c]), (int(bb[i - 1, 1]), int(bb[i - 1, 2])),
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
                                cv2.putText(img2, 'P{}'.format(labelCam2[c]),
                                            (int(bb[i, 1] + bb[i, 3] / 2), int(bb[i, 2] + bb[i, 4] / 2)),
                                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 2555, 0), 5, cv2.LINE_AA)
                                cv2.rectangle(img2, (int(bb[i, 1]), int(bb[i, 2])),
                                              (int(bb[i, 1] + bb[i, 3]), int(bb[i, 2] + bb[i, 4])),
                                              (255, 0, 0), 5, cv2.LINE_AA)
                    c = c + 1
                if cam2cam == 'H9to2':
                    final_img = cv2.vconcat([img2, img])
                if cam2cam == 'H5to2':
                    final_img = cv2.hconcat([img2, img])  # img: c5, img2: c2
                if cam2cam == 'H5to11':
                    final_img = cv2.vconcat([img2, img])
                if cam2cam == 'H5to13':
                    final_img = cv2.hconcat([img2, img])  # img: c11, img2: c13
                if cam2cam == 'H13to11':
                    img2 = cv2.copyMakeBorder(img2, 0, 0, 320, 0, cv2.BORDER_CONSTANT, value=0)
                    img = cv2.copyMakeBorder(img, 0, 0, 0, 320, cv2.BORDER_CONSTANT, value=0)
                    final_img = cv2.hconcat([img2, img])  # img: c11, img2: c13
                if cam2cam == 'H13to14':
                    img = cv2.copyMakeBorder(img, 0, 0, 700, 0, cv2.BORDER_CONSTANT, value=0)
                    img2 = cv2.copyMakeBorder(img2, 0, 0, 0, 700, cv2.BORDER_CONSTANT, value=0)
                    final_img = cv2.vconcat([img2, img])  # img: c13, img2: c14

                #cv2.imshow("image", final_img)
                if save_imgs:
                    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(MC_trklt_C9C2['outPath'] + '/{:06d}.png'.format(fr), final_img)
                #cv2.waitKey(30)
            except:
                # print("Frame {} not found".format(fr))
                continue
    #cv2.destroyAllWindows()


def dafault_params(cam2cam, experiment, cams, mhtTrackers, tracker_motion=False):
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
                      [268, -1402, 1.00]]
        param['fr_offsetAVis'] = 0  # cam9 delay
        param['fr_offsetPVis'] = 46  ## cam2 delay
        param['H'] = np.transpose(param['H'])
        param['d_th'] = 0.30  # #multi-task cost: 0.4, only loc: 0.25

        param['primary'] = benchmark + 'exp2/Results_train/tracktor_trained_app/cam02exp2.mp4.txt'
        param['auxiliary'] = benchmark + 'exp2/Results_train/tracktor_trained_app/cam09exp2.mp4.txt'

        param['finalTrackFile'] = 'cam09exp2_MCTA.txt'
        # param['finalTrackFileC2'] = 'cam02exp2_MCTA.txt'
        param['PImgsPath'] = benchmark + experiment + '/imgs/cam{:02d}'.format(cams[
                                                                                   'C2']) + experiment + '.mp4/'  # benchmark + 'test_data/exp1'+ '/imgs/cam{:02d}'.format(cams['C2']) + 'exp1' + '.mp4/'
        # benchmark + experiment + '/imgs/cam{:02d}'.format(cams['C2']) + tracklet_exp + '.mp4/'
        param['AImgsPath'] = benchmark + experiment + '/imgs/cam{:02d}'.format(cams[
                                                                                   'C9']) + experiment + '.mp4/'  # benchmark + 'test_data/exp1'+ '/imgs/cam{:02d}'.format(cams['C9']) + 'exp1' + '.mp4/'
        # benchmark + experiment + '/imgs/cam{:02d}'.format(cams['C9']) + tracklet_exp + '.mp4/'
        param['outPath'] = benchmark + experiment + '/Results/MCTA/vis/cam{}to{}/'.format(cams['C9'], cams['C2'])

    if cam2cam == 'H5to2':
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
        param['d_th'] = 0.32  #multi-task cost: 0.32, only loc: 0.2,
        param['primary'] = benchmark+'exp2/Results_train/tracktor_trained_app/cam02exp2.mp4.txt'
        param['auxiliary'] = benchmark+'exp2/Results_train/tracktor_trained_app/cam05exp2.mp4.txt'
        #param['primary'] = '/home/siddique/Desktop/NEU_Data/May_Demo_Trackers/exp2-train/cam02exp2.mp4imgtestTracksv3.txt' #v2
        #param['auxiliary'] = '/home/siddique/Desktop/NEU_Data/May_Demo_Trackers/exp2-train/cam05exp2.mp4imgtestTracksv3.txt'
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
        param['Pid'] = 5
        param['Aid'] = 11
        # C11: pri, C5: aux : t_c5 - t_c11 = 22
        param['tOffsetPstart'] = -22 #(24-1) #always subtract from time advance camera
        param['tOffsetAstart'] = 0   #(1-1)
        # similarity based homography: TODO: compute homography using undistorted image
        # H5to11
        '''
                param['H'] = [[0.9704, 0.1027, 0.00],
                      [-0.1027, 0.9704, 0.00],
                      [37.211, 514.028, 1.00]]
            0.9453   -0.0154         0
    0.0154    0.9453         0
   23.8510  508.6742    1.0000'''
        param['H'] = [[0.935, -0.011, 0.00],
                      [0.011, 0.935, 0.00],
                      [35.86, 511.47, 1.00]]

        param['fr_offsetPVis'] = 22
        param['fr_offsetAVis'] = 0## C5: 24_frame_at_time == C11: 1_frame_at_time > for 29 at C5 we will pick 6 from C11
        param['H'] = np.transpose(param['H'])
        # H11to5
        if param['Pid'] == 5:
            param['H'] = np.linalg.inv(param['H'])
        param['d_th'] = 0.37 # multi-task cost: 0.45, only loc: 0.17, l+m+a=1.03, l_c (weighted ghd) = 0.22

        param['primary'] = benchmark + 'exp2/Results_train/tracktor_trained_app/cam05exp2.mp4.txt'
        param['auxiliary'] = benchmark + 'exp2/Results_train/tracktor_trained_app/cam11exp2.mp4.txt'
        param['finalTrackFile'] = 'cam11exp2_MCTA.txt'
        # param['finalTrackFileC2'] = 'cam02exp2_MCTA.txt'
        param['PImgsPath'] = benchmark + experiment + '/imgs/cam{:02d}'.format(cams[
                                                                                   'C5']) + experiment + '.mp4/'  # benchmark + 'test_data/exp1'+ '/imgs/cam{:02d}'.format(cams['C2']) + 'exp1' + '.mp4/'
        # benchmark + experiment + '/imgs/cam{:02d}'.format(cams['C2']) + tracklet_exp + '.mp4/'
        param['AImgsPath'] = benchmark + experiment + '/imgs/cam{:02d}'.format(cams[
                                                                                   'C11']) + experiment + '.mp4/'  # benchmark + 'test_data/exp1'+ '/imgs/cam{:02d}'.format(cams['C9']) + 'exp1' + '.mp4/'
        # benchmark + experiment + '/imgs/cam{:02d}'.format(cams['C9']) + tracklet_exp + '.mp4/'
        param['outPath'] = benchmark + experiment + '/Results/MCTA/vis/cam{}to{}/'.format(cams['C11'], cams['C5'])

    if cam2cam == 'H5to13':
        param = {}
        param['Pid'] = 5
        param['Aid'] = 13
        # C11: pri, C5: aux
        param['tOffsetPstart'] = 0  # 13>1 == 5>71
        param['tOffsetAstart'] = 72  #
        # similarity based homography: TODO: compute homography using undistorted image
        # H5to13
        '''
    0.9019   -0.1053         0
    0.1053    0.9019         0
  789.2232  539.3645    1.0000
        '''
        param['H'] = [[0.9, -0.105, 0.00],
                      [0.105, 0.9, 0.00],
                      [950.22, 650.36, 1.00]]

        param['fr_offsetPVis'] = 0
        param['fr_offsetAVis'] = -72  ## T_c2 - T_c11 = offset
        param['H'] = np.transpose(param['H'])
        # H13to5
        if param['Pid'] == 5:
            param['H'] = np.linalg.inv(param['H'])
        param['d_th'] = 0.37  # multi-task cost: 0.45, only loc: 0.17
        param['primary'] = benchmark + 'exp2/Results_train/tracktor_trained_app/cam05exp2.mp4.txt'
        param['auxiliary'] = benchmark + 'exp2/Results_train/tracktor_trained_app/cam13exp2.mp4.txt'
        # param['primary'] = '/home/siddique/Desktop/NEU_Data/May_Demo_Trackers/exp2-train/cam05exp2.mp4imgtestTracksv3.txt'
        # param['auxiliary'] = '/home/siddique/Desktop/NEU_Data/May_Demo_Trackers/exp2-train/cam13exp2.mp4imgtestTracksv3.txt'
        param['finalTrackFile'] = 'cam13exp2_MCTA.txt'
        # param['finalTrackFileC2'] = 'cam02exp2_MCTA.txt'
        param['PImgsPath'] = benchmark + experiment + '/imgs/cam{:02d}'.format(cams[
                                                                                   'C5']) + experiment + '.mp4/'  # benchmark + 'test_data/exp1'+ '/imgs/cam{:02d}'.format(cams['C2']) + 'exp1' + '.mp4/'
        # benchmark + experiment + '/imgs/cam{:02d}'.format(cams['C2']) + tracklet_exp + '.mp4/'
        param['AImgsPath'] = benchmark + experiment + '/imgs/cam{:02d}'.format(cams[
                                                                                   'C13']) + experiment + '.mp4/'  # benchmark + 'test_data/exp1'+ '/imgs/cam{:02d}'.format(cams['C9']) + 'exp1' + '.mp4/'
        # benchmark + experiment + '/imgs/cam{:02d}'.format(cams['C9']) + tracklet_exp + '.mp4/'
        param['outPath'] = benchmark + experiment + '/Results/MCTA/vis/cam{}to{}/'.format(cams['C13'], cams['C5'])

    if cam2cam == 'H13to11':
        param = {}
        param['Pid'] = 11
        param['Aid'] = 13
        # C11: pri, C5: aux
        param['tOffsetPstart'] = -47  # 11 delayed 5 by 23: 11>1 == 5>24
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
                      [-1200.1, 0.5, 1.0]]

        param['fr_offsetPVis'] = 47
        param['fr_offsetAVis'] = 0  ## T_c2 - T_c11 = offset
        param['H'] = np.transpose(param['H'])
        # param['H'] = np.linalg.inv(param['H'])
        param['d_th'] = 0.37  # multi-task cost: 0.45, only loc: 0.17
        param['primary'] = benchmark + 'exp2/Results_train/tracktor_trained_app/cam11exp2.mp4.txt'
        param['auxiliary'] = benchmark + 'exp2/Results_train/tracktor_trained_app/cam13exp2.mp4.txt'
        # param['primary'] = '/home/siddique/Desktop/NEU_Data/May_Demo_Trackers/exp2-train/cam11exp2.mp4imgtestTracksv3.txt'
        # param['auxiliary'] = '/home/siddique/Desktop/NEU_Data/May_Demo_Trackers/exp2-train/cam13exp2.mp4imgtestTracksv3.txt'
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
        param['Pid'] = 13
        param['Aid'] = 14
        # C5: primary, C2: auxiliary
        param['tOffsetPstart'] = 0  # 13 delayed 14 by 5: 13>1 == 14>5
        param['tOffsetAstart'] = -4  #
        # similarity based homography: TODO: compute homography using undistorted image
        # H13to14
        param['H'] = [[1.1163, 0.0542, 0],
                      [-0.0542, 1.1163, 0],
                      [589.7222, -87.6723, 1.0000]]

        param['fr_offsetPVis'] = 0
        param['fr_offsetAVis'] = 4  ## T_c2 - T_c11 = offset
        param['H'] = np.transpose(param['H'])
        #H14to13
        if param['Pid'] == 13:
           param['H'] = np.linalg.inv(param['H'])
        param['d_th'] = 0.37

        param['primary'] = benchmark+'exp2/Results_train/tracktor_trained_app/cam13exp2.mp4.txt'
        param['auxiliary'] = benchmark+'exp2/Results_train/tracktor_trained_app/cam14exp2.mp4.txt'
        #param['primary'] = '/home/siddique/Desktop/NEU_Data/May_Demo_Trackers/exp2-train/cam13exp2.mp4imgtestTracksv3.txt'
        #param['auxiliary'] = '/home/siddique/Desktop/NEU_Data/May_Demo_Trackers/exp2-train/cam14exp2.mp4imgtestTracksv3.txt'
        param['finalTrackFile'] = 'cam14exp2_MCTA.txt'
        # param['finalTrackFileC2'] = 'cam02exp2_MCTA.txt'
        param['PImgsPath'] = benchmark + 'exp2' + '/imgs/cam{:02d}'.format(cams[
                                                                               'C13']) + 'exp2' + '.mp4/'  # benchmark + 'test_data/exp1'+ '/imgs/cam{:02d}'.format(cams['C2']) + 'exp1' + '.mp4/'
        # benchmark + experiment + '/imgs/cam{:02d}'.format(cams['C2']) + tracklet_exp + '.mp4/'
        param['AImgsPath'] = benchmark + 'exp2' + '/imgs/cam{:02d}'.format(cams[
                                                                               'C14']) + 'exp2' + '.mp4/'  # benchmark + 'test_data/exp1'+ '/imgs/cam{:02d}'.format(cams['C9']) + 'exp1' + '.mp4/'
        # benchmark + experiment + '/imgs/cam{:02d}'.format(cams['C9']) + tracklet_exp + '.mp4/'
        param['outPath'] = benchmark + experiment + '/Results/MCTA/vis/cam{}to{}/'.format(cams['C14'], cams['C13'])

    return param


def assignID2matches(params, mt, isseqlID):
    # form edges from matches
    params['adjacencyPA'] = np.zeros((len(params['PAproj']), len(params['PAproj'])), dtype='int')
    edges = {keys: [] for keys in range(len(params['PAproj']))}
    for (c, tStampComnMin, k, l) in mt:
        edges[k].append(l)
        edges[l].append(k)
        print ('k {} l {}'.format(k, l))
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
        print ('making identity sequential.....')
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

def get_cam2cam_matchedTrklt(params, cam2cam, isDistorted=True, isseqID=False, tracker_motion=False, tracker_app=False):
    # collect primary camera: P:C2 features
    import pandas as pd
    trackers = np.loadtxt(params['primary'], dtype='float32', delimiter=',')
    #trackers = pd.read_csv(params['primary'], header=None)
    #trackers = np.array(trackers)
    trklt_size = 90
    trackers[:, 0] = trackers[:, 0] + params['tOffsetPstart']
    fr_startP = min(trackers[:, 0][trackers[:, 0] > 0])
    fr_endP = trackers[-1, 0]
    isSCA = 0
    print('Frame start and end {}, {}'.format(fr_startP, fr_endP))
    # use [cx,cy] to compute distance and d_th = 10 for single camera association
    # Apply SCTA and Filtering
    mht_tracklets, totalTrack = form_tracklets(trackers, fr_startP, fr_endP, tracker_min_size=trklt_size,
                                               t_thr=30, d_thr=50, motion=tracker_motion, t_app=tracker_app, single_cam_association=isSCA).tracker_mht()
    # list C2 tracklets
    trackletsP = convert_centroids(mht_tracklets)

    # A:C9:------------------------------------------------------------------
    # Collect auxiliary camera features
    trackers = np.loadtxt(params['auxiliary'], dtype='float32', delimiter=',')
    #trackers = pd.read_csv(params['auxiliary'], header=None)
    #trackers = np.array(trackers)
    trklt_size = 90  # cam9: 30
    trackers[:, 0] = trackers[:, 0] + params['tOffsetAstart']
    fr_startA = min(trackers[:, 0][trackers[:, 0] > 0])
    fr_endA = trackers[-1, 0]
    isSCA = 0

    print('Frame start and end {}, {}'.format(trackers[0, 0], trackers[-1, 0]))
    # use [cx,cy] to compute distance and d_th = 10 for single camera association
    # Apply SCTA and Filtering
    mht_tracklets, totalTrack = form_tracklets(trackers, fr_startA, fr_endA, tracker_min_size=trklt_size,
                                               t_thr=30, d_thr=20, motion=tracker_motion, t_app=tracker_app, single_cam_association=isSCA).tracker_mht()
    # keep both original and projected tracklets for auxiliary
    tracklet_2 = copy.deepcopy(mht_tracklets)

    #tracklet_2 = xywh2x1y1x2y2(tracklet_2)
    #inter_track = interpolate_left_track(tracklet_2[0][:,0:5])
    # tracklet_2 = convert_centroids(tracklet_2)
    # Project: center>(cx,cy), top-left>(x,y), velocity>(vx,vy) [fr,cx,cy,x,y,vx,vy]
    # active_trklt: indexes of tracklets whose projected onto the destination (primary) image boundary
    trackletsA, active_trklt = project_tracklets(mht_tracklets, params['H'], tracker_motion,
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
                                               cam2cam,
                                               isDistorted,
                                               motion = tracker_motion,
                                               appearance=tracker_app,
                                               max_dist=params['d_th']
                                               )

    label_map_update, params = assignID2matches(params, mt, isseqID)

    return mt, label_map_update, min(fr_startP, fr_startA), max(fr_endP, fr_endA)


def merge_edges(mt,G_mc,index_factor, cam_trklts_size, cp, ca):
    # G_mc: empty graph which is updated iteratively for each camera pair association
    # mt: association between
    # k, l: index of camera pair tracklets should be started from 0: k>0 to n, l> l-cam_trklts_size[cp]> 0 to m
    assert cp in index_factor.keys(), 'graph initialization parameters for MC graph is not ready for C{} and C{}'.format(cp,ca)
    assert ca in index_factor.keys()
    for (c, tStampComnMin, k, l) in mt:
        l = l - cam_trklts_size[cp]
        #pdb.set_trace()
        #initialize empty association list when tracklet first appear in camera system
        if k+index_factor[cp] not in G_mc.keys():
            G_mc[k + index_factor[cp]] = []
        if l+index_factor[ca] not in G_mc.keys():
            G_mc[l+index_factor[ca]] = []
        #update the association list
        G_mc[k+index_factor[cp]].append(l+index_factor[ca])
        G_mc[l+index_factor[ca]].append(k+index_factor[cp])
        print('pairwise association k {} l {}'.format(k, l))
        print('multi-camera system association k {} l {}'.format(k+index_factor[cp], l+index_factor[ca]))
    return G_mc


def expand_from_temporal_list(box_all=None, mask_30=None):
    if box_all is not None:
        box_list = [b for b in box_all if len(b) > 0]
        box_all = np.concatenate(box_list)
    if mask_30 is not None:
        mask_list = [m for m in mask_30 if len(m) > 0]
        masks_30 = np.concatenate(mask_list)
    else:
        masks_30 =[]
    return box_all, masks_30

if __name__ == "__main__":
    import time
    dataset = 'clasp2'
    benchmark = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/'
    mhtTrackers = '/home/siddique/Desktop/NEU_Data/TrackResult/'
    DFS = 1
    DistCorrect = True
    eval = True
    tracker = 'mht'
    vis_pair = True
    motion_model = True
    app_model = True

    if dataset == 'clasp2':
        experiment = 'exp2'
        tracklet_exp = 'exp2'

    cams = {}
    MCparams = {}
    MCparams['demo_path'] =  benchmark+experiment+'/Results/MCTA/vis/demo'
    cam2cam = {}

    cams['C2'] = 2  # t_p
    cams['C9'] = 9  # t_a
    cams['C5'] = 5  # H5to2
    cams['C11'] = 11  # H5to11
    cams['C13'] = 13  # H13to11
    cams['C14'] = 14  # H13to14
    #C_caC_cp
    cam2cam['C9C2'] = 'H9to2'
    cam2cam['C5C2'] = 'H5to2'
    cam2cam['C11C5'] = 'H5to11'
    cam2cam['C13C5'] = 'H5to13'
    cam2cam['C13C11'] = 'H13to11'
    cam2cam['C14C13'] = 'H13to14'
    prim_list = [2,11,14]
    prim_list_trklts = []
    aux_list = [9,5,13]
    aux_list_trklts = []
    mc_all_trklts = []
    folder = []
    MC_tracklets = {}
    MC_labelMap = []
    cam2cam_vis = 1
    camP_Q = {2:[9,5],5:[11,13],11:[13],13:[14]}
    cam_visited = []
    index_factor = {}
    cam_trklts_size = {}
    G_mc = {}
    for cp in sorted(camP_Q.keys()):
        Auxs = camP_Q[cp]
        for ca in Auxs:
            start_time = time.time()
            cacp = 'C'+str(ca)+'C'+str(cp)
            MCparams[cam2cam[cacp]] = dafault_params(cam2cam[cacp], experiment, cams, mhtTrackers, tracker_motion = motion_model)
            mt, label_map_update, fr_start, fr_end = get_cam2cam_matchedTrklt(MCparams[cam2cam[cacp]],
                                                                              cam2cam[cacp],
                                                                              isDistorted=DistCorrect,
                                                                              isseqID=False,
                                                                              tracker_motion = motion_model,
                                                                              tracker_app = app_model
                                                                              )


            if vis_pair and cacp=='C140C13':
                assert len(MCparams[cam2cam[cacp]]['PAorg']) == len(label_map_update)
                labelsAux, labelsPri = split_labelMap(label_map_update, MCparams[cam2cam[cacp]])

                # call plot function to visualize the association correct
                fr_start = 100
                fr_end = 13400
                plot_trajectories_pairs(labelsAux,
                                        labelsPri,
                                        fr_start,
                                        fr_end,
                                        MCparams[cam2cam[cacp]],
                                        cam2cam[cacp],
                                        cams,
                                        vis_rate=30,
                                        isDistorted=DistCorrect,
                                        metrics=eval,
                                        save_imgs=1,
                                        motion=0
                                        )
                pdb.set_trace()

            #save camera pair graph
            MCparams[cacp+'_graph'] = mt

            # prepare traklets info for mc system
            cam2cam_params = MCparams[cam2cam[cacp]]

            # get factors for individual camera position in G_mc when first appear in BFS
            print('ca {} cp {}'.format(ca,cp))
            if cp == 2:
                index_factor[cp] = 0
                cam_trklts_size[cp] = cam2cam_params['trkltFamilySizeP']
            if ca == 9:
                index_factor[ca] = cam_trklts_size[2]
                cam_trklts_size[ca] = cam2cam_params['trkltFamilySizeA']
            if ca==5:
                index_factor[ca] = cam_trklts_size[2] + cam_trklts_size[9]
                cam_trklts_size[ca] = cam2cam_params['trkltFamilySizeA']

            if ca == 11:
                index_factor[ca] = cam_trklts_size[2] + cam_trklts_size[9] + cam_trklts_size[5]
                cam_trklts_size[ca] = cam2cam_params['trkltFamilySizeA']

            if ca == 13:
                index_factor[ca] = cam_trklts_size[2] + cam_trklts_size[9] + cam_trklts_size[5] + cam_trklts_size[11]
                cam_trklts_size[ca] = cam2cam_params['trkltFamilySizeA']

            if ca == 14:
                index_factor[ca] = cam_trklts_size[2] + cam_trklts_size[9] + cam_trklts_size[5] + cam_trklts_size[11] + cam_trklts_size[13]
                cam_trklts_size[ca] = cam2cam_params['trkltFamilySizeA']

            if cp not in cam_visited:
                mc_all_trklts.append(cam2cam_params['PAorg'][:cam2cam_params['trkltFamilySizeP']])
                cam_visited.append(cp)

            if ca not in cam_visited:
                mc_all_trklts.append(cam2cam_params['PAorg'][cam2cam_params['trkltFamilySizeP']:])
                cam_visited.append(ca)

            # update system graph from pair association (generated from Hungarian)
            G_mc = merge_edges(mt, G_mc, index_factor, cam_trklts_size, cp, ca)
            print('Average Speed: {:.2f} Hz'.format(fr_end/ (time.time() - start_time)))

    mc_all_trklts,_ =  expand_from_temporal_list(box_all=mc_all_trklts, mask_30=None)
    #apply DFS on mc system graph
    label_map = {keys: [] for keys in range(len(mc_all_trklts))}
    # TODO: each track identity should be unique
    for label_tp, tklt in enumerate(mc_all_trklts):
        if label_tp in G_mc.keys():
            #pdb.set_trace()
            solnPath = dfs(G_mc, label_tp, [])
            label_map[label_tp].append(min(solnPath))
        else:
            label_map[label_tp].append(label_tp)


    if not os.path.exists(MCparams['demo_path']):
        os.makedirs(MCparams['demo_path'])
    delete_all(MCparams['demo_path'])
    if eval:
        track_for_mot = open(MCparams['demo_path'] + '/MC_TrackFile', 'w')
    # start and end frames are based on primary camera
    fr_start, fr_end = 2000, 13400
    #assert len(cam_trklts_size)==sum(cam_trklts_size.values())
    plot_trajectories_MC(label_map,
                         mc_all_trklts,
                         cam_trklts_size,
                         index_factor,
                         fr_start,
                         fr_end,
                         MCparams,
                         cam2cam,
                         cams,
                         vis_rate=60,
                         isDistorted=DistCorrect,
                         metrics=eval,
                         save_imgs=1,
                         motion=0
                         )


