from __future__ import division
import numpy as np
from scipy.spatial.distance import directed_hausdorff,cosine,mahalanobis
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
np.set_printoptions(threshold=sys.maxsize)

__version__ = 0.4

def camera_intrinsics(cam):
    # A: 3*3 camera matrix
    # dist_coeff: distortion coefficient k1, k2
    if cam==9:
        A = np.array([[1217.6, 0.0, 972.3],
                       [0, 1217.8, 550.9],
                       [0.0, 0.0, 1.0]])
        dist_coeff = np.array([-0.3235, 0.0887, 0, 0])  # why 4 or 5 coefficients are used in opencv????
    if cam==11: # assume that 9 and 11 have similar distortion
        A = np.array([[1217.6, 0.0, 972.3],
                       [0, 1217.8, 550.9],
                       [0.0, 0.0, 1.0]])
        dist_coeff = np.array([-0.3235, 0.0887, 0, 0])  # why 4 or 5 coefficients are used in opencv????
    if cam==2:
        A = np.array([[1216.5, 0.0, 989.0],
                       [0, 1214.5, 595.0],
                       [0.0, 0.0, 1.0]])
        dist_coeff = np.array([-0.3655, 0.1296, 0, 0])  # why 4 or 5 coefficients are used in opencv????
    if cam==5: #assume that c2 and c5 have similar distortion
        A = np.array([[1216.5, 0.0, 989.0],
                       [0, 1214.5, 595.0],
                       [0.0, 0.0, 1.0]])
        dist_coeff = np.array([-0.3655, 0.1296, 0, 0])  # why 4 or 5 coefficients are used in opencv????
    if cam==4: #assume that c2 and c5 have similar distortion
        A = np.array([[1216.5, 0.0, 989.0],
                       [0, 1214.5, 595.0],
                       [0.0, 0.0, 1.0]])
        dist_coeff = np.array([-0.3655, 0.1296, 0, 0])  # why 4 or 5 coefficients are used in opencv????

    return dist_coeff, A


def undistorted_coords(trklt, dist_coeff, A):
    # use a copy of traclet centroids to convert into undistorted format
    # ***A: Camera Intrinsic Matrix
    # https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html
    #https: // www.mathworks.com / help / vision / ref / estimatecameraparameters.html
    # new camMatrix
    im_shape = (1920, 1080)  # img.shape[:2] : (w,h)
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(A, dist_coeff,im_shape,1,im_shape)
    trklt = cv2.undistortPoints(trklt.reshape(trklt.shape[0],1,2), A, dist_coeff, 0, newcameramtx)
    return trklt.reshape(trklt.shape[0],2)

def filter_tracklets(trackers, min_len=60):
    # Remove tracklets with <60 detections
    return [tl for tl in trackers if len(tl) >= min_len]

def get_tracklets(trackers):
    tracklets = []
    for tid in range(len(trackers)):
        #tracker_i = trackers.get(str(tid + 1))
        tracker_i = trackers[tid]
        tracker_i = np.array(list(tracker_i))
        tracker_i = np.array(sorted(tracker_i, key=lambda x: x[0]))
        tracklets.append(tracker_i)
    return tracklets

def filter_tracklets(trackers, min_len=60):
    # Remove tracklets with <60 detections
    return [tl for tl in trackers if len(tl) >= min_len]


def convert_centroids(tracklet):
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


def project_tracklets(in_tracklets, H, isDistorted=False, cam=9):
    out_tracklets = []
    active_trklt = []
    Hv = np.copy(H)
    Hv[0,2],Hv[1,2] = 0,0
    #[fr,cx,cy,x,y,vx,vy]<<< projected coordinates: center is used
    for i,trklt in enumerate(in_tracklets):
        #TODO: apply undistorted coords on H or undistort coords after applying H: here H is based distorted image
        if isDistorted:
            dist_coeff, A = camera_intrinsics(cam)
            xy = np.copy(trklt[:, 1:3])
            xy = undistorted_coords(xy, dist_coeff, A)
            trklt[:,1:3] = xy
        for bb in trklt:
            bbt1 = applyTransform(bb[1:3], H)
            cxcy = applyTransform( bb[1:3] + bb[3:5] / 2.0, H)
            vxvy = applyTransform(bb[5:7], Hv)
            bbt2 = cxcy - bbt1# projected w/2,h/2
            bb[1:7] = np.concatenate([cxcy, bbt2, vxvy]) #projected: [cx,cy,w/2,h/2,vx,vy]
        # Delete tracklets that don't have any detection visible in the second camera
        # TODO: There must be a smarter and faster way to do that
        #cy+h/2 > 0 and w/2>0
        if max(trklt[:, 2]-trklt[:, 4]) > 0 and max(trklt[:, 3]) > 0:
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
        masks_30 =[]
    return box_all, masks_30

def augment_feature_with_velocity(trackletAux,trackletPri):
    trackletAuxDelayed = np.insert(trackletAux,0,np.array([trackletAux[0,0]-1,0,0]),axis=0)
    trackletPriDelayed = np.insert(trackletPri,0,np.array([trackletPri[0,0]-1,0,0]),axis=0)

    #[delta.Cx/delta.t,delta.Cy/delta.t]
    # Due to the smaller overlap between camera pairs, Vy changes are dissimilar
    trackletAuxVelocity = (trackletAux[:, 1:3] - trackletAuxDelayed[:, 1:3][:-1,:]) / np.transpose(
        [trackletAux[:, 0] - trackletAuxDelayed[:, 0][:-1],
         trackletAux[:, 0] - trackletAuxDelayed[:, 0][:-1]])
    trackletPriVelocity = (trackletPri[:, 1:3] - trackletPriDelayed[:, 1:3][:-1,:]) / np.transpose(
        [trackletPri[:, 0] - trackletPriDelayed[:, 0][:-1],
         trackletPri[:, 0] - trackletPriDelayed[:, 0][:-1]])
    # size check
    assert trackletAuxVelocity.shape[0]==trackletAux.shape[0],'number of observation in velocity vector and centroids vector shuld same' \
                                                              ' but found {},{} respectively'.format(trackletAuxVelocity.shape[0],trackletAux.shape[0])
    # append to form [fr,Cx,Cy,dot.Cx,dot.Cy]
    trackletAuxAug = np.append(trackletAux,trackletAuxVelocity,axis=1)
    trackletPriAug = np.append(trackletPri,trackletPriVelocity,axis=1)
    return trackletAuxAug[:,0:5],trackletPriAug[:,0:5]

def mahalanobis21(final_tracklet1,final_tracklet2):
    diff = final_tracklet1[:, 1:6] - final_tracklet2[:, 1:6]
    X12 = np.stack((final_tracklet1[:, 1:6], final_tracklet2[:, 1:6]), axis=1)
    iv = np.linalg.inv( np.cov(X12.reshape(X12.shape[0],8)))
    MD = np.linalg.norm(np.sqrt(np.diag(np.dot(np.dot(diff.T, iv), diff))))
    return MD

def refine_cost_row(cost,tilde_t_pa,common_t_stamps,indP):
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

def refine_cost_column(cost,tilde_t_pa,common_t_stamps,indA):
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

def associate_tracklets_DFS(cam1,cam2,cam2to1_tracklets, cam1Tracklets, cam2Tracklets,
                            active_trklt, isDistorted=False, max_dist=400):
    #cam2to1_tracklets = [cam1,cam2]
    print('Total tracklets in a camera pairs {}'.format(len(cam2to1_tracklets)))
    #cost = np.empty((len(cam2to1_tracklets), len(cam2to1_tracklets)))
    cost = np.ones((len(cam2to1_tracklets), len(cam2to1_tracklets)))*np.inf
    tilde_t_pa = np.ones((len(cam2to1_tracklets), len(cam2to1_tracklets)),dtype='float')*np.inf
    #it = np.nditer(cost, op_flags=['readwrite'])
    #it_cosine = np.nditer(cosine_cost, op_flags=['readwrite'])
    i = 0
    matches = []
    for tracklet1 in cam2to1_tracklets:
        j = 0
        for tracklet2 in cam2to1_tracklets:
            print('Processing tracklets {} and {}'.format(i, j))
            #condition to check multicamera trackets have any overlap
            if i!=j and i<len(cam1Tracklets) and j>=len(cam1Tracklets) and j in len(cam1Tracklets)+np.array(active_trklt):
                if tracklet1[-1, 0] < tracklet2[0, 0] or tracklet1[0, 0] > tracklet2[-1, 0]:# cam1_end<cam2_start or cam1_start>cam2_end
                    #it[0] = np.inf
                    cost[i,j] = np.inf
                # Search for the overlapping portions of multicamera tracklets
                else:
                    print 'Trackets are overlapped'
                    # TODO: There must be a more compact way of doing this (find the overlapped region)
                    # TODO: Assess whether the timestamp should be considered in the distance computation as well
                    common_t_stamps = np.array(list(set(list(tracklet2[:, 0])).intersection(list(tracklet1[:, 0]))))
                    assert len(common_t_stamps)>0,'Non-overlaped T_p, T_a can not be associated'
                    #tilde_t_pa[j, i] = min(common_t_stamps)
                    #2D or 4D feature: [fr,cx,cy,w/2,h/2,5-vx,6-vy]
                    # augment feature (4D) with velocity (aux,primary): [Cx,Cy,dot.Cx,dot.Cy]
                    #tracklet2,tracklet1 = augment_feature_with_velocity(tracklet2[:,0:3], tracklet1[:,0:3])
                    featIndx = [0,1,2,5,6]#,5,6]
                    mask1 = np.isin(tracklet1[:, 0], common_t_stamps)
                    final_tracklet1 = tracklet1[mask1][:,featIndx]
                    if isDistorted:
                        dist_coeff, A = camera_intrinsics(cam=cam1) # 2:c5 or 9:c11
                        xy = np.copy(final_tracklet1[:, 1:3])
                        xy = undistorted_coords(xy, dist_coeff, A)
                        final_tracklet1[:,1:3] = xy
                    # compute the location from velocity and time difference
                    final_tracklet1[:,1:5] = final_tracklet1[:,1:5]/[1920.0,1080.0,50.0,50.0]#max(abs(final_tracklet1[:,3]))

                    mask2 = np.isin(tracklet2[:, 0], common_t_stamps)
                    final_tracklet2 = tracklet2[mask2][:,featIndx]
                    final_tracklet2[:,1:5] = final_tracklet2[:,1:5]/ [1920.0, 1080.0,50.0,50.0]

                    d_h = directed_hausdorff(final_tracklet1[:, 1:3], final_tracklet2[:, 1:3])[0]
                    dOverlap = 0.3#final_tracklet1.shape[0]/float(max(tracklet1.shape[0],tracklet2.shape[0]))
                    #d_h = (1-dOverlap)*directed_hausdorff(final_tracklet1[:,1:3], final_tracklet2[:,1:3])[0]  \
                     # + (dOverlap)*cosine(final_tracklet2[:,3:5].flatten(), final_tracklet1[:,3:5].flatten())
                    #d_cos = 1-cosine(final_tracklet2.flatten(), final_tracklet1.flatten())
                    #d_mah = mahalanobis21(final_tracklet1, final_tracklet2)
                    if d_h<=max_dist:
                        #it[0] = d_h
                        cost[i,j] = d_h
                        tilde_t_pa[i, j] = min(common_t_stamps)
                        # tracklet orientation distance
                        #print 'Cosine distance between {} in cam9 and {} in cam2 is {}'.format(j,i,d_cos)
            else:
                cost[i, j] = np.inf
                #it[0] = np.inf
            #it.iternext()
            #it_cosine.iternext()
            j = j + 1
        # refine the cost for each primary with all auxiliary
        i = i + 1
    # DEBUG
    # import matplotlib.pyplot as plt
    # hist = np.histogram(cost[cost < 10000])
    # plt.hist(cost[cost < 10000])
    iteration = 0
    while len(cost[cost<=max_dist])>0:
        row_ind, col_ind = linear_sum_assignment_with_inf(cost)
        Mincosts = cost[row_ind,col_ind]
        idx = np.where(Mincosts <= max_dist)[0]
        for i in idx:
            matches.append((Mincosts[i], tilde_t_pa[row_ind[i], col_ind[i]], row_ind[i],
                            col_ind[i]))  # tilde_t_pa[row_ind[i],col_ind[i]]  cam2_tracklets[col_ind[i]][0,0]
            # t_p fixed> search over t_a to get any overlapping with T_a
            indx_t_a = np.where(cost[row_ind[i], :] <= max_dist)[0]
            if len(indx_t_a)>0:
                T_a = cam2to1_tracklets[indx_t_a]
                isIntersect = [len(set(cam2to1_tracklets[col_ind[i]][:,0]).intersection(t_a[:,0]))>0 for t_a in T_a]
                cost[row_ind[i], indx_t_a[isIntersect]] = np.inf

            indx_t_p = np.where(cost[:, col_ind[i]] <= max_dist)[0]
            if len(indx_t_p)>0:
                T_p = cam2to1_tracklets[indx_t_p]
                isIntersect = [len(set(cam2to1_tracklets[row_ind[i]][:, 0]).intersection(t_p[:, 0])) > 0 for t_p in T_p]
                cost[indx_t_p[isIntersect], col_ind[i]] = np.inf

    return matches, tilde_t_pa

def associate_tracklets(cam2to1_tracklets,cam1_tracklets, cam2_tracklets, max_dist=400):
    cost = np.empty((len(cam1_tracklets), len(cam2_tracklets)))
    #cosine_cost = np.empty((len(cam1_tracklets), len(cam2_tracklets)))
    tilde_t_pa = np.ones((len(cam1_tracklets), len(cam2_tracklets)),dtype='float')*np.inf
    it = np.nditer(cost, op_flags=['readwrite'])
    #it_cosine = np.nditer(cosine_cost, op_flags=['readwrite'])
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
            #condition to check trackets have any overlap
            if tracklet1[-1, 0] < tracklet2[0, 0] or tracklet1[0, 0] > tracklet2[-1, 0]:# cam1_end<cam2_start or cam1_start>cam2_end
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
                tilde_t_pa[i,j] = min(common_t_stamps)
                mask1 = np.isin(tracklet1[:, 0], common_t_stamps)
                final_tracklet1 = tracklet1[mask1]

                mask2 = np.isin(tracklet2[:, 0], common_t_stamps)
                final_tracklet2 = tracklet2[mask2]
                d_h = directed_hausdorff(final_tracklet1[:, 1:3], final_tracklet2[:, 1:3])[0]
                d_cos = cosine(final_tracklet2[:, 1:3].flatten(), final_tracklet1[:, 1:3].flatten())
                if d_h<max_dist:
                    it[0] = d_h
                    #it_cosine[0] = cosine(final_tracklet2[:,1:3].flatten(), final_tracklet1[:,1:3].flatten())
                    # tracklet orientation distance
                    print 'Cosine distance between {} in cam9 and {} in cam2 is {}'.format(j,i,d_cos)
                else:
                    it[0] = np.inf
            it.iternext()
            #it_cosine.iternext()
            j = j + 1
        i = i + 1
    # DEBUG
    # hist = np.histogram(cost[cost < 10000])
    # plt.hist(cost[cost < 10000])
    while len(cost[cost<max_dist])>0:
        row_ind, col_ind = linear_sum_assignment_with_inf(cost)
        mcost = cost[row_ind,col_ind]
        idx = np.where(mcost < max_dist)[0]
        for i in idx:
            matches.append((mcost[i],tilde_t_pa[row_ind[i],col_ind[i]], row_ind[i], col_ind[i]))# tilde_t_pa[row_ind[i],col_ind[i]]  cam2_tracklets[col_ind[i]][0,0]
            # for multiple association restrict currently associated pairs from further association
        cost[row_ind,col_ind] = np.inf
    return matches


def plot_trajectories(label_map,fr_start, fr_end,cam2to1_tracklets,tracklet_cam9, tracklets,
                      fr_offsetP,fr_offsetA,folder,folder2,out_path,cam2cam,vis_rate=30,
                      isDistorted=False,metrics=False,save_imgs=False,motion=False):
    # TODO: Figure out why I need the reshape here
    labelCam1 = dict(list(label_map.items())[:len(tracklets)])
    labelCam2 = dict(list(label_map.items())[len(tracklets):])
    color = np.column_stack([[255.0 * np.random.rand(1, len(cam2to1_tracklets))],
                             [255.0 * np.random.rand(1, len(cam2to1_tracklets))],
                             [255.0 * np.random.rand(1, len(cam2to1_tracklets))]]).reshape(3, len(cam2to1_tracklets))
    #BGR
    #magenta = (255,0,255)
    #yellow = (0,255,255)
    #green = (0,255,0)
    #blue = (255,0,0)
    # plt.imshow(np.ones((1080, 1920, 3)))
    #fr_offset = 48#11to13
    #fr_offset = 29  # 9to2, cam2 lag = 29
    if save_imgs:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 640, 480)


    for fr in range(fr_start, fr_end):
        #9A:9 fr-15, 11 fr-15
        if (fr)%vis_rate==0: #15 for 6A,7A,.. 3 for 5A, 5B.. 30 for 9A
            try:
                img = cv2.imread('{}{:06d}.png'.format(folder, fr+fr_offsetP)) # target camera 2,11 - folder:
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img2 = cv2.imread('{}{:06d}.png'.format(folder2, fr+fr_offsetA)) # source camera 9,5 - folder2: + fr_offset for 9to2, + fr_offset for 5to11
                # visualize the projections on primary camera (11,2)
                c = len(tracklets)
                for bb in cam2to1_tracklets[len(tracklets):]: # show projection of auxiliary camera
                    # TODO: Set the colors of the rectangles
                    for i in range(1, len(bb)):
                        if fr_start <= bb[i, 0] <= fr:
                            # image = cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength = 0.5)
                            if save_imgs and motion and bb[i, 0]%5==0:
                                #2d velocity plot
                                xx = int(bb[i, 1]+bb[i, 5]*2*(bb[i, 0]-bb[i-1, 0]))#xx=x0+vxt
                                yy = int(bb[i, 2]+bb[i, 6]*2*(bb[i, 0]-bb[i-1, 0]))
                                cv2.arrowedLine(img,(int(bb[i, 1]), int(bb[i, 2])),
                                                (xx,yy),
                                                (255,0,255), thickness=4, tipLength = 1)
                                #line between centroids
                                #cv2.arrowedLine(img, (int(bb[i - 1, 1]), int(bb[i - 1, 2])), (int(bb[i, 1]), int(bb[i, 2])),
                                         #magenta, thickness=5, tipLength = 1)#lineType=8

                            # TODO: Think about how to show this only at the last detection in the current frame
                            if bb[i, 0] == fr :#and len(label_map_aux[c])>0
                                cv2.putText(img, '{}'.format(labelCam2[c]), (int(bb[i - 1, 1]), int(bb[i - 1, 2])),
                                            cv2.FONT_HERSHEY_SIMPLEX, 3,  (255,0,255), 5, cv2.LINE_AA)
                    c = c + 1
                c=0
                # visulaize the identity handover in primary camera: C2
                for i_p,bb in enumerate(cam2to1_tracklets[:len(tracklets)]): #primary camera
                    # To show the association with the projections (tracklets2)
                    # discard added delay during association
                    for i in range(1, len(bb)):
                        if fr_start <= bb[i, 0] <= fr:
                            if save_imgs and motion and bb[i, 0]%5==0:
                                #2d velocity plot, bb[i,1]
                                xx = int((bb[i, 1] + bb[i, 3] / 2.0)+bb[i, 5]*2*(bb[i, 0]-bb[i-1, 0]))#xx=x0+vxt
                                yy = int((bb[i, 2] + bb[i, 4] / 2.0)+bb[i, 6]*2*(bb[i, 0]-bb[i-1, 0]))
                                cv2.arrowedLine(img,(int(bb[i, 1] + bb[i, 3] / 2.0), int(bb[i, 2] + bb[i, 4] / 2.0)),
                                                (xx,yy), (0,255,255),
                                                thickness=4, tipLength=1)
                                #line between centroids
                                #cv2.arrowedLine(img,
                                        #(int(bb[i - 1, 1] + bb[i - 1, 3] / 2.0), int(bb[i - 1, 2] + bb[i - 1, 4] / 2.0)),
                                        #(int(bb[i, 1] + bb[i, 3] / 2.0), int(bb[i, 2] + bb[i, 4] / 2.0)),
                                        #(0, 255, 255), thickness=5, tipLength = 1)#lineType=8
                            if metrics:
                                if bb[i, 0] == fr:#9A:9 fr-15
                                    #if bb[i, 0]>=5415:
                                        #label_map[c]=0
                                    #Adjust frame delay and frame rate to find matching with GT
                                    # 6A:9,11>bb[i, 0]+30, 5A:9,11>bb[i, 0]+30, 7A:9>bb[i, 0],11>bb[i, 0]+30, 10A: 9,11>bb[i, 0]+30,9A: 9,11>bb[i, 0]+15,frdelay=15

                                    cv2.rectangle(img, (int(bb[i, 1]), int(bb[i, 2])),
                                                  (int(bb[i, 1] + bb[i, 3]), int(bb[i, 2] + bb[i, 4])),
                                                  (255, 0, 0), 5, cv2.LINE_AA)
                                    # show identity at undistorted centroid
                                    if isDistorted:
                                        dist_coeff, A = camera_intrinsics(cam=2)
                                        cxy_undist = np.copy(bb[i, 1:3]+bb[i, 3:5]/2.0)
                                        cxy_undist = undistorted_coords(cxy_undist.reshape(1,1,2), dist_coeff, A)
                                    else:
                                        cxy_undist = np.copy(bb[i, 1:3]+bb[i, 3:5]/2.0).reshape(1,2)
                                    cv2.putText(img, 'P{}'.format(labelCam1[c]),
                                                (int(cxy_undist[0,0]), int(cxy_undist[0,1])),
                                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5, cv2.LINE_AA)
                                    if cam2cam == 'H5to11':
                                        track_for_mot.writelines(
                                            str((bb[i, 0]+30)//15) + ',' + str(labelCam1[c]+1) + ',' + str(bb[i,1]) + ',' + str(
                                                bb[i,2]) + ',' + str(bb[i,3]) + ',' + str(bb[i,4]) + ',' +
                                                '-1' + ',' + '-1' + ',' + '-1' + ',' + '-1' + '\n')

                            else:
                                if bb[i, 0] == fr and label_map[c]==-1:
                                    cv2.putText(img, '{}'.format(c), (int(bb[i - 1, 1]), int(bb[i - 1, 2])),
                                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 5, cv2.LINE_AA)
                                if bb[i, 0] == fr and label_map[c] != -1:
                                    cv2.putText(img, '{}'.format(label_map[c]), (int(bb[i - 1, 1]), int(bb[i - 1, 2])),
                                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5, cv2.LINE_AA)

                    c = c + 1
                # visualize the actual target identity in auxiliary camera (9,5)
                c = len(tracklets)
                for bb in tracklet_cam9[len(tracklets):]: # source camera : yellow
                    # TODO: Set the colors of the rectangles
                    for i in range(1, len(bb)):
                        if fr_start <= bb[i, 0] <= fr:
                            #image = cv2.arrowedLine(image, start_point, end_point, color, thickness, tipLength = 0.5)
                            if save_imgs and motion and bb[i, 0]%5==0:
                                #2d velocity plot
                                xx = int((bb[i, 1] + bb[i, 3] / 2.0)+bb[i, 5]*2*(bb[i, 0]-bb[i-1, 0]))#xx=x0+vxt
                                yy = int((bb[i, 2] + bb[i, 4] / 2.0)+bb[i, 6]*2*(bb[i, 0]-bb[i-1, 0]))
                                cv2.arrowedLine(img2,(int(bb[i, 1] + bb[i, 3] / 2.0), int(bb[i, 2] + bb[i, 4] / 2.0)),
                                                (xx,yy), (0,255,0),
                                                thickness=4, tipLength=1)
                                #line between centroids
                                #cv2.arrowedLine(img2,
                                        #(int(bb[i - 1, 1] + bb[i - 1, 3] / 2.0), int(bb[i - 1, 2] + bb[i - 1, 4] / 2.0)),
                                        #(int(bb[i, 1] + bb[i, 3] / 2.0), int(bb[i, 2] + bb[i, 4] / 2.0)),
                                        #(0, 255, 0), thickness=5, tipLength = 1)#lineType=8
                            if metrics:
                                # TODO: Think about how to show this only at the last detection in the current frame
                                if bb[i, 0] == fr:
                                    cv2.putText(img2, '{}'.format(labelCam2[c]),  (int(bb[i, 1] + bb[i, 3] / 2), int(bb[i, 2] + bb[i, 4] / 2)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0,2555,0), 5, cv2.LINE_AA)
                                    cv2.rectangle(img2, (int(bb[i, 1]), int(bb[i, 2])),
                                                  (int(bb[i, 1] + bb[i, 3]), int(bb[i, 2] + bb[i, 4])),
                                                  (255,0,0), 5, cv2.LINE_AA)
                                    if cam2cam == 'H9to2' and metrics:
                                        track_for_mot.writelines(
                                            str((bb[i, 0]+30) // 15) + ',' + str(labelCam2[c] + 1) + ',' + str(
                                                bb[i, 1]) + ',' + str(
                                                bb[i, 2]) + ',' + str(bb[i, 3]) + ',' + str(bb[i, 4]) + ',' +
                                            '-1' + ',' + '-1' + ',' + '-1' + ',' + '-1' + '\n')
                    c = c + 1
                if cam2cam == 'H9to2':
                    final_img = cv2.vconcat([img2,img])
                if cam2cam == 'H2to5':
                    final_img = cv2.hconcat([img,img2]) # img: c5, img2: c2
                if cam2cam == 'H2to4':
                    final_img = cv2.hconcat([img,img2]) # img: c4, img2: c2
                else:
                    final_img = cv2.vconcat([img2, img])
                cv2.imshow("image", final_img)
                #final_img = cv2.resize(final_img,(960,540))
                if save_imgs:
                    #img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
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
    cost = np.full((len(prev_track), len(new_track)),np.inf)
    i = 0
    for prev_det in prev_track:
        j = 0
        for new_det in new_track:
            # This is the only difference from the single-camera function
            # TODO: Make both a common function that take this test as a parameter
            delta_t = new_det[0] - prev_det[0]
            dist = np.linalg.norm(new_det[1:5]-prev_det[1:5])
            if 0 < delta_t < t_thr and dist < d_thr:
                cost[i,j] = dist
            j = j + 1
        i = i + 1

    row_ind, col_ind = linear_sum_assignment_with_inf(cost)

    # Find the maximum IoU for each pair
    for i in row_ind:
        if cost[i,col_ind[i]] < 50:#1
            matches.append((cost[i, col_ind[i]], i, col_ind[i]))

    return matches


def merge_tracklets(filtered_tracker, unassigned_tracklets):
    # Iterate a few times to join pairs of tracklets that correspond to multiple
    # fragments of the same track
    # TODO: Find a more elegant solution -- Possibly using DFS
    del_tl = []

    for merge_steps in range(0,5):

        mt = match_tl(filtered_tracker,unassigned_tracklets)

        for (c, k, l) in mt:
            filtered_tracker[k] = np.vstack((filtered_tracker[k],unassigned_tracklets[l]))
            if l not in del_tl:
                del_tl.append(l)

    del_tl.sort(reverse=True)
    # delete associated tracklets to discard the duplicity
    for l in del_tl:
        del(filtered_tracker[l])
        del(unassigned_tracklets[l])

    return filtered_tracker, unassigned_tracklets

def dfs(graph, node, visited):
    if node not in visited:
        visited.append(node)
        for n in graph[node]:
            dfs(graph,n, visited)
    return visited

def computeEdges(vertices,matchList):
    edges = {keys: [] for keys in vertices}
    for i,vi in enumerate(vertices):
        for j,vj in enumerate(vertices):
            if i != j:
                if (vi,vj) in [('0','6'),('0','8'),('2','7'),('6','2'),('7','3')]:
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

if __name__ == "__main__":
    dataset = 'exp2'
    benchmark = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/'
    mhtTrackers =  '/home/siddique/Desktop/NEU_Data/TrackResult/'
    DFS = 1
    isDistCorrect = True
    eval = True
    tracker = 'mht'
    sequentialID = 1
    # 9A cam11: time_offset_11 = +12 (100 in camera 11 will be 112 w.r.t camera 5)
    # 9A cam2: time_offset_2 = -55 (100 in camera 2 will be 155 w.r.t camera 9)
    #time_offset_Psca = -40#9Acam11: +12, C5>-12, 5acam11: +24, 10A: +12, 6A: +30, cam5>-25, 7A cam55>0
    #time_offset_Asca= 0#5Acam2: 29, 5Bcam2: -31#9Acam2: -55, 10A:C5> -9, 7Acam2: -60, 6Acam2: -220, exp1cam2: -40
    if dataset =='exp1':
        experiment = 'exp1'
        tracklet_exp = 'exp1'
        raw_mht = 'det' + 'exp1'
    if dataset =='exp2':
        experiment = 'exp2'
        tracklet_exp = 'exp2'
        raw_mht = 'det' + 'exp2'
    cam1 = 2 #t_p
    cam2 = 9 #t_a
    cam2cam = 'H'+str(cam2)+'to'+str(cam1)
    if cam2cam == 'H9to2':
        time_offset_Psca = -46#-46 #2
        time_offset_Asca= 0   #9
        # similarity based homography
        '''
    1.1361   -0.0754         0
    0.0754    1.1361         0
  159.5726 -684.3318    1.0000
        '''
        #exp2
        H9to2 = [[1.136, -0.075, 0.00],
                 [0.075, 1.136, 0.00],
                 [159.57, -681.38, 1.00]]
        H9to2 = [[0.9812, 0.1426, 0.00],
                 [-0.1426, 0.9812, 0.00],
                 [463.84, -782.41, 1.00]]
        fr_start = 1
        fr_end = 13400
        fr_offsetP = 46#46## cam2 delay
        fr_offsetA = 0# cam9 delay
        H = np.transpose(H9to2)
        d_th = 0.2#0.17#0.17#0.45#200 # multi-task cost: 0.45, only loc: 0.17, loc+velocity 4d hd: 0.4
        if experiment == 'exp1':
            camP = mhtTrackers + '/cam{:02d}{}PersonMASK_30FPSrotv1.txt'.format(cam1, experiment)#cam2: test_aug: v1
            camA = mhtTrackers + '/cam{:02d}{}PersonMASK_30FPSrot.txt'.format(cam2, experiment)#cam9-test_aug:v2exp1
        if experiment=='exp2':
            camA ='/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results_train/exp2-train-track/cam09exp2.mp4testTracks.txt'
            camP =  '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results_train/exp2-train-track/cam02exp2.mp4testTracks.txt'
        finalTrackFile = 'cam09.txt'
    if cam2cam == 'H2to5' and experiment == 'exp1':
        #similarity based homography
        H2to5 = [[1.2, -0.2, 0.0],
                [0.2, 1.2, 0.0],
                [1422.4, 46.8, 1.0]]
        fr_offsetP = 0
        fr_offsetA = 0
        time_offset_Psca = 0
        time_offset_Asca =  0   ## T_c2 - T_c5 = offset
        H = np.transpose(H2to5)
        d_th = 0.5  # 0.17#0.17#0.45#200 # multi-task cost: 0.45, only loc: 0.17, loc+velocity 4d hd: 0.4
        camP = mhtTrackers + '/cam{:02d}{}PersonMASK_30FPSrot.txt'.format(cam1, experiment)#cam5
        camA = mhtTrackers + '/cam{:02d}{}PersonMASK_30FPSrotv1.txt'.format(cam2, experiment)#cam2
        finalTrackFile = 'detexp1Cam09PersonMASK_30FPSrot_SCAexp.txt'
    if cam2cam == 'H2to4' and experiment == 'exp1':
        #similarity based homography

        H2to4 = [[0.72, -0.41, 0.0],
                [0.41, 0.72, 0.0],
                [660.5, 311.7, 1.0]]
        fr_offsetP = 7
        fr_offsetA = 0
        time_offset_Psca = -7
        time_offset_Asca =  0   ## T_c4 - T_c2 = offset = 7
        H = np.transpose(H2to4)
        d_th = 0.5  # 0.17#0.17#0.45#200 # multi-task cost: 0.45, only loc: 0.17, loc+velocity 4d hd: 0.4
        camP = mhtTrackers + '/cam{:02d}{}PersonMASK_30FPSrot.txt'.format(cam1, experiment)#cam4
        camA = mhtTrackers + '/cam{:02d}{}PersonMASK_30FPSrot.txt'.format(cam2, experiment)#cam2
        finalTrackFile = 'detexp1Cam09PersonMASK_30FPSrot_SCAexp.txt'


    if cam2cam in ['H5to11','H11to5'] and experiment == 'exp1':
        # similarity based homography
        H5to11 = [[0.9704, 0.1027, 0.00],
                 [-0.1027, 0.9704, 0.00],
                 [37.211, 514.028, 1.00]]
        fr_start = 800
        fr_end = 7000  # 13400
        fr_offsetP = -89  ##cam11 delay
        fr_offsetA = 0
        time_offset_Psca = 89
        time_offset_Asca =  0   ## T_c5 - T_c11 = offset
        H = np.transpose(H5to11)
        if cam2cam=='H11to5':
            H = np.linalg.inv(H)
        d_th = 0.5  # 0.17#0.17#0.45#200 # multi-task cost: 0.45, only loc: 0.17, loc+velocity 4d hd: 0.4
        camP = mhtTrackers + '/cam{:02d}{}PersonMASK_30FPSrot.txt'.format(cam1, experiment)#cam11
        camA = mhtTrackers + '/cam{:02d}{}PersonMASK_30FPSrot.txt'.format(cam2, experiment)#cam5
        finalTrackFile = 'detexp1Cam09PersonMASK_30FPSrot_SCAexp.txt'

    if cam2cam == 'H9to2' and experiment == 'exp2':
        # similarity based homography
        H9to2 = [[1.0914, -0.1503, 0],
                 [0.1503, 1.0914, 0],
                 [228.4156, -714.1929, 1.0000]]
        fr_start = 1
        fr_end = 13400
        fr_offsetP = 70  ## cam2 delay
        fr_offsetA = 0  # cam9 delay
        H = np.transpose(H9to2)
        d_th = 0.2

    if cam2cam == 'H9to2' and experiment == 'exp5B':
        H9to2 =  [[1.0560,   -0.0823,    0.0003],
                  [-0.3904,    0.6757,   -0.0005],
                  [508.6211, -448.6223,    1.0000]]
        '''
            1.0914   -0.1503         0
    0.1503    1.0914         0
  228.4156 -714.1929    1.0000'''
        #similarity based homography
        H9to2 =  [[1.0914,   -0.1503,         0],
                  [0.1503,    1.0914,         0],
                  [228.4156, -714.1929,    1.0000]]
        fr_start = 180
        fr_end = 2660
        fr_offset = 31  # cam2 delay
        H = np.transpose(H9to2)
        d_th = 480

    if cam2cam == 'H5to11' and experiment == 'exp5B':
        '''
            0.8600    0.2691         0
   -0.2691    0.8600         0
   67.7767  488.8727    1.0000'''
        H11to5 = [[0.8600,    0.2691,         0],
                  [-0.2691,    0.8600,         0],
                  [67.7767,  488.8727,    1.0000]]
        H = np.transpose(H11to5)
        fr_start = 800
        fr_end = 2900
        fr_offset = 111 # cam11 delay: i.e, if cam5>>211,cam11>>100
        d_th = 480

    if cam2cam in ['H9to2','H2to9'] and experiment == 'exp5A':
        #projective homography
        '''
        H9to2 =  [[1.0560,   -0.0823,    0.0003],
          [-0.3904,    0.6757,   -0.0005],
          [508.6211, -448.6223,    1.0000]]
        '''

        #similarty based homography
        H9to2 =  [[1.0914,   -0.1503,         0],
                  [0.1503,    1.0914,         0],
                  [228.4156, -714.1929,    1.0000]]
        fr_start = 200
        fr_end = 3000
        if cam2cam == 'H2to9':
            H = np.linalg.inv(np.transpose(H9to2))
        else:
            H = np.transpose(H9to2)
        fr_offsetP = -29  # cam2-361, cam9-390
        fr_offsetA = 0
        d_th = 0.2105#0.2105#0.4#d_h = 0.203, 0.2105 with velocity, 0.12 w/o velocity, d_mah = 2, multi-task = 0.39
        finalTrackFile = 'det9ACam09PersonMASK_2FPSrot_SCA.txt'

    if cam2cam in ['H9to2','H2to9'] and experiment == 'exp10A':
        #projective
        '''
        H9to2 =  [[1.0560,   -0.0823,    0.0003],
                  [-0.3904,    0.6757,   -0.0005],
                  [508.6211, -448.6223,    1.0000]]
        '''
        #Similarity transformation

        H9to2 =  [[1.0914,   -0.1503,         0],
                  [0.1503,    1.0914,         0],
                  [228.4156, -714.1929,    1.0000]]
        fr_start = 300
        fr_end = 4800
        if cam2cam == 'H2to9':
            H = np.linalg.inv(np.transpose(H9to2))
        else:
            H = np.transpose(H9to2)
        fr_offsetP = 3  # cam2-, cam9-
        fr_offsetA = 0
        d_th = 0.171#d_h = 300
        finalTrackFile  = 'det10ACam09PersonMASK_2FPSrot_SCA.txt'

    if cam2cam == 'H9to2' and experiment == 'exp7A':
        H9to2 =  [[1.0560,   -0.0823,    0.0003],
                  [-0.3904,    0.6757,   -0.0005],
                  [508.6211, -448.6223,    1.0000]]
        fr_start = 200
        fr_end = 6500
        #H = np.linalg.inv(np.transpose(H9to2))
        H = np.transpose(H9to2)
        fr_offsetP = 60#cam2 delay
        fr_offsetA = 0
        d_th = 0#300
        finalTrackFile = 'det7ACam09PersonMASK_2FPSrot.txt'

    if cam2cam in ['H9to2','H2to9']  and experiment == 'exp6A':
        H9to2 =  [[1.0560,   -0.0823,    0.0003],
                  [-0.3904,    0.6757,   -0.0005],
                  [508.6211, -448.6223,    1.0000]]
        fr_start = 800#2300#800
        fr_end = 7000
        if cam2cam == 'H2to9':
            H = np.linalg.inv(np.transpose(H9to2))
        else:
            H = np.transpose(H9to2)
        fr_offsetP = 300 ## cam2 delay
        fr_offsetA = 0
        d_th =0# 250#350#300
        finalTrackFile = 'det6ACam09PersonMASK_2FPSrot_SCA.txt'

    if cam2cam in ['H9to2','H2to9'] and experiment == 'exp9A':
        '''
        #use projective transformation
        H9to2 = [[1.0560, -0.0823, 0.0003],
         [-0.3904, 0.6757, -0.0005],
         [508.6211, -448.6223, 1.0000]]
        H = np.transpose(H9to2)
        '''
        #use similarity transformation
        H9to2 = [[0.9945,   -0.1652,         0],
         [0.1652,    0.9945,        0],
         [ 199.6502, -635.9232,    1.0000]]
        if cam2cam == 'H2to9':
            H = np.linalg.inv(np.transpose(H9to2))
        else:
            H = np.transpose(H9to2)

        fr_start = 200
        fr_end = 4700
        fr_offsetP = 60 ## cam2 delay
        fr_offsetA = 0# cam9 delay
        d_th = 0.18#0.12#0.33#295 h_d:0.12,h_d_4D:0.28, multi-task: 0.45
        finalTrackFile = 'det9ACam09PersonMASK_1FPSrot_SCA.txt'

    if cam2cam == 'H5to11' and experiment == 'exp5A':
        '''
        H11to5 = [[0.85, -0.088, 0],
                  [-0.225, 0.7, -0.0002],
                  [202.18, -315.5, 1.0]]
        H = np.linalg.inv(np.transpose(H11to5))

    0.8571    0.2681         0

   -0.2681    0.8571         0

   70.9060  494.1304    1.0000
        '''
        H5to11 = [[0.8571,    0.2681,         0],
                  [-0.2681,    0.8571,         0 ],
                  [70.9060,  494.1304,    1.0000]]
        H = np.transpose(H5to11)
        fr_start = 900
        fr_end = 3500
        fr_offsetA = 24 # cam2: 2584, cam11: 2560
        fr_offsetP = 0  # cam2: 2584, cam11: 2560
        d_th = 270 #200
        finalTrackFile = 'det5ACam11PersonMASK_10FPSrot_SCA.txt'
    if cam2cam == 'H5to11' and experiment == 'exp10A':
        '''
        H11to5 = [[0.85, -0.088, 0],
                  [-0.225, 0.7, -0.0002],
                  [202.18, -315.5, 1.0]]
        H = np.linalg.inv(np.transpose(H11to5))
        '''
        H5to11 = [[0.8571,    0.2681,         0],
                  [-0.2681,    0.8571,         0 ],
                  [70.9060,  494.1304,    1.0000]]
        H = np.transpose(H5to11)
        fr_start = 1300
        fr_end = 5600
        fr_offsetP = 0#cam5:112, cam11:100
        fr_offsetA = 9
        d_th = 245#270 #250
        finalTrackFile = 'det10ACam11PersonMASK_2FPSrot_SCAexp.txt'
    if cam2cam == 'H5to11' and experiment == 'exp6A':

        H11to5 = [[0.85, -0.088, 0],
                  [-0.225, 0.7, -0.0002],
                  [202.18, -315.5, 1.0]]

        H = np.linalg.inv(np.transpose(H11to5))
        fr_start = 2030
        fr_end = 6800
        fr_offsetP = 0 ## cam11 delay: i.e, if cam5>>280,cam11>>250
        fr_offsetA = 25# cam5 delay
        d_th = 280#0.16#280#450
        finalTrackFile = 'det6ACam11PersonMASK_2FPSrot_SCA.txt'
    if cam2cam == 'H5to11' and experiment == 'exp7A':
        H11to5 = [[0.85, -0.088, 0],
                  [-0.225, 0.7, -0.0002],
                  [202.18, -315.5, 1.0]]
        H = np.linalg.inv(np.transpose(H11to5))
        fr_start = 1400
        fr_end = 6500
        fr_offsetA = 0
        fr_offsetP = 0
        d_th = 250
        finalTrackFile = 'det7ACam11PersonMASK_2FPSrot_SCA.txt'
    if cam2cam in ['H5to11','H11to5'] and experiment == 'exp9A':
        '''
        #using projective transformation
        H11to5 = [[0.85, -0.088, 0],
          [-0.225, 0.7, -0.0002],
          [202.18, -315.5, 1.0]]
        H = np.linalg.inv(np.transpose(H11to5))
        '''

        H5to11 = [[ 0.9603,    0.1652,         0 ],
                  [-0.1652,    0.9603 ,        0 ],
                  [-89.2048,  603.4781,    1.0000 ]]
        H = np.transpose(H5to11)
        #H = np.linalg.inv(np.transpose(H5to11))
        fr_start = 1200
        fr_end = 5700
        fr_offsetA = -12# cam5 delay
        fr_offsetP = 0 # cam11 delay: i.e, if cam5>>211,cam11>>100
        d_th =269#269.5#230 0.2#269.3, 0.2035 w/o vel,
        finalTrackFile = 'det9ACam11PersonMASK_1FPSrot_SCAexp.txt'


    folder = []
    tracklets = []
    for cam in [cam1, cam2]:
        # convert mht trajectory into tracklets: cam02exp1PersonMASK_30FPSrotv1
        if cam==cam1:
            trackers = np.loadtxt(camP, dtype='float32',delimiter=',')
        if cam==cam2:
            trackers = np.loadtxt(camA, dtype='float32',delimiter=',')
        # img_path = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/ourdataset/'
        # +experiment'/imgs/cam{}exp5a.mp4/'.format(cam)
        # demo_path = '/media/siddique/CLASP2019/CLASP_tracking_results/cam{}exp5a/'.format(cam)
        fr_start = trackers[0, 0]
        fr_end = trackers[-1, 0]
        if cam == cam1:
            trackers[:, 0] = trackers[:, 0] + time_offset_Psca
            trklt_size = 120  # cam5:120, cam2:120
            fr_start = trackers[0, 0]
            fr_end = trackers[-1, 0]
            isSCA = 0
        else:
            trklt_size = 0  # 9A: 0, 10A:30, 5A:30, 6A:60
            trackers[:, 0] = trackers[:, 0] + time_offset_Asca
            trklt_size = 120  # 9A: 15, 10A: 60, 5A: 30
            fr_start = trackers[0, 0]
            fr_end = trackers[-1, 0]
            isSCA = 0
            print 'Tracklets are Synchronized'
        print 'Frame start and end {}, {}'.format(trackers[0, 0], trackers[-1, 0])
        # use [cx,cy] to compute distance and d_th = 10 for single camera association
        # Apply SCTA and Filtering
        mht_tracklets, totalTrack = form_tracklets(trackers, fr_start, fr_end, tracker_min_size=trklt_size,
                              t_thr=120,d_thr=50,single_cam_association=isSCA).tracker_mht()
        '''
        if experiment!='exp1':
            folder = '/home/siddique/multi-camera-association/mht_trajectory/'+experiment+'/'
            detfile = 'cam{:01d}/trackers_mht_cam{:01d}'.format(cam,cam)+tracklet_exp+'.npy'
            trackers = np.load('{}{}'.format(folder, detfile), allow_pickle=True)
        '''
        #apply single camera tracklet assocaitation
        #trackers,_ = merge_tracklets(trackers, trackers)
        #trklt = get_tracklets(trackers)label_map = {keys: [] for keys in range(len(tracklets[0]))}
        if cam==cam2:
            #tracklet_2 = np.load('{}{}'.format(folder, detfile), allow_pickle=True)
            tracklet_2 = copy.deepcopy(mht_tracklets)
            tracklet_2 = convert_centroids(tracklet_2)
        if cam == cam2:
            #Project: center>(cx,cy), top-left>(x,y), velocity>(vx,vy) [fr,cx,cy,x,y,vx,vy]
            # active_trklt: number of source tracklets and their projections onto target camera (bounded in image frame) might be different
            trackletsA,active_trklt = project_tracklets(mht_tracklets, H, isDistorted=isDistCorrect, cam=cam2)
            tracklets.append(trackletsA)
            #tracklet_2 = tracklet_2[active_trklt]
        else:
            trackletsP = convert_centroids(mht_tracklets)
            tracklets.append(trackletsP)


    cam2to1_tracklets = np.append(tracklets[0], tracklet_2)
    # TODO: need to verify for 9to2, 5to11
    cam2to1_tracklets[len(tracklets[0])+np.array(active_trklt)] = tracklets[1]
    # use raw tracklets for visualization and metrics evaluation
    cam2to1Original = np.append(tracklets[0], tracklet_2)
    if DFS:
        mt,tCommonPA = associate_tracklets_DFS(cam1,cam2,cam2to1_tracklets, tracklets[0], tracklet_2,active_trklt,
                                               isDistorted=isDistCorrect, max_dist=d_th)
        # refine association based on heuristics
        tStampStart = {keys: [] for keys in range(len(cam2to1_tracklets))}
        #mt = refine_association(mt,tStampStart)
        edges = {keys: [] for keys in range(len(cam2to1_tracklets))}

        for (c, tStampComnMin, k, l) in mt:
            edges[k].append(l)
            edges[l].append(k)
            print '{} in camera {} associate with {} in camea {}'.format(k, cam1, l, cam2)
        # TODO: instead of mapping label append tracklets based on label similarity
        label_map = {keys: [] for keys in range(len(cam2to1_tracklets))}

        for label_tp,tklt in enumerate(cam2to1_tracklets):
            solnPath  = dfs(edges,label_tp,[])
            label_map[label_tp].append(solnPath)
            for tStampInd in solnPath:
                tStamp =cam2to1_tracklets[tStampInd][:,0].min()
                tStampStart[label_tp].append(tStamp)
        # update tracklet identity after association
        label_map_update = {keys: [] for keys in range(len(cam2to1_tracklets))}

        for k in label_map.keys():
            label_tp = label_map[k][0]
            if len(label_tp) >= 2:
                label_map_update[k] = np.array(label_tp)[tStampStart[k] == min(tStampStart[k])][0]
            #if len(label_tp) == 2:
                #label_map_update[k] = label_tp[1]
            if len(label_tp) == 1:
                label_map_update[k] =  label_tp[0]

        if sequentialID:
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

                #pdb.set_trace()
    else:
        mt = associate_tracklets(cam2to1_tracklets,tracklets[0], tracklets[1],max_dist=d_th)
        # TODO: compute the edges from the association and apply DFS

        # Note that for camera-to-camera we don't need to merge the trajectories, only associate target IDs.
        # The labels of cam1 (target camera) mapped to the labels of cam2 (source camera)
        #label_map = dict.fromkeys(range(len(tracklets[0])), 0)
        #label_map = {}
        label_map_aux = {keys: [] for keys in range(len(tracklets[1]))}
        t_stamps_start_aux = {keys: [] for keys in range(len(tracklets[1]))}

        label_map = {keys: [] for keys in range(len(tracklets[0]))}
        t_stamps_start = {keys: [] for keys in range(len(tracklets[0]))}
        costs = {keys: [] for keys in range(len(tracklets[0]))}
        for (c,id_first_used, k, l) in mt:
            #tracklets[0][k] = np.vstack((tracklets[0][k], tracklets[1][l]))
            costs[k].append(c)
            t_stamps_start[k].append(id_first_used)
            label_map[k].append(l)

            label_map_aux[l].append(k)
            t_stamps_start_aux[l].append(id_first_used)
            print '{} in camera {} associate with {} in camea {}'.format(k,cam1,l,cam2)

        # Update labels in auxiliary camera, T_a
        label_map_update_aux = {keys: [] for keys in range(len(tracklets[1]))}
        for k in label_map_update_aux.keys():
            label_ta = label_map_aux[k]
            if len(label_ta)>1:
                label_map_update_aux[k] = np.array(label_ta)[t_stamps_start_aux[k]==min(t_stamps_start_aux[k])][0]
            if len(label_ta)==1:
                label_map_update_aux[k] = label_ta[0]
        # Update labels in Primary camera, T_p
        label_map_update = {keys: [] for keys in range(len(tracklets[0]))}
        for k in label_map.keys():
            label_tp = label_map[k]
            if len(label_tp)>1:
                label_map_update[k] = np.array(label_tp)[t_stamps_start[k]==min(t_stamps_start[k])][0]
            if len(label_tp)==1:
                label_map_update[k] = label_tp[0]
            if len(label_tp)==0:
                label_map_update[k] = k

    if experiment=='exp1':
        cam1_path = benchmark + experiment + '/imgs/cam{:02d}'.format(cam1) + tracklet_exp + '.mp4/'
        cam2_path = benchmark + experiment + '/imgs/cam{:02d}'.format(cam2) + tracklet_exp + '.mp4/'
        out_path = benchmark + experiment + '/Results/MCTA/vis/cam{}to{}/'.format(cam2, cam1)

    if experiment=='exp2':
        cam1_path = benchmark + experiment + '/imgs/cam{:02d}'.format(cam1) + tracklet_exp + '.mp4/'
        cam2_path = benchmark + experiment + '/imgs/cam{:02d}'.format(cam2) + tracklet_exp + '.mp4/'
        out_path = benchmark + experiment + '/Results/MCTA/vis/cam{}to{}/'.format(cam2, cam1)
    '''
    else:
        cam1_path =  '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/ourdataset/'\
                     +experiment+'/imgs/cam{:01d}'.format(cam1)+tracklet_exp+'.mp4/'
        cam2_path = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/ourdataset/' \
                    + experiment + '/imgs/cam{:01d}'.format(cam2)+tracklet_exp+'.mp4/'
        out_path = '/media/siddique/CLASP2019/CLASP_tracking_results/Multi-camera_tracking/Results/'\
                   +experiment+'/cam{}to{}/'.format(cam2,cam1)
    '''
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if eval:
        tracklets[0] = centroids2xywh(tracklets[0])
        tracklet_2 = centroids2xywh(tracklet_2)
        track_for_mot = open(out_path + finalTrackFile, 'w')
    # start and end frames are based on primary camera
    fr_start = 1
    fr_end = 13400
    plot_trajectories(label_map_update,fr_start, fr_end,cam2to1_tracklets, cam2to1Original, tracklets[0],
                      fr_offsetP,fr_offsetA,cam1_path,cam2_path,out_path,cam2cam,vis_rate=30,isDistorted=isDistCorrect,
                      metrics=eval,save_imgs=1,motion=0)
                   # '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/ourdataset/'+experiment+'/cam{:01d}/30FPS/'.format(cam2))

