import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from MCTA.read_TU import bb_intersection_over_union
from scipy.optimize import linear_sum_assignment
#import cv2
'''
trackers_cam1 = np.load('./trackers_cam01exp1.mp4.npy', allow_pickle=True)
trackers = trackers_cam1.item()

fr_start = 2300  # 2321#6228
fr_end = 2900  # 6650
'''
def linear_sum_assignment_with_inf(cost_matrix):
    cost_matrix = np.asarray(cost_matrix)
    min_inf = np.isneginf(cost_matrix).any()
    max_inf = np.isposinf(cost_matrix).any()
    if min_inf and max_inf:
        raise ValueError("matrix contains both inf and -inf")

    if min_inf or max_inf:
        values = cost_matrix[~np.isinf(cost_matrix)]
        if values.size == 0:
            cost_matrix = np.full(cost_matrix.shape,1000) #workaround for the cast of no finite costs
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

def match_tl(det1, det2, t_thr=150, d_thr=200, metric='iou'):
    # Associate pairs of tracklets with maximum overlap (minimum euclidean distance)
    #  between the last and first detections
    # using the Hungarian algorithm
    WH=[1920., 1080.]
    if metric == 'iou':
        assert 0<=d_thr<=1, 'distance threshold should be [0, 1] for iou metric'

    new_track = []
    prev_track = []
    matches = []

    for tr in det1:
        prev_track.append(tr[-1])
    for tr in det2:
        new_track.append(tr[0])

    # Compute the cost matrix
    cost = np.full((len(prev_track), len(new_track)),np.inf)
    i = 0
    for prev_det in prev_track:
        j = 0
        for new_det in new_track:
            delta_t = new_det[0] - prev_det[0]
            if metric=='euclidean':
                dist = np.linalg.norm((new_det[1:3]+new_det[3:5]/2)/WH - (prev_det[1:3]+ prev_det[3:5]/2)/WH)
            elif metric=='iou':
                iou = bb_intersection_over_union(
                    [new_det[1], new_det[2], new_det[1] + new_det[3], new_det[1] + new_det[4]],
                    [prev_det[1], prev_det[2], prev_det[1] + prev_det[3], prev_det[1] + prev_det[4]])
                # for iou=1, cost=0, for iou=0, cost=1
                dist = 1 - iou
            else:
                print('metric should be iou or euclidean')

            if 0 < delta_t < t_thr and 0 < dist < d_thr:
                cost[i,j] = dist
            j = j + 1
        i = i + 1

    row_ind, col_ind = linear_sum_assignment_with_inf(cost)

    for i in row_ind:
        if cost[i,col_ind[i]] < d_thr:
            matches.append((cost[i, col_ind[i]], i, col_ind[i]))

    return matches


def merge_tracklets_old(filtered_tracker, unassigned_tracklets,t_thr=150, d_thr=60):
    # Iterate a few times to join pairs of tracklets that correspond to multiple
    # fragments of the same track
    # TODO: Find a more elegant solution -- Possibly using DFS
    del_tl = []

    for merge_steps in range(0,5):

        mt = match_tl(filtered_tracker,unassigned_tracklets,t_thr, d_thr)

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

def merge_tracklets(filtered_tracker, unassigned_tracklets,t_thr=150, d_thr=60, metric='iou'):
    G_sc =  {tr[0,-1]: [] for tr in filtered_tracker}

    mt = match_tl(filtered_tracker, unassigned_tracklets, t_thr, d_thr, metric=metric)
    for (c, k, l) in mt:
        l_id = filtered_tracker[l][0, -1]
        k_id = filtered_tracker[k][0, -1]
        assert k_id!=l_id
        G_sc[k_id].append(l_id)
        G_sc[l_id].append(k_id)
        print('found SCA match (k, l, c): {}-{}:{}'.format(k_id, l_id, c))

    return G_sc

def relabel_SCT(tracklets, G_sc):
    label_map = {tr[0, -1]: [] for tr in tracklets}
    # tStampStart = {tr[0,1]: [] for tr in tracklets}
    for tklt in tracklets:
        solnPath = dfs(G_sc, tklt[0, -1], [])
        tklt[:,-1] = min(solnPath)  # ids are assigned sequentially
    return tracklets

def get_tracklets(trackers):
    tracklets = []
    for tracker_i in trackers:
        tracker_i = np.array(sorted(tracker_i, key=lambda x: x[0]))
        tracklets.append(tracker_i)
    return tracklets


def filter_tracklets(trackers,min_size):
    # Remove tracklets with <60 detections
    return [tl for tl in trackers if len(tl)>=min_size]