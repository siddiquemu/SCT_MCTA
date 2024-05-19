import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from read_TU import bb_intersection_over_union
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

def match_tl(det1, det2, t_thr = 90):
# Associate pairs of tracklets with maximum overlap between the last and first detections
# using the Hungarian algorithm
# track_format: [fr x y w h]

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
            if new_det[0] > prev_det[0] and (new_det[0] - prev_det[0])<t_thr:
                iou = bb_intersection_over_union(
                    [new_det[1], new_det[2], new_det[1]+new_det[3], new_det[1]+new_det[4]],
                    [prev_det[1], prev_det[2], prev_det[1]+prev_det[3], prev_det[1]+prev_det[4]])
                if iou > 0 :
                    # for iou=1, cost=0, for iou=0, cost=1
                    cost[i,j] = 1-iou
            j = j + 1
        i = i + 1

    row_ind, col_ind = linear_sum_assignment_with_inf(cost)

    # Find the maximum IoU for each pair
    for i in row_ind:
        if cost[i,col_ind[i]] < 1:
            matches.append((cost[i, col_ind[i]], i, col_ind[i]))

    return matches



def get_tracklets(trackers):
    tracklets = []

    for tid in range(len(trackers)):
        tracker_i = trackers.get(str(tid))
        tracker_i = np.array(list(tracker_i))
        tracker_i = np.array(sorted(tracker_i, key=lambda x: x[0]))
        tracklets.append(tracker_i)
    return tracklets


def filter_tracklets(trackers,min_size):
    # Remove tracklets with <60 detections
    return [tl for tl in trackers if len(tl)>=min_size]

def merge_tracklets(filtered_tracker, unassigned_tracklets, t_thr=90):
    # Iterate a few times to join pairs of tracklets that correspond to multiple
    # fragments of the same track
    # TODO: Find a more elegant solution
    del_tl = []
    for merge_steps in range(0,5):

        mt = match_tl(filtered_tracker,unassigned_tracklets, t_thr=90)

	pairs = []
        for pair_n in range(len(mt)):
            print 'Pairs: ',mt[pair_n][1:3]
            pairs.append([mt[pair_n][1:3][0], mt[pair_n][1:3][1]])
        pairs = np.array(pairs)

        for (c, k, l) in mt:
            t_e = filtered_tracker[k][:,0][-1]
            t_s = unassigned_tracklets[l][:,0][0]
            delta_t = t_s-t_e# t_s>t_e for occlusion
            print 'Delta_t {}'.format(delta_t)
            filtered_tracker[k] = np.vstack((filtered_tracker[k],unassigned_tracklets[l]))
            del_tl.append(l)

    del_tl.sort(reverse=True)
    for l in np.unique(del_tl)[::-1]:
        del(filtered_tracker[l])
        del(unassigned_tracklets[l])

    return filtered_tracker, unassigned_tracklets


def show_trajectories(fr_start, fr_end, tracklets):
    #TODO: This is insanely slow! Implement something faster.
    first_flag = [False]* len(tracklets)

    color = plt.cm.hsv(np.linspace(0, 1, len(tracklets)))
    plt.imshow(np.ones((1080, 1920, 3)))
    ca = plt.gca()

    for fr in range(fr_start,fr_end):
        c = 0
        for trklt in tracklets:
            #TODO: Set the colors of the rectangles
            for bb in trklt:
                if bb[0] == fr:
                    rect = patches.Rectangle((bb[1],bb[2]),bb[3],bb[4], fill=None, color=color[c])
                    ca.add_patch(rect)
                    if not first_flag[c]:
                        plt.text(bb[1]+bb[3], bb[2], 'P{}'.format(c+1), fontsize=12, color=color[c])
                        first_flag[c] = True
            c = c + 1
        plt.pause(0.01)

'''
if __name__ == "__main__":

    long_tracklets = get_tracklets(trackers)
    short_tracklets = get_tracklets(trackers)

    [long_tracklets, short_tracklets] = merge_tracklets(long_tracklets, short_tracklets)

    final_tracklets = filter_tracklets(long_tracklets)

    show_trajectories(fr_start, fr_end, long_tracklets)

    print('Done!')
'''
