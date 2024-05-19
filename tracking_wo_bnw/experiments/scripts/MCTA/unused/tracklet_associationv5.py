import numpy as np
from scipy.spatial.distance import directed_hausdorff
from scipy.optimize import linear_sum_assignment
import cv2
import sys

np.set_printoptions(threshold=sys.maxsize)

__version__ = 0.7

def get_tracklets(trackers):
    tracklets = []

    for tid in range(len(trackers)):
        tracker_i = trackers.get(str(tid + 1))
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


def applyTransform(source_corners, H):
    dest_corners = np.empty(2)

    w = H[2][0] * source_corners[0] + H[2][1] * source_corners[1] + H[2][2] * 1
    dest_corners[0] = (H[0][0] * source_corners[0] + H[0][1] * source_corners[1] + H[0][2] * 1) / w
    dest_corners[1] = (H[1][0] * source_corners[0] + H[1][1] * source_corners[1] + H[1][2] * 1) / w

    return dest_corners


def project_tracklets(in_tracklets, H):
    out_tracklets = []

    for trklt in in_tracklets:
        for bb in trklt:
            bbt1 = applyTransform(bb[1:3], H)
            bbt2 = applyTransform(bb[1:3] + bb[3:5] / 2, H) - bbt1
            bb[1:5] = np.concatenate([bbt1, bbt2])
        # Delete tracklets that don't have any detection visible in the second camera
        # TODO: There must be a smarter and faster way to do that
        if max(trklt[:, 2]) > 0 and max(trklt[:, 3]) > 0:
            out_tracklets.append(trklt)

    return out_tracklets


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


def associate_tracklets(cam1_tracklets, cam2_tracklets, max_dist=250):
    cost = np.empty((len(cam1_tracklets), len(cam2_tracklets)))
    it = np.nditer(cost, op_flags=['readwrite'])

    i = 0
    for tracklet1 in cam1_tracklets:
        j = 0
        for tracklet2 in cam2_tracklets:
            print('Processing tracklets {} and {}'.format(i, j))
            # Quick check if tracklets overlap
            # TODO: This condition (and the loop below) could be relaxed to allow a maximum offset between two
            #       tracklets. That could be useful to apply the same procedure for same camera association
            if tracklet1[-1, 0] < tracklet2[0, 0] or tracklet1[0, 0] > tracklet2[-1, 0]:
                it[0] = np.inf
            # Search for the overlapping portions of both tracklets
            else:
                # TODO: There must be a more compact way of doing this
                s = [0, 0]
                e = [len(tracklet1) - 1, len(tracklet2) - 1]

                while e[0] > 0 and tracklet1[e[0]][0] > tracklet2[e[1]][0]:
                    e[0] = e[0] - 1
                while e[1] > 0 and tracklet2[e[1]][0] > tracklet1[e[0]][0] and e[1] >= 0:
                    e[1] = e[1] - 1
                while s[0] < e[0] and tracklet1[s[0]][0] < tracklet2[s[1]][0]:
                    s[0] = s[0] + 1
                while s[1] < e[1] and tracklet2[s[1]][0] < tracklet1[s[0]][0]:
                    s[1] = s[1] + 1

                # TODO: Assess whether the timestamp should be considered in the distance computation as well
                # This paper uses Hausdorff distances: https://www.ijcai.org/Proceedings/16/Papers/479.pdf
                #it[0] = directed_hausdorff(tracklet1[s[0]:e[0], 1:], tracklet2[s[1]:e[1], 1:])[0]
                it[0] = directed_hausdorff(tracklet1[s[0]:e[0]+1], tracklet2[s[1]:e[1]+1])[0]
            it.iternext()
            j = j + 1
        i = i + 1

    row_ind, col_ind = linear_sum_assignment_with_inf(cost)

    # DEBUG
    # hist = np.histogram(cost[cost < 10000])
    # plt.hist(cost[cost < 10000])

    matches = []

    for (i, j) in zip(row_ind, col_ind):
        if cost[i, j] < max_dist:
            matches.append((cost[i, j], i, j))

    return matches


def plot_trajectories(fr_start, fr_end, tracklets, folder, folder2=None, fr_offset=2320):
    # TODO: Figure out why I need the reshape here
    color = np.column_stack([[255.0 * np.random.rand(1, len(tracklets))],
                             [255.0 * np.random.rand(1, len(tracklets))],
                             [255.0 * np.random.rand(1, len(tracklets))]]).reshape(3, len(tracklets))
    # plt.imshow(np.ones((1080, 1920, 3)))
    fr_offset = 0

    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 640, 480)

    for fr in range(fr_start, fr_end):
        try:
            img = cv2.imread('{}{:06d}.png'.format(folder, fr - fr_offset + 5))
            if folder2:
                img2 = cv2.imread('{}{:06d}.png'.format(folder2, fr + 5))

            c = 0
            for bb in tracklets:
                # TODO: Set the colors of the rectangles
                for i in range(1, len(bb)):
                    if fr_start <= bb[i, 0] <= fr:
                        cv2.line(img, (int(bb[i - 1, 1]), int(bb[i - 1, 2])), (int(bb[i, 1]), int(bb[i, 2])),
                                 (color[:, c]), thickness=5, lineType=8)
                        # TODO: Think about how to show this only at the last detection in the current frame
                        if bb[i, 0] == fr:
                            cv2.putText(img, '{}'.format(c), (int(bb[i - 1, 1]), int(bb[i - 1, 2])),
                                        cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 0, 0), 5, cv2.LINE_AA)
                c = c + 1

            if folder2:
                cv2.imshow("image", cv2.vconcat([img,img2]))
            else:
                cv2.imshow("image", img)
            cv2.waitKey(30)

        except:
            print("Frame {} not found".format(fr))
            continue

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
        if cost[i,col_ind[i]] < 1:
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

    for l in del_tl:
        del(filtered_tracker[l])
        del(unassigned_tracklets[l])

    return filtered_tracker, unassigned_tracklets


if __name__ == "__main__":

    camA = 13
    camB = 11

    '''H9to2 = [[1.92798313033608, -0.206244644719952, 0.00061786025610996],
             [0.16363727596408, 1.16058885545216, -0.000138947293184879],
             [64.8383967198011, -781.41777598123, 1]]'''

    H = np.transpose([[1.0877, -0.0943, 0],
                      [0.0943,  1.0877, 0],
                      [676.0355, 18.2077, 1]])

    '''H = np.transpose([[0.9125, 0.0791, 0.0000],
                      [-0.0791, 0.9125, 0.0000],
                      [-615.4503, -70.0971, 1.0000]])'''

    fr_start = 2500 # 4704  # 4131 #2320  # 2321#6228
    fr_end =  2811  #5702  # 4274 #2900  # 6650

    folder = []
    tracklets = []

    for cam in [camA, camB]:
        folder.append('../Data/cam{}/'.format(cam))
        detfile = 'trackers_cam{:02d}exp2.npy'.format(cam)
        trackers = np.load('{}{}'.format(folder[-1], detfile), allow_pickle=True).item()

        trklt = get_tracklets(trackers)

        if cam == camB:
            trklt = project_tracklets(trklt, H)

        # TODO: It would be more elegant to create new trajectories only with the centroids
        tracklets.append(convert_centroids(trklt))

        trklt = filter_tracklets(merge_tracklets(trklt, trklt), 5)


    mt = associate_tracklets(tracklets[0], tracklets[1])

    # Note that for camera-to-camera we don't need to merge the trajectories, only associate target IDs.
    # This is just for testing
    for (c, k, l) in mt:
        tracklets[0][k] = np.vstack((tracklets[0][k], tracklets[1][l]))

    plot_trajectories(fr_start, fr_end, tracklets[0], folder[0])

