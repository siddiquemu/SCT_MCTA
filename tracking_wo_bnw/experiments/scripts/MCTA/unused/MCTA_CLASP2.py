import numpy as np
from scipy.spatial.distance import directed_hausdorff,cosine
from scipy.optimize import linear_sum_assignment
import cv2
import sys
import os
import glob
np.set_printoptions(threshold=sys.maxsize)

__version__ = 0.4

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


def project_tracklets(in_tracklets, H):
    out_tracklets = []
    active_trklt = []

    for i,trklt in enumerate(in_tracklets):
        for bb in trklt:
            bbt1 = applyTransform(bb[1:3], H)
            bbt2 = applyTransform(bb[1:3] + bb[3:5] / 2, H) - bbt1
            bb[1:5] = np.concatenate([bbt1, bbt2])
        # Delete tracklets that don't have any detection visible in the second camera
        # TODO: There must be a smarter and faster way to do that
        if max(trklt[:, 2]) > 0 and max(trklt[:, 3]) > 0:
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


def associate_tracklets(cam1_tracklets, cam2_tracklets, max_dist=400):
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


def plot_trajectories(label_map,fr_start, fr_end,tracklet_cam9, tracklets,tracklets2,fr_offset,folder,folder2,out_path,metrics=False,save_imgs=False):
    # TODO: Figure out why I need the reshape here
    color = np.column_stack([[255.0 * np.random.rand(1, len(tracklets2))],
                             [255.0 * np.random.rand(1, len(tracklets2))],
                             [255.0 * np.random.rand(1, len(tracklets2))]]).reshape(3, len(tracklets2))
    # plt.imshow(np.ones((1080, 1920, 3)))
    #fr_offset = 48#11to13
    #fr_offset = 29  # 9to2, cam2 lag = 29
    if save_imgs:
        if not os.path.exists(out_path):
            os.makedirs(out_path)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 640, 480)
    for fr in range(fr_start, fr_end):
        if (fr)%10==0:
            try:
                img = cv2.imread('{}{:06d}.png'.format(folder, fr)) # target camera 2,11 - folder:
                #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img2 = cv2.imread('{}{:06d}.png'.format(folder2, fr)) # source camera 9,5 - folder2: + fr_offset for 9to2, + fr_offset for 5to11

                c = 0
                for bb in tracklets2: # projection of auxiliary camera
                    # TODO: Set the colors of the rectangles
                    for i in range(1, len(bb)):
                        if fr_start <= bb[i, 0] <= fr:
                            #cv2.line(img, (int(bb[i - 1, 1]), int(bb[i - 1, 2])), (int(bb[i, 1]), int(bb[i, 2])),
                                     #(color[:, c]), thickness=5, lineType=8)
                            # TODO: Think about how to show this only at the last detection in the current frame
                            if bb[i, 0] == fr :#and len(label_map_aux[c])>0
                                cv2.putText(img, '{}'.format(c), (int(bb[i - 1, 1]), int(bb[i - 1, 2])),
                                            cv2.FONT_HERSHEY_SIMPLEX, 3,  (0, 255, 0), 5, cv2.LINE_AA)
                    c = c + 1
                c=0
                for i_p,bb in enumerate(tracklets): #primary camera
                    # To show the association with the projections (tracklets2)
                    for i in range(1, len(bb)):
                        if fr_start <= bb[i, 0] <= fr:
                            if metrics:
                                if bb[i, 0] == fr:#9A: fr-15
                                    #if bb[i, 0]>=5415:
                                        #label_map[c]=0
                                    #Adjust frame delay and frame rate to find matching with GT
                                    #5A:9,11>bb[i, 0]+30, 7A:9,11>bb[i, 0]+30, 10A: 9,11>bb[i, 0]+30,9A: 9,11>bb[i, 0]+15,frdelay=15
                                    cv2.rectangle(img, (int(bb[i, 1]), int(bb[i, 2])),
                                                  (int(bb[i, 1] + bb[i, 3]), int(bb[i, 2] + bb[i, 4])),
                                                  (255, 0, 0), 5, cv2.LINE_AA)
                                    cv2.putText(img, 'P{}'.format(c),
                                                (int(bb[i, 1] + bb[i, 3] / 2), int(bb[i, 2] + bb[i, 4] / 2)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 8, cv2.LINE_AA)
                                    track_for_mot.writelines(
                                        str((bb[i, 0]+30)//15) + ',' + str(c+1) + ',' + str(bb[i,1]) + ',' + str(
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
                c = 0
                for bb in tracklet_cam9: # source camera : yellow
                    # TODO: Set the colors of the rectangles
                    for i in range(1, len(bb)):
                        if fr_start <= bb[i, 0] <= fr:
                            #cv2.line(img, (int(bb[i - 1, 1]), int(bb[i - 1, 2])), (int(bb[i, 1]), int(bb[i, 2])),
                                     #(color[:, c]), thickness=5, lineType=8)
                            # TODO: Think about how to show this only at the last detection in the current frame
                            if bb[i, 0] == fr:
                                cv2.putText(img2, '{}'.format(c), (int(bb[i - 1, 1]), int(bb[i - 1, 2])),
                                            cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 5, cv2.LINE_AA)
                    c = c + 1

                final_img = cv2.vconcat([img2,img])
                cv2.imshow("image", final_img)
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


if __name__ == "__main__":
    benchmark2 = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/'
    #benchmark1 = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/ourdataset/'
    eval = True
    experiment = 'exp5A'
    tracklet_exp = 'exp5a'
    cam1 = 2 #t_p
    cam2 = 9 #t_a
    cam2cam = 'H'+str(cam2)+'to'+str(cam1)
    if cam2cam == 'H9to2' and experiment == 'exp2':
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

    if cam2cam == 'H5to11' and experiment == 'exp2':
        '''
        0.9304    0.1079         0
       -0.1079    0.9304         0
       83.0023  519.2747    1.0000
        '''
        H5to11 = [[0.93,    0.108,         0],
                  [-0.108,    0.93,         0],
                  [83,  519.27,    1.0000]]
        H = np.transpose(H5to11)
        fr_start = 3905  # 4131 #2320  # 2321#6228
        fr_end = 7000  # 4274 #2900  # 6650
        fr_offset = 30 # cam11 delay: i.e, if cam5>>211,cam11>>100
        d_th = 400

    if cam2cam == 'H2to9' and experiment == 'exp5A':
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
        fr_start = 300
        fr_end = 2800
        H = np.linalg.inv(np.transpose(H9to2))
        fr_offset = 31  # cam2-361, cam9-390
        d_th = 300

    if cam2cam == 'H2to9' and experiment == 'exp10A':
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
        fr_start = 400
        fr_end = 4200
        H = np.linalg.inv(np.transpose(H9to2))
        fr_offset = 0  # cam2 delay
        d_th = 300
    if cam2cam == 'H2to9' and experiment == 'exp7A':
        H9to2 =  [[1.0560,   -0.0823,    0.0003],
                  [-0.3904,    0.6757,   -0.0005],
                  [508.6211, -448.6223,    1.0000]]
        fr_start = 370
        fr_end = 5720
        H = np.linalg.inv(np.transpose(H9to2))
        fr_offset = 31  # cam2 delay
        d_th = 350

    if cam2cam == 'H2to9' and experiment == 'exp6A':
        H9to2 =  [[1.0560,   -0.0823,    0.0003],
                  [-0.3904,    0.6757,   -0.0005],
                  [508.6211, -448.6223,    1.0000]]
        fr_start = 1000
        fr_end = 7000
        H = np.linalg.inv(np.transpose(H9to2))
        fr_offset = 31  # cam2 delay
        d_th = 300

    if cam2cam == 'H2to9' and experiment == 'exp9A':
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
        H = np.linalg.inv(np.transpose(H9to2))

        fr_start = 300
        fr_end = 4700
        fr_offset = 55  # cam9 delay
        d_th = 400

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
        fr_start = 800
        fr_end = 3500
        fr_offset = 24 # cam2: 2584, cam11: 2560
        d_th = 400

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
        fr_offset = 12#cam5:112, cam11:100
        d_th = 350
    if cam2cam == 'H5to11' and experiment == 'exp6A':

        H11to5 = [[0.85, -0.088, 0],
                  [-0.225, 0.7, -0.0002],
                  [202.18, -315.5, 1.0]]

        H = np.linalg.inv(np.transpose(H11to5))
        fr_start = 1800
        fr_end = 6800
        fr_offset = 111 # cam11 delay: i.e, if cam5>>211,cam11>>100
        d_th = 350
    if cam2cam == 'H5to11' and experiment == 'exp7A':
        H11to5 = [[0.85, -0.088, 0],
                  [-0.225, 0.7, -0.0002],
                  [202.18, -315.5, 1.0]]
        H = np.linalg.inv(np.transpose(H11to5))
        fr_start = 1500
        fr_end = 6400
        fr_offset = 111 # cam11 delay: i.e, if cam5>>211,cam11>>100
        d_th = 350
    if cam2cam == 'H5to11' and experiment == 'exp9A':
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
        fr_start = 1300
        fr_end = 5600
        fr_offset = 12 # cam11 delay: i.e, if cam5>>211,cam11>>100
        d_th = 250


    folder = []
    tracklets = []
    for cam in [cam1, cam2]:
        folder = benchmark+experiment+'/Results'
        detfile = '/trackers_mht_cam{:02d}'.format(cam)+tracklet_exp+'.npy'
        trackers = np.load('{}{}'.format(folder, detfile), allow_pickle=True)
        #apply single camera tracklet assocaitation
        #trackers,_ = merge_tracklets(trackers, trackers)
        #trklt = get_tracklets(trackers)label_map = {keys: [] for keys in range(len(tracklets[0]))}
        if cam==cam2:
            tracklet_2 = np.load('{}{}'.format(folder, detfile), allow_pickle=True)
            tracklet_2 = convert_centroids( tracklet_2)
            #number of source tracklets and their projections onto target camera (bounded in image frame) might be different
        if cam == cam2:
            trackers,active_trklt = project_tracklets(trackers, H)
            tracklet_2 = tracklet_2[active_trklt]
        # TODO: It would be more elegant to create new trajectories only with the centroids
        tracklets.append(convert_centroids(trackers))
    mt = associate_tracklets(tracklets[0], tracklets[1],max_dist=d_th)
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

    # Update labels in Primary camera, T_a
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
            label_map_update[k] = -1
    cam1_path =  benchmark+experiment+'/imgs/cam{:02d}'.format(cam1)+tracklet_exp+'.mp4/'
    cam2_path = benchmark+experiment + '/imgs/cam{:02d}'.format(cam2)+tracklet_exp+'.mp4/'
    out_path = benchmark+experiment+'/Results/MCTA/cam{}to{}/'.format(cam2,cam1)

    if os.path.exists(out_path):
        result_list = glob.glob(out_path + '*.png')
        result_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        for path in result_list:
            os.remove(path)

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    if eval:
        tracklets[0] = centroids2xywh(tracklets[0])
        track_for_mot = open(out_path + 'det5ACam11PersonMASK_10FPSrot_SCA.txt', 'w')
    plot_trajectories(label_map_update,fr_start, fr_end, tracklet_2, tracklets[0],tracklets[1],fr_offset,cam1_path,cam2_path,out_path,metrics=eval,save_imgs=True)
                   # '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/ourdataset/'+experiment+'/cam{:01d}/30FPS/'.format(cam2))

