import  cv2
import numpy as np
import collections
import os
class global_tracker(object):
    def __init__(self, fr, id=None, global_tracker=None, first_used='false'):
        self.global_tracker = global_tracker
        self.fr = fr
        self.id = id
        self.first_used = first_used
    def update_state(self, bb):
        camera = str(bb[-1])
        event_type = 'LOC'
        type = 'PAX'
        pax_id = 'P'+str(self.id)
        self.global_tracker.writelines(
            event_type + ' ' + 'type: ' + type + ' ' + 'camera-num: ' + camera + ' ' + 'frame: ' + str(self.fr) + ' '
            'time-offset: ' + '{:.2f}'.format(self.fr / 30.0) + ' ' + 'BB: ' + str(int(bb[2])) + ', ' + str(int(bb[3])) + ', '
            + str(int(bb[2] + bb[4])) + ', ' + str(int(bb[3] + bb[5])) + ' ' + 'ID: ' + pax_id + ' ' + 'PAX-ID: ' + pax_id
            + ' ' + 'first-used: ' + self.first_used + ' ' + 'partial-complete: ' + 'description: ' + '\n')

def convert_centroids(tracklet):
    # [fr,id,x,y,w,h, cam] = [fr,id, x+w/2.y+h/2,w,h, cam]
    if len(tracklet) > 0:
        for tl in tracklet:
            for bb in tl:
                bb[2] = bb[2] + bb[4] / 2.0
                bb[3] = bb[3] + bb[5] / 2.0
    return tracklet


def centroids2xywh(tracklet):
    if len(tracklet)>0:
        for tl in tracklet:
            for bb in tl:
                bb[2] = bb[2] - bb[4] / 2.0
                bb[3] = bb[3] - bb[5] / 2.0
    return tracklet

def xywh2x1y1x2y2(tracklet):
    for tl in tracklet:
        for bb in tl:
            bb[3] = bb[1] + bb[3]
            bb[4] = bb[2] + bb[4]
    return tracklet

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

def vis_overlapped_trklt(final_tracklet1,final_tracklet2,i,j,out_dir):
    from mpl_toolkits.mplot3d import Axes3D
    out_dir = out_dir+'/feature_comp/'
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    import matplotlib.pyplot as plt
    Dh = directed_hausdorff(final_tracklet1[:, 1:3], final_tracklet2[:, 1:3])[0]
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

    plt.title('Computed Hausdorff distance {:.2f}'.format(Dh))
    plt.legend(loc=2)
    plt.savefig(out_dir+'{}_{}.png'.format(i,j), dpi=300)
    plt.close()

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
    delete_all(MC_trklt_C9C2['outPath'])
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow("image", 640, 480)
    for fr in range(fr_start, fr_end):
        if (fr) % vis_rate == 0:
            try:
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

                cv2.imshow("image", final_img)
                if save_imgs:
                    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(MC_trklt_C9C2['outPath'] + '/{:06d}.png'.format(fr), final_img)
                cv2.waitKey(30)
            except:
                # print("Frame {} not found".format(fr))
                continue
    cv2.destroyAllWindows()

def plot_global_tracks(labelCam2,
                        labelCam1,
                        batch_start,
                        batch_end,
                        MC_trklt_C9C2,
                        cam2cam,
                        track_saver,
                        vis_rate=30,
                        isDistorted=0,
                        metrics=eval,
                        save_imgs=0,
                        motion=0
                        ):
    assert len(MC_trklt_C9C2['PAproj']) == len(MC_trklt_C9C2['PAorg']) == MC_trklt_C9C2['trkltFamilySizeP'] + \
                                                                          MC_trklt_C9C2['trkltFamilySizeA']
    #labelCam1 = dict(list(label_map_update.items())[:MC_trklt_C9C2['trkltFamilySizeP']])
    #labelCam2 = dict(list(label_map_update.items())[MC_trklt_C9C2['trkltFamilySizeP']:])
    # BGR
    # magenta = (255,0,255)
    # yellow = (0,255,255)
    # green = (0,255,0)
    # blue = (255,0,0)

    #if not os.path.exists(MC_trklt_C9C2['outPath']):
       # os.makedirs(MC_trklt_C9C2['outPath'])
    #delete_all(MC_trklt_C9C2['outPath'])
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("image", 640, 480)
    for fr in range(batch_start, batch_end+1):
        if (fr) % vis_rate == 0:
            #try:
            if save_imgs:
                imgP = cv2.imread('{}{:06d}.png'.format(MC_trklt_C9C2['PImgsPath'], fr + MC_trklt_C9C2['fr_offsetPVis']))
                imgA = cv2.imread('{}{:06d}.png'.format(MC_trklt_C9C2['AImgsPath'], fr + MC_trklt_C9C2['fr_offsetAVis']))
            # visulaize the identity handover in primary camera
            # get detection from primary
            if MC_trklt_C9C2['trkltFamilySizeP']>0:
                c = 0
                for i_p, bb in enumerate(MC_trklt_C9C2['PAorg'][:MC_trklt_C9C2['trkltFamilySizeP']]):  # primary camera
                    # To show the association with the projections (tracklets2)
                    for i in range(1, len(bb)):
                        if batch_start <= bb[i, 0] <= fr:
                            if bb[i, 0] == fr:
                                #update global tracks
                                global_tracker(fr, id=labelCam1[c], global_tracker=track_saver, first_used='false').update_state(bb[i])
                                if save_imgs:
                                    cv2.rectangle(imgP, (int(bb[i, 1]), int(bb[i, 2])),
                                                  (int(bb[i, 1] + bb[i, 3]), int(bb[i, 2] + bb[i, 4])),
                                                  (255, 0, 0), 5, cv2.LINE_AA)
                                    cv2.putText(imgP, 'P{}'.format(labelCam2[c]),
                                                (int(bb[i, 1] + bb[i, 3] / 2), int(bb[i, 2] + bb[i, 4] / 2)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 2555, 0), 5, cv2.LINE_AA)
                    c = c + 1

            # visualize the actual target identity in auxiliary
            if MC_trklt_C9C2['trkltFamilySizeA'] > 0: # if primary tracks family is zero, aux will start from 0 to end
                c = MC_trklt_C9C2['trkltFamilySizeP']
                for bb in MC_trklt_C9C2['PAorg'][MC_trklt_C9C2['trkltFamilySizeP']:]:  # source camera : yellow
                    # TODO: Set the colors of the rectangles
                    for i in range(1, len(bb)):
                        if batch_start <= bb[i, 0] <= fr:
                            # TODO: Think about how to show this only at the last detection in the current frame
                            if bb[i, 0] == fr:
                                global_tracker(fr, id=labelCam2[c], global_tracker=track_saver, first_used='false').update_state(bb[i])
                                if save_imgs:
                                    cv2.putText(imgA, 'P{}'.format(labelCam2[c]),
                                                (int(bb[i, 1] + bb[i, 3] / 2), int(bb[i, 2] + bb[i, 4] / 2)),
                                                cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 2555, 0), 5, cv2.LINE_AA)
                                    cv2.rectangle(imgA, (int(bb[i, 1]), int(bb[i, 2])),
                                                  (int(bb[i, 1] + bb[i, 3]), int(bb[i, 2] + bb[i, 4])),
                                                  (255, 0, 0), 5, cv2.LINE_AA)
                    c = c + 1
            if save_imgs:
                if cam2cam == 'C9C2':
                    final_img = cv2.vconcat([imgA, imgP])
                if cam2cam == 'C2C5':
                    final_img = cv2.hconcat([imgA, imgP])  # img: c5, img2: c2
                if cam2cam == 'C11C5':
                    final_img = cv2.vconcat([imgA, imgP])
                if cam2cam == 'C13C5':
                    final_img = cv2.hconcat([imgA, imgP])  # img: c11, img2: c13
                if cam2cam == 'C13C11':
                    img2 = cv2.copyMakeBorder(imgA, 0, 0, 320, 0, cv2.BORDER_CONSTANT, value=0)
                    img = cv2.copyMakeBorder(imgP, 0, 0, 0, 320, cv2.BORDER_CONSTANT, value=0)
                    final_img = cv2.hconcat([imgA, imgP])  # img: c11, img2: c13
                if cam2cam == 'C13C14':
                    img = cv2.copyMakeBorder(imgP, 0, 0, 700, 0, cv2.BORDER_CONSTANT, value=0)
                    img2 = cv2.copyMakeBorder(imgA, 0, 0, 0, 700, cv2.BORDER_CONSTANT, value=0)
                    final_img = cv2.vconcat([imgA, imgP])  # img: c13, img2: c14

                    cv2.imshow("image", final_img)
                    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
                    cv2.imwrite(MC_trklt_C9C2['outPath'] + '/{:06d}.png'.format(fr), final_img)
                #cv2.waitKey(30)
            #except:
                #print("Frame {} has no global tracker state".format(fr))
                #continue
    #cv2.destroyAllWindows()

def dafault_params(cam2cam=None, tracker_motion=False):
    #setting pair-wise camera parameters
    if cam2cam == 'C9C2':
        # C2: primary, C9: auxiliary
        param = {}
        param['Pid'] = 2
        param['Aid'] = 9
        param['tOffsetPstart'] = -44  # 2 will start at 40 wrt 9 is 1: exp1-test: x2-x9=44, exp1-train: 40
        param['tOffsetAstart'] = 0  # 9
        # similarity based homography: TODO: compute homography using undistorted image
        # H9to2
        param['H'] = [[0.9812, 0.1426, 0.00],
                      [-0.1426, 0.9812, 0.00],
                      [463.84, -782.41, 1.00]]

        param['fr_offsetPVis'] = 44
        param['fr_offsetAVis'] = 0

        param['H'] = np.transpose(param['H'])
        param['d_th'] = 0.4#  #multi-task cost: 0.4, only loc: 0.25
        param['primary'] = []
        param['auxiliary'] = []
        param['finalTrackFile'] = 'cam09exp2_MCTA.txt'
        param['PImgsPath'] = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/cam02exp2.mp4'
        param['AImgsPath'] = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/cam09exp2.mp4'

    if cam2cam == 'C5C2':
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
        param['primary'] = []
        param['auxiliary'] = []
        param['finalTrackFile'] = 'cam09exp2_MCTA.txt'
        param['PImgsPath'] = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/cam02exp2.mp4'
        param['AImgsPath'] = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/cam05exp2.mp4'

    if cam2cam == 'C11C5':
        param = {}
        param['Pid'] = 5
        param['Aid'] = 11
        # C11: pri, C5: aux
        param['tOffsetPstart'] = -41  # 11 delayed 5 by 23: 11>1 == 5>24
        param['tOffsetAstart'] = 0  #
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

        param['fr_offsetPVis'] = 41
        param['fr_offsetAVis'] = 0  ## T_c2 - T_c11 = offset
        param['H'] = np.transpose(param['H'])
        # H11to5
        if param['Pid'] == 5:
            param['H'] = np.linalg.inv(param['H'])
        param['d_th'] = 0.35  # multi-task cost: 0.45, only loc: 0.17
        param['primary'] = []
        param['auxiliary'] = []
        param['finalTrackFile'] = 'cam09exp2_MCTA.txt'
        param['PImgsPath'] = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/cam05exp2.mp4'
        param['AImgsPath'] = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/cam011exp2.mp4'

    if cam2cam == 'C13C5':
        param = {}
        param['Pid'] = 5
        param['Aid'] = 13
        # C11: pri, C5: aux
        param['tOffsetPstart'] = -139  # 13>1 == 5>71
        param['tOffsetAstart'] = 0  #
        # similarity based homography: TODO: compute homography using undistorted image
        # H5to13
        '''
    0.9214   -0.1088         0
    0.1088    0.9214         0
  768.4175  536.0207    1.0000

        '''
        param['H'] = [[0.92, -0.108, 0.00],
                      [0.108, 0.92, 0.00],
                      [968.41, 736.02, 1.00]]

        param['fr_offsetPVis'] = 139
        param['fr_offsetAVis'] = 0  ## T_c2 - T_c11 = offset
        param['H'] = np.transpose(param['H'])
        # H13to5
        if param['Pid'] == 5:
            param['H'] = np.linalg.inv(param['H'])
        param['d_th'] = 0.37  # multi-task cost: 0.45, only loc: 0.17
        param['finalTrackFile'] = 'cam13exp2_MCTA.txt'
        param['PImgsPath'] = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/cam05exp2.mp4'
        param['AImgsPath'] = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/cam13exp2.mp4'

    if cam2cam == 'C13C11':
        param = {}
        param['Pid'] = 11
        param['Aid'] = 13
        # C11: pri, C5: aux
        param['tOffsetPstart'] = -98  # 11 delayed 5 by 23: 11>1 == 5>24
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

        param['fr_offsetPVis'] = 98
        param['fr_offsetAVis'] = 0  ## T_c2 - T_c11 = offset
        param['H'] = np.transpose(param['H'])
        # param['H'] = np.linalg.inv(param['H'])
        param['d_th'] = 0.33 # multi-task cost: 0.45, only loc: 0.17
        param['finalTrackFile'] = 'cam13exp2_MCTA.txt'
        param['PImgsPath'] = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/cam11exp2.mp4'
        param['AImgsPath'] = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/cam13exp2.mp4'

    if cam2cam == 'C14C13':
        param = {}
        param['Pid'] = 13
        param['Aid'] = 14
        # C5: primary, C2: auxiliary
        param['tOffsetPstart'] = -8  # 13 delayed 14 by 5: 13>1 == 14>5
        param['tOffsetAstart'] = 0  #
        # similarity based homography: TODO: compute homography using undistorted image
        # H13to14
        '''
                param['H'] = [[1.1163, 0.0542, 0],
                      [-0.0542, 1.1163, 0],
                      [589.7222, -87.6723, 1.0000]]
    0.8937    0.0484         0
   -0.0484    0.8937         0
  617.0456  -13.7563    1.0000'''
        param['H'] = [[0.995, 0.165, 0],
                      [-0.165,  0.995, 0],
                      [705.78, -142.36, 1.0000]]

        param['fr_offsetPVis'] = 8
        param['fr_offsetAVis'] = 0  ## T_c2 - T_c11 = offset
        param['H'] = np.transpose(param['H'])
        #H14to13
        if param['Pid'] == 13:
           param['H'] = np.linalg.inv(param['H'])
        param['d_th'] = 0.35
        param['finalTrackFile'] = 'cam14exp2_MCTA.txt'
        param['PImgsPath'] = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/cam13exp2.mp4'
        param['AImgsPath'] = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/cam14exp2.mp4'

    return param

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
