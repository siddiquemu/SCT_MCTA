import  cv2
import numpy as np
import collections
import os
import glob
from collections import deque
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

def convert_bottom_center(tracklet):
    # [fr,id,x,y,w,h, cam] = [fr,id, x+w/2.y+h/2,w,h, cam]
    if len(tracklet) > 0:
        for tl in tracklet:
            for bb in tl:
                bb[2] = bb[2] + bb[4] / 2.0
                bb[3] = bb[3] + bb[5] / 1.0
    return tracklet


def centroids2xywh(tracklet):
    if len(tracklet)>0:
        for tl in tracklet:
            for bb in tl:
                bb[2] = bb[2] - bb[4] / 2.0
                bb[3] = bb[3] - bb[5] / 2.0
    return tracklet

def botcenter2xywh(tracklet):
    if len(tracklet)>0:
        for tl in tracklet:
            for bb in tl:
                bb[2] = bb[2] - bb[4] / 2.0
                bb[3] = bb[3] - bb[5] / 1.0
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

def camera_intrinsics(cam, img_HW):
    # currently used camera: 2,4,5,9,11,13
    # A: 3*3 camera matrix
    # dist_coeff: distortion coefficient k1, k2
    ##https://dsp.stackexchange.com/questions/6055/how-does-resizing-an-image-affect-the-intrinsic-camera-matrix
    if img_HW[0]==720:
        factor_x = 1080/1920.
        factor_y = 720/1080.
        fx = 1217.6 * factor_x
        fy = 1217.8 * factor_y
        if cam == 9:
            cx = 972.3*factor_x
            cy = 550.9*factor_y
            A = np.array([[fx, 0.0, 540.0],
                          [0.0, fy, 360.0],
                          [0.0, 0.0, 1.0]])
            dist_coeff = np.array([-0.3235, 0.0887, 0, 0])  # why 4 or 5 coefficients are used in opencv????
        if cam == 11:  # assume that 9 and 11 have similar distortion
            cx = 972.3*factor_x
            cy = 550.9*factor_y
            A = np.array([[fx, 0.0, 540.0],
                          [0.0, fy, 360.0],
                          [0.0, 0.0, 1.0]])
            dist_coeff = np.array([-0.3235, 0.0887, 0, 0])  # why 4 or 5 coefficients are used in opencv????
        if cam == 13:  # assume that 9 and 11 have similar distortion
            cx = 972.3*factor_x
            cy = 550.9*factor_y
            A = np.array([[fx, 0.0, 540.0],
                          [0.0, fy, 360.0],
                          [0.0, 0.0, 1.0]])
            dist_coeff = np.array([-0.3235, 0.0887, 0, 0])  # why 4 or 5 coefficients are used in opencv????

        if cam == 14:  # assume that 9 and 11 have similar distortion
            A = np.array([[fx, 0.0, 540.0],
                          [0.0, fy, 360.0],
                          [0.0, 0.0, 1.0]])
            dist_coeff = np.array([-0.3235, 0.0887, 0, 0])  # why 4 or 5 coefficients are used in opencv????

        fx = 1216.5 * factor_x
        fy = 1214.5 * factor_y
        if cam == 2:
            cx = 989*factor_x
            cy = 595*factor_y
            A = np.array([[fx, 0.0, 540.0],
                          [0.0, fy, 360.0],
                          [0.0, 0.0, 1.0]])
            dist_coeff = np.array([-0.3655, 0.1296, 0, 0])  # why 4 or 5 coefficients are used in opencv????
        if cam == 5:  # assume that c2 and c5 have similar distortion
            cx = 989*factor_x
            cy = 595*factor_y
            A = np.array([[fx, 0.0, 540.0],
                          [0.0, fy, 360.0],
                          [0.0, 0.0, 1.0]])
            dist_coeff = np.array([-0.3655, 0.1296, 0, 0])  # why 4 or 5 coefficients are used in opencv????
        if cam == 4:  # assume that c2 and c5 have similar distortion
            A2 = np.array([[973.0, 0.0, 968.0],
                           [0, 966.0, 540.0],
                           [0.0, 0.0, 1.0]])
            A = np.array([[fx, 0.0, 540.0],
                          [0.0, fy, 360.0],
                          [0.0, 0.0, 1.0]])
            dist_coeff = np.array([-0.3655, 0.1296, 0, 0])  # why 4 or 5 coefficients are used in opencv????

    else:
        if cam == 9:
            A = np.array([[1217.6, 0.0, 972.3],
                          [0.0, 1217.8, 550.9],
                          [0.0, 0.0, 1.0]])
            dist_coeff = np.array([-0.3235, 0.0887, 0, 0])  # why 4 or 5 coefficients are used in opencv????
        if cam == 11:  # assume that 9 and 11 have similar distortion
            A = np.array([[1217.6, 0.0, 972.3],
                          [0.0, 1217.8, 550.9],
                          [0.0, 0.0, 1.0]])
            dist_coeff = np.array([-0.3235, 0.0887, 0, 0])  # why 4 or 5 coefficients are used in opencv????
        if cam == 13:  # assume that 9 and 11 have similar distortion
            A = np.array([[1217.6, 0.0, 972.3],
                          [0.0, 1217.8, 550.9],
                          [0.0, 0.0, 1.0]])
            dist_coeff = np.array([-0.3235, 0.0887, 0, 0])  # why 4 or 5 coefficients are used in opencv????

        if cam == 14:  # assume that 9 and 11 have similar distortion
            A = np.array([[1217.6, 0.0, 972.3],
                          [0.0, 1217.8, 550.9],
                          [0.0, 0.0, 1.0]])
            dist_coeff = np.array([-0.3235, 0.0887, 0, 0])  # why 4 or 5 coefficients are used in opencv????

        if cam == 2:
            A = np.array([[1216.5, 0.0, 989.0],
                          [0.0, 1214.5, 595.0],
                          [0.0, 0.0, 1.0]])
            dist_coeff = np.array([-0.3655, 0.1296, 0, 0])  # why 4 or 5 coefficients are used in opencv????
        if cam == 5:  # assume that c2 and c5 have similar distortion
            A = np.array([[1216.5, 0.0, 989.0],
                          [0.0, 1214.5, 595.0],
                          [0.0, 0.0, 1.0]])
            dist_coeff = np.array([-0.3655, 0.1296, 0, 0])  # why 4 or 5 coefficients are used in opencv????
        if cam == 4:  # assume that c2 and c5 have similar distortion
            A = np.array([[1216.5, 0.0, 989.0],
                          [0.0, 1214.5, 595.0],
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


def default_params_pvd_new(cam2cam=None):
    """
    deque:
    append() :- This function is used to insert the value in its argument to the right end of deque.
    appendleft() :- This function is used to insert the value in its argument to the left end of deque.
    pop() :- This function is used to delete an argument from the right end of deque.
    popleft() :- This function is used to delete an argument from the left end of deque.
    """
    param = {}
    if cam2cam=='C300C340':
        '''

    1.1783   -0.0710         0
    0.0710    1.1783         0
  418.3418 -118.0142    1.0000
        '''
        param['Pid'] = 340
        param['Aid'] = 300

        param['H'] = [[1.17, -0.07, 0.00],
                      [0.07, 1.17, 0.00],
                      [458.34, -118.01, 1.00]]
        param['H'] = np.transpose(param['H'])
        param['d_th'] = 0.2

    if cam2cam == 'C300C440':
        '''   1.0e+03 *

   -0.0010   -0.0005         0
    0.0005   -0.0010         0
    0.8354    1.1157    0.0010'''
        param['Pid'] = 440
        param['Aid'] = 300

        param['H'] = [[-1.0, -0.5, 0.00],
                      [0.5, -1.0, 0.00],
                      [750.4, 1115.7, 1.00]]
        param['H'] = np.transpose(param['H'])
        param['d_th'] = 0.15

    if cam2cam == 'C361C440':
        '''    1.2608   -0.8733         0
    0.8733    1.2608         0
  185.8588  472.0882    1.0000'''
        param['Pid'] = 440
        param['Aid'] = 361

        param['H'] = [[1.3, -0.75, 0.00],
                      [0.75, 1.3, 0.00],
                      [225.78, 460.03, 1.00]]
        param['H'] = np.transpose(param['H'])
        param['d_th'] = 0.2

    if cam2cam == 'C360C361':
        param['Pid'] = 361
        param['Aid'] = 360
        param['H'] = None
        param['d_th'] = 20

    if cam2cam == 'C330C340':
        param['Pid'] = 340
        param['Aid'] = 330
        param['H'] = None
        param['d_th'] = 60

    if cam2cam == 'C360C340':
        param['Pid'] = 340
        param['Aid'] = 360
        param['H'] = None
        param['d_th'] = 30

    if cam2cam == 'C330C360':
        '''
    0.0179   -0.5302         0
    0.5302    0.0179         0
  815.3029  729.2142    1.0000
  
[[-9.10135385e-01 -1.16852535e+00  1.78174701e+03]
 [ 4.14669709e-01  2.91543488e-01 -3.78034582e+02]
 [-6.74120343e-04  3.63795490e-04  1.00000000e+00]]
 
 H = [[0.627, 0.05, 321.9],
      [-0.05, 0.627, 50],
      [0.0, 0.0, 1.00]]
        '''
        param['Pid'] = 360
        param['Aid'] = 330

        param['H'] =  [[0.15179, -0.6302 , 0],
                        [0.6302, 0.15179, 0],
                        [650.3, 750.21, 1.0000]]
        param['H'] = np.transpose(param['H'])
        #Zparam['H'] = np.linalg.inv(param['H'])
        param['d_th'] = 0.15 #integ: 0.15

    if cam2cam == 'C361C360..incorrect H':
        '''
    0.1126    0.2944         0
   -0.2944    0.1126         0
  950.0960   -8.7541    1.0000
        '''
        param['Pid'] = 360
        param['Aid'] = 361
        param['H'] = [[0.11, 0.29, 0.00],
                      [-0.29, 0.11, 0.00],
                      [950.1, -8.75, 1.00]]
        param['H'] = np.transpose(param['H'])

        param['d_th'] = 0.15

    if cam2cam == 'C340C440':
        '''
       1.0e+03 *

   -0.0009   -0.0004         0
    0.0004   -0.0009         0
    1.2955    1.1883    0.0010
    
        '''
        param['Pid'] = 440
        param['Aid'] = 340

        param['H'] = [[-0.9, -0.4, 0.00],
                      [0.4, -0.9, 0.00],
                      [1235.5, 1188.3, 1.00]]
        param['H'] = np.transpose(param['H'])
        param['d_th'] = 0.15

    return param

def default_params_pvd(cam2cam=None):
    """
    deque:
    append() :- This function is used to insert the value in its argument to the right end of deque.
    appendleft() :- This function is used to insert the value in its argument to the left end of deque.
    pop() :- This function is used to delete an argument from the right end of deque.
    popleft() :- This function is used to delete an argument from the left end of deque.
    """
    if cam2cam=='C1C3':
        '''
        PVD:         param['H'] = [[1.3, -0.32, 0.00],
                      [0.32, 1.3, 0.00],
                      [398.68, -100.93, 1.00]]
HDPVD:    0.7486   -0.5658         0
    0.5658    0.7486         0
  532.2412   99.4252    1.0000

        '''
        param = {}
        param['Pid'] = 3
        param['Aid'] = 1

        param['H'] = [[0.75, -0.56, 0.00],
                      [0.56, 0.75, 0.00],
                      [580.24, 60.42, 1.00]]
        param['H'] = np.transpose(param['H'])
        param['d_th'] = 0.2


    if cam2cam == 'C1C10':
        '''   1.0e+03 *

       -0.0013   -0.0004         0
        0.0004   -0.0013         0
        0.8783    1.1952    0.0010'''
        param = {}
        param['Pid'] = 10
        param['Aid'] = 1

        param['H'] = [[-0.9, -0.4, 0.00],
                      [0.4, -0.9, 0.00],
                      [837.9, 1039.1, 1.00]]
        param['H'] = np.transpose(param['H'])
        param['d_th'] = 0.2


    if cam2cam == 'C4C10':
        param = {}
        param['Pid'] = 10
        param['Aid'] = 4

        param['H'] = [[1.3, -0.75, 0.00],
                      [0.75, 1.3, 0.00],
                      [225.78, 460.03, 1.00]]
        param['H'] = np.transpose(param['H'])
        param['d_th'] = 0.2

    if cam2cam=='C5C3':
        param = {}
        param['Pid'] = 3
        param['Aid'] = 5
        param['d_th'] = None
        param['H'] = None
        param['out_path'] = '/media/NasServer/LabFiles/CLASP2/2019_10_24/exp2/Results/Online_MCTA/'+cam2cam

    if cam2cam=='C2C3':
        '''
            0.6033   -0.0280         0
    0.0280    0.6033         0
  296.8080  429.1330    1.0000
        '''
        param = {}
        param['Pid'] = 3
        param['Aid'] = 2

        param['H'] = [[0.603, -0.028, 0.00],
                      [0.028, 0.603, 0.00],
                      [296.8, 380.13, 1.00]] #429.13
        param['H'] = np.transpose(param['H'])
        param['d_th'] = 0.37
        param['out_path'] = '/media/NasServer/LabFiles/CLASP2/2019_10_24/exp2/Results/Online_MCTA/'+cam2cam
    if cam2cam=='C5C4':
        '''
           1.0e+03 *

    0.0002    0.0015         0
   -0.0015    0.0002         0
    1.4554   -1.1048    0.0010
        '''
        param = {}
        param['Pid'] = 4
        param['Aid'] = 5

        param['H'] = [[0.2, 1.5, 0.00],
                      [-1.5, 0.2, 0.00],
                      [1455.4, -1104.8, 1.00]] #429.13
        param['H'] = np.transpose(param['H'])
        param['d_th'] = 0.37
        param['out_path'] = '/media/NasServer/LabFiles/CLASP2/2019_10_24/exp2/Results/Online_MCTA/'+cam2cam

    if cam2cam == 'C5C8':
        '''
   1.0e+03 *

   -0.0005    0.0017         0
   -0.0017   -0.0005         0
    1.6560   -0.9067    0.0010
        '''
        param = {}
        param['Pid'] = 8
        param['Aid'] = 5

        param['H'] = [[-0.5, 1.7, 0.00],
                      [-1.7, -0.5, 0.00],
                      [1656, -906.7, 1.00]]  # 429.13
        param['H'] = np.transpose(param['H'])
        param['d_th'] = 0.37
        param['out_path'] = '/media/NasServer/LabFiles/CLASP2/2019_10_24/exp2/Results/Online_MCTA/' + cam2cam

    if cam2cam == 'C4C3':
        '''
   1.0e+03 *

   -0.0001    0.0012         0
   -0.0012   -0.0001         0
    1.1447   -0.2915    0.0010
        '''
        param = {}
        param['Pid'] = 3
        param['Aid'] = 4

        param['H'] = [[-0.1, 1.2, 0.00],
                      [-1.2, -0.1, 0.00],
                      [1144.7, -291.5, 1.00]]  # 429.13
        param['H'] = np.transpose(param['H'])
        param['d_th'] = 0.37

    if cam2cam == 'C5C3':
        '''
    1.2774    2.3037         0
   -2.3037    1.2774         0
   94.9309 -602.3705    1.0000
        '''
        param = {}
        param['Pid'] = 3
        param['Aid'] = 5
        param['BS_ROI'] = {}
        param['AIT_ROI'] = {}
        param['BS_ROI'][param['Pid']] = []
        param['BS_ROI'][param['Aid']] = []
        param['AIT_ROI'][param['Pid']] = []
        param['AIT_ROI'][param['Aid']] = []
        param['H'] = [[1.27, 2.0, 0.00],
                      [-2.3, 1.27, 0.00],
                      [94.93, -602.37, 1.00]]  # 429.13
        param['H'] = np.transpose(param['H'])
        param['d_th'] = 0.37

    return param


def dafault_params(cam2cam=None, img_HW=None, dataset='clasp1'):
    #setting pair-wise camera parameters
    if img_HW[0]==720:
        factor_x = (1080/1920.0)
        factor_y = (720/1080.0)
    else:
        factor_x = 1
        factor_y = 1
    if cam2cam == 'C9C2':
        # C2: primary, C9: auxiliary
        param = {}
        param['Pid'] = 2
        param['Aid'] = 9
        # similarity based homography:
        # H9to2
        if dataset=='clasp2':
            param['H'] = [[0.9812, 0.1426, 0.00],
                          [-0.1426, 0.9812, 0.00],
                          [factor_x*463.84, -782.41*factor_y, 1.00]]
        if dataset=='clasp1':
            param['H'] = [[0.99, -0.16, 0.00],
                          [0.16, 0.99, 0.00],
                          [factor_x*199.65, -635.92*factor_y, 1.00]]

        param['H'] = np.transpose(param['H'])
        param['d_th'] = 0.35#  #multi-task cost: 0.4, only loc: 0.25


    if cam2cam == 'C4C2':
        param = {}
        param['Pid'] = 2
        param['Aid'] = 4
        # similarity based homography:
        # H2to4
        '''
            1.0590   -0.3486         0 

    0.3486    1.0590         0 

  436.0147   77.4740    1.0000 
      1.0331   -0.3674         0
    0.3674    1.0331         0
  434.5345   99.9885    1.0000
        '''
        param['H'] = [[1.06, -0.35, 0.0],
                      [0.35, 1.06, 0.0],
                      [factor_x*436.01, factor_y*77.98, 1.0]]
        param['H'] = np.transpose(param['H'])
        param['H'] = np.linalg.inv(param['H'])
        param['d_th'] = 0.3  # multi-task cost: 0.45, only loc: 0.17

    if cam2cam == 'C5C2':
        param = {}
        param['Pid'] = 2
        param['Aid'] = 5
        # similarity based homography
        # H2to5
        param['H'] = [[1.2, -0.2, 0.0],
                      [0.2, 1.2, 0.0],
                      [factor_x*1422.4, factor_y*46.8, 1.0]]
        param['H'] = np.transpose(param['H'])
        param['H'] = np.linalg.inv(param['H'])
        param['d_th'] = 0.22  # multi-task cost: 0.45, only loc: 0.17

    if cam2cam == 'C5C4':
        param = {}
        param['Pid'] = 4
        param['Aid'] = 5
        # similarity based homography: TODO: compute homography using undistorted image
        # H5to4 TODO: compute homography using undistorted image
        '''
            0.9171   -0.1523         0
    0.1523    0.9171         0
 -764.7323  263.8579    1.0000
 
    0.9871   -0.1841         0
    0.1841    0.9871         0
 -843.1104  254.2725    1.0000
        '''
        param['H'] = [[0.98, -0.18, 0.0],
                      [0.18, 0.98, 0.0],
                      [-843.11*factor_x, factor_y*254.27, 1.0]]
        param['H'] = np.transpose(param['H'])
        #param['H'] = np.linalg.inv(param['H'])
        param['d_th'] = 0.3  # multi-task cost: 0.45, only loc: 0.17

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

        if dataset=='clasp2':
            param['H'] = [[0.935, -0.011, 0.00],
                      [0.011, 0.935, 0.00],
                      [factor_x*35.86, factor_y*611.47, 1.00]]
            param['H'] = np.transpose(param['H'])
        if dataset=='clasp1':
            param['H'] = [[0.96, 0.165, 0.00],
                          [-0.165, 0.96, 0.00],
                          [-factor_x * 89.2, factor_y * 603.47, 1.00]]
            param['H'] = np.transpose(param['H'])
        # H11to5
        if param['Pid'] == 5:
            param['H'] = np.linalg.inv(param['H'])
        param['d_th'] = 0.35  # multi-task cost: 0.45, only loc: 0.17

    if cam2cam == 'C13C5':
        param = {}
        param['Pid'] = 5
        param['Aid'] = 13
        # similarity based homography: TODO: compute homography using undistorted image
        # H5to13
        '''
    0.9214   -0.1088         0
    0.1088    0.9214         0
  768.4175  536.0207    1.0000

        '''
        param['H'] = [[0.9, -0.105, 0.00],
                      [0.105, 0.9, 0.00],
                      [factor_x*950.22, factor_y*650.36, 1.00]]

        param['H'] = np.transpose(param['H'])
        # H13to5
        if param['Pid'] == 5:
            param['H'] = np.linalg.inv(param['H'])
        param['d_th'] = 0.37  # multi-task cost: 0.45, only loc: 0.17

    if cam2cam == 'C13C11':
        param = {}
        param['Pid'] = 11
        param['Aid'] = 13
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
                      [-1200.1*factor_x, 0.5*factor_y, 1.0]]

        param['H'] = np.transpose(param['H'])
        # param['H'] = np.linalg.inv(param['H'])
        param['d_th'] = 0.37 # multi-task cost: 0.45, only loc: 0.17

    if cam2cam == 'C14C13':
        param = {}
        param['Pid'] = 13
        param['Aid'] = 14
        # similarity based homography: TODO: compute homography using undistorted image
        # H13to14
        '''
                param['H'] = [[1.1163, 0.0542, 0],
                      [-0.0542, 1.1163, 0],
                      [589.7222, -87.6723, 1.0000]]
    0.8937    0.0484         0
   -0.0484    0.8937         0
  617.0456  -13.7563    1.0000'''
        param['H'] = [[1.1163, 0.0542, 0],
                      [-0.0542, 1.1163, 0],
                      [factor_x*589.7222, -87.6723*factor_y, 1.0000]]

        param['H'] = np.transpose(param['H'])
        #H14to13
        if param['Pid'] == 13:
           param['H'] = np.linalg.inv(param['H'])
        param['d_th'] = 0.37

    return param

def merge_edges(mt, G_mc, batch_ids=None,cam_batch_ids=None, ca=None, cp=None):
    # G_mc: empty graph which is updated iteratively for each camera pair association
    # k: global track id from primary camera
    # l: global track id from auxiliary camera
    # mt: association between
    for (c, tStampComnMin, k, l) in mt:
        #initialize empty association list when tracklet first appear in camera system
        # both ids are new
        if k not in G_mc and l not in G_mc:
            G_mc[k] = []
            G_mc[l] = []

            G_mc[k].append(l)
            G_mc[l].append(k)
            continue
        #l is new
        if k in G_mc and l not in G_mc:#check the previous association of k present in current batch ids
            #check k has the previously associated l in current batch ids
            #TODO: check batch_ids only in current batch: current_batch_ids
            # TODO: before updating association: verify duplicity
            if len(set(G_mc[k]).intersection(cam_batch_ids[ca]))==0:# and k not in cam_batch_ids[ca]:
                G_mc[l] = []
                #k should not already in [ca current batch ids]
                G_mc[l].append(k)
                G_mc[k].append(l)
                continue
        #k: primary is new
        if l in G_mc and k not in G_mc:#check the previous association of l present in current batch ids
            #check l has the previously associated k in current batch ids
            # TODO: before updating association: verify duplicity
            if len(set(G_mc[l]).intersection(cam_batch_ids[cp]))==0:# and l not in cam_batch_ids[cp]:
                G_mc[k] = []
                # l should not already in [cp current batch ids]
                G_mc[k].append(l)
                G_mc[l].append(k)
                continue
        # both ids are already in Gmc:

        #if k in G_mc and l in G_mc:
           #if len(set(G_mc[k]).intersection(batch_ids))==0 and len(set(G_mc[l]).intersection(batch_ids))==0:
               #if l not in cam_batch_ids[cp]:
                   # G_mc[k].append(l)
               #if k not in cam_batch_ids[ca]:
                    #G_mc[l].append(k)

    return G_mc

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