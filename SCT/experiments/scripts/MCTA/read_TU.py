#!/usr/bin/python
from __future__ import division
import csv
import copy
import numpy as np



def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def readTU(fname):

    csv_file = open(fname, mode='r')
    gt_csv = csv.DictReader(csv_file)

    gt_TUs = []
    entry = {}

    for row in gt_csv:
        if row['TU'] != '0':
            entry['Frame'] = int(row['Frame'])
            entry['TU'] = row['TU']
            entry['BB'] = list(map(int, row['BB'].split(',')))
            gt_TUs.append(copy.deepcopy(entry))

    return gt_TUs


def matchTU(gt_bbs, currFrame, t_bbs):
    matches = []

    for row in gt_bbs:
        if row['Frame'] == currFrame:
            maxIoU = 0
            maxIoUidx = -1
            TU = []

            idx = 0
            gt_bb = row['BB']
            for det_bb in t_bbs:
                tmpIoU = bb_intersection_over_union(gt_bb,det_bb)
                if tmpIoU > maxIoU:
                    maxIoU = tmpIoU
                    maxIoUidx = idx
                    TU = row['TU']
                idx = idx + 1
            if maxIoU > 0:
                matches.append((maxIoU, maxIoUidx, TU))

    return matches

'''
# Detection bounding boxes at current frame -- Should be updated at every frame
#target_bbs = [[675, 167, 990, 751], [1085, 489, 1471, 1073], [816, 617, 1220, 923], [1333, 576, 1604, 1044]]
trackers_cam01exp2 = np.loadtxt('/media/siddique/RemoteServer/CLASP2/2019_04_16/exp1/Results/cam01exp1.mp4/tracking_resultscam01exp1.mp4.txt')
#target_bbs = trackers_cam01exp2[:,2:6]
#target_bbs = [target_bbs[]]
TU_bbs = readTU('/media/siddique/RemoteServer/CLASP2/2019_04_16/exp1/Results/cam01exp1.mp4/04162019_Exp1Cam1_People_metadata.csv')

for fnum in range(0,7900,100):
    if fnum in trackers_cam01exp2[:,0]:
        target_bbs =trackers_cam01exp2[trackers_cam01exp2[:, 0]==fnum][:,2:6]
        target_bbs = np.array(([target_bbs[:,0],target_bbs[:,1],target_bbs[:,0]+target_bbs[:,2],target_bbs[:,1]+target_bbs[:,3]]),dtype='float')
        target_bbs = np.transpose(target_bbs)
        match = matchTU(TU_bbs, fnum,target_bbs )
        if match:
            print('Frame {}, Match {}'.format(fnum, match))

'''