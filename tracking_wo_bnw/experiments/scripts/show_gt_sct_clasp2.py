from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
import sys
import glob
import os
import imutils
import pandas as pd
import csv
import random


# from evaluate_3D_tracks import *

def get_split(GT, train_split=0.8, cam=None):
    """

    :param GT:
    :param train_splt:
    :param cam:
    :return:
    """
    GTp = [gt for gt in GT if gt[9] == 1]
    GTb = [gt for gt in GT if gt[9] == 2]
    gt_frameset = np.unique(GT[:, 0].astype('int'))
    gt_len = len(gt_frameset)
    print('full set {}: Nfr {}, person {} bag {}'.format(cam, gt_len, len(GTp), len(GTb)))

    # random split: keep split similar for training and testing forever
    random.seed(42)
    train_subset = random.sample(list(gt_frameset), int(gt_len * train_split))
    print('random sample {}'.format(train_subset[2]))
    # print(subset)
    train_GTp = [gt for gt in GT if gt[0] in train_subset and gt[9] == 1]
    train_GTb = [gt for gt in GT if gt[0] in train_subset and gt[9] == 2]
    print('train split {}: Nfr {}, person {} bag {}'.format(cam, len(train_subset), len(train_GTp), len(train_GTb)))

    test_subset = np.array([t for t in gt_frameset if t not in train_subset])
    test_GTp = [gt for gt in GT if gt[0] not in train_subset and gt[9] == 1]
    test_GTb = [gt for gt in GT if gt[0] not in train_subset and gt[9] == 2]
    print(
        'test split {}: Nfr {}, person {} bag {}'.format(cam, len(test_subset), len(test_GTp), len(test_GTb)))
    print('-------------------------------------------------')
    return train_subset, test_subset


def delete_all(demo_path, fmt='png'):
    import glob
    filelist = glob.glob(os.path.join(demo_path, '*.' + fmt))
    if len(filelist) > 0:
        for f in filelist:
            os.remove(f)


def find_frame_jump(files):
    frs = np.array([float(os.path.basename(name).split('.')[0]) for name in files])
    for i, fr in enumerate(frs):
        if i == 0:
            fr_prev = fr
        diff = fr - fr_prev
        if diff != 10:
            print(fr)
        fr_prev = fr

    # return jump_at


if __name__ == '__main__':
    # Read GT
    dataset = 'clasp2-gt'
    gt_type = 'SCT'  # 'MC'
    show_det = 1
    vis = 1
    cam_num = 9
    exp_name = 'I'

    # local
    lags = {'G': {9: 10, 11: 15}, 'H': {9: 0, 11: 0}, 'I': {9: 0, 11: 0}}
    pb_lags = {'G': {9: 3, 11: 4}, 'H': {9: 0, 11: 0}, 'I': {9: 0, 11: 0}}
    bag_lag = {'G': {9: 0, 11: 0}}
    # lags = {2: 5, 9: 0, 5: 0, 11: 0, 13: 10, 14: 18}
    offset = lags[exp_name][cam_num]
    cam = '{}_{}'.format(exp_name, cam_num)
    category = {1: 'person', 2: 'bag'}

    server = '/media/siddique/RemoteServer/LabFiles/CLASP2/exp2/'
    storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw'
    data_dir = storage + '/data/CLASP/train_gt_all/PB_gt/'
    result_dir = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/output/clasp2/'

    if gt_type == 'MC':
        #trcaks
        if show_det:
            detPerson1FPS = np.loadtxt(result_dir + 'SSL_alpha_tracks/{}/2_C{}.txt'.format(
                exp_name, cam_num), delimiter=',')
            detBag1FPS = np.loadtxt(result_dir + 'SSL_alpha_tracks/{}/2_C{}.txt'.format(
                exp_name, cam_num), delimiter=',')
        #gts
        GT = np.loadtxt(os.path.join(data_dir, '{}_{}/gt/gt.txt'.format(exp_name, cam_num)), delimiter=',')
        gtPerson1FPS = GT[GT[:,-1]==1]
        gtBag1FPS = GT[GT[:,-1]==2]
        # path of the output image
        outpath = storage + '/data/CLASP/train_gt_all/GTVsTrack/' + cam + '/'

    if gt_type in ['SCT']:
        #trcaks
        if show_det:
            detPerson1FPS = np.loadtxt(result_dir + 'SSL_alpha_tracks/{}/1_C{}.txt'.format(
                exp_name, cam_num), delimiter=',')
            detBag1FPS = np.loadtxt(result_dir + 'SSL_alpha_tracks/{}/2_C{}.txt'.format(
                exp_name, cam_num), delimiter=',')
        #gts
        GT = np.loadtxt(os.path.join(data_dir, '{}_{}/gt_sct/gt.txt'.format(exp_name, cam_num)), delimiter=',')
        #-1: cam, -2: clas id
        GT = GT[GT[:, -1] == cam_num]
        gtPerson1FPS = GT[GT[:,-2]==1]
        gtBag1FPS = GT[GT[:,-2]==2]
        # path of the output image
        outpath = storage + '/data/CLASP/train_gt_all/GTVsDet/' + cam + '/'

    if not os.path.exists(outpath):
        os.makedirs(outpath)
    else:
        delete_all(outpath, fmt='jpg')

    # path of the image
    path = storage + '/data/CLASP/train_gt_all/PB_gt/' + cam + '/img1/*'

    files = glob.glob(path)
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    det = []
    frNum = 0
    classGT = 'PGT'
    classDet = 'P'
    tFactor = 1  # 30
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('image', 960, 540)

    gt_framesetP = gtPerson1FPS[:, 0]  # +offset
    gtPerson1FPS[:, 0] = gt_framesetP
    gt_framesetB = gtBag1FPS[:, 0]
    # if gt_type=='SCT':
    # test_set = list(gt_framesetP) + list(gt_framesetB)
    for name in files:
        frNum = float(os.path.basename(name).split('.')[0])

        if frNum in gt_framesetP or frNum in gt_framesetB:
            imgcv = cv2.imread(name)
            print('gt frame {}'.format(frNum - offset))
            # find bbox for person
            if frNum in gtPerson1FPS[:, 0]:  # search actual frame in gt
                # imgcv = imutils.rotate_bound(imgcv, 180)
                # imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)
                # find bbox for gt.person
                Frgt = gtPerson1FPS[gtPerson1FPS[:, 0] == frNum]
                for bb in Frgt:
                    xmin, ymin, w, h = bb[2:6]
                    xmax = xmin + w
                    ymax = ymin + h
                    color = (0, 0, 255)
                    classGT = 'P' + str(int(bb[1]))
                    class_id = 1
                    #if (xmin + w / 2) > 50 and (ymin + h / 2) < 1030:
                    if vis:
                        imgcv = cv2.rectangle(imgcv, (np.int(xmin), np.int(ymin)), (np.int(xmax), np.int(ymax)),
                                              color, 4)
                        cv2.putText(imgcv, classGT, (int(xmin + w / 2), int(ymin + h / 2)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

            # find bbox for bag
            if frNum in gtBag1FPS[:, 0]:
                # imgcv = imutils.rotate_bound(imgcv, 180)
                # imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)
                # find bbox for gt.person
                Frgt = gtBag1FPS[gtBag1FPS[:, 0] == frNum]
                for bb in Frgt:
                    class_id = 2
                    xmin, ymin, w, h = bb[2:6]
                    xmax = xmin + w
                    ymax = ymin + h
                    color = (0, 0, 255)
                    classGT = 'B' + str(int(bb[1]))
                    class_id = 2
                    # if (xmin+w/2)<1500 or (ymin+h/2)<700:
                    if vis:
                        imgcv = cv2.rectangle(imgcv, (np.int(xmin), np.int(ymin)), (np.int(xmax), np.int(ymax)),
                                              (255, 0, 255), 4)
                        cv2.putText(imgcv, classGT, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

            #overaly person track if available
            if show_det and frNum in detPerson1FPS[:, 0]:
                detFr = detPerson1FPS[detPerson1FPS[:, 0] == frNum]
                for bb in detFr:
                    xmin, ymin, w, h = bb[2:6]
                    xmax = xmin + w
                    ymax = ymin + h
                    color = (255, 0, 0)  # red
                    # if xmin+w/2 > 100 and 1000>ymin+h/2>300: #C11
                    # if (xmin + w / 2 > 200 or ymin + h / 2>300)  and 1000>ymin + h / 2 : #C9
                    # score = detPerson1FPS[indFr[0][i]][6]
                    if dataset == 'clasp2-gt':
                        # display_txt = category[bb[7]]  +'%.2f' % (bb[6])
                        display_txt = 'P{}'.format(int(bb[1]))
                    else:
                        display_txt = 'P{}'.format(int(bb[1]))
                    # (xx(i,3)+xx(i,5)/2)>50 && (xx(i,4)+xx(i,6)/2)<1050
                    #if bb[6] >= 0.0 and (xmin + w / 2) > 50 and (ymin + h / 2) < 1030:  # C11
                    if vis:
                        imgcv = cv2.rectangle(imgcv, (np.int(xmin), np.int(ymin)), (np.int(xmax), np.int(ymax)),
                                              color, 5)
                        cv2.putText(imgcv, display_txt, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                    (255, 255, 0), 4)
            #overlay bag track if available
            if show_det and frNum in detBag1FPS[:, 0]:
                detFr = detBag1FPS[detBag1FPS[:, 0] == frNum]
                for bb in detFr:
                    xmin, ymin, w, h = bb[2:6]
                    xmax = xmin + w
                    ymax = ymin + h
                    color = (255, 0, 0)  # red
                    # if xmin+w/2 > 100 and 1000>ymin+h/2>300: #C11
                    # if (xmin + w / 2 > 200 or ymin + h / 2>300)  and 1000>ymin + h / 2 : #C9
                    # score = detPerson1FPS[indFr[0][i]][6]
                    if dataset == 'clasp2-gt':
                        # display_txt = category[bb[7]]  +'%.2f' % (bb[6])
                        display_txt = 'B{}'.format(int(bb[1]))
                    else:
                        display_txt = 'B{}'.format(int(bb[1]))
                    # (xx(i,3)+xx(i,5)/2)>50 && (xx(i,4)+xx(i,6)/2)<1050
                    #if bb[6] >= 0.0 and (xmin + w / 2) > 50 and (ymin + h / 2) < 1030:  # C11
                    if vis:
                        imgcv = cv2.rectangle(imgcv, (np.int(xmin), np.int(ymin)), (np.int(xmax), np.int(ymax)),
                                              color, 5)
                        cv2.putText(imgcv, display_txt, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 2,
                                    (255, 255, 0),
                                    4)
            cv2.imwrite(os.path.join(outpath, '{:06d}.jpg'.format(int(frNum))), imgcv)
            cv2.imshow('image', imgcv)
            cv2.waitKey(10)
        # frNum+=1
