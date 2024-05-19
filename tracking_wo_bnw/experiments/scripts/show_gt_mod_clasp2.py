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
#from evaluate_3D_tracks import *
def delete_all(demo_path, fmt='png'):
    import glob
    filelist = glob.glob(os.path.join(demo_path, '*.' + fmt))
    if len(filelist) > 0:
        for f in filelist:
            os.remove(f)

if __name__ == '__main__':
    # Read GT
    dataset='clasp2-det'
    cam_num =4
    show_img = 0
    #remote
    #lags = {2:-224, 9:-175, 5:-200, 11:-176, 13:-176}
    #local
    lags = {2: 5, 4:0, 9: 10, 5: 10, 11: 15, 13: 10, 14: 18}
    offset = lags[cam_num]
    save_tracks = 1
    cam = 'cam{:02d}exp2.mp4'.format(cam_num)
    if dataset=='clasp2-det':
        server = '/media/siddique/RemoteServer/LabFiles/CLASP2/exp2/'
        storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw'
        try:
            detPerson1FPS = np.loadtxt(storage+'/output/tracktor-clasp2-det/exp2train/C{}_1nms0aug.txt'.format(cam_num), delimiter=',')
        except:
            print('detection file is not available')
            detPerson1FPS = None

        gtPerson1FPS = np.loadtxt(storage+'/data/CLASP/train_gt/'+cam+'/gt/gt.txt', delimiter=',')
        # path of the output image
        outpath = storage+'/data/CLASP/GTVsDet/'+cam+'/'
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        else:
            delete_all(outpath, fmt='jpg')
        # path of the image
        # local
        path = storage+'/data/CLASP/train_gt/'+cam+'/img1/*'

    if dataset=='clasp2-sct':
        t_lag = cam_lag[exp][cam_num]
        eval_online = 0
        save_tracks = 0
        data_folder = 'exp{}'.format(exp.lower())
        server = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/ourdataset'
        storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw'

        gtPerson1FPS = np.loadtxt(storage+'/data/CLASP1/train_gt/{}/person_label_{}.txt'.format(cam_folder, cam_folder),
                                  delimiter=',')
        gtBag1FPS = np.loadtxt(storage+'/data/CLASP1/train_gt/{}/bag_label_{}.txt'.format(cam_folder, cam_folder),
                                  delimiter=',')

        if show_det:
            category = {1: 'person', 2:'bag'}
            detPerson1FPS = np.loadtxt(storage + '/output/clasp2/SL_tracks/{}/1_C{}.txt'.format(
                data_folder, cam_num), delimiter=',')
            detBag1FPS = np.loadtxt(storage + '/output/clasp2/SL_tracks/{}/2_C{}.txt'.format(
                data_folder, cam_num), delimiter=',')

        #path of the output image
        outpath = storage+'/data/CLASP/GTvsTrackSCT/{}/cam{:02d}'.format(exp, cam_num)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        else:
            delete_all(outpath, fmt='png')
        #path of the image
        path = storage+'/data/CLASP1/train_gt/{}_{}/img1/*.png'.format(data_name[exp], cam_num)

    if dataset=='clasp2-mcta':
        eval_online = 0
        cam_tracks = []
        server = '/media/siddique/RemoteServer/LabFiles/CLASP2/exp2/'
        storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw'

        # read global tracks
        bbs = pd.read_csv(storage+'/output/tracktor/online_SCT/global_tracks_30FPS_R50.csv', index_col=False)
        all_tracks = np.transpose(np.vstack(
        (bbs['frame'].array,
         bbs['id'].array,
         bbs['x1'].array,
         bbs['y1'].array,
         bbs['w'].array,
         bbs['h'].array,
         bbs['cam'].array)))
        # get monocular tracks from gloabl tracks
        detPerson1FPS = all_tracks[all_tracks[:,-1]==cam_num]

        # READ GT
        gtPerson1FPS = np.loadtxt(storage+'/data/CLASP/train_gt/'+cam+'/gt/gt.txt', delimiter=',')
        # path of the output image
        outpath = storage+'/data/CLASP/GTvsTrack/'+cam+'/'
        if save_tracks:
            result_path = storage+'/data/CLASP/eval_tracks_30fps/'
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            det_file  = open(os.path.join(result_path , '{}.txt'.format(cam)), "w")
            det_file = csv.writer(det_file, delimiter=',')

        if not os.path.exists(outpath):
            os.makedirs(outpath)
        else:
            delete_all(outpath, fmt='jpg')
        # path of the image
        # remote
        #path = storage+'/data/CLASP/imgs_30fps/cam{}_*.jpg'.format(cam_num)
        #local
        path = storage+'/data/CLASP/train_gt/cam{:02d}exp2.mp4/img1/*.jpg'.format(cam_num)

    files = glob.glob(path)
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    det = []
    frNum = 0

    classGT = 'PGT'
    classDet = 'P'
    tFactor = 1 # 30
    if show_img:
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image',960,540)

    gt_frameset = gtPerson1FPS[:, 0]+offset
    print(gt_frameset)
    for name in files:
        print (os.path.basename((name)))
        if dataset=='clasp2-det':
            frNum = float(os.path.basename(name).split('.')[0])
        else:
            frNum = frNum

        if frNum in gt_frameset:
            if dataset=='clasp2-det':
                try:
                    imgcv = cv2.imread(name.split(os.path.basename(name))[0]+'{:05d}.jpg'.format(int(frNum)))
                except:
                    imgcv = cv2.imread(name)
            else:
                imgcv = cv2.imread(name)
            # find bbox for det.person
            if detPerson1FPS:
                detFr = detPerson1FPS[detPerson1FPS[:, 0] == frNum ]
                print('det frmae {}'.format(frNum))
                for bb in detFr:
                    print(classDet)
                    xmin = bb[2]
                    ymin = bb[3]
                    #xmax = bb[4]
                    #ymax = bb[5]
                    #w = xmax-xmin
                    #h = ymax-ymin
                    w = bb[4]
                    h = bb[5]
                    xmax = xmin + w
                    ymax = ymin + h
                    color = (255, 0, 0)  # red
                    # score = detPerson1FPS[indFr[0][i]][6]
                    display_txt = classDet + str(int(bb[1]))  # +'%.2f' % (score)
                    # if (np.int(detPerson1FPS[indFr[0][i]][7]) == 1):
                    #imgcv = cv2.rectangle(imgcv, (np.int(xmin), np.int(ymin)), (np.int(xmax), np.int(ymax)), color, 5)
                    #cv2.putText(imgcv, display_txt, (int(xmin+w/2), int(ymin+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255),5)
                    if save_tracks:
                        # maintain actual gt frames
                        t_feature = [frNum-offset, bb[1], xmin, ymin, w, h, -1, -1, -1, -1]
                        det_file.writerow(t_feature)
                        cam_tracks.append(t_feature)

            if frNum-offset in gtPerson1FPS[:, 0]: #search actual frame in gt
                print('gt frame {}'.format(frNum-offset))
                # imgcv = imutils.rotate_bound(imgcv, 180)
                # imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)
                # find bbox for gt.person
                Frgt = gtPerson1FPS[gtPerson1FPS[:, 0] == frNum-offset]
                for bb in Frgt:
                    print(classGT)
                    xmin = bb[2]
                    ymin = bb[3]
                    w = bb[4]
                    h = bb[5]
                    xmax = xmin + w
                    ymax = ymin + h
                    color = (0, 0, 255)
                    classGT = 'P'+str(int(bb[1]))
                    imgcv = cv2.rectangle(imgcv, (np.int(xmin), np.int(ymin)), (np.int(xmax), np.int(ymax)), color, 4)
                    cv2.putText(imgcv, classGT, (int(xmin+w/2), int(ymin+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

            cv2.imwrite(os.path.join(outpath, os.path.basename(name)), imgcv)
            if show_img:
                cv2.imshow('image', imgcv)
                cv2.waitKey(10)
        print('time offset {}'.format('{:.2f}'.format((frNum)/30.0)))
        frNum+=1