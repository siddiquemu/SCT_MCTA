from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
import sys
import glob
import os
#import imutils
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
    dataset='pvd'
    cam_num =360
    eval_online = 0
    #remote
    #lags = {2:-224, 9:-175, 5:-200, 11:-176, 13:-176}
    #local
    lags = {330: 0, 340: 0, 300: 0, 360: 0, 361: 0, 440: 0, 8:0}
    offset = lags[cam_num]
    save_tracks = 0
    cam = 'C{}'.format(cam_num)
    if dataset=='pvd':
        server = '/media/siddique/RemoteServer/LabFiles/CLASP2/exp2/'
        storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61'
        #detPerson1FPS = None #np.loadtxt(storage+'/tracking_wo_bnw/data/PVD/HDPVD_new/train_gt/C{}/1_C{}.txt'.format(cam_num, cam_num), delimiter=',')
        gtPerson1FPS = np.loadtxt(storage+'/tracking_wo_bnw/data/PVD/HDPVD_new/train_gt/'+cam+'/gt_tso/gt.txt', delimiter=',')
        detPerson1FPS = None
        #detPerson1FPS = np.loadtxt(storage+'/tracking_wo_bnw/data/PVD/HDPVD_new/train_gt/C{}/1_C{}.txt'.format(cam_num, cam_num), delimiter=',')

        # path of the output image
        outpath = storage+'/tracking_wo_bnw/data/PVD/HDPVD_new/GTVsDet/'+cam+'/'
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        else:
            delete_all(outpath, fmt='jpg')
        # path of the image
        # local
        path = storage+'/tracking_wo_bnw/data/PVD/HDPVD_new/train_gt/'+cam+'/img1/*'

    if dataset=='pvd-mcta':
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
    # for raw det from baseline
    # detPerson1FPS = detPerson1FPS[np.where(detPerson1FPS[:,8]==0),:][0]
    #detFr = detPerson1FPS[:,0]
    classGT = 'PGT'
    classDet = 'P'
    tFactor = 1 # 30
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow('image',960,540)

    gt_frameset = gtPerson1FPS[:, 0]+offset
    for name in files:
        print (os.path.basename((name)))
        if dataset=='pvd':
            frNum = float(os.path.basename(name).split('.')[0])
        else:
            frNum = frNum

        if frNum - offset in gtPerson1FPS[:, 0]:  # search actual frame in gt

            if dataset == 'pvd':
                imgcv = cv2.imread(name.split(os.path.basename(name))[0] + '{:06d}.png'.format(int(frNum)))
            else:
                imgcv = cv2.imread(name)

            print('gt frame {}'.format(frNum - offset))
            # imgcv = imutils.rotate_bound(imgcv, 180)
            # imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)
            # find bbox for gt.person
            Frgt = gtPerson1FPS[gtPerson1FPS[:, 0] == frNum - offset]
            for bb in Frgt:
                print(classGT)
                xmin = bb[2]
                ymin = bb[3]
                w = bb[4]
                h = bb[5]
                xmax = xmin + w
                ymax = ymin + h
                color = (0, 0, 255)
                classGT = 'P' + str(int(bb[1]))
                imgcv = cv2.rectangle(imgcv, (np.int(xmin), np.int(ymin)), (np.int(xmax), np.int(ymax)), color, 4)
                cv2.putText(imgcv, classGT, (int(xmin + w / 2), int(ymin + h / 2)), cv2.FONT_HERSHEY_SIMPLEX, 2,
                            (0, 255, 0), 4)

        if frNum in gt_frameset and detPerson1FPS is not None:
            # find bbox for det.person
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
                imgcv = cv2.rectangle(imgcv, (np.int(xmin), np.int(ymin)), (np.int(xmax), np.int(ymax)), color, 5)
                cv2.putText(imgcv, display_txt, (int(xmin+w/2), int(ymin+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 255),5)
                if save_tracks:
                    # maintain actual gt frames
                    t_feature = [frNum-offset, bb[1], xmin, ymin, w, h, -1, -1, -1, -1]
                    det_file.writerow(t_feature)
                    cam_tracks.append(t_feature)

        if frNum - offset in gtPerson1FPS[:, 0]:
            cv2.imwrite(os.path.join(outpath, '{:06d}.jpg'.format(int(frNum))), imgcv)
            #cv2.imshow('image', imgcv)
            #cv2.waitKey(10)
        print('time offset {}'.format('{:.2f}'.format((frNum)/10.0)))
        frNum+=1
    # evaluate camera
    if eval_online:
        #C11: 4400-1300, C2: 2100-12900
        results = {}
        mot_evaluation = EvaluateMOT(gt_path=storage,
                                     result_path=outpath,
                                     vis_path=outpath,
                                     vis=False,
                                     proj_plane='2D',
                                     radious=100,
                                     iou_max=0.5,
                                     isMCTA=1,
                                     max_gt_frame=13200,
                                     min_gt_frame=1,
                                     auto_id=True)

        results['gt'] = gtPerson1FPS
        results['tracks'] = np.array(cam_tracks)
        print('unique tragets detected in {} is {}'.format(cam, len(np.unique(results['tracks'][:,1]))))
        # do mot accumulation
        # TODO: SCT evaluation
        # python -m motmetrics.apps.eval_motchallenge /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/wild-track/train_gt   /media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/multicamera_wildtrack/wildtrack/Wildtrack_dataset/results-track/C5C1
        # evaluate mot accumations
        mot_evaluation.evaluate_mot_accums(results, cam, generate_overall=True)