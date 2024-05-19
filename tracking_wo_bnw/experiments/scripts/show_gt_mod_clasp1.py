from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
import sys
import glob
import os
import csv
import imutils
import pandas as pd

def delete_all(demo_path, fmt='png'):
    import glob
    filelist = glob.glob(os.path.join(demo_path, '*.' + fmt))
    if len(filelist) > 0:
        for f in filelist:
            os.remove(f)
#combined PB GT
#exp9A: 1FPS, A | exp6A:2FPS, B | exp7A: 2FPS, C
dataset='clasp1-sct' #'clasp1-det'
vis=1
PAcam = {9:2,11:5}
cam_list = [9,9,9,9,9,11,11,11,11,11,]
exp_list = ['9A', '6A', '7A', '10A', '5A', '9A', '6A', '7A', '10A', '5A']

cam_lag = {'9A':{9:14, 11:10}, '6A':{9:30, 11:30}, '7A':{9:30, 11:30},'10A':{9:30, 11:30},'5A':{9:4, 11:4}}
frame_rate = {'9A': 1, '6A': 2, '7A': 2, '10A':2, '5A': 10}
data_name = {'9A': 'A', '6A': 'B', '7A': 'C', '10A':'D', '5A': 'E'}
fr_rate_factor = {'9A': 30, '6A': 15, '7A': 15, '10A': 15, '5A': 3}


save_gt_p = 0
save_gt_b = 0

offset = 0
show_det = 1
out_dir = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/CLASP1/train_gt'#reid_person
#run for full clasp1 dataset
for cam_num, exp in zip(cam_list, exp_list):
    cam_pri = PAcam[cam_num]
    cam_folder = '{}_{}'.format(data_name[exp], cam_num)
    if save_gt_p or save_gt_b:
        PB_file  = open(os.path.join(out_dir, '{}/gt/gt.txt'.format(cam_folder)), "w")
        PB_file = csv.writer(PB_file, delimiter=',')

    if dataset=='clasp1-det':
        eval_online = 0
        save_tracks = 0
        server = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/ourdataset'
        storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw'
        #detPerson1FPS = np.loadtxt(server+'/Results/1FPS/ResNet50FPNPANet_tuned/C11_pb_0aug0nms.txt', delimiter=',')

        #gtPerson1FPS = np.loadtxt(server+'/exp'+exp+'/GT_{}/det/det{}Cam{:02d}PersonMASK_{}FPS/gt/gt.txt'.format(exp, exp, cam_num,
        #                                                                                          frame_rate[exp]), delimiter=',')
        gtPerson1FPS = np.loadtxt(storage+'/data/CLASP1/train_gt/{}/person_label_{}.txt'.format(cam_folder, cam_folder),
                                  delimiter=',')

        #gtBag1FPS = np.loadtxt(server+'/exp'+exp+'/GT_{}/det/det{}Cam{:02d}BagMASK_{}FPS/gt/gt.txt'.format(exp, exp, cam_num,
        #                                                                                    frame_rate[exp]), delimiter=',')
        gtBag1FPS = np.loadtxt(storage+'/data/CLASP1/train_gt/{}/bag_label_{}.txt'.format(cam_folder, cam_folder),
                                  delimiter=',')

        if show_det:
            category = {1: 'person', 2:'bag'}
            #detPerson1FPS = np.loadtxt(storage + '/output/tracktor-clasp1-det/{}_{}/C{}_1nms0aug.txt'.format(
               # data_name[exp], cam_num, cam_num), delimiter=',' )

            #detPerson1FPS = np.loadtxt(server + '/clasp1_detections/panet_ssl/iter8/{}_{}_pb_0aug1nms.txt'.format(
                #data_name[exp], cam_num), delimiter=',')
            detPerson1FPS = np.loadtxt(server + '/clasp1_detections/panet_ssl_score/iter8/{}_{}_pb_0aug1nms.txt'.format(
                data_name[exp], cam_num), delimiter=',')
        #path of the output image
        outpath = storage+'/data/CLASP1/GTvsDet/{}/cam{:02d}'.format(exp, cam_num)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        else:
            delete_all(outpath, fmt='png')
        #path of the image
        path = storage+'/data/CLASP1/train_gt/{}_{}/img1/*.png'.format(data_name[exp], cam_num)

    if dataset == 'clasp1-sct':
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
            detPerson1FPS = np.loadtxt(storage + '/output/clasp1/SSL_alpha_tracks/{}/1_C{}.txt'.format(
                data_folder, cam_num), delimiter=',')
            detBag1FPS = np.loadtxt(storage + '/output/clasp1/SSL_alpha_tracks/{}/2_C{}.txt'.format(
                data_folder, cam_num), delimiter=',')

        #path of the output image
        outpath = storage+'/data/CLASP1/GTvsTrackSCT/{}/cam{:02d}'.format(exp, cam_num)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        else:
            delete_all(outpath, fmt='png')
        #path of the image
        path = storage+'/data/CLASP1/train_gt/{}_{}/img1/*.png'.format(data_name[exp], cam_num)

    if dataset=='clasp1-mcta':
        tFactor = 1  # 30
        t_lag = cam_lag[exp][cam_num]  # C11: 10, C9:14
        eval_online = 0
        save_tracks = 1
        cam_tracks = []
        server = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/ourdataset/exp'+exp
        storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw'

        # read global tracks
        bbs = pd.read_csv(storage+'/output/tracktor/online_SCT/clasp1/global_tracks_clasp1_30FPSC{}{}_R50.csv'.format(cam_num, cam_pri), index_col=False)
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
        gtPerson1FPS = np.loadtxt(server+'/GT_{}/track/det{}Cam{:02d}PersonMASK_{}FPSrot/gt/gt.txt'.format(exp, exp, cam_num,
                                                                                                 frame_rate[exp]), delimiter=',')
        gtBag1FPS = np.loadtxt(server+'/GT_{}/track/det{}Cam{:02d}BagMASK_{}FPSrot/gt/gt.txt'.format(exp, exp, cam_num,
                                                                                            frame_rate[exp]), delimiter=',')
        # path of the output image
        outpath = storage+'/data/CLASP1/GTvsTrack/{}_{}'.format(exp, cam_num)
        if save_tracks:
            result_path = storage+'/data/CLASP1/eval_tracks_30fps/'
            if not os.path.exists(result_path):
                os.makedirs(result_path)
            #det_file  = open(os.path.join(result_path , '{}.txt'.format(cam)), "w")
            #det_file = csv.writer(det_file, delimiter=',')

        if not os.path.exists(outpath):
            os.makedirs(outpath)
        else:
            delete_all(outpath, fmt='jpg')
        # path of the image
        # remote
        path = storage+'/data/CLASP1/train_gt/{}_{}/img1/*.png'.format(data_name[exp], cam_num)

    files = glob.glob(path)
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    det = []
    frNum = 1
    # for raw det from baseline
    #detPerson1FPS = detPerson1FPS[np.where(detPerson1FPS[:,8]==0),:][0]

    classGT = 'PGT'
    classDet = 'P'
    if vis:
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('image',960,540)

    gt_framesetP = gtPerson1FPS[:, 0]+offset
    gt_framesetB = gtBag1FPS[:, 0]+offset

    #test frames:
    if dataset=='clasp1-det' and not (save_gt_p and save_gt_b):
        storage + '/data/CLASP1/train_gt/{}_{}/img1/*.png'.format(data_name[exp], cam_num)
        test_frames = pd.read_csv(os.path.join(storage +
                                               '/data/CLASP1/train_gt/{}/test_frames/{}.csv'.format(
                                                   cam_folder, cam_folder)))
        test_set = test_frames.values.squeeze()
        print('cam {}, total test frames: {}'.format(cam_folder, len(test_set)))
    else:
        test_set = list(gt_framesetP) + list(gt_framesetB)
        print('cam {}, total train+test frames: {}'.format(cam_folder, len(test_set)))

    for name in files:
        #print (name)
        frNum = float(os.path.basename(name).split('.')[0])
        imgcv = cv2.imread(name.split(os.path.basename(name))[0] + '{:06d}.png'.format(int(frNum)))
        if frNum in test_set: #gt_framesetP or frNum in gt_framesetB:
            # show person
            if dataset in ['clasp1-mcta', 'clasp1-sct']:
                detFr = detPerson1FPS[detPerson1FPS[:, 0] == frNum*fr_rate_factor[exp] - t_lag ]
            else:
                detFr = detPerson1FPS[detPerson1FPS[:, 0] == frNum]
            print('det frmae {}'.format(frNum))
            for bb in detFr:
                xmin = bb[2]
                ymin = bb[3]
                w = bb[4]
                h = bb[5]
                xmax = xmin + w
                ymax = ymin + h
                color = (0, 255, 0)  # green: bgr
                #if xmin+w/2 > 100 and 1000>ymin+h/2>300: #C11
                #if (xmin + w / 2 > 200 or ymin + h / 2>300)  and 1000>ymin + h / 2 : #C9
                    # score = detPerson1FPS[indFr[0][i]][6]
                if dataset=='clasp1-det':
                    display_txt = category[bb[7]]  +'%.2f' % (bb[6])
                else:
                    display_txt = 'P{}'.format(int(bb[1]))
                #if xmin + w / 2 > 100 and 1000 > ymin + h / 2 > 300:  # C11
                if vis:
                    imgcv = cv2.rectangle(imgcv, (np.int(xmin), np.int(ymin)), (np.int(xmax), np.int(ymax)), color, 8)
                    #cv2.putText(imgcv, display_txt, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
                if save_tracks:
                    # maintain actual gt frames
                    t_feature = [frNum-offset, bb[1], xmin, ymin, w, h, -1, -1, -1, -1]
                    cam_tracks.append(t_feature)

            # show bag
            if dataset in ['clasp1-mcta', 'clasp1-sct']:
                detFr = detBag1FPS[detBag1FPS[:, 0] == frNum*fr_rate_factor[exp] - t_lag ]
            else:
                detFr = detBag1FPS[detBag1FPS[:, 0] == frNum]
            if len(detFr)>0:
                print('det frmae {}'.format(frNum))
                for bb in detFr:
                    xmin = bb[2]
                    ymin = bb[3]
                    w = bb[4]
                    h = bb[5]
                    xmax = xmin + w
                    ymax = ymin + h
                    color =(255, 0, 255)  # magenta: bgr
                    #if xmin+w/2 > 100 and 1000>ymin+h/2>300: #C11
                    #if (xmin + w / 2 > 200 or ymin + h / 2>300)  and 1000>ymin + h / 2 : #C9
                        # score = detPerson1FPS[indFr[0][i]][6]
                    if dataset=='clasp1-det':
                        display_txt = category[bb[7]]  +'%.2f' % (bb[6])
                    else:
                        display_txt = 'B{}'.format(int(bb[1]))
                    #if xmin + w / 2 > 100 and 1000 > ymin + h / 2 > 300:  # C11
                    if vis:
                        imgcv = cv2.rectangle(imgcv, (np.int(xmin), np.int(ymin)), (np.int(xmax), np.int(ymax)), color, 8)
                        #cv2.putText(imgcv, display_txt, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
                    if save_tracks:
                        # maintain actual gt frames
                        t_feature = [frNum-offset, bb[1], xmin, ymin, w, h, -1, -1, -1, -1]
                        cam_tracks.append(t_feature)

            #baggage
            if dataset in ['clasp1-det', 'clasp1-sct']:
                if frNum-offset in gtBag1FPS[:, 0]:
                    print('gt frame {}'.format(frNum-offset))
                    # imgcv = imutils.rotate_bound(imgcv, 180)
                    #imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)
                    #find bbox for gt.person

                    Frgt = gtBag1FPS[gtBag1FPS[:, 0] == frNum-offset]
                    for bb in Frgt:
                        print('bGT')
                        class_id = 2
                        xmin = bb[2]
                        ymin = bb[3]
                        w = bb[4]
                        h = bb[5]
                        xmax = xmin + w
                        ymax = ymin + h
                        color = (0, 0, 255) #red: bgr
                        classGT = 'B' + str(int(bb[1]))
                        if vis:
                            imgcv = cv2.rectangle(imgcv, (np.int(xmin), np.int(ymin)), (np.int(xmax), np.int(ymax)), color, 8)
                            #cv2.putText(imgcv, classGT, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
                        if save_gt_b:
                            t_feature = [bb[0], bb[1] , xmin, ymin, w, h, 1, 1, 1, class_id, cam_num]
                            PB_file.writerow(t_feature)


            if frNum-offset in gtPerson1FPS[:, 0]:
                print('gt frame {}'.format(frNum-offset))
                # imgcv = imutils.rotate_bound(imgcv, 180)
                #imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)
                #find bbox for gt.person
                Frgt = gtPerson1FPS[gtPerson1FPS[:, 0] == frNum-offset]
                for bb in Frgt:
                    print(classGT)
                    class_id=1
                    xmin = bb[2]
                    ymin = bb[3]
                    w = bb[4]
                    h = bb[5]
                    xmax = xmin + w
                    ymax = ymin + h
                    color = (0, 0, 255) # red: bgr
                    classGT = 'P' + str(int(bb[1]))
                    if vis:
                        imgcv = cv2.rectangle(imgcv, (np.int(xmin), np.int(ymin)), (np.int(xmax), np.int(ymax)), color, 8)
                        #cv2.putText(imgcv, classGT, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4)
                    if save_gt_p:
                        t_feature = [bb[0], bb[1] , xmin, ymin, w, h, 1, 1, 1, class_id, cam_num]
                        PB_file.writerow(t_feature)
            if vis:
                cv2.imwrite(os.path.join(outpath, os.path.basename(name)), imgcv)
                cv2.imshow('image', imgcv)
                cv2.waitKey(10)

    # evaluate camera
    if eval_online:
        from evaluate_3D_tracks import *
        results = {}
        mot_evaluation = EvaluateMOT(gt_path=storage,
                                     result_path=outpath,
                                     vis_path=outpath,
                                     vis=False,
                                     proj_plane='2D',
                                     radious=100,
                                     iou_max=0.5,
                                     isMCTA=1,
                                     max_gt_frame=6500,
                                     min_gt_frame=1,
                                     auto_id=True)

        results['gt'] = gtPerson1FPS
        results['tracks'] = np.array(cam_tracks)
        cam = '{}_{}'.format(exp, cam_num)
        print('unique tragets detected in {} is {}'.format(cam, len(np.unique(results['tracks'][:,1]))))
        # do mot accumulation
        # TODO: SCT evaluation
        # python -m motmetrics.apps.eval_motchallenge /media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/wild-track/train_gt   /media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/multicamera_wildtrack/wildtrack/Wildtrack_dataset/results-track/C5C1
        # evaluate mot accumations
        mot_evaluation.evaluate_mot_accums(results, cam, generate_overall=True)

