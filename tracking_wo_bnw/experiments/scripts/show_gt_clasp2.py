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
#from evaluate_3D_tracks import *

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
        if i==0:
            fr_prev = fr
        diff = fr - fr_prev
        if diff!=10:
            pass
            #print(fr)
        fr_prev = fr

def apply_partial_constraints(gt, tr, database='clasp2', cam='G_9'):
    if cam in ['G_11', 'H_11', 'I_11']:
        cxy = tr[:, 2:4] + tr[:, 4:6] / 2
        tr = tr[cxy[:, 1] > 450]
    if cam in ['G_9']:  # filter person like appearance on the belt
        cxy = tr[:, 2:4] + tr[:, 4:6] / 2
        tr = tr[cxy[:, 1] > 600]
    if cam in ['G_9', 'H_9', 'I_9']:  # filter person like appearance on the belt
        ymax = 1010
        cxy = tr[:, 2:4] + tr[:, 4:6] / 2
        tr = tr[cxy[:, 1] < ymax]
        cxy = cxy[cxy[:, 1] < ymax]
        tr = tr[cxy[:, 0] > 130]

        gcxy = gt[:, 2:4] + gt[:, 4:6] / 2
        gt = gt[gcxy[:, 1] < ymax]
        gcxy = gcxy[gcxy[:, 1] < ymax]
        gt = gt[gcxy[:, 0] > 130]
    return gt, tr

    #return jump_at
if __name__ == '__main__':
    # Read GT
    dataset='clasp2-gt'
    gt_type = 'SCT'#'MC' # MC: person, SCT: person, bag
    save_pb_gt =1 #when save gt is ON, generate train+test otherwise test
    
    show_det=1
    vis=1
    cam_num =9
    exp_name = 'H'
    #remote
    #lags = {2:-224, 9:-175, 5:-200, 11:-176, 13:-176}
    #local
    lags = {'G':{9:10, 11:15}, 'H':{9:0, 11:0}, 'I':{9:0, 11:0}}
    pb_lags = {'G':{9:3, 11:4}, 'H':{9:0, 11:0}, 'I':{9:0, 11:0}}
    bag_lag = {'G':{9:0, 11:0}}
    #lags = {2: 5, 9: 0, 5: 0, 11: 0, 13: 10, 14: 18}
    offset = lags[exp_name][cam_num]
    cam = '{}_{}'.format(exp_name, cam_num)

    server = '/media/siddique/RemoteServer/LabFiles/CLASP2/exp2/'
    storage = '/media/6TB_local/tracking_wo_bnw'
    data_dir = storage + '/data/CLASP/train_gt_all/PB_gt/'
    #result_dir = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/PANet_Results/CLASP2/PANet_mask/'
    #result_dir = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/ourdataset/clasp2_detections/panet_ssl_score/iter7/'
    result_dir = '/media/6TB_local/tracking_wo_bnw/output/clasp2/'
    if gt_type=='MC':
        if cam_num==9:
            gtPerson1FPS = np.loadtxt(data_dir + cam + '/person_label_{}{}_mc_partial_correct.txt'.format(exp_name, cam_num), delimiter=',')#_partial_correct
        else:
            gtPerson1FPS = np.loadtxt(
                data_dir + cam + '/person_label_{}{}_mc.txt'.format(exp_name, cam_num),
                delimiter=',')  # _partial_correct
        # path of the output image
        outpath = storage + '/data/CLASP/train_gt_all/GTVsDet/' + cam + '/'
        
        

    if gt_type in ['SCT', 'det']:
        gtPerson1FPS = np.loadtxt(
            data_dir + cam + '/person_label_{}{}_sct.txt'.format(exp_name, cam_num),
            delimiter=',')

        gtBag1FPS = np.loadtxt(data_dir + cam + '/bag_label_{}{}.txt'.format(exp_name, cam_num), delimiter=',')
        # path of the output image
        outpath = storage + '/data/CLASP/train_gt_all/GTVsDet_SCT/' + cam + '/'

    if not os.path.exists(outpath):
        os.makedirs(outpath)
    else:
        delete_all(outpath, fmt='jpg')
        
    gt_img_path = storage + '/data/CLASP/train_gt_all/gt_imgs/' + cam + '/'    
    if not os.path.exists(gt_img_path):
            os.makedirs(gt_img_path)
    # path of the image
    # local
    path = storage+'/data/CLASP/train_gt_all/PB_gt/'+cam+'/img1/*'
    if show_det:
        category = {1: 'person', 2: 'bag'}
        #detPerson1FPS = np.loadtxt(result_dir+'{}_{}_pb_0aug1nms.txt'.format(
            #exp_name, cam_num), delimiter=',')
        detPerson1FPS = np.loadtxt(result_dir + 'SSL_alpha_tracks/{}/1_C{}.txt'.format(
            exp_name, cam_num), delimiter=',')
        detBag1FPS = np.loadtxt(result_dir + 'SSL_alpha_tracks/{}/2_C{}.txt'.format(
            exp_name, cam_num), delimiter=',')

    if save_pb_gt:
        if gt_type=='MC':
            PB_file  = open(os.path.join(data_dir, '{}_{}/gt/gt.txt'.format(exp_name, cam_num)), "w")
        if gt_type=='SCT':
            PB_file = open(os.path.join(data_dir, '{}_{}/gt_sct/gt.txt'.format(exp_name, cam_num)), "w")
        PB_file = csv.writer(PB_file, delimiter=',')
    else:
        if gt_type == 'MC':
            GT = np.loadtxt(os.path.join(data_dir, '{}_{}/gt/gt.txt'.format(exp_name, cam_num)), delimiter=',')
        if gt_type == 'SCT':
            GT = np.loadtxt(os.path.join(data_dir, '{}_{}/gt_sct/gt.txt'.format(exp_name, cam_num)), delimiter=',')
        #_, test_set = get_split(GT, train_split=0.8, cam=cam_num)
        cam_folder = '{}_{}'.format(exp_name, cam_num)
        test_frames = pd.read_csv(os.path.join(data_dir,'{}/test_frames/{}.csv'.format(cam_folder, cam_folder)))
        test_set = test_frames.values.squeeze()
        print('cam {}, total frames: {}'.format(cam_folder, len(test_set)))

    files = glob.glob(path)
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
    find_frame_jump(files)
    det = []
    frNum = 0

    classGT = 'PGT'
    classDet = 'P'
    tFactor = 1 # 30
    # cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow('image',960,540)

    gt_framesetP = gtPerson1FPS[:, 0]#+offset
    gtPerson1FPS[:, 0] = gt_framesetP
    if gt_type in ['SCT', 'det']:
        gt_framesetB = gtBag1FPS[:, 0]
    else:
        gt_framesetB = []
    #apply partial constraints
    if show_det:
        gtPerson1FPS, detPerson1FPS = apply_partial_constraints(gtPerson1FPS, detPerson1FPS, cam=cam)

    for name in files:
        #print (os.path.basename((name)))

        frNum = float(os.path.basename(name).split('.')[0])

        #if frNum in  test_set: #gt_framesetP or frNum in gt_framesetB: #detPerson1FPS[:,0]:#test_set: #
        if frNum in gt_framesetP or frNum in gt_framesetB:
            imgcv = cv2.imread(name)
            #save gt rgb frames
            #cv2.imwrite(f'{gt_img_path}/{os.path.basename(name)}', imgcv) 
            # find bbox for person
            if frNum in gtPerson1FPS[:, 0]: #search actual frame in gt
                #print('gt frame {}'.format(frNum))
                # imgcv = imutils.rotate_bound(imgcv, 180)
                # imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)
                # find bbox for gt.person
                Frgt = gtPerson1FPS[gtPerson1FPS[:, 0] == frNum]
                for bb in Frgt:
                    xmin = bb[2]
                    ymin = bb[3]
                    w = bb[4]
                    h = bb[5]
                    xmax = xmin + w
                    ymax = ymin + h
                    color = (0, 0, 255) #red: bgr
                    classGT = 'P'+str(int(bb[1]))
                    class_id = 1
                    if vis:
                        imgcv = cv2.rectangle(imgcv, (np.int(xmin), np.int(ymin)), (np.int(xmax), np.int(ymax)), color, 8)
                        cv2.putText(imgcv, f"{bb[1]}", (int(xmin+w/2), int(ymin+h/2)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)

                    if save_pb_gt:
                        t_feature = [bb[0], bb[1] , xmin, ymin, w, h, 1, 1, 1, class_id, cam_num]
                        PB_file.writerow(t_feature)
            #find bbox for bag
            if frNum in gtBag1FPS[:, 0]:
                print('gt frame {}'.format(frNum-offset))
                # imgcv = imutils.rotate_bound(imgcv, 180)
                #imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)
                #find bbox for gt.person
                Frgt = gtBag1FPS[gtBag1FPS[:, 0]== frNum]
                for bb in Frgt:
                    #print('bGT')
                    class_id = 2
                    xmin = bb[2]
                    ymin = bb[3]
                    w = bb[4]
                    h = bb[5]
                    xmax = xmin + w
                    ymax = ymin + h
                    color = (0, 0, 255) #red: bgr
                    classGT = 'B' + str(int(bb[1]))
                    class_id = 2

                    #if (xmin+w/2)<1500 or (ymin+h/2)<700:

                    if vis:
                        imgcv = cv2.rectangle(imgcv, (np.int(xmin), np.int(ymin)), (np.int(xmax), np.int(ymax)), color, 8)
                        #cv2.putText(imgcv, classGT, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 4)
                    if save_pb_gt:
                        t_feature = [bb[0], bb[1] , xmin, ymin, w, h, 1, 1, 1, class_id, cam_num]
                        PB_file.writerow(t_feature)

            if show_det and frNum in detPerson1FPS[:, 0]:
                detFr = detPerson1FPS[detPerson1FPS[:, 0] == frNum]
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
                    if dataset=='clasp2-gt':
                        #display_txt = category[bb[7]]  +'%.2f' % (bb[6])
                        display_txt = 'P{}'.format(int(bb[1]))
                    else:
                        display_txt = 'P{}'.format(int(bb[1]))
                    #(xx(i,3)+xx(i,5)/2)>50 && (xx(i,4)+xx(i,6)/2)<1050
                    if vis:
                        imgcv = cv2.rectangle(imgcv, (np.int(xmin), np.int(ymin)), (np.int(xmax), np.int(ymax)), color, 8)
                        #cv2.putText(imgcv, display_txt, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 2, color,4)
            if show_det:
                if frNum in detBag1FPS[:, 0]:
                    detFr = detBag1FPS[detBag1FPS[:, 0] == frNum]
                    for bb in detFr:
                        xmin = bb[2]
                        ymin = bb[3]
                        w = bb[4]
                        h = bb[5]
                        xmax = xmin + w
                        ymax = ymin + h
                        color = (255, 0, 255)  # green: bgr
                        #if xmin+w/2 > 100 and 1000>ymin+h/2>300: #C11
                        #if (xmin + w / 2 > 200 or ymin + h / 2>300)  and 1000>ymin + h / 2 : #C9
                            # score = detPerson1FPS[indFr[0][i]][6]
                        if dataset=='clasp2-gt':
                            #display_txt = category[bb[7]]  +'%.2f' % (bb[6])
                            display_txt = 'B{}'.format(int(bb[1]))
                        else:
                            display_txt = 'B{}'.format(int(bb[1]))
                        #(xx(i,3)+xx(i,5)/2)>50 && (xx(i,4)+xx(i,6)/2)<1050
                        if vis:
                            imgcv = cv2.rectangle(imgcv, (np.int(xmin), np.int(ymin)), (np.int(xmax), np.int(ymax)), color, 8)
                            #cv2.putText(imgcv, display_txt, (int(xmin), int(ymin)), cv2.FONT_HERSHEY_SIMPLEX, 2, color,4)

            cv2.imwrite(os.path.join(outpath, '{:06d}.jpg'.format(int(frNum))), imgcv)
            # cv2.imshow('image', imgcv)
            # cv2.waitKey(10)
        #frNum+=1