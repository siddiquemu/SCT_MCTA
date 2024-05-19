import numpy as np
import pandas as pd
import cv2
import glob
import os

dataset = 'clasp1' #'clasp2'
if dataset=='clasp2':
    test_gt_frames = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/ourdataset/clasp2_detections/panet_ssl_score/iter6'

    storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw'
    data_dir = storage + '/data/CLASP/train_gt_all/PB_gt'
    pax_count = []
    bag_count = []
    for cam in ['G_9', 'G_11', 'H_9', 'H_11', 'I_9','I_11']:
        test_set_dict = {}
        test_set_dict[cam] = []
        train_set_dict = {}
        train_set_dict[cam] = []

        cam_save_path = os.path.join(data_dir, '{}/test_frames'.format(cam))
        if not os.path.exists(cam_save_path):
            os.makedirs(cam_save_path)

        cam_save_path_train = os.path.join(data_dir, '{}/train_frames'.format(cam))
        if not os.path.exists(cam_save_path_train):
            os.makedirs(cam_save_path_train)

        path = os.path.join(test_gt_frames, '{}_test_gt.txt'.format(cam))
        path_all = os.path.join(data_dir, '{}/gt/gt.txt'.format(cam))
        frames = np.loadtxt(path, delimiter=',')
        frames_all = np.loadtxt(path_all, delimiter=',')

        print('cam {}, total pax: {}'.format(cam, len(frames[frames[:, 9]==1])))
        pax_count.append(len(frames[frames[:, 9]==1]))
        print('cam {}, total bag: {}'.format(cam, len(frames[frames[:, 9] == 2])))
        bag_count.append(len(frames[frames[:, 9]==2]))

        for frNum in np.unique(frames_all[:,0]):
            #frNum = float(os.path.basename(name).split('.')[0])
            if frNum not in np.unique(frames[:,0]):
                train_set_dict[cam].append(frNum)
            else:
                test_set_dict[cam].append(frNum)
        #data = pd.DataFrame(test_set_dict)
        #data.to_csv(os.path.join(cam_save_path, '{}.csv'.format(cam)), index=False)

        data = pd.DataFrame(train_set_dict)
        data.to_csv(os.path.join(cam_save_path_train, '{}.csv'.format(cam)), index=False)

if dataset == 'clasp1':
    test_gt_frames = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/ourdataset/clasp1_detections/panet_ssl/ResNet50FPNPANet_tuned06'

    storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw'
    data_dir = storage + '/data/CLASP1/train_gt'
    pax_count = []
    bag_count = []
    for cam in ['A_9', 'A_11', 'B_9', 'B_11', 'C_9', 'C_11', 'D_9', 'D_11', 'E_9', 'E_11']:
        test_set_dict = {}
        test_set_dict[cam] = []
        train_set_dict = {}
        train_set_dict[cam] = []

        cam_save_path = os.path.join(data_dir, '{}/test_frames'.format(cam))
        if not os.path.exists(cam_save_path):
            os.makedirs(cam_save_path)

        cam_save_path_train = os.path.join(data_dir, '{}/train_frames'.format(cam))
        if not os.path.exists(cam_save_path_train):
            os.makedirs(cam_save_path_train)

        path = os.path.join(test_gt_frames, '{}_test_gt.txt'.format(cam))
        path_all = os.path.join(data_dir, '{}/gt/gt.txt'.format(cam))
        frames = np.loadtxt(path, delimiter=',')
        frames_all = np.loadtxt(path_all, delimiter=',')

        print('cam {}, total pax: {}'.format(cam, len(frames[frames[:, 9] == 1])))
        pax_count.append(len(frames[frames[:, 9] == 1]))
        print('cam {}, total bag: {}'.format(cam, len(frames[frames[:, 9] == 2])))
        bag_count.append(len(frames[frames[:, 9] == 2]))

        for frNum in np.unique(frames_all[:, 0]):
            # frNum = float(os.path.basename(name).split('.')[0])
            if frNum not in np.unique(frames[:, 0]):
                train_set_dict[cam].append(frNum)
            else:
                test_set_dict[cam].append(frNum)
        test_data = pd.DataFrame(test_set_dict)
        test_data.to_csv(os.path.join(cam_save_path, '{}.csv'.format(cam)), index=False)

        train_data = pd.DataFrame(train_set_dict)
        train_data.to_csv(os.path.join(cam_save_path_train, '{}.csv'.format(cam)), index=False)

print('total pax: {}'.format(sum(pax_count)))
print('total bag: {}'.format(sum(bag_count)))