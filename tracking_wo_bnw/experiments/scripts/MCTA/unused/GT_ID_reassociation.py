from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import cv2
import logging
import sys
import os
import glob
from scipy.misc import imsave
import imutils
import random
#create color space
def hex_to_rgb(hex):
    hex = hex.lstrip('#')
    hlen = len(hex)
    return tuple(int(hex[i:i + hlen // 3], 16) for i in range(0, hlen, hlen // 3))

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

color = ["#" + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
         for i in range(200)]

#imgs Path
imgsPath = ['exp9A/cam11']#['exp5A/cam11/10FPS','exp6A/cam11/2fps','exp7A/cam11/2fps','exp9A/cam11','exp10A/cam11/2fps'] #'exp5A/cam9/10FPS','exp6A/cam9/2fps','exp7A/cam9/2fps','exp9A/cam9','exp10A/cam9/2fps'
#Read GT_xCam: PAX
allGT_path =  '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/ourdataset/GT_MU/'+imgsPath[0].split('/')[1]+'/'
gtFiles = glob.glob(allGT_path + '*.txt')
gtFiles.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))

for j,gtPath in enumerate(gtFiles[-2:-1]):
    gt = np.loadtxt(gtPath, delimiter=',')

    #path of the output image
    outpath = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/ourdataset/'+imgsPath[j].split('/')[0]+'/GT/vis/'+imgsPath[0].split('/')[1]+'/'
    #delete the files from the current path
    if os.path.exists(outpath):
        result_list = glob.glob(outpath+'/*.png')
        result_list.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        for path in result_list:
            os.remove(path)

    if not os.path.exists(outpath):
        os.makedirs(outpath)
    offset = 0
    if imgsPath[j] == 'exp9A/cam11':
        offset = 2
        # reassociate ID
        gt[:, 1][gt[:, 1] == 11] = 10

    if imgsPath[j] == 'exp5A/cam9/10FPS':
        # reassociate ID
        for id in [5,8]:
            gt[:, 1][gt[:, 1] == id] = 3

        gt[:, 1][gt[:, 1] == 4] = 2
        gt10 = gt[gt[:,0]>=844]
        gt10 = gt10[gt10[:, 1] == 10]
        gt10[:,1] = 9
        gt[gt[:, 0] >= 844] = gt10

    if imgsPath[j] == 'exp5A/cam11/10FPS':
        # reassociate ID
        gt[:, 1][gt[:, 1] == 8] = 6

    if imgsPath[j] == 'exp6A/cam11/2fps':
        # reassociate ID
        for id in [3,4,9,13,15]:
            gt[:, 1][gt[:, 1] == id] = 1
    if imgsPath[j] == 'exp6A/cam9/2fps':
        # reassociate ID
        gt[:, 1][gt[:, 1] == 10] = 9
    if imgsPath[j] == 'exp7A/cam9/2fps':
        # reassociate ID
        gt[:, 1][gt[:, 1] == 3] = 2
        gt[:, 1][gt[:, 1] == 5] = 4
    if imgsPath[j] == 'exp9A/cam9':
        offset = 2
        # reassociate ID
        gt[:, 1][gt[:, 1] == 6] = 5
        gt[:, 1][gt[:, 1] == 17] = 8
        gt[:, 1][gt[:, 1] == 19] = 15

    imgs_path = gtFiles[0].split('GT_MU')[0] +imgsPath[j]+'/*.png'
    files = glob.glob(imgs_path)
    files.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    det = []
    frNum = 1
    gtAsso = []
    for name in files:
        print (name)
        #find bbox at frNum
        Frgt = gt[gt[:, 0] == frNum-offset]
        if imgsPath[j] == 'exp9A/cam11':
            if frNum>=157 and frNum<=179:
                #Frgt = gt[gt[:, 0]==frNum]
                gt10 = Frgt[Frgt[:,1]==10]
                gt10[:,1][gt10[:,4]*gt10[:,5] == max(gt10[:,4]*gt10[:,5])] = 11
                Frgt[Frgt[:,1]==10] = gt10

        if len(Frgt) == 0:
            print
            'no box found'
            #imsave(outpath + name[-10:-4] + '.png', imgcv)
        if len(Frgt) > 0:
            #save GT with reassociation
            # visual results
            imgcv = cv2.imread(name)
            # imgcv = imutils.rotate_bound(imgcv, 180)
            imgcv = cv2.cvtColor(imgcv, cv2.COLOR_BGR2RGB)
            for i,box in enumerate(Frgt):
                #if imgsPath[j]=='exp9A/cam11':
                    #box = np.insert(box, 6, values=-1, axis=0)
                    #box[-4:] = box[-4:] * -1
                gtAsso.append(box)
                ID = int(box[1])
                print 'ID {}'.format(ID)
                xmin = np.int(box[2])
                ymin = np.int(box[3])
                w = box[4]
                h = box[5]
                xmax = xmin + w
                ymax = ymin + h
                colorRGB = hex_to_rgb(color[ID])
                imgcv = cv2.rectangle(imgcv, (xmin, ymin), (np.int(xmax), np.int(ymax)),colorRGB, 8)
                cv2.putText(imgcv,'P'+str(ID), (int(xmin+w/2), int(ymin+h//2)), cv2.FONT_HERSHEY_SIMPLEX, 2, colorRGB,3, cv2.LINE_AA)
            imsave(outpath + name[-10:], imgcv)
        frNum = frNum + 1
    if len(gtAsso)>0:
        #sort based on ID and the time stamps
        gtAsso = np.array(gtAsso)
        GT = []
        for ID in np.unique(gtAsso[:,1]):
            for box in gtAsso[gtAsso[:,1]==ID]:
                GT.append(box)

        np.savetxt(outpath + 'gt.txt',np.array(GT), fmt='%.3e', delimiter=',')
