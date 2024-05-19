#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import sys
import pprint

# Use a non-interactive backend
#import matplotlib
#matplotlib.use('Agg')

import numpy as np
import cv2
import time
import torch
import imutils
import pycocotools.mask as mask_util
from skimage import measure
import imageio
import glob
import _init_paths
from utils.timer import Timer
from clasp2coco import define_dataset_dictionary, Write_To_Json, Write_ImagesInfo,  Write_AnnotationInfo

def delete_all(demo_path, fmt='png'):
    filelist = glob.glob(os.path.join(demo_path, '*.' + fmt))
    if len(filelist) > 0:
        for f in filelist:
            os.remove(f)

def main(folders, dataset_clasp, output_dir ,imgResultDir, cam_data_split=0.8, data_type='train'):
    """main function"""
    # read imgaes from multi-camera clasp2 data folders
    fr = 1
    annIDcount = 1
    for cam_path in folders:
        image_dir = os.path.join(cam_path, 'img1')
        box_gt = np.loadtxt(os.path.join(cam_path, 'gt/gt.txt'), delimiter=',')
        gt_frameset = box_gt[:, 0].astype('int')
        # Apply split 80% for train and 20% for test
        split_ind = int(len(gt_frameset) * cam_data_split)
        if data_type == 'train':
            splitted_frameset = gt_frameset[:split_ind]
            print('80% train: {} from cam {} of total {}'.format(len(splitted_frameset), cam_path.split('/')[-1], len(gt_frameset)))
        if data_type == 'test':
            splitted_frameset = gt_frameset[split_ind:]
            print('20% train set: {} to {}'.format(min(splitted_frameset), max(splitted_frameset)))
        try:
            imglist = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        except NotADirectoryError: print('{} not found'.format(image_dir))

        num_images = len(imglist)
        cam_fr = 1
        for i, im_name in enumerate(imglist):
            if cam_fr in splitted_frameset:
                print(os.path.basename(im_name))
                fr_num = float(os.path.basename(im_name).split('.')[0])
                assert cam_fr==fr_num

                im = cv2.imread(im_name)
                fr_box_gt = box_gt[box_gt[:,0]==fr_num]

                imgIdnew = 1000*int('%06d'%fr) # fr is the gloabl frame index
                imgname = str(imgIdnew) + '.png'

                dataset_clasp = Write_ImagesInfo(im, imgname, int(imgIdnew), dataset_clasp)

                imf = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                # TODO: save images only for new angles used in dataset: save training data preparation time
                img_write_path = output_dir+'/'+imgname
                if not os.path.exists(img_write_path):
                    print('Writing image {}'.format(imgname))
                    imageio.imwrite(img_write_path, imf)

                print('total gt {} box at {}'.format(len(fr_box_gt), fr_num))
                for box in fr_box_gt:
                    #nox: x, y, w, h
                    bboxfinal = [round(x, 2) for x in box[2:6]]
                    area = box[4]*box[5]
                    #mask = masks[ib]
                    #area = mask_util.area(mask)#polygon area
                    catID = box[-2].astype('int')
                    assert  catID in [1, 2], 'catID {}is not applicable in clasp1'.format(catID)
                    annID = 1000 * int('%06d' % (annIDcount))
                    annIDcount+=1
                    #box = mask_util.toBbox(mask)
                    # convert rle mask into polygon mask
                    #bmask = mask_util.decode(mask)
                    #contours = measure.find_contours(bmask, 0.5)
                    #contours = cv2.findContours(bmask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    segmPolys = []
                    #for contour in contours:
                        #contour = np.flip(contour, axis=1)
                        #segmentation = contour.ravel().tolist()
                        #segmPolys.append(segmentation)
                    dataset_clasp = Write_AnnotationInfo(bboxfinal, segmPolys, int(imgIdnew),
                                                         int(annID), catID, int(area), dataset_clasp)
                fr += 1
            cam_fr+=1

    return dataset_clasp

if __name__ == '__main__':
    imgResultDir = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/CLASP1/train_gt_mask'
    savefilename = imgResultDir + '/clasp1_gt_mask.json'
    SaveImgDir = imgResultDir+'/img1'
    # clear path files
    if os.path.exists(imgResultDir):
        delete_all(SaveImgDir, fmt='png')
    else:
        os.makedirs(SaveImgDir)
    dataset_clasp = define_dataset_dictionary()

    benchmark_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/CLASP1/train_gt/'
    folders = glob.glob(benchmark_path + '*')
    folders.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))

    dataset_clasp = main(folders, dataset_clasp, SaveImgDir, imgResultDir)
    Write_To_Json(savefilename, dataset_clasp)