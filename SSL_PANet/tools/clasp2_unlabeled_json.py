# from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import distutils.util
import os
import sys
import numpy as np
import cv2
import time
import torch
import imutils
import glob
from clasp2coco import define_dataset_dictionary, Write_To_Json, Write_ImagesInfo, Write_AnnotationInfo
import random
import copy


# OpenCL may be enabled by default in OpenCV3; disable it because it's not
# thread safe and causes unwanted GPU memory allocations.
cv2.ocl.setUseOpenCL(False)

def main(folders, angleSet, dataset_clasp, output_dir):
    """main function"""

    # save all images in one folder
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fr = 1
    folders = sorted(folders)
    for cam_path in folders:
        # get training set
        image_dir = os.path.join(cam_path, 'img1')
        imglist = sorted(glob.glob(os.path.join(image_dir, '*')))
        num_images = len(imglist)

        # loop over all the annotated image
        for i, im_name in enumerate(imglist):
            fr_num = float(os.path.basename(im_name).split('.')[0])

            # search training set frames for augmented detections
            if fr_num % 40 == 0:
                im = cv2.imread(im_name)

                blankImg = im
                rot_imgs_dict = {}
                for angle in angleSet:
                    if angle > 0:
                        print('Image: {}, Cam: {}, Rotated by: {},'.format(os.path.basename(im_name),
                                                                           im_name.split('/')[-3], angle))
                        imgrot = imutils.rotate_bound(blankImg, angle)
                    else:
                        imgrot = blankImg
                    rot_imgs_dict[angle] = copy.deepcopy(imgrot)
                    for angle in angleSet:
                        # save image info
                        imgrot = rot_imgs_dict[angle]
                        imgIdnew = 10000 * int('%06d' % fr) + angle
                        imgname = '{:08d}.png'.format(imgIdnew)
                        img_write_path = output_dir + '/' + imgname

                        if not os.path.exists(img_write_path):
                            dataset_clasp = Write_ImagesInfo(imgrot, imgname, int(imgIdnew), dataset_clasp)
                            print('{}: writing image {}'.format(cam_path.split('/')[-1], imgname))
                            cv2.imwrite(img_write_path, imgrot)

                        dataset_clasp = Write_ImagesInfo(imgrot, imgname, int(imgIdnew), dataset_clasp)

                    fr += 1
    return dataset_clasp


def delete_all(demo_path, fmt='png'):
    filelist = glob.glob(os.path.join(demo_path, '*.' + fmt))
    if len(filelist) > 0:
        for f in filelist:
            os.remove(f)


if __name__ == '__main__':
    storage = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61'
    # required inputs
    data_type = 'clasp2_unlabeled'
    save_data = 1
    test_aug = 0

    if data_type == 'clasp2_unlabeled':
        benchmark_path = storage + '/tracking_wo_bnw/data/CLASP/train_gt_all/PB_no_gt/'
        save_data_dir = storage + '/SoftTeacher/data/clasp2/'
        imgResultDir = save_data_dir + 'unlabeledCLASP2/'
        savefilename = save_data_dir + 'annotations/instances_unlabeledCLASP2.json'
        SaveImgDir = imgResultDir

    elif data_type == 'clasp1_unlabeled':
        benchmark_path = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/' \
                 'ourdataset/exp6a/'
        cams = ['cam05exp6a.mp4', 'cam09exp6a.mp4', 'cam02exp6a.mp4', 'cam11exp6a.mp4']
        save_data_dir = storage + '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/SoftTeacher/data/clasp2/'
        imgResultDir = save_data_dir + 'unlabeledCLASP2/'
        savefilename = imgResultDir + '/instances_unlabeledCLASP2.json'
        SaveImgDir = imgResultDir
    # clear path files
    if os.path.exists(imgResultDir):
        delete_all(imgResultDir, fmt='png')
    else:
        os.makedirs(imgResultDir)
    if test_aug:
        # angleSet =[0, 12, 84, 90, 180, 186, 264, 270, 348, 354]
        angleSet = [0, 6, 12, 78, 84, 90, 96, 102, 168, 174, 180, 186, 192, 258, 264, 270, 276, 342, 348, 354]
        saveAugResult = 1

    else:
        angleSet = [0]
        saveAugResult = 0
    dataset_clasp = define_dataset_dictionary()


    folders = glob.glob(benchmark_path + '*')
    folders.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))

    dataset_clasp = main(folders, angleSet, dataset_clasp, SaveImgDir)
    if save_data:
        Write_To_Json(savefilename, dataset_clasp)

