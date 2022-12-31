# Copyright (c) 2017-present, Facebook, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
##############################################################################

"""Collection of available datasets."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

# Path to data dir
#clasp2
#_DATA_DIR = cfg.DATA_DIR
#_DATA_DIR = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT'
#clasp1
_DATA_DIR = '/media/6TB_local/tracking_wo_bnw/data'
#_DATA_DIR = '/media/abubakarsiddique/tracking_wo_bnw/data'
_DATA_DIR_FLOWER = '/media/abubakarsiddique/tracking_wo_bnw/data'
#_DATA_DIR='/media/siddique/RemoteServer/LabFiles/Walden/trainTestSplit/train/dataFormattedProperly/splitImages4x3'
# Required dataset entry keys
IM_DIR = 'image_directory'
ANN_FN = 'annotation_file'
#flower
IM_DIR_FL = 'image_directory'
ANN_FN_FL = 'annotation_file'

# Optional dataset entry keys
IM_PREFIX = 'image_prefix'
DEVKIT_DIR = 'devkit_directory'
RAW_DIR = 'raw_dir'

# Available datasets
DATASETS = {
    'cityscapes_fine_instanceonly_seg_train': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_train.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_val': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        # use filtered validation as there is an issue converting contours
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'cityscapes_fine_instanceonly_seg_test': {
        IM_DIR:
            _DATA_DIR + '/cityscapes/images',
        ANN_FN:
            _DATA_DIR + '/cityscapes/annotations/instancesonly_gtFine_test.json',
        RAW_DIR:
            _DATA_DIR + '/cityscapes/raw'
    },
    'coco_2014_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_train2014.json'
    },
    'coco_2014_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_val2014.json'
    },
    'coco_2014_minival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_minival2014.json'
    },
    'coco_2014_valminusminival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/instances_valminusminival2014.json'
    },
    'coco_2015_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'coco_2015_test-dev': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'coco_2017_train': {# finetune for generating rotation invariant model using CLASP2 data
        IM_DIR:
            _DATA_DIR + '/coco_images/coco_2017/train2017',
        ANN_FN:
            _DATA_DIR + '/coco_images/coco_2017/annotations/instances_train2017.json',
    },
    'coco_2017_val': {
        IM_DIR:
            _DATA_DIR + '/coco_images/coco_2017/val2017',
        ANN_FN:
            _DATA_DIR + '/coco_images/coco_2017/annotations/instances_val2017.json',
    },
    'coco_val_2017_aug': {
        IM_DIR:
            _DATA_DIR + '/coco_images/coco_2017/val2017_aug',
        ANN_FN:
            _DATA_DIR + '/coco_images/coco_2017/val_2017_annotation/coco_2017_aug_val_2_classes.json',
    },
#finetune using clasp
    'clasp2_2020_test': {  # 2020 test uses train2020_CLASP_aug images: TODO: use different set of test images (only at angle==0)
        IM_DIR:
            _DATA_DIR + '/coco_images/coco_2017/train2020_CLASP_aug',
        ANN_FN:
            _DATA_DIR + '/coco_images/coco_2017/train2020_CLASP_aug_annotation/test2020_CLASP_aug_annotation_0_anglev2.json',
    },
    #clasp1 SL model data
    'clasp1_2021_train': {
        IM_DIR:
            _DATA_DIR + '/CLASP1/train_gt_det/img1',
        ANN_FN:
            _DATA_DIR + '/CLASP1/train_gt_det/clasp1_gt_det.json',
    },
    #clasp2 SL model data
    'clasp2_2021_train': {
        IM_DIR:
            _DATA_DIR + '/CLASP/train_gt_det/img1',
        ANN_FN:
            _DATA_DIR + '/CLASP/train_gt_det/clasp2_gt_det.json',
    },
    'clasp1_2021_aug_train': {
        IM_DIR:
            _DATA_DIR + '/CLASP1/test_augMS_gt/img1_9C11.',
        ANN_FN:
            _DATA_DIR + '/CLASP1/test_augMS_gt/clasp1_test_aug1_9C11..json',
    },
    'clasp1_2021_aug_train_wo_regress': {
        IM_DIR:
            _DATA_DIR + '/CLASP1/test_augMS_gt_wo_regress',
        ANN_FN:
            _DATA_DIR + '/CLASP1/test_augMS_gt_wo_regress',
    },
    #rotation aug, instance certainty
    'clasp1_2021_aug_train_score': {
        IM_DIR:
            _DATA_DIR + '/CLASP1/test_augMS_gt_score',
        ANN_FN:
            _DATA_DIR + '/CLASP1/test_augMS_gt_score',
    },
    #rotation aug, motion blur and color jitter, instance certainty
    'clasp1_2021_mixed_aug_train_score': {
        IM_DIR:
            _DATA_DIR + '/CLASP1/test_augMS_gt_score',
        ANN_FN:
            _DATA_DIR + '/CLASP1/test_augMS_gt_score',
    },

    'clasp2_2021_mixed_aug_train_score': {
        IM_DIR:
            _DATA_DIR + '/CLASP/test_augMS_gt_score',
        ANN_FN:
            _DATA_DIR + '/CLASP/test_augMS_gt_score',
    },
    'clasp1_2021_aug_train_score_wo_regress': {
        IM_DIR:
            _DATA_DIR + '/CLASP1/test_augMS_gt_score_wo_regress',
        ANN_FN:
            _DATA_DIR + '/CLASP1/test_augMS_gt_score_wo_regress',
    },
    'clasp2_2021_aug_train': {
        IM_DIR:
            _DATA_DIR + '/CLASP/test_augMS_gt/img1_9',
        ANN_FN:
            _DATA_DIR + '/CLASP/test_augMS_gt/clasp2_test_aug_9.json',
    },
    'clasp2_2021_aug_train_score': {
        IM_DIR:
            _DATA_DIR + '/CLASP/test_augMS_gt_score',
        ANN_FN:
            _DATA_DIR + '/CLASP/test_augMS_gt_score',
    },
    'clasp2_2021_aug_train_score_wo_regress': {
        IM_DIR:
            _DATA_DIR + '/CLASP/test_augMS_gt_score_wo_regress',
        ANN_FN:
            _DATA_DIR + '/CLASP/test_augMS_gt_score_wo_regress',
    },
    'clasp2_2021_aug_train_wo_regress': {
        IM_DIR:
            _DATA_DIR + '/CLASP/test_augMS_gt_wo_regress',
        ANN_FN:
            _DATA_DIR + '/CLASP/test_augMS_gt_wo_regress',
    },
    'clasp2_2021_aug_train_score_sigm': {
        IM_DIR:
            _DATA_DIR + '/CLASP/test_augMS_gt_score_sigm/iter4/img1_4',
        ANN_FN:
            _DATA_DIR + '/CLASP/test_augMS_gt_score_sigm/iter4/clasp2_test_aug_4.json',
    },
    'PVD_2021_aug_train': {
        IM_DIR:
            _DATA_DIR + '/PVD/test_augMS_gt/img1_1C4',
        ANN_FN:
            _DATA_DIR + '/PVD/test_augMS_gt/PVD_test_aug1_1C4.json',
    },
    'PVD_train': {
        IM_DIR:
            _DATA_DIR + '/PVD/train_gt_det/img1',
        ANN_FN:
            _DATA_DIR + '/PVD/train_gt_det/pvd_gt_det.json',
    },
    
    'flower_2021_labeled': {
        IM_DIR:
            _DATA_DIR + '/flower/train_gt_sw/trainFlowerAug',
        ANN_FN:
            _DATA_DIR + '/flower/train_gt_sw/instances_train_2021.json',
    },
    'flower_2021_aug_score': {

        IM_DIR:
            _DATA_DIR + '/flower/test_augMS_gt_score',
        ANN_FN:
            _DATA_DIR + '/flower/test_augMS_gt_score',
    },
    'flower_2021_aug_score_old': {

        IM_DIR_FL:
            _DATA_DIR_FLOWER + '/flower/test_augMS_gt_score_old',
        ANN_FN_FL:
            _DATA_DIR_FLOWER + '/flower/test_augMS_gt_score_old',
    },
    'mot_2020_aug_train': {
        IM_DIR:
            _DATA_DIR + '/MOT/MOT20Det/test_augMS_gt/img1_2C2',
        ANN_FN:
            _DATA_DIR + '/MOT/MOT20Det/test_augMS_gt/MOT20_test_aug1_2C2.json',
    },
    'coco_2017_test': {  # 2017 test uses 2015 test images
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json',
        IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_2017_test-dev': {  # 2017 test-dev uses 2015 test images
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2017.json',
        IM_PREFIX:
            'COCO_test2015_'
    },
    'coco_stuff_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/stuff_train.json'
    },
    'coco_stuff_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/stuff_val.json'
    },
    'keypoints_coco_2014_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2014.json'
    },
    'keypoints_coco_2014_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2014.json'
    },
    'keypoints_coco_2014_minival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_minival2014.json'
    },
    'keypoints_coco_2014_valminusminival': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2014',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_valminusminival2014.json'
    },
    'keypoints_coco_2015_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2015.json'
    },
    'keypoints_coco_2015_test-dev': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2015',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test-dev2015.json'
    },
    'keypoints_coco_2017_train': {
        IM_DIR:
            _DATA_DIR + '/coco/images/train2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_train2017.json'
    },
    'keypoints_coco_2017_val': {
        IM_DIR:
            _DATA_DIR + '/coco/images/val2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/person_keypoints_val2017.json'
    },
    'keypoints_coco_2017_test': {
        IM_DIR:
            _DATA_DIR + '/coco/images/test2017',
        ANN_FN:
            _DATA_DIR + '/coco/annotations/image_info_test2017.json'
    },
    'voc_2007_trainval': {
        IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_trainval.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2007_test': {
        IM_DIR:
            _DATA_DIR + '/VOC2007/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2007/annotations/voc_2007_test.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2007/VOCdevkit2007'
    },
    'voc_2012_trainval': {
        IM_DIR:
            _DATA_DIR + '/VOC2012/JPEGImages',
        ANN_FN:
            _DATA_DIR + '/VOC2012/annotations/voc_2012_trainval.json',
        DEVKIT_DIR:
            _DATA_DIR + '/VOC2012/VOCdevkit2012'
    }
}
