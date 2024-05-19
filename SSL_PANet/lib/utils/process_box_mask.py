# Written by Roy Tseng
#
# Based on:
# --------------------------------------------------------
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


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import os
import sys
sys.path.insert(0, '/home/siddique/PANet/rgr-public/Python')
sys.path.insert(0, '/media/abubakarsiddique/PANet/rgr-public/Python')
import pycocotools.mask as mask_util
from runRGR import RGR
from utils.colormap import colormap
import utils.keypoints as keypoint_utils
import pdb
#from pudb.remote import set_trace

# Use a non-interactive backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import warnings
warnings.filterwarnings("ignore")
plt.rcParams['pdf.fonttype'] = 42  # For editing in Adobe Illustrator

    

_GRAY = (218, 227, 218)
_GREEN = (18, 127, 15)
_WHITE = (255, 255, 255)


def kp_connections(keypoints):
    kp_lines = [
        [keypoints.index('left_eye'), keypoints.index('right_eye')],
        [keypoints.index('left_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('nose')],
        [keypoints.index('right_eye'), keypoints.index('right_ear')],
        [keypoints.index('left_eye'), keypoints.index('left_ear')],
        [keypoints.index('right_shoulder'), keypoints.index('right_elbow')],
        [keypoints.index('right_elbow'), keypoints.index('right_wrist')],
        [keypoints.index('left_shoulder'), keypoints.index('left_elbow')],
        [keypoints.index('left_elbow'), keypoints.index('left_wrist')],
        [keypoints.index('right_hip'), keypoints.index('right_knee')],
        [keypoints.index('right_knee'), keypoints.index('right_ankle')],
        [keypoints.index('left_hip'), keypoints.index('left_knee')],
        [keypoints.index('left_knee'), keypoints.index('left_ankle')],
        [keypoints.index('right_shoulder'), keypoints.index('left_shoulder')],
        [keypoints.index('right_hip'), keypoints.index('left_hip')],
    ]
    return kp_lines


def convert_from_cls_format(cls_boxes, cls_segms, cls_keyps):
    """Convert from the class boxes/segms/keyps format generated by the testing
    code.
    """
    box_list = [b for b in cls_boxes if len(b) > 0]
    if len(box_list) > 0:
        boxes = np.concatenate(box_list)
    else:
        boxes = None
    if cls_segms is not None:
        segms = [s for slist in cls_segms for s in slist]
    else:
        segms = None
    if cls_keyps is not None:
        keyps = [k for klist in cls_keyps for k in klist]
    else:
        keyps = None
    classes = []
    for j in range(len(cls_boxes)):
        classes += [j] * len(cls_boxes[j])
    return boxes, segms, keyps, classes


def vis_bbox_opencv(img, bbox, thick=1):
    """Visualizes a bounding box."""
    (x0, y0, w, h) = bbox
    x1, y1 = int(x0 + w), int(y0 + h)
    x0, y0 = int(x0), int(y0)
    cv2.rectangle(img, (x0, y0), (x1, y1), _GREEN, thickness=thick)
    return img


def get_class_string(class_index, score, dataset):
    class_text = dataset.classes[class_index] if dataset is not None else \
        'id{:d}'.format(class_index)
    return class_text + ' {:0.2f}'.format(score).lstrip('0')


def get_box_mask(fr, boxes, segms=None, keypoints=None, thresh=0.5, angle=None,
                 class_list=None, dataset=None,out_when_no_box=False, verbose=False):
    PAXdet = []
    PAXmask = []

    if isinstance(boxes, list):
        boxes, segms, keypoints, classes = convert_from_cls_format(
            boxes, segms, keypoints)

    if boxes is None or boxes.shape[0] == 0 and out_when_no_box:
        return PAXdet, PAXmask

    if segms is not None:
        masks = mask_util.decode(segms)

    if boxes is None:
        sorted_inds = [] # avoid crash when 'boxes' is None
        return PAXdet, PAXmask
    else:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


    mask_color_id = 0
    for i,_ in enumerate(boxes):
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        if score <= thresh:
            continue
        #save pax boxs
        w = bbox[2] - bbox[0] + 1
        h = bbox[3] - bbox[1] + 1
        w = np.maximum(w, 1)
        h = np.maximum(h, 1)
        if classes[i] in class_list:
            if areas[i]>10:
                PAXdet.append([fr, i, bbox[0], bbox[1], w, h, score, classes[i], angle])

                if segms is not None:
                    mask_image = masks[:, :, i]
                    rle = mask_util.encode(np.asfortranarray(mask_image))
                    PAXmask.append(rle)
                #print(dataset.classes[classes[i]], score)
            else:
                print(f'regressed boxes are ignored due to area<10: {boxes[i,:]}')

    assert len(PAXdet) == len(PAXmask)
    return np.array(PAXdet), np.array(PAXmask)


def get_box_mask_single_cat(fr, boxes, segms=None, keypoints=None, proposals=None, thresh=0.5, angle=None,
                 class_id=None, dataset=None, cls_segms_coarse=None, out_when_no_box=False, 
                 img=None, rgr_refine=False, verbose=False, vis_dir=None):
    PAXdet = []
    PAXmask = []
    assert len(boxes[class_id]) == len(proposals)
    boxes, segms = boxes[class_id], segms[class_id]
    classes = [class_id] * len(boxes)

    if segms is not None:
        masks = mask_util.decode(segms)

    if cls_segms_coarse:
        coarses_28 = cls_segms_coarse[class_id]

    if boxes is None:
        sorted_inds = [] # avoid crash when 'boxes' is None
        return PAXdet, PAXmask
    else:
        areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    if rgr_refine and angle==0:
        mask_debug = np.zeros(img.shape[:2], dtype='uint8')
    else:
        mask_debug = None
    mask_color_id = 0
    for i,_ in enumerate(boxes):
        bbox = boxes[i, :4]
        score = boxes[i, -1]
        #use rotation invariance instead of regressed score
        if score <= thresh:
            continue
        #save pax boxs
        w = bbox[2] - bbox[0] + 1
        h = bbox[3] - bbox[1] + 1
        w = np.maximum(w, 1)
        h = np.maximum(h, 1)
        #print(f"class: {classes[i]} regressed boxes: {boxes[i]} \t class: {proposals[i, 7]} cluster_modes: {proposals[i, 2:6]} ")
        if areas[i]>0:
            if segms is not None:
                # apply RGR for flower
                if rgr_refine:
                    assert cls_segms_coarse is not None

                    mask_image = np.zeros(img.shape[:2], dtype='uint8')
                    ## RGR parameters
                    # fixed parameters
                    numSets = 10    # number of seeds sets (samplings)
                    cellSize = 10   # average spacing between samples

                    ## RGR parameters
                    # thresholds
                    tau0 = 0.5  # original CNN threshold
                    tauF = 0.8  # high confidence foreground
                    tauB = 0.01     # high confidence background
                    m = 0.1
                    #set_trace()
                    pos = np.array([ bbox[0],  bbox[1], w, h]).astype('int')
                    # convert 28x28 coarse to wxh instance
                    pred = cv2.resize(coarses_28[i], (pos[2],pos[3]))
                    # Get the corresponding RGB from img
                    rgb = img[pos[1]:pos[1]+pos[3], pos[0]:pos[0]+pos[2], :]
                    # background mask = 1 - Prediction
                    #TODO: try 0:background class mask
                    bck = 1 - pred
                    bck[bck>=1] = 0.005
                    assert rgb.shape[:2]==bck.shape==pred.shape

                    bck = np.reshape(bck, (bck.shape[0], bck.shape[1], 1))
                    pred = np.reshape(pred, (pred.shape[0], pred.shape[1], 1))
                    pred = np.concatenate((bck, pred), axis=2)
                    # refine instance mask using RGR
                    warnings.filterwarnings("ignore")
                    im_color, finalMask = RGR(rgb, pred, m, numSets, cellSize, tau0, tauF, tauB)
                    finalMask = cv2.cvtColor(finalMask, cv2.COLOR_RGB2GRAY)
                    # instance mask to mask_image
                    mask_image[pos[1]:pos[1]+pos[3], pos[0]:pos[0]+pos[2]] = finalMask
                    mask_image[mask_image>0]=255
                    #for debug
                    if rgr_refine and angle==0:
                        mask_debug+= mask_image
                    #RGR refined binary mask
                    rle = mask_util.encode(np.asfortranarray(mask_image))
                    x,y,w,h = mask_util.toBbox(rle)
                    PAXdet.append([fr, i, x, y, w, h, score, classes[i], angle])
                    PAXmask.append(rle)
                    print(f'refined instacne {i} in {fr}')
                else:
                    # thresholded binary mask 
                    PAXdet.append([fr, i, bbox[0], bbox[1], w, h, score, classes[i], angle])   
                    mask_image = masks[:, :, i]
                    rle = mask_util.encode(np.asfortranarray(mask_image))
                    PAXmask.append(rle)
            #print(dataset.classes[classes[i]], score)
        else:
            print(f'regressed box having area: {areas[i]} ignored even though the cluster modes are available')

    assert len(PAXdet) == len(PAXmask)
    if rgr_refine and angle==0:
        mask_debug[mask_debug>0] = 255
        cv2.imwrite(f'{vis_dir}/rgr_{fr}.png', mask_debug)
    return np.array(PAXdet), np.array(PAXmask), mask_debug
