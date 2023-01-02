#!/usr/bin/python
from __future__ import division
import pandas as pd
import copy
import numpy as np

def bb_intersection_over_union(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)

    # return the intersection over union value
    return iou


def match_id_pvd(meta_data, currFrame, t_bb, cam=None, matched_label = None, matched_tu=0, max_iou=0.5):
    for i, row in enumerate(meta_data['BB']):
        if meta_data['Frame'][i] == currFrame and meta_data['cam'][i]==cam:
            #row = [row[0]*(1080.0/1920.0), row[1]*(720.0/1080.0),row[2]*(1080.0/1920.0),row[3]*(720.0/1080.0)]
            row = [row[0] , row[1], row[2],row[3]]
            tmpIoU = bb_intersection_over_union(row,t_bb)
            if tmpIoU >= max_iou:
                matched_label = meta_data['ID'][i]
                if meta_data['TU'][i]!=0:
                    matched_tu = meta_data['TU'][i]#meta_data['ID'][i] #meta_data['TU'][i]
    return matched_label, matched_tu

def read_metadata_pvd(fname):

    data_frame = pd.read_excel(fname,encoding='latin-1')
    meta_data = {'Frame':[], 'Lane3':{}, 'ID': [], 'TU':[],'BB':[],'cam':[]}

    for row in data_frame.values:
        #start frame
        meta_data['cam'].append(float(row[0]))
        meta_data['ID'].append(row[1])
        meta_data['Frame'].append(row[4])
        meta_data['BB'].append(list(map(int, row[5].split(','))))
        meta_data['TU'].append(row[8])
        if row[3]==3:
            meta_data['Lane3'][row[1]] = row[3]
        #end frame
        meta_data['cam'].append(float(row[0]))
        meta_data['ID'].append(row[1])
        meta_data['Frame'].append(row[6])
        meta_data['BB'].append(list(map(int, row[7].split(','))))
        meta_data['TU'].append(row[8])


    return meta_data

def read_metadata(fname):

    data_frame = pd.read_excel(fname,encoding='latin-1')
    meta_data = {'Frame':[], 'ID': [], 'TU':[],'BB':[],'cam':[]}

    for row in data_frame.values:
        meta_data['Frame'].append(row[3])
        meta_data['ID'].append(row[0])
        meta_data['TU'].append(row[-1])
        meta_data['BB'].append(list(map(int, row[4].split(','))))
        meta_data['cam'].append(float(row[2].split('cam')[-1]))
    return meta_data

def match_id(meta_data, currFrame, t_bb, cam=None, matched_label = None, matched_tu=0, max_iou=0.5):
    for i, row in enumerate(meta_data['BB']):
        if meta_data['Frame'][i] == currFrame and meta_data['cam'][i]==cam:
            row = [row[0]*(1080.0/1920.0), row[1]*(720.0/1080.0),row[2]*(1080.0/1920.0),row[3]*(720.0/1080.0)]
            tmpIoU = bb_intersection_over_union(row,t_bb)
            if tmpIoU >= max_iou:
                matched_label = meta_data['ID'][i]
                if meta_data['TU'][i]!=0:
                    matched_tu = meta_data['TU'][i]
    return matched_label, matched_tu


