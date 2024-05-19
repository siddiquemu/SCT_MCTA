import numpy as np
import json
import pickle
import os
import sys
import cv2
from PIL import Image
import glob
from collections import OrderedDict
import collections
import matplotlib.pyplot as plt
from read_TU import matchTU, readTU
from tracker_merge import *
import csv

def tracker_mht(trackers_cam,fr_start,fr_end,tracker_min_size):
    mht_trackers = []
    track_id = 0
    trackers_cam = trackers_cam[trackers_cam[:, 0] >= fr_start]
    trackers_cam = trackers_cam[trackers_cam[:, 0] <= fr_end]
    for i in np.unique(trackers_cam[:,1]):
        tracker_i = trackers_cam[np.where(trackers_cam[:,1]==i)][:,[0,2,3,4,5]]#1-id
        if len(tracker_i)>=tracker_min_size:
            print ('PAX Tracker of size {}'.format(len(tracker_i)))
            mht_trackers.append(tracker_i)
            track_id+=1
    return mht_trackers, track_id



def log_files(fr,pax_id,box,cam,travel_unit,type):
    info = {"frame_id":fr,
            "id": pax_id,
            "bbox": box,
            "camera": cam,
            "type": type,
            "events": [],
            "TU": travel_unit
            }
    new_dict = collections.OrderedDict()
    new_dict["frame_id"] = info["frame_id"]
    new_dict["id"] = info["id"]
    new_dict["bbox"] = info["bbox"]
    new_dict["camera"] = info["camera"]
    new_dict["events"] = info["events"]
    new_dict["TU"] = info["TU"]
    new_dict["type"] = info["type"]
    return new_dict

def Write_To_Json(path, filename, data):
    file = path + filename + '.json'
    with open(file,'w') as fp:
        json.dump(data, fp)
def delete_all(demo_path,fmt='png'):
        filelist = glob.glob(os.path.join(demo_path, '*.'+fmt))
        if len(filelist) > 0:
            for f in filelist:
                os.remove(f)

# read multi-camera tracking results for camera 1, 9, 11, 13 and generate log file
if __name__ == '__main__':
    logs = {
        'passengers': []
    }
    tracker = 'mht'
    out_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/test_data/exp1/Results/'
    for cam in ['cam05exp1',]: #'cam09exp1','cam11exp1','cam13exp1',, 'cam02exp2','cam04exp2','cam05exp2']:

        if tracker == 'TCT':
            tracker_file = '/media/siddique/RemoteServer/CLASP2/2019_04_16/exp2/Results/' + cam + \
                           '.mp4/tracking_results/trackers_' + cam + '.npy'
            img_path = '/media/siddique/RemoteServer/CLASP2/2019_04_16/exp2/imgs/' + cam + '.mp4/'
            demo_path = '/media/siddique/RemoteServer/CLASP2/2019_04_16/exp2/Results/' + cam + '.mp4/demo/'#final_demo/wc_demo/'
            trackers_cam1 = np.load(tracker_file, allow_pickle=True)
            trackers = trackers_cam1.item()
        # map consistent id from camera to camera using manual labels or homography projections
        pax = 1
        vis = 1
        sequential = 1
        TU_detect = 0
        tracklet_association = 0
        if TU_detect:
            TU_bbs = readTU('/media/siddique/CLASP2019/dataset_2019/10242019_Exp2Cam1_People_metadata(2).csv')
            lock_TU = {}
        if cam =='cam02exp1':
            #trackers_cam9 = np.loadtxt(out_path + 'cam02exp1.mp4testTracks.txt',dtype='float32',delimiter=',')
            trackers_cam9 = np.loadtxt(
                out_path + 'exp1_tracks_tracktor/cam02exp1.mp4.txt',
                dtype='float32', delimiter=',')
            fr_start = int(trackers_cam9[0, 0])
            fr_end = int(trackers_cam9[-1, 0])
            img_path = out_path.split('Results')[0]+'imgs/' + cam + '.mp4/'
            demo_path = out_path.split('Results')[0]+'Results/' + cam + '.mp4/demov/'
            if not os.path.exists(demo_path):
                os.makedirs(demo_path)
            if os.path.exists(demo_path):
                delete_all(demo_path)
        if cam =='cam04exp1':
            trackers_cam9 = np.loadtxt(
                 out_path + 'cam04exp2.mp4testTracks.txt',
                dtype='float32',
                delimiter=',')
            fr_start = int(trackers_cam9[0, 0])
            fr_end = int(trackers_cam9[-1, 0])
            img_path = out_path.split('Results')[0]+'imgs/' + cam + '.mp4/'
            demo_path = out_path.split('Results')[0]+'Results/' + cam + '.mp4/demo/'
            if os.path.exists(demo_path):
                delete_all(demo_path)
        if cam =='cam05exp1':
            trackers_cam9 = np.loadtxt(
                out_path + 'exp1_tracks_tracktor/cam05exp1.mp4.txt',
                dtype='float32',
                delimiter=',')
            fr_start = int(trackers_cam9[0, 0])
            fr_end = int(trackers_cam9[-1, 0])
            img_path = out_path.split('Results')[0]+'imgs/' + cam + '.mp4/'
            demo_path = out_path.split('Results')[0]+'Results/' + cam + '.mp4/demov/'
            if not os.path.exists(demo_path):
                os.makedirs(demo_path)
            if os.path.exists(demo_path):
                delete_all(demo_path)

        if cam =='cam09exp1':
            #time-stamp: 09.20.20 - 09.20.50
            camera = '9'
            if tracker == 'mht':  # 9-v7
                trackers_cam9 = np.loadtxt(
                     out_path + 'cam09exp1.mp4testTracks.txt', dtype='float32',
                    delimiter=',')
                fr_start = 800#int(trackers_cam9[0,0])
                fr_end = 11300#int(trackers_cam9[-1,0])
                img_path = out_path.split('Results')[0] + 'imgs/' + cam + '.mp4/'
                demo_path = out_path.split('Results')[0] + 'Results/' + cam + '.mp4/demo/'


        if cam =='cam11exp1':
            camera = '11'
            #time-stamp: 09.20.20 - 09.20.50
            if tracker == 'mht':  # 9-v5
                trackers_cam9 = np.loadtxt(
                      out_path + 'cam11exp1.mp4testTracks.txt', dtype='float32',
                    delimiter=',')
                fr_start = 700#int(trackers_cam9[0, 0])
                fr_end = 111400#int(trackers_cam9[-1, 0])
                img_path = out_path.split('Results')[0] + 'imgs/' + cam + '.mp4/'
                demo_path = out_path.split('Results')[0] + 'Results/' + cam + '.mp4/demo/'


        if cam =='cam13exp1':
            if tracker == 'mht':  # 9-v5
                trackers_cam9 = np.loadtxt(
                     out_path + 'cam13exp1.mp4testTracks.txt', dtype='float32',
                    delimiter=',')
                fr_start = 2300#int(trackers_cam9[0,0])
                fr_end = 11900#int(trackers_cam9[-1,0])
                img_path = out_path.split('Results')[0] + 'imgs/' + cam + '.mp4/'
                demo_path = out_path.split('Results')[0] + 'Results/' + cam + '.mp4/demo/'

        if cam =='cam14exp1':
            if tracker == 'mht':  # 9-v5
                trackers_cam9 = np.loadtxt(
                    out_path + 'exp1_tracks_tracktor/cam14exp1.mp4.txt', dtype='float32',
                    delimiter=',')
                fr_start = int(trackers_cam9[0,0])
                fr_end = int(trackers_cam9[-1,0])
                img_path = out_path.split('Results')[0] + 'imgs/' + cam + '.mp4/'
                demo_path = out_path.split('Results')[0] + 'Results/' + cam + '.mp4/demo/'
            if os.path.exists(demo_path):
                delete_all(demo_path)

        # write result as csv file
        PAX_Tracker = open(out_path +cam+ '_logs_full_may14.txt', 'w')
        if not os.path.exists(demo_path):
            os.makedirs(demo_path)

        if tracker=='TCT' and tracklet_association:

            long_tracklets = get_tracklets(trackers)
            short_tracklets = get_tracklets(trackers)
            long_tracklets = filter_tracklets(long_tracklets, min_size=30)
            short_tracklets = filter_tracklets(short_tracklets, min_size=30)

            [long_tracklets, short_tracklets] = merge_tracklets(long_tracklets, short_tracklets)

            filtered_tracker = filter_tracklets(long_tracklets, min_size=30)
            # show_trajectories(fr_start, fr_end, filtered_tracker)
            track_id = len(filtered_tracker)
            print 'Total Number Of Trackers: {} , in Camera: {}'.format(track_id, cam)

        elif tracker=='TCT' and not tracklet_association:
            filtered_tracker = []
            for id in range(len(trackers)):
                tracker_i = trackers.get(str(id+1))
                tracker_i = np.array(list(tracker_i))
                tracker_i = np.array(sorted(tracker_i, key=lambda x: x[0]))
                if (len(tracker_i)>=20): #cam11-100-TCT
                    print ('PAX Tracker of size {}'.format(len(tracker_i)))
                    filtered_tracker.append(tracker_i)
            filtered_tracker = np.array(filtered_tracker)

        # For MHT
        else:
            filtered_tracker, track_id = tracker_mht(trackers_cam9,fr_start,fr_end, tracker_min_size=30) #full - 120
            if tracklet_association:
                trackers = {}
                for tid in range(len(filtered_tracker)):
                    trackers[str(tid+1)]=filtered_tracker[tid]
                long_tracklets = get_tracklets(trackers)
                short_tracklets = get_tracklets(trackers)
                #long_tracklets = filter_tracklets(long_tracklets, min_size=60)
                #short_tracklets = filter_tracklets(short_tracklets, min_size=60)

                [long_tracklets, short_tracklets] = merge_tracklets(long_tracklets, short_tracklets, t_thr=30)

                filtered_tracker = filter_tracklets(long_tracklets, min_size=0)
                track_id = len(filtered_tracker)

        #save tracklets for association
        #np.save('/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results/trackers_mht_'+cam,
                #filtered_tracker, allow_pickle=True, fix_imports=True)

        keep_track_id = dict.fromkeys(range(track_id),[])
        seq_id = 0
        keep_id = []
        for fr in xrange(fr_start,fr_end+1,1):
            print 'frame:',fr
            fr_logs = {
                'frame': []
            }
            img = cv2.imread(img_path + "%06d" % fr + '.png')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            #img = np.zeros((im.shape[0], im.shape[1] + 400, 3), dtype=np.uint8) + 255
            #img[0:1080, 0:1920, :] = im
            # search fr in trackers
            if tracker=='TCT':
                track_id = len(filtered_tracker)
            for k in xrange(0,track_id,1):#

                # Determine PAX is in TU
                if fr in filtered_tracker[k][:,0]: # k+1 belongs to tracker id
                    TU = 0
                    event_type = 'LOC:'
                    type = 'PAX'
                    # Manually set id from camera 9
                    #cam9_ids = ids_from_cam9()
                    track_fr = filtered_tracker[k][np.where(filtered_tracker[k][:,0]==fr)][0]

                    if tracker=='TCT' and cam=='cam09exp1':
                        box = list(track_fr[1:5])
                        pax_id = 'P' + str(int(k+1))
                        first_used = 'false'
                        if int(k+1) not in keep_id:
                            keep_id.append(int(k+1))
                            first_used = 'true'
                        color = (255, 0, 0)
                        if sequential:
                            if str(int(k + 1)) == '15' and fr>=2905:
                                pax_id = 'P21'
                            if str(int(k + 1)) == '31':
                                pax_id = 'P29'
                            if str(int(k + 1)) == '34':
                                pax_id = 'P1'
                            if str(int(k + 1)) == '20':
                                continue
                            if str(int(k + 1)) == '24':
                                continue
                            if str(int(k + 1)) == '32':
                                continue
                            if str(int(k + 1)) == '34':
                                continue


                    elif (tracker == 'mht' and cam == 'cam08exp1'):
                        camera='08'
                        type = 'PAX'
                        box = list(track_fr[2:6])
                        pax_id = 'P' + str(int(k + 1))
                        color = (0, 0, 255)  # y
                        first_used = 'false'
                        if int(k+1) not in keep_id:
                            keep_id.append(int(k+1))
                            first_used = 'true'
                        if sequential:
                            if str(int(k + 1)) == '50':
                                pax_id = 'P1'
                            if str(int(k + 1)) == '52':
                                pax_id = 'P2'
                            if str(int(k + 1)) in ['54', '49']:
                                pax_id = 'P3'
                            if str(int(k + 1)) == '53':
                                pax_id = 'P4'
                            if str(int(k + 1)) in ['40', '43']:
                                pax_id = 'P5'
                            if str(int(k + 1)) == '45':
                                pax_id = 'P6'
                            if str(int(k + 1)) == '44':
                                pax_id = 'P7'
                            if str(int(k + 1)) == '42':
                                pax_id = 'P8'

                            if str(int(k + 1)) == '38':
                                pax_id = 'P9'
                            if str(int(k + 1)) == '38' and fr >= 7905:
                                continue
                            if str(int(k + 1)) == '37' and fr > 7905:
                                pax_id = 'P9'
                            if str(int(k + 1)) == '36':
                                pax_id = 'P10'

                            if str(int(k + 1)) == '29':
                                pax_id = 'P11'
                            if str(int(k + 1)) in ['30', '28', '24', '21', '20', '16']:
                                pax_id = 'P12'
                            if str(int(k + 1)) in ['33', '25', '22', '17', '19']:
                                pax_id = 'P13'

                            if str(int(k + 1)) == '31' and fr < 9135:
                                continue
                            if str(int(k + 1)) == '31' and fr >= 9135:
                                pax_id = 'P13'
                            if str(int(k + 1)) == '33' and fr >= 9135:
                                continue
                            if str(int(k + 1)) == '1' and fr >= 12410:
                                continue
                            if str(int(k + 1)) == '4' and fr >= 12730:
                                continue
                            if str(int(k + 1)) == '17' and fr >= 10520:
                                pax_id = 'P13'
                                continue

                            if str(int(k + 1)) == '14':
                                pax_id = 'P14'
                            if str(int(k + 1)) in ['13', '6']:
                                pax_id = 'P15'
                            if str(int(k + 1)) == '4':
                                pax_id = 'P16'
                            if str(int(k + 1)) == '9':
                                pax_id = 'P17'

                            if str(int(k + 1)) in ['5', '35', '12', '2', '46', '15']:
                                continue

                            if str(int(k + 1)) in ['39', '55', '48', '23', '7', '1']:
                                pax_id = 'TSO1'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['56', '34', '51', '41', '32', '27', '8', '3']:
                                pax_id = 'TSO2'
                                color = (0, 255, 255)

                    elif(tracker=='mht' and cam=='cam13exp1'):
                        camera='13'
                        box = list(track_fr[1:5])
                        pax_id = 'P' + str(int(k+1))
                        color = (0, 0, 255)#y
                        first_appear = 4415
                        first_used = 'false'
                        if int(k+1) not in keep_id:
                            keep_id.append(int(k+1))
                            first_used = 'true'
                        if sequential:
                            if str(int(k+1)) == '33':
                                pax_id = 'P1'
                            if str(int(k+1)) in ['34','31','29']:
                                pax_id = 'P2'
                            if str(int(k+1)) in ['28']:
                                pax_id = 'P3'
                            if str(int(k+1)) == '24':
                                pax_id = 'P4'
                            if str(int(k+1)) in ['25']:
                                pax_id = 'P5'
                            if str(int(k+1)) in ['22']:
                                pax_id = 'P6'
                            if str(int(k+1)) == '19':
                                pax_id = 'P7'
                            if str(int(k+1)) in ['21']:
                                pax_id = 'P8'
                            if str(int(k+1)) == '17':
                                pax_id = 'P9'
                            if str(int(k+1)) == '10':
                                pax_id = 'P10'
                            if str(int(k+1)) in ['15','16']:
                                pax_id = 'P11'
                            if str(int(k+1)) in ['13']:
                                pax_id = 'P12'
                            if str(int(k+1)) in ['40']:
                                pax_id = 'P13'
                            if str(int(k + 1)) in ['8']:
                                pax_id = 'P14'

                            if str(int(k + 1)) in ['7']:
                                pax_id = 'P15'
                            if str(int(k + 1)) in ['4','3']:
                                pax_id = 'P16'
                            if str(int(k + 1)) == '2':
                                pax_id = 'P17'


                            if str(int(k + 1)) in ['42', '27']:
                                pax_id = 'TSO5'
                                color = (0, 255, 255)
                                type = 'TSO'
                            if str(int(k + 1)) in ['26', '20','32', '18', '14', '43', '9', '1']:
                                type='TSO'
                                pax_id = 'TSO4'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['41']:
                                continue

                    elif(tracker=='mht' and cam=='cam01exp1'):
                        camera = '1'
                        box = list(track_fr[2:6])
                        pax_id = 'P' + str(int(k+1))
                        color = (0, 0, 255)
                        first_used = 'false'
                        if int(k+1) not in keep_id:
                            keep_id.append(int(k+1))
                            first_used = 'true'
                        if sequential:

                            if str(int(k+1)) == '13':
                                pax_id = 'P1'
                            if str(int(k+1)) == '9':
                                pax_id = 'P2'
                            if str(int(k+1)) == '14':
                                pax_id = 'P3'
                            if str(int(k+1)) == '10':
                                pax_id = 'P3'
                            if str(int(k+1)) == '12':
                                pax_id = 'P4'
                            if str(int(k+1)) == '11':
                                pax_id = 'P4'
                            if str(int(k+1)) == '5':
                                pax_id = 'P5'
                            if str(int(k+1)) == '6':
                                pax_id = 'P6'
                            if str(int(k+1)) == '4':
                                pax_id = 'P6'
                            if str(int(k+1)) == '3':
                                pax_id = 'P7'
                            if str(int(k+1)) == '16':
                                pax_id = 'P8'
                            if str(int(k+1)) == '17':
                                continue
                            if str(int(k+1)) == '7':
                                continue
                            if str(int(k+1)) == '1':
                                continue
                            if str(int(k+1)) == '2':
                                continue
                            if str(int(k+1)) == '15':
                                pax_id = 'TSO1'
                                color = (0, 255, 255)
                            if str(int(k+1)) == '8':
                                pax_id = 'TSO1'
                                color = (0, 255, 255)

                    elif(tracker=='mht' and cam=='cam09exp1'):
                        box = list(track_fr[1:5])
                        pax_id = 'P' + str(int(k+1))
                        color = (0, 0, 255)
                        first_used = 'false'
                        if int(k+1) not in keep_id:
                            keep_id.append(int(k+1))
                            first_used = 'true'
                        if sequential:

                            if str(int(k+1)) in['35','34']:
                                pax_id = 'P1'
                            if str(int(k+1)) == '33':
                                pax_id = 'P2'
                            if str(int(k+1)) == '32':
                                pax_id = 'P3'
                            if str(int(k+1)) == '30':
                                pax_id = 'P4'
                            if  str(int(k+1)) == '29':
                                pax_id = 'P5'
                            if str(int(k+1)) == '28':
                                pax_id = 'P6'
                            if str(int(k+1)) == '26':
                                pax_id = 'P7'
                            if str(int(k+1)) == '24':
                                pax_id = 'P8'
                            if str(int(k+1)) in ['27','16']:
                                pax_id = 'P9'
                            if str(int(k+1)) == '17':
                                pax_id = 'P10'
                            if str(int(k + 1)) == '18':
                                pax_id = 'P11'
                            if str(int(k+1)) in ['20','14']:
                                pax_id = 'P12'
                            if str(int(k + 1)) in ['9']:
                                pax_id = 'P13'
                            if str(int(k + 1)) in ['11','10','6']:
                                pax_id = 'P14'
                            if str(int(k + 1)) == '7':
                                pax_id = 'P15'
                            if str(int(k + 1)) == '3':
                                pax_id = 'P16'
                            if str(int(k + 1)) == '2':
                                pax_id = 'P17'

                            if str(int(k+1)) in ['31','25','22','15','8','5','1']:
                                pax_id = 'TSO1'
                                type = 'TSO'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['23', '21', '19', '13', '12', '4']:
                                continue
                            if fr>=8230 and str(int(k+1)) in ['6']:
                                pax_id = 'P15'
                            if fr>=8230 and str(int(k+1)) in ['7']:
                                pax_id = 'P14'

                    elif(tracker=='mht' and cam=='cam05exp1'):
                        camera='05'
                        box = list(track_fr[1:5])
                        pax_id = 'P' + str(int(k+1))
                        color = (0, 0, 255)#y
                        first_appear = 4415
                        first_used = 'false'
                        if int(k+1) not in keep_id:
                            keep_id.append(int(k+1))
                            first_used = 'true'
                        if sequential:
                            if str(int(k+1)) in ['15','10','17']:
                                pax_id = 'P1'
                            if str(int(k+1)) in ['11']:
                                pax_id = 'P2'
                            if str(int(k+1)) in ['14']:
                                pax_id = 'P3'
                            if str(int(k+1)) in ['16']:
                                pax_id = 'P4'
                            if str(int(k+1)) in ['21']:
                                pax_id = 'P5'
                            if str(int(k+1)) in ['29','24']:
                                pax_id = 'P6'
                            if str(int(k+1)) in ['25']:
                                pax_id = 'P7'
                            if str(int(k+1)) in ['27']:
                                pax_id = 'P8'

                            if str(int(k+1)) in ['30']:
                                pax_id = 'P9'

                            if str(int(k+1)) in ['32','37']:
                                pax_id = 'P10'

                            if str(int(k+1)) in ['33','39','41']:
                                pax_id = 'P11'
                            if str(int(k+1)) in ['36','40']:
                                pax_id = 'P12'
                            if str(int(k+1)) in ['44']:
                                pax_id = 'P13'

                            if str(int(k + 1)) in ['46','50']:
                                pax_id = 'P14'
                            if str(int(k + 1)) in ['48','53','51']:
                                pax_id = 'P15'
                            if str(int(k + 1)) in ['55','58']:
                                pax_id = 'P16'
                            if str(int(k + 1)) in ['57','59']:
                                pax_id = 'P17'

                            if str(int(k + 1)) in ['4', '5', '6','7','8','9','13','18','20','22','23','28','31','34','38','42','45','60','52']:
                                pax_id = 'TSO5'
                                type = 'TSO'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['3']:
                                pax_id = 'TSO1'
                                type = 'TSO'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['1','61']:
                                pax_id = 'TSO2'
                                type = 'TSO'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['2','35','56']:
                                pax_id = 'TSO3'
                                type = 'TSO'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['19', '26', '43','47','49']:
                                continue
                    elif (tracker == 'mht' and cam == 'cam04exp1'):
                        camera = '05'
                        box = list(track_fr[1:5])
                        pax_id = 'P' + str(int(k + 1))
                        color = (0, 0, 255)  # y
                        first_appear = 4415
                        first_used = 'false'
                        if int(k + 1) not in keep_id:
                            keep_id.append(int(k + 1))
                            first_used = 'true'
                        if sequential:
                            if str(int(k + 1)) == '54':
                                pax_id = 'P1'
                            if str(int(k + 1)) == '56':
                                pax_id = 'TU1'  # 'P2'
                            if str(int(k + 1)) in ['58', '53']:
                                pax_id = 'TU1'  # 'P3'
                            if str(int(k + 1)) == '57':
                                pax_id = 'TU1'  # 'P4'
                            if str(int(k + 1)) in ['40', '43']:
                                pax_id = 'P5'
                            if str(int(k + 1)) == '45':
                                pax_id = 'P6'
                            if str(int(k + 1)) == '44':
                                pax_id = 'P7'
                            if str(int(k + 1)) == '42':
                                pax_id = 'P8'

                            if str(int(k + 1)) == '38':
                                pax_id = 'P9'
                            if str(int(k + 1)) == '38' and fr >= 7905:
                                continue
                            if str(int(k + 1)) == '37' and fr > 7905:
                                pax_id = 'P9'
                            if str(int(k + 1)) == '36':
                                pax_id = 'P10'

                            if str(int(k + 1)) == '29':
                                pax_id = 'P11'
                            if str(int(k + 1)) in ['30', '28', '24', '21', '20', '16']:
                                pax_id = 'P12'
                            if str(int(k + 1)) in ['33', '25', '22', '17', '19']:
                                pax_id = 'P13'

                            if str(int(k + 1)) == '31' and fr < 9135:
                                continue
                            if str(int(k + 1)) == '31' and fr >= 9135:
                                pax_id = 'P13'
                            if str(int(k + 1)) == '33' and fr >= 9135:
                                continue
                            if str(int(k + 1)) == '1' and fr >= 12410:
                                continue
                            if str(int(k + 1)) == '4' and fr >= 12730:
                                continue
                            if str(int(k + 1)) == '17' and fr >= 10520:
                                pax_id = 'P13'
                                continue

                            if str(int(k + 1)) == '14':
                                pax_id = 'P14'
                            if str(int(k + 1)) in ['13', '6']:
                                pax_id = 'P15'
                            if str(int(k + 1)) == '4':
                                pax_id = 'P16'
                            if str(int(k + 1)) == '9':
                                pax_id = 'P17'

                            if str(int(k + 1)) in ['5', '35', '12', '2', '46', '15']:
                                continue

                            if str(int(k + 1)) in ['39', '55', '48', '23', '7', '1']:
                                pax_id = 'TSO5'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['60', '34', '51', '41', '32', '27', '8', '3']:
                                pax_id = 'TSO4'
                                color = (0, 255, 255)

                    elif (tracker == 'mht' and cam == 'cam02exp1'):
                        camera = '02'
                        box = list(track_fr[1:5])
                        pax_id = 'P' + str(int(k+1))
                        color = (0, 0, 255)#y
                        first_appear = 4415
                        first_used = 'false'
                        if int(k+1) not in keep_id:
                            keep_id.append(int(k+1))
                            first_used = 'true'
                        if sequential:
                            if str(int(k+1)) == '5':
                                pax_id = 'P1'
                            if str(int(k+1)) in ['7','10']:
                                pax_id = 'P2'
                            if str(int(k+1)) in ['9','11']:
                                pax_id = 'P3'
                            if str(int(k+1)) in ['12','16']:
                                pax_id = 'P4'
                            if str(int(k+1)) in ['13','17']:
                                pax_id = 'P5'
                            if str(int(k+1)) in ['14','19','21']:
                                pax_id = 'P6'
                            if str(int(k+1)) in ['18','23']:
                                pax_id = 'P7'
                            if str(int(k+1)) in ['20','27','31']:
                                pax_id = 'P8'

                            if str(int(k+1)) in ['25','28','37']:
                                pax_id = 'P9'

                            if str(int(k+1)) in ['24','29','38']:
                                pax_id = 'P10'

                            if str(int(k+1)) in ['32','34','40','42','44','49']:
                                pax_id = 'P11'
                            if str(int(k+1)) in ['36','41','45']:
                                pax_id = 'P12'
                            if str(int(k+1)) in ['33','50']:
                                pax_id = 'P13'

                            if str(int(k + 1)) in ['46','51','53','56','57']:
                                pax_id = 'P14'
                            if str(int(k + 1)) in ['52','55','58']:
                                pax_id = 'P15'
                            if str(int(k + 1)) in ['54','59','60','62','63','64']:
                                pax_id = 'P16'
                            if str(int(k + 1)) in ['61','65']:
                                pax_id = 'P17'

                            if str(int(k + 1)) in ['6', '66', '67']:
                                pax_id = 'TSO5'
                                type = 'TSO'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['2', '8', '35', '39']:
                                pax_id = 'TSO1'
                                type = 'TSO'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['3']:
                                pax_id = 'TSO2'
                                type = 'TSO'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['1','4', '15', '48']:
                                continue
                    elif(tracker=='mht' and cam=='cam11exp1'):
                        camera = '11'
                        box = list(track_fr[1:5])
                        '''
                        if k not in keep_track_id[k]:
                            seq_id += 1
                            keep_track_id[k].append(k)
                        pax_id = 'P' + str(keep_track_id[k][0])
                        '''
                        pax_id = 'P' + str(int(k+1))
                        color = (0, 0, 255)
                        type = 'PAX'
                        first_appear = 4480#TU1
                        first_used = 'false'
                        if int(k+1) not in keep_id:
                            keep_id.append(int(k+1))
                            first_used = 'true'
                        if sequential:
                            if str(int(k+1)) in ['37','36']:
                                pax_id = 'P1'
                            if str(int(k+1)) == '32':
                                pax_id = 'P3'
                            if str(int(k+1)) == '29':
                                pax_id = 'P4'
                            if str(int(k+1)) in ['35']:
                                pax_id = 'P2'
                            if str(int(k+1)) in ['26']:
                                pax_id = 'P5'
                            if str(int(k+1)) in ['28','24','18']:
                                pax_id = 'P6'
                            if str(int(k+1)) in ['22']:
                                pax_id = 'P7'

                            if str(int(k + 1)) in ['21']:
                                pax_id = 'P8'
                            if str(int(k + 1)) == '17':
                                pax_id = 'P9'
                            if str(int(k + 1)) == '19':
                                pax_id = 'P10'
                            if str(int(k+1)) == '14':
                                pax_id = 'P11'

                            if str(int(k + 1)) == '16':
                                pax_id = 'P12'
                            if str(int(k + 1)) == '11':
                                pax_id = 'P13'
                            if str(int(k + 1)) in ['9']:
                                pax_id = 'P14'
                            if str(int(k + 1)) == '8':
                                pax_id = 'P15'
                            if str(int(k + 1)) == '3':
                                pax_id = 'P16'
                            if str(int(k + 1)) == '4':
                                pax_id = 'P17'

                            if str(int(k+1)) in ['5','7','13','2','38','30','25']:
                                pax_id = 'TSO3'
                                type = 'TSO'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['39', '34', '31', '27', '20', '6','1']:
                                pax_id = 'TSO4'
                                type = 'TSO'
                                color = (0, 255, 255)

                            if str(int(k + 1)) in ['42']:
                                pax_id = 'TSO1'
                                type = 'TSO'
                                color = (0, 255, 255)

                            if str(int(k + 1)) in ['33', '25']:
                                continue

                    #tracker info
                    #if not TU_detect and sequential:
                        #pax_id = id

                    elif (tracker == 'mht' and cam == 'cam14exp1'):
                        camera = '11'
                        box = list(track_fr[1:5])
                        '''
                        if k not in keep_track_id[k]:
                            seq_id += 1
                            keep_track_id[k].append(k)
                        pax_id = 'P' + str(keep_track_id[k][0])
                        '''
                        pax_id = 'P' + str(int(k + 1))
                        color = (0, 0, 255)
                        type = 'PAX'
                        first_appear = 4480  # TU1
                        first_used = 'false'
                        if int(k + 1) not in keep_id:
                            keep_id.append(int(k + 1))
                            first_used = 'true'
                        if sequential:
                            if str(int(k + 1)) in ['56', '58']:
                                pax_id = 'P1'
                            if str(int(k + 1)) == '48':
                                pax_id = 'TU1'  # 'P3'
                            if str(int(k + 1)) == '51':
                                pax_id = 'TU1'  # 'P4'
                            if str(int(k + 1)) in ['52', '49']:
                                pax_id = 'TU1'  # 'P2'
                            if str(int(k + 1)) == '37':
                                pax_id = 'P5'

                            if str(int(k + 1)) == '35' and fr >= 6625:
                                pax_id = 'P5'

                            if str(int(k + 1)) == '35' or str(int(k + 1)) == '33':
                                pax_id = 'P6'
                            if str(int(k + 1)) == '31':
                                pax_id = 'P8'
                            if str(int(k + 1)) == '25':
                                pax_id = 'P9'
                            if str(int(k + 1)) == '27' or str(int(k + 1)) == '24' or str(int(k + 1)) == '22':
                                pax_id = 'P10'
                            if str(int(k + 1)) == '23' and fr >= 8475:
                                pax_id = 'P11'

                            if str(int(k + 1)) == '18':
                                pax_id = 'P12'
                            if str(int(k + 1)) == '15':
                                pax_id = 'P13'
                            if str(int(k + 1)) == '14' or str(int(k + 1)) == '13':
                                pax_id = 'P14'
                            if str(int(k + 1)) == '6':
                                pax_id = 'P15'
                            if str(int(k + 1)) == '5' or str(int(k + 1)) == '3':
                                pax_id = 'P16'
                            if str(int(k + 1)) == '4':
                                pax_id = 'P17'

                            if str(int(k + 1)) in ['57', '53', '45', '50', '38', '29', '26', '2', '23', '21', '20',
                                                   '16', '9']:
                                pax_id = 'TSO3'
                                type = 'TSO'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['1', '36', '39']:
                                pax_id = 'TSO4'
                                type = 'TSO'
                                color = (0, 255, 255)

                            if str(int(k + 1)) in ['54', '19', '10', '7', '47']:
                                pax_id = 'TSO5'
                                type = 'TSO'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['32', '17', '8']:
                                continue

                                # tracker info
                                # if not TU_detect and sequential:
                                # pax_id = id
                    #frame_logs = log_files(fr,pax_id, box, cam, TU, event_type)
                    #fr_logs['frame'].append(frame_logs)
                    print 'PAX: ', pax_id
                    #for demo
                    if pax_id =='TU1':
                        color = (255,255,0)
                    if pax_id not in ['TU1', 'TU2', 'TU3']:
                        try:
                            id = pax_id.split('P')[1]
                        except:
                            id = pax_id.split('TSO')[1]

                    PAX_Tracker.writelines(event_type+' '+'type: '+type+' '+ 'camera-num: '+camera+ ' '+ 'frame: '+str(fr)+' '
                                           'time-offset: '+'{:.2f}'.format(fr/30.0)+' '+'BB: '+str(int(box[0]))+', '+str(int(box[1]))+', '+str(int(box[0]+box[2]))
                                           +', '+str(int(box[1]+box[3]))+' '+'ID: '+pax_id+' '+'PAX-ID: '+pax_id
                                           +' '+'first-used: '+first_used+' '+'partial-complete: '+'description: '+'\n')

                    if fr>0:
                        if vis and fr%10==0:

                            img = cv2.rectangle(img, (np.int(box[0]), np.int(box[1])), (np.int(box[0]+box[2]),
                                                                                        np.int(box[1]+box[3])), color, 4)
                            img = cv2.putText(img, pax_id, (int(box[0]+box[2]/2),np.int(box[1]+box[3]/2)), cv2.FONT_HERSHEY_SIMPLEX,
                                                2, (255,0,0), 2, cv2.LINE_AA)
                            if pax_id=='TU1':
                                img = cv2.putText(img, pax_id+':'+' Travel time ' + str((fr - first_appear) / 30), (1920 + 50, 0+50),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            logs['passengers'].append(fr_logs)
            if fr>0:
                if vis and fr % 10 == 0:
                    print (cam)
                    img = cv2.resize(img,(960,540))
                    plt.imsave(demo_path + "%06d" % fr + '.png', img, dpi=200)