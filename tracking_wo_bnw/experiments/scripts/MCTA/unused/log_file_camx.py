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
    out_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results_train/'
    for cam in ['cam02exp2']: #,'cam11exp2','cam13exp2','cam14exp2', 'cam02exp2','cam04exp2','cam05exp2']:

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
        sequential = 0
        TU_detect = 0
        tracklet_association = 0
        auto_sequential = 0
        if TU_detect:
            TU_bbs = readTU('/media/siddique/CLASP2019/dataset_2019/10242019_Exp2Cam1_People_metadata(2).csv')
            lock_TU = {}
        if cam =='cam01exp1':
            #time-stamp: 09.20.20 - 09.20.50
            fr_start = 2321#2321#6228
            fr_end = 2850#6650
        if cam =='cam01exp2':
            #time-stamp: 09.20.20 - 09.20.50
            fr_start = 2000#6230#2321#6228#child-2321
            fr_end = 13502#6645#6650#child-2850
            if tracker == 'mht':  # 9-v7
                trackers_cam9 = np.loadtxt(
                    '/home/siddique/Desktop/NEU_Data/TrackResult/' + cam + 'PersonMASK_30FPSrot_augv1.txt', dtype='float32',
                    delimiter=',')
                img_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/' + cam + '.mp4/'
                demo_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results/' + cam + '.mp4/demo/'
        if cam =='cam02exp2':
            trackers_cam9 = np.loadtxt('/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results_train/tracktor_trained/cam02exp2.mp4.txt', dtype='float32', delimiter=',')
            fr_start = int(trackers_cam9[0, 0])
            fr_end = int(trackers_cam9[-1, 0])
            #fr_start = 1340
            #fr_end = 12190  #

            img_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/' + cam + '.mp4/'
            demo_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results/' + cam + '.mp4/demov/'
            if os.path.exists(demo_path):
                delete_all(demo_path)
        if cam =='cam04exp2':
            trackers_cam9 = np.loadtxt(
                '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results_train/exp2-train-track/cam04exp2.mp4testTracksv2.txt',
                dtype='float32',
                delimiter=',')
            fr_start = int(trackers_cam9[0, 0])
            fr_end = int(trackers_cam9[-1, 0])
            img_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/' + cam + '.mp4/'
            demo_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results/' + cam + '.mp4/demo/'
            if os.path.exists(demo_path):
                delete_all(demo_path)
        if cam =='cam05exp2':
            #fr_start = 200
            #fr_end = 12990
            trackers_cam9 = np.loadtxt(
                '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results_train/exp2-train-track/cam05exp2.mp4testTracksv2.txt',
                dtype='float32',
                delimiter=',')
            fr_start = int(trackers_cam9[0, 0])
            fr_end = int(trackers_cam9[-1, 0])
            img_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/' + cam + '.mp4/'
            demo_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results/' + cam + '.mp4/demov/'
            if os.path.exists(demo_path):
                delete_all(demo_path)

        if cam =='cam08exp2':
            #time-stamp: 09.20.20 - 09.20.50
            if tracker == 'mht':  # 9-v7
                trackers_cam9 = np.loadtxt(
                    '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results_train/exp2-train-track/cam08exp2.mp4testTracksv2.txt', dtype='float32',
                    delimiter=',')
                fr_start = 2000#int(trackers_cam9[0, 0])
                fr_end = int(trackers_cam9[-1, 0])
                img_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/' + cam + '.mp4/'
                demo_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results/' + cam + '.mp4/demo/'
                if os.path.exists(demo_path):
                    delete_all(demo_path)
        if cam =='cam09exp2':
            #time-stamp: 09.20.20 - 09.20.50
            camera = '9'
            if tracker == 'mht':  # 9-v7
                trackers_cam9 = np.loadtxt(
                    '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results_train/exp2-train-track/cam09exp2.mp4testTracks.txt', dtype='float32',
                    delimiter=',')
                fr_start = 1560#int(trackers_cam9[0,0])
                fr_end = 12920#int(trackers_cam9[-1,0])
                img_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/' + cam + '.mp4/'
                demo_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results/' + cam + '.mp4/demo/'


        if cam =='cam11exp2':
            camera = '11'
            #time-stamp: 09.20.20 - 09.20.50
            if tracker == 'mht':  # 9-v5
                trackers_cam9 = np.loadtxt(
                     '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results_train/exp2-train-track/cam11exp2.mp4testTracks.txt', dtype='float32',
                    delimiter=',')
                fr_start =1560# int(trackers_cam9[0, 0])
                fr_end = 13200#int(trackers_cam9[-1, 0])
                img_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/' + cam + '.mp4/'
                demo_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results/' + cam + '.mp4/demo/'


        if cam =='cam13exp2':
            if tracker == 'mht':  # 9-v5
                trackers_cam9 = np.loadtxt(
                    '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results_train/exp2-train-track/cam13exp2.mp4testTracks.txt', dtype='float32',
                    delimiter=',')
                fr_start = 4900#int(trackers_cam9[0,0])
                fr_end = 13320#int(trackers_cam9[-1,0])
                img_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/' + cam + '.mp4/'
                demo_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results/' + cam + '.mp4/demo/'

        if cam =='cam14exp2':
            if tracker == 'mht':  # 9-v5
                trackers_cam9 = np.loadtxt(
                    '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results_train/exp2-train-track/cam14exp2.mp4testTracks.txt', dtype='float32',
                    delimiter=',')
                fr_start = 4000#int(trackers_cam9[0,0])
                fr_end = int(trackers_cam9[-1,0])
                img_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/' + cam + '.mp4/'
                demo_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results/' + cam + '.mp4/demo/'

        if cam =='cam20exp2':
            #time-stamp: 09.20.20 - 09.20.50
            if tracker == 'mht':  # 9-v7
                trackers_cam9 = np.loadtxt(
                      '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results_train/exp2-train-track/cam20exp2.mp4testTracksv2.txt', dtype='float32',
                    delimiter=',')
                fr_start = int(trackers_cam9[0, 0])
                fr_end = int(trackers_cam9[-1, 0])
                img_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/' + cam + '.mp4/'
                demo_path = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results/' + cam + '.mp4/demo/'
                if os.path.exists(demo_path):
                    delete_all(demo_path)

        if cam =='cam22exp2':
            #time-stamp: 09.20.20 - 09.20.50
            fr_start = 1500
            fr_end = 9000
        if cam =='cam18exp2':
            #time-stamp: 09.20.20 - 09.20.50
            fr_start = 1
            fr_end = 9300
        # write result as csv file
        PAX_Tracker = open(out_path +cam+ '_logs_full.txt', 'w')
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
            filtered_tracker, track_id = tracker_mht(trackers_cam9,fr_start,fr_end, tracker_min_size=120) #full - 120, cam05-30
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
        seq_id_map = {}
        fr_start = 1340
        fr_end = 12190  #
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

                    if tracker=='TCT' and cam=='cam09exp2':
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


                    elif (tracker == 'mht' and cam == 'cam08exp2'):
                        camera='08'
                        type = 'PAX'
                        box = list(track_fr[1:5])
                        pax_id = 'P' + str(int(k + 1))
                        color = (0, 0, 255)  # y
                        first_used = 'false'
                        if int(k+1) not in keep_id:
                            keep_id.append(int(k+1))
                            first_used = 'true'
                        if sequential:
                            if str(int(k + 1)) in ['55','52']:
                                pax_id = 'P1'
                            if str(int(k + 1)) in ['56','49','47']:
                                pax_id = 'TU1'#'P2'
                            if str(int(k + 1)) in ['54', '53']:
                                pax_id = 'TU1'#'P3'
                            if str(int(k + 1)) == '48':
                                pax_id = 'TU1'#'P4'
                            if str(int(k + 1)) in ['41']:
                                pax_id = 'P5'
                            if str(int(k + 1)) in ['44','45','34']:
                                pax_id = 'P6'
                            if str(int(k + 1)) in ['46','38','39']:
                                pax_id = 'TU2'#'P7'
                            if str(int(k + 1)) in ['43','37','33','31']:
                                pax_id = 'TU2'#'P8'

                            if str(int(k + 1)) in ['35','18']:
                                pax_id = 'P9'
                            if str(int(k + 1)) == '27':
                                pax_id = 'P10'

                            if str(int(k + 1)) in ['26','24']:
                                pax_id = 'P11'
                            if str(int(k + 1)) in ['12']:
                                pax_id = 'P12'
                            if str(int(k + 1)) in ['21']:
                                pax_id = 'TU3'#'P13'

                            if str(int(k + 1)) == '40' and fr >= 4150:
                                continue
                            if str(int(k + 1)) == '52' and fr >= 3460:
                                pax_id = 'TU1'#'P2'
                                continue
                            if str(int(k + 1)) == '53' and fr >= 3460:
                                continue

                            if str(int(k + 1)) in ['23','17','15']:
                                pax_id = 'TU3'#'P14'
                            if str(int(k + 1)) in ['19', '16','13']:
                                pax_id = 'TU3'#'P15'

                            if str(int(k + 1)) == '10':
                                pax_id = 'P16'
                            if str(int(k + 1)) == '9':
                                pax_id = 'P17'
                            if str(int(k + 1)) == '8':
                                pax_id = 'P18'

                            if str(int(k + 1)) in ['6', '14', '22', '58', '50','42','30','25']:
                                continue

                            if str(int(k + 1)) in ['4']:
                                type = 'TSO'
                                pax_id = 'TSO1'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['51', '40', '36', '33', '29', '11', '5']:
                                type = 'TSO'
                                pax_id = 'TSO2'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['12'] and fr>=7700:
                                type = 'TSO'
                                pax_id = 'TSO2'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['18'] and fr>=6250:
                                type = 'TSO'
                                pax_id = 'TSO2'
                                color = (0, 255, 255)

                    elif(tracker=='mht' and cam=='cam13exp2'):
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
                            if str(int(k+1)) == '40':
                                pax_id = 'P1'
                            if str(int(k+1)) == '41':
                                pax_id = 'P2'
                            if str(int(k+1)) in ['43']:
                                pax_id = 'P3'
                            if str(int(k+1)) == '45':
                                pax_id = 'P4'
                            if str(int(k+1)) in ['35']:
                                pax_id = 'P5'
                            if str(int(k+1)) in ['38','34']:
                                pax_id = 'P6'
                            if str(int(k+1)) == '37':
                                pax_id = 'P7'
                            if str(int(k+1)) in ['36','39']:
                                pax_id = 'P8'
                            if str(int(k+1)) == '30':
                                pax_id = 'P9'
                            if str(int(k+1)) == '26':
                                pax_id = 'P10'
                            if str(int(k+1)) == '25':
                                pax_id = 'P11'
                            if str(int(k+1)) in ['22']:
                                pax_id = 'P12'
                            if str(int(k+1)) in ['14','21','16']:
                                pax_id = 'P13'
                            if str(int(k + 1)) in ['18', '17', '15']:
                                pax_id = 'P14'

                            if str(int(k + 1)) in ['13']:
                                pax_id = 'P15'
                            if str(int(k + 1)) == '7':
                                pax_id = 'P16'
                            if str(int(k + 1)) == '10':
                                pax_id = 'P17'
                            if str(int(k + 1)) == '3':
                                pax_id = 'P18'


                            if str(int(k + 1)) in ['46', '28', '19', '1', '2']:
                                pax_id = 'TSO5'
                                color = (0, 255, 255)
                                type = 'TSO'
                            if str(int(k + 1)) in ['44', '42', '29', '23', '20', '6', '5']:
                                type='TSO'
                                pax_id = 'TSO4'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['9']:
                                type='TSO'
                                pax_id = 'TSO4'
                                color = (0, 255, 255)


                    elif(tracker=='mht' and cam=='cam01exp2'):
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

                    elif(tracker=='mht' and cam=='cam09exp2'):
                        box = list(track_fr[1:5])
                        pax_id = 'P' + str(int(k+1))
                        color = (0, 0, 255)
                        first_used = 'false'
                        if int(k+1) not in seq_id_map.keys():
                            seq_id+=1
                            seq_id_map[int(k+1)] = 'P' + str(seq_id)

                        if int(k+1) not in keep_id:
                            keep_id.append(int(k+1))
                            first_used = 'true'

                        if auto_sequential:
                            pax_id = seq_id_map[int(k+1)]
                        if sequential:

                            if str(int(k+1)) == '26':
                                pax_id = 'P1'
                            if str(int(k+1)) == '23':
                                pax_id = 'P2'
                            if str(int(k+1)) == '25':
                                pax_id = 'P3'
                            if str(int(k+1)) == '24':
                                pax_id = 'P4'
                            if  str(int(k+1)) == '20':
                                pax_id = 'P5'
                            if str(int(k+1)) == '21' or str(int(k+1)) == '18' or str(int(k+1)) == '16':
                                pax_id = 'P6'
                            if str(int(k+1)) == '19':
                                pax_id = 'P7'
                            if str(int(k+1)) == '17':
                                pax_id = 'P8'
                            if str(int(k+1)) == '15':
                                pax_id = 'P9'
                            if str(int(k+1)) == '14':
                                pax_id = 'P10'
                            if str(int(k + 1)) == '12':
                                pax_id = 'P11'
                            if str(int(k+1)) == '10':
                                pax_id = 'P12'
                            if str(int(k + 1)) in ['7','11']:
                                pax_id = 'P13'
                            if str(int(k + 1)) == '8':
                                pax_id = 'P14'
                            if str(int(k + 1)) == '6':
                                pax_id = 'P15'
                            if str(int(k + 1)) == '4':
                                pax_id = 'P16'
                            if str(int(k + 1)) == '3':
                                pax_id = 'P17'
                            if str(int(k + 1)) == '2':
                                pax_id = 'P18'

                            if str(int(k+1)) in ['38','27','22','13','9','5','1']:
                                pax_id = 'TSO2'
                                type = 'TSO'
                                color = (0, 255, 255)

                    elif(tracker=='mht' and cam=='cam05exp2'):
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
                            if str(int(k + 1)) in ['73']:
                                pax_id = 'P1'
                            if str(int(k + 1)) in ['72','69']:
                                pax_id =  'P2'
                            if str(int(k + 1)) in ['80', '76', '74','68']:
                                pax_id =  'P3'
                            if str(int(k + 1)) in ['81','78','70']:
                                pax_id =  'P4'
                            if str(int(k + 1)) in ['63', '57']:
                                pax_id = 'P5'
                            if str(int(k + 1)) in ['59', '54']:
                                pax_id = 'P6'
                            if str(int(k + 1)) in ['64', '61']:
                                pax_id = 'P7'
                            if str(int(k + 1)) in ['66', '62', '58']:
                                pax_id = 'P8'

                            if str(int(k + 1)) in ['55', '51','49']:
                                pax_id = 'P9'
                            if str(int(k + 1)) in ['2','86','82','79','65','60','48','44','42','29']:
                                continue
                            if str(int(k + 1)) in ['43']:
                                pax_id = 'P10'

                            if str(int(k + 1)) in ['46', '45', '40']:
                                pax_id = 'P11'
                            if str(int(k + 1)) in ['41', '37']:
                                pax_id = 'P12'
                            if str(int(k + 1)) in ['38', '31', '26', '25']:
                                pax_id = 'P13'

                            if str(int(k + 1)) in ['35','27']:
                                pax_id = 'P14'
                            if str(int(k + 1)) in ['33','28','24']:
                                pax_id = 'P15'
                            if str(int(k + 1)) in ['23','12']:
                                pax_id = 'P16'
                            if str(int(k + 1)) in ['19','16','13','10']:
                                pax_id = 'P17'
                            if str(int(k + 1)) in ['18','9']:
                                pax_id = 'P18'

                            if str(int(k + 1)) in ['85', '75', '56', '39', '20', '17','15','3','92']:
                                pax_id = 'TSO3'
                                type = 'TSO'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['87', '84', '83', '71', '67', '53', '52','50','47','36','34','21','7']:
                                pax_id = 'TSO5'
                                type = 'TSO'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['87', '84', '83', '71', '67', '53', '52','50','47','36','34','21','7']:
                                pax_id = 'TSO5'
                                type = 'TSO'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['90', '89', '58', '8', '6', '4']:
                                pax_id = 'TSO2'
                                type = 'TSO'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['91']:
                                pax_id = 'TSO1'
                                type = 'TSO'
                                color = (0, 255, 255)

                            if str(int(k + 1)) == '20' and fr >= 10890:
                                pax_id = 'P16'

                    elif (tracker == 'mht' and cam == 'cam04exp2'):
                        camera = '04'
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

                    elif (tracker == 'mht' and cam == 'cam02exp2'):
                        camera = '02'
                        box = list(track_fr[1:5])
                        pax_id = 'P' + str(int(k + 1))
                        color = (0, 0, 255)  # y
                        first_appear = 4415
                        first_used = 'false'
                        if int(k + 1) not in keep_id:
                            keep_id.append(int(k + 1))
                            first_used = 'true'
                        if sequential:
                            if str(int(k + 1)) in ['71','66']:
                                pax_id = 'P1'
                            if str(int(k + 1)) in ['73','53']:
                                pax_id =  'P2'
                            if str(int(k + 1)) in ['67', '65', '60']:
                                pax_id =  'P3'
                            if str(int(k + 1)) in ['68','64','57','55']:
                                pax_id =  'P4'
                            if str(int(k + 1)) in ['63', '48', '44']:
                                pax_id = 'P5'
                            if str(int(k + 1)) in ['61', '56', '54','51','42','40']:
                                pax_id = 'P6'
                            if str(int(k + 1)) in ['50', '49']:
                                pax_id = 'P7'
                            if str(int(k + 1)) in ['52', '45']:
                                pax_id = 'P8'

                            if str(int(k + 1)) in ['43', '38']:
                                pax_id = 'P9'
                            if str(int(k + 1)) in ['2']:
                                continue
                            if str(int(k + 1)) in ['39', '37']:
                                pax_id = 'P10'

                            if str(int(k + 1)) in ['36', '33', '30', '27']:
                                pax_id = 'P11'
                            if str(int(k + 1)) in ['32', '31', '28']:
                                pax_id = 'P12'
                            if str(int(k + 1)) in ['29', '22', '21', '18']:
                                pax_id = 'P13'

                            if str(int(k + 1)) == '24':
                                pax_id = 'P14'
                            if str(int(k + 1)) in ['15']:
                                pax_id = 'P15'
                            if str(int(k + 1)) in ['16','14','11','10']:
                                pax_id = 'P16'
                            if str(int(k + 1)) in ['13','9','6']:
                                pax_id = 'P17'
                            if str(int(k + 1)) in ['5']:
                                pax_id = 'P18'

                            if str(int(k + 1)) in ['74', '35', '34', '23', '20', '17','13','3']:
                                pax_id = 'TSO1'
                                type = 'TSO'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['51', '41', '47', '26', '19', '8', '4','7']:
                                pax_id = 'TSO2'
                                type = 'TSO'
                                color = (0, 255, 255)

                    elif(tracker=='mht' and cam=='cam11exp2'):
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
                            if str(int(k+1)) in ['42']:
                                pax_id = 'P1'
                            if str(int(k+1)) == '37':
                                pax_id = 'P3'
                            if str(int(k+1)) == '41':
                                pax_id = 'P4'
                            if str(int(k+1)) in ['40','39']:
                                pax_id = 'P2'
                            if str(int(k+1)) in ['32','32']:
                                pax_id = 'P5'
                            if str(int(k+1)) == '31':
                                pax_id = 'P6'
                            if str(int(k + 1)) in ['35','33']:
                                pax_id = 'P8'
                            if str(int(k + 1)) == '28':
                                pax_id = 'P9'
                            if str(int(k + 1)) == '25':
                                pax_id = 'P10'
                            if str(int(k+1)) == '24':
                                pax_id = 'P11'

                            if str(int(k + 1)) == '22':
                                pax_id = 'P12'
                            if str(int(k + 1)) == '16':
                                pax_id = 'P13'
                            if str(int(k + 1)) in ['19', '17', '15']:
                                pax_id = 'P14'
                            if str(int(k + 1)) == '14':
                                pax_id = 'P15'
                            if str(int(k + 1)) == '9':
                                pax_id = 'P16'
                            if str(int(k + 1)) == '7':
                                pax_id = 'P17'
                            if str(int(k + 1)) == '6':
                                pax_id = 'P18'

                            if str(int(k+1)) in ['36','23','20','11']:
                                pax_id = 'TSO3'
                                type = 'TSO'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['43', '30', '26', '18', '12', '8', '2']:
                                pax_id = 'TSO4'
                                type = 'TSO'
                                color = (0, 255, 255)

                            if str(int(k + 1)) in ['3']:
                                pax_id = 'TSO5'
                                type = 'TSO'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['4','1']:
                                pax_id = 'TSO1'
                                type = 'TSO'
                                color = (0, 255, 255)

                            if str(int(k + 1)) in ['38', '21', '5', '48']:
                                continue

                    #tracker info
                    #if not TU_detect and sequential:
                        #pax_id = id

                    elif (tracker == 'mht' and cam == 'cam14exp2'):
                        camera = '14'
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
                            if str(int(k + 1)) in ['31']:
                                pax_id = 'P1'
                            if str(int(k + 1)) == '30':
                                pax_id = 'P2'
                            if str(int(k + 1)) == '29':
                                pax_id = 'P3'
                            if str(int(k + 1)) in ['33']:
                                pax_id = 'P4'
                            if str(int(k + 1)) in ['27','27']:
                                pax_id = 'P5'
                            if str(int(k + 1)) == '23':
                                pax_id = 'P6'
                            if str(int(k + 1)) == '25':
                                pax_id = 'P7'
                            if str(int(k + 1)) == '26':
                                pax_id = 'P8'
                            if str(int(k + 1)) == '22':
                                pax_id = 'P9'
                            if str(int(k + 1)) == '19':
                                pax_id = 'P10'
                            if str(int(k + 1)) == '18':
                                pax_id = 'P11'

                            if str(int(k + 1)) == '16':
                                pax_id = 'P12'
                            if str(int(k + 1)) in ['12','10']:
                                pax_id = 'P13'
                            if str(int(k + 1)) in ['13','11']:
                                pax_id = 'P14'
                            if str(int(k + 1)) == '9':
                                pax_id = 'P15'
                            if str(int(k + 1)) == '7':
                                pax_id = 'P16'
                            if str(int(k + 1)) in ['6','2','36']:
                                pax_id = 'P17'
                            if str(int(k + 1)) in ['37']:
                                pax_id = 'P18'


                            if str(int(k + 1)) in ['1']:
                                pax_id = 'TSO3'
                                type = 'TSO'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['17', '32', '15','5']:
                                pax_id = 'TSO4'
                                type = 'TSO'
                                color = (0, 255, 255)

                            if str(int(k + 1)) in ['35', '20', '14', '4']:
                                pax_id = 'TSO5'
                                type = 'TSO'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['21']:
                                continue

                                # tracker info
                                # if not TU_detect and sequential:
                                # pax_id = id
                    elif (tracker == 'mht' and cam == 'cam20exp2'):
                        camera = '20'
                        type = 'PAX'
                        box = list(track_fr[1:5])
                        pax_id = 'P' + str(int(k + 1))
                        color = (0, 0, 255)  # y
                        first_used = 'false'
                        if int(k + 1) not in keep_id:
                            keep_id.append(int(k + 1))
                            first_used = 'true'
                        if sequential:
                            if str(int(k + 1)) == '67':
                                pax_id = 'P1'
                            if str(int(k + 1)) == '66':
                                pax_id = 'TU1'#'P2'
                            if str(int(k + 1)) in ['60', '59']:
                                pax_id = 'TU1'#'P3'
                            if str(int(k + 1)) == '69':
                                pax_id = 'TU1'#'P4'
                            if str(int(k + 1)) in ['54', '47']:
                                pax_id = 'P5'
                            if str(int(k + 1)) == '48':
                                pax_id = 'P6'
                            if str(int(k + 1)) == '57':
                                pax_id = 'TU2'#'P7'
                            if str(int(k + 1)) in ['51','49']:
                                pax_id = 'TU2'#'P8'

                            if str(int(k + 1)) in ['44','43','40']:
                                pax_id = 'P9'

                            if str(int(k + 1)) == '37':
                                pax_id = 'P10'



                            if str(int(k + 1)) in ['33']:
                                pax_id = 'P12'
                            if str(int(k + 1)) in ['27']:
                                pax_id = 'TU3'#'P13'


                            if str(int(k + 1)) == '26':
                                pax_id = 'TU3'#'P14'


                            if str(int(k + 1)) in ['28', '24']:
                                pax_id = 'TU3'#'P15'

                            if str(int(k + 1)) == '11':
                                pax_id = 'P16'
                            if str(int(k + 1)) in ['20','7']:
                                pax_id = 'P17'
                            if str(int(k + 1)) == '10':
                                pax_id = 'P18'

                            if str(int(k + 1)) in ['77', '72', '71', '70', '56', '46','39','36','31','18','19']:
                                continue

                            if str(int(k + 1)) == '33' and fr >= 8720:
                                pax_id = 'TU3'#'P13'

                            if str(int(k + 1)) in ['78', '74', '57', '45', '35','21','29','25','23','17','14']:
                                type = 'TSO'
                                pax_id = 'TSO3'
                                color = (0, 255, 255)
                            if str(int(k + 1)) == '21' and fr >= 9100:
                                type = 'TSO'
                                pax_id = 'TSO5'
                                color = (0, 255, 255)

                            if str(int(k + 1)) in ['68', '58', '38', '30', '16', '3']:
                                type = 'TSO'
                                pax_id = 'TSO4'
                                color = (0, 255, 255)
                            if str(int(k + 1)) in ['76', '73', '64', '63', '55', '42','41','32','15','13']:
                                type = 'TSO'
                                pax_id = 'TSO5'
                                color = (0, 255, 255)
                            if str(int(k + 1)) == '35' and fr >= 7810:
                                pax_id = 'P11'
                    #frame_logs = log_files(fr,pax_id, box, cam, TU, event_type)
                    #fr_logs['frame'].append(frame_logs)
                    print 'PAX: ', pax_id
                    #for demo
                    if pax_id in ['TU1','TU2','TU3']:
                        color = (255,255,0)
                        type = 'TU'
                        id = pax_id.split('TU')[1]
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
                            if pax_id=='TU4':
                                img = cv2.putText(img, pax_id+':'+' Travel time ' + str((fr - first_appear) / 30), (1920 + 50, 0+50),
                                                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            logs['passengers'].append(fr_logs)
            if fr>0:
                if vis and fr % 10 == 0:
                    print (cam)
                    img = cv2.resize(img, (1920 // 2, 1080 // 2))
                    plt.imsave(demo_path + "%06d" % fr + '.png', img, dpi=200)