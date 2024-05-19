import copy

import numpy as np
from MCTA.Tracker_Merge_SingleCam import *
import pdb

class form_tracklets:

    def __init__(self,trackers, cam=None, tracker_min_size=60,t_thr = 150,d_thr=60,
                 dataset='clasp2',motion=False, t_app=False, single_cam_association=False,
                 pvd_id_ait=None, cam2cam=None, global_track_start=None, lane3_ids=None):
        self.trackers = trackers
        self.fr_start = None
        self.fr_end = None
        self.tracker_min_size = tracker_min_size
        self.single_cam_association = single_cam_association
        self.t_thr = t_thr
        self.d_thr = d_thr
        self.tracker_motion = motion
        self.tracker_app = t_app
        self.cam=cam
        self.cam_trklts_id = []
        self.keep_id_pvd = pvd_id_ait
        self.cam2cam = cam2cam
        self.global_track_start = global_track_start
        self.lane3_ids = lane3_ids
        self.metric = 'euclidean'

    def _format_trackers(self):
        #We generally use static _methods to create private function
        #format trackers from single camera tracker for MCTA
        global_feature = []
        for i, track in self.trackers.items():
            for frame, bb in track.items():
                #print(frame)
                x1 = bb[0]
                y1 = bb[1]
                x2 = bb[2]
                y2 = bb[3]
                score = bb[4]
                if self.tracker_motion and self.tracker_app is not None:
                    v_cx = bb[5]
                    v_cy = bb[6]
                    app = bb[7::]
                # [fr,id,x,y,w,h]
                global_feature.append([frame, float(i), x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, float(self.cam)])
        return np.array(global_feature)

    def get_trklts(self):
        tracklets = []
        totalTrack = 0
        self.trackers = form_tracklets._format_trackers(self)
        assert len(self.trackers)!=0, 'empty trackers if not valid but found {}'.format(self.trackers)
        #self.trackers = self.trackers[self.trackers[:, 0] >= self.fr_start]
        #self.trackers = self.trackers[self.trackers[:, 0] <= self.fr_end]
        for i in np.unique(self.trackers[:,1]):
            #trackers with motion
            if self.tracker_motion and not self.tracker_app:
               # [fr,x,y,w,h,10-vx,11-vy,12:141-app]
               tracker_i = self.trackers[self.trackers[:, 1] == i][:, [0, 2, 3, 4, 5, 10, 11]]
               #tracker_i = np.vstack(tracker_i[:, :]).astype(np.float)
            #for trackers with motion and appearance
            elif self.tracker_motion and self.tracker_app:
               # [fr,x,y,w,h,10-vx,11-vy,12:141-app]
               assert self.trackers.shape[1]==140
               feature_colms = [j for j in range(0, self.trackers.shape[1]) if j not in [1,6,7,8,9]]
               tracker_i = self.trackers[self.trackers[:, 1] == i][:, feature_colms]
            else:
                #for trackers without motion and apperance: need to keep raw_id and camera information
                #filter tracks for supporting cameras
                tracker_i = self.trackers[self.trackers[:, 1] == i] #[:, [0, 2, 3, 4, 5]]
                if self.cam in [4]:
                    tracker_i = tracker_i[(tracker_i[:, 3]+tracker_i[:, 5]) > 165]#ymax<250
                if self.cam in [2] and self.cam2cam in ['C5C2']:
                    tracker_i = tracker_i[tracker_i[:, 3] < 357]
                    
                if self.cam in [440]:# and self.cam2cam in ['C300C440', 'C361C440']:
                    tracker_i = tracker_i[(tracker_i[:,2]+tracker_i[:,4]/2)>640]#cx>580
                    tracker_i = tracker_i[(tracker_i[:, 3]+tracker_i[:, 5]) > 250]#ymax<250

                if self.cam in [340] and self.cam2cam in ['C360C340']:
                    tracker_i = tracker_i[(tracker_i[:,2]+tracker_i[:,4]/2)>270]
                    tracker_i = tracker_i[(tracker_i[:, 2] + tracker_i[:, 4] / 2) < 1000]

                if self.cam in [340] and self.cam2cam not in ['C330C340']:
                    tracker_i = tracker_i[(tracker_i[:,2]+tracker_i[:,4]/2)>270]
                    tracker_i = tracker_i[(tracker_i[:, 2] + tracker_i[:, 4] / 2) < 1000]


                if self.cam in [361] and self.cam2cam in ['C360C361']:
                    #tracker_i = tracker_i[(tracker_i[:,2]+tracker_i[:,4]/2)>640]#cx>580
                    tracker_i = tracker_i[(tracker_i[:, 3]+tracker_i[:, 5]) > 350]#ymax>350

                #separate lane34: cam 300 ids
                #ymax>450
                if self.cam in [300] and tracker_i[0, 1] in self.global_track_start and \
                        (tracker_i[0, 3] + tracker_i[0, 5]) > 450 and self.lane3_ids is not None:#(tracker_i[0, 3]+tracker_i[0, 5])>450 and \
                    if tracker_i[0, 1] not in self.lane3_ids:
                        self.lane3_ids.append(tracker_i[0, 1])

            if len(tracker_i)>0:
                tracklets.append(tracker_i)
                self.cam_trklts_id.append(i)
                #connect enry and exit cameras
                #Issue 1: start time usually found in the current batch
                # Issue 2: However, end time usually found in the current batch
                # start of track at 360
                if self.cam in [360] and self.cam2cam in ['C360C340'] and \
                        (tracker_i[0, :][2] + tracker_i[0, :][4] / 2) > 870 \
                        and (tracker_i[0, :][3] + tracker_i[0, :][5] / 2) < 200 \
                        and tracker_i[0,1] not in self.keep_id_pvd[self.cam2cam][self.cam]:

                    self.keep_id_pvd[self.cam2cam][self.cam][tracker_i[0,1]] = {}
                    self.keep_id_pvd[self.cam2cam][self.cam][tracker_i[0,1]]['start'] = tracker_i[0,0]#self.global_track_start[tracker_i[0,1]]
                    self.keep_id_pvd[self.cam2cam][self.cam][tracker_i[0,1]]['first_pos'] = tracker_i[0,:]
                    print('Person {} starts at {} in cam {}'.format(tracker_i[0,1], self.global_track_start[tracker_i[0,1]], self.cam))
                # end of track at 340
                if self.cam in [340] and self.cam2cam in ['C360C340'] and \
                        270<(tracker_i[-1, :][2] + tracker_i[-1, :][4] / 2) < 520 \
                        and (tracker_i[-1, :][3] + tracker_i[-1, :][5] / 2) > 160:
                    self.keep_id_pvd[self.cam2cam][self.cam][tracker_i[-1, 1]] = {}
                    #end time will be updated based on the current batch only
                    self.keep_id_pvd[self.cam2cam][self.cam][tracker_i[-1, 1]]['end'] = max(tracker_i[:, 0]) #tracker_i[-1, 0]
                    self.keep_id_pvd[self.cam2cam][self.cam][tracker_i[-1, 1]]['last_pos'] = max(tracker_i[:, 0]) #tracker_i[-1,:]
                    print('Person {} ends at {} in cam {}'.format(tracker_i[-1, 1], max(tracker_i[:, 0]), self.cam))

                # start of track at 330
                if self.cam in [330] and self.cam2cam in ['C330C340...'] and \
                        (tracker_i[0, :][2] + tracker_i[0, :][4] / 2) > 870 \
                        and (tracker_i[0, :][3] + tracker_i[0, :][5] / 2) < 200 \
                        and tracker_i[0,1] not in self.keep_id_pvd[self.cam]:

                    self.keep_id_pvd[self.cam][tracker_i[0,1]] = {}
                    self.keep_id_pvd[self.cam][tracker_i[0,1]]['start'] = tracker_i[0,0]#self.global_track_start[tracker_i[0,1]]
                    self.keep_id_pvd[self.cam][tracker_i[0,1]]['first_pos'] = tracker_i[0,:]
                    print('Person {} starts at {} in cam {}'.format(tracker_i[0,1], tracker_i[0,0], self.cam))
                # end of track at 340
                if self.cam in [340] and self.cam2cam in ['C330C340..'] and \
                        270<(tracker_i[-1, :][2] + tracker_i[-1, :][4] / 2) < 520 \
                        and (tracker_i[-1, :][3] + tracker_i[-1, :][5] / 2) > 160:
                    self.keep_id_pvd[self.cam][tracker_i[-1, 1]] = {}
                    self.keep_id_pvd[self.cam][tracker_i[-1, 1]]['end'] = tracker_i[-1, 0]
                    self.keep_id_pvd[self.cam][tracker_i[-1, 1]]['last_pos'] = tracker_i[-1,:]
                    print('Person {} ends at {} in cam {}'.format(tracker_i[-1, 1], tracker_i[-1, 0], self.cam))


                #start of track at 360
                if self.cam in [360] and self.cam2cam in ['C360C361'] and \
                        540<(tracker_i[0,:][2]+tracker_i[0,:][4]/2)<720 \
                        and (tracker_i[0,:][3]+tracker_i[0,:][5]/2)<280 \
                        and tracker_i[0,1] not in self.keep_id_pvd[self.cam2cam][self.cam]:

                    self.keep_id_pvd[self.cam2cam][self.cam][tracker_i[0,1]] = {}
                    self.keep_id_pvd[self.cam2cam][self.cam][tracker_i[0,1]]['start'] = tracker_i[0,0] #self.global_track_start[tracker_i[0,1]]
                    #start pose should be in the AIT ROI
                    self.keep_id_pvd[self.cam2cam][self.cam][tracker_i[0,1]]['first_pos'] = tracker_i[0,:]
                    print('Person {} starts at {} in cam {}'.format(tracker_i[0,1], self.global_track_start[tracker_i[0,1]], self.cam))
                #end of track at 361
                #TODO: Need to know the eaxct time when track disappear
                if self.cam in [361] and self.cam2cam in ['C360C361'] and \
                        650<(tracker_i[-1, :][2] + tracker_i[-1, :][4]/2) \
                        and 390<(tracker_i[-1, :][3] + tracker_i[-1, :][5]/2):

                    self.keep_id_pvd[self.cam2cam][self.cam][tracker_i[-1, 1]] = {}
                    # end time will be updated based on the current batch only
                    self.keep_id_pvd[self.cam2cam][self.cam][tracker_i[-1, 1]]['end'] = max(tracker_i[:, 0]) #tracker_i[-1, 0]
                    self.keep_id_pvd[self.cam2cam][self.cam][tracker_i[-1, 1]]['last_pos'] = max(tracker_i[:, 0]) #tracker_i[-1,:]
                    print('Person {} ends at {} in cam {}'.format(tracker_i[-1, 1], max(tracker_i[:, 0]), self.cam))

                totalTrack += 1

        if self.single_cam_association:
            long_tracklets = get_tracklets(tracklets)
            short_tracklets = copy.deepcopy(tracklets)
            tracklets = merge_tracklets(long_tracklets, short_tracklets,
                                            self.t_thr, self.d_thr, metric=self.metric)
            totalTrack = len(tracklets)
        return tracklets, totalTrack, self.cam_trklts_id, self.keep_id_pvd, self.lane3_ids
