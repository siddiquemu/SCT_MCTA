from __future__ import division
import numpy as np
from scipy.spatial.distance import directed_hausdorff, cosine, mahalanobis
from scipy.optimize import linear_sum_assignment
import cv2
import sys
import os
import glob
import copy
import pdb
import matplotlib.pyplot as plt
import collections

class SCT_ReID(object):
    def __init__(self, id_init_map=None, all_tracklets=None, curr_frame=None, t_thr=0, d_thr=200, reid_map=None, new_id=None):
        self.id_init_map = id_init_map #list of global_track_start_time
        self.all_tracklets = all_tracklets
        self.t_thr=t_thr
        self.d_thr=d_thr
        self.reid_map = reid_map
        self.new_id = new_id
        #updated at each frame/batch
        self.curr_frame = curr_frame
        self.new_track = {}
        self.prev_track = {}

    @staticmethod
    def linear_sum_assignment_with_inf(cost_matrix):
        cost_matrix = np.asarray(cost_matrix)
        min_inf = np.isneginf(cost_matrix).any()
        max_inf = np.isposinf(cost_matrix).any()
        if min_inf and max_inf:
            raise ValueError("matrix contains both inf and -inf")
        if min_inf or max_inf:
            values = cost_matrix[~np.isinf(cost_matrix)]
            if values.size == 0:
                cost_matrix = np.full(cost_matrix.shape, 1000)  # workaround for the cast of no finite costs
            else:
                m = values.min()
                M = values.max()
                n = min(cost_matrix.shape)
                # strictly positive constant even when added
                # to elements of the cost matrix
                positive = n * (M - m + np.abs(M) + np.abs(m) + 1)
                if max_inf:
                    place_holder = (M + (n - 1) * (M - m)) + positive
                if min_inf:
                    place_holder = (m + (n - 1) * (m - M)) - positive
                cost_matrix[np.isinf(cost_matrix)] = place_holder
        return linear_sum_assignment(cost_matrix)

    def _dfs(self, graph, node, visited):
        if node not in visited:
            visited.append(node)
            for n in graph[node]:
                self._dfs(graph, n, visited)
        return visited

    def get_poses(self):

        for j, t in self.all_tracklets.items():
            # new query track: id appear at current frame
            if self.curr_frame == self.id_init_map[j] and j not in self.new_id:
                #self.curr_frame - 1 is the first pose
                #apply boundary condition on new track: TODO: FOV ROI, camera id
                t_pose = t[self.curr_frame-1][:4].astype('float')
                if self.curr_frame==1542:
                    print('debug')
                if t_pose[1]>100 and 0<t_pose[2]<1180: #H: 960.0, W: 1280.0
                    self.new_track[j]= np.append(self.curr_frame-1, t_pose)
                    self.new_id.append(j)
            else:
                #TODO: work with batches since this condition will collect all the ids in the history
                # TODO: predict dummy using interpolation
                # gallery traacks: all lost tracks in history
                # t[-1] is the last pose of lost track: frame of last pose< current frame
                t = list(t.items())
                if t[-1][0]<self.curr_frame - 1:
                    self.prev_track[j]= np.append(t[-1][0], t[-1][1][:4].astype('float'))


    def apply_dfs(self):
        # apply DFS on mc system graph
        label_map = {keys: [] for keys in self.reid_map.keys()}
        for s_id, t_id in self.reid_map.items():
            # each track identity is unique
            solnPath = self._dfs(self.reid_map, s_id, [])
            # min id (used) or min time
            label_map[s_id] = [min(solnPath)]
        return label_map

    def match_tl(self):
        # Associate pairs of tracklets with maximum overlap between the last and first detections
        # using the Hungarian algorithm
        self.get_poses()
        #update all_tracklets based on the association when new id appeared in current frame
        if self.new_track and self.prev_track:
            # Compute the cost matrix IoU between each pair of last and first detections
            cost = np.full((len(self.prev_track), len(self.new_track)), np.inf)
            id_map = {}
            i = 0
            for id_i, prev_det in self.prev_track.items():

                j = 0
                for id_j, new_det in self.new_track.items():
                    #TODO: do id map separately for row and column
                    id_map[i] = id_i
                    #self.reid_map[id_i] = id_i
                    id_map[j] = id_j
                    #self.reid_map[id_j] = id_j
                    # This is the only difference from the single-camera function
                    # TODO: Make both a common function that take this test as a parameter
                    delta_t = new_det[0] - prev_det[0]
                    new_cent = new_det[1:3] + (new_det[3:5]-new_det[1:3])/2.
                    prev_cent = prev_det[1:3] + (prev_det[3:5] - prev_det[1:3]) / 2.
                    dist = np.linalg.norm(new_cent - prev_cent)
                    if 0 < delta_t < self.t_thr and dist < self.d_thr:
                        cost[i, j] = dist
                    j = j + 1
                i = i + 1

            row_ind, col_ind = self.linear_sum_assignment_with_inf(cost)
            # Find the maximum IoU for each pair
            self.do_dfs = False
            for r, c in zip(row_ind, col_ind):
                if cost[r, c] < self.d_thr:
                    #matches.append((cost[r, c], id_map[r], id_map[c]))
                    if id_map[r] not in self.reid_map.keys():
                        self.reid_map[id_map[r]] = []
                    self.reid_map[id_map[r]].append(id_map[c])

                    if id_map[c] not in self.reid_map.keys():
                        self.reid_map[id_map[c]] = []
                    self.reid_map[id_map[c]].append(id_map[r])

                    self.do_dfs = True
            if self.do_dfs:
                self.reid_map = self.apply_dfs()
        return self.reid_map

    def merge_tracklets(self):
        # Iterate a few times to join pairs of tracklets that correspond to multiple
        # fragments of the same track
        # TODO: Find a more elegant solution -- Possibly using DFS in offline
        self.get_poses()
        #update all_tracklets based on the association when new id appeared in current frame
        if self.new_track and self.prev_track:
            mt = self.match_tl()

            for (c, k, l) in mt:
                self.filtered_tracker[k] = np.vstack((self.filtered_tracker[k], self.unassigned_tracklets[l]))
                if l not in del_tl:
                    del_tl.append(l)

            del_tl.sort(reverse=True)
            # delete associated tracklets to discard the duplicity
            for l in del_tl:
                del (self.filtered_tracker[l])
                del (self.unassigned_tracklets[l])

        return self.reid_map