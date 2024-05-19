import time
from collections import deque

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from scipy.optimize import linear_sum_assignment
import cv2

from .utils import bbox_overlaps, center_distances, ReID_search_distances, warp_pos, \
	get_center, get_height, get_width, make_pos, ReID_search_distances_linear

from torchvision.ops.boxes import clip_boxes_to_image, nms
import  pdb
import glob

class Tracker:
	"""The main tracking file, here is where magic happens."""
	# only track pedestrian
	cl = 1

	def __init__(self, obj_detect, reid_network, tracker_cfg, global_track_num,
				 global_track_start_time, reid_rads_patience, class_id=None,  start_frame=None,
				 det_time_counter=None, isSCT_ReID=False):
		self.obj_detect = obj_detect
		self.reid_network = reid_network
		self.detection_person_thresh = tracker_cfg['detection_person_thresh']
		self.regression_person_thresh = tracker_cfg['regression_person_thresh']
		self.detection_nms_thresh = tracker_cfg['detection_nms_thresh']
		self.regression_nms_thresh = tracker_cfg['regression_nms_thresh']
		self.public_detections = tracker_cfg['public_detections']
		self.inactive_patience = tracker_cfg['inactive_patience']
		self.do_reid = tracker_cfg['do_reid']
		self.max_features_num = tracker_cfg['max_features_num']
		self.reid_sim_threshold = tracker_cfg['reid_sim_threshold']
		self.reid_iou_threshold = tracker_cfg['reid_iou_threshold']
		self.do_align = tracker_cfg['do_align']
		self.motion_model_cfg = tracker_cfg['motion_model']

		self.warp_mode = eval(tracker_cfg['warp_mode'])
		self.number_of_iterations = tracker_cfg['number_of_iterations']
		self.termination_eps = tracker_cfg['termination_eps']

		self.tracks = []
		self.inactive_tracks = []
		self.track_num = 1
		self.global_track_num = global_track_num
		self.global_track_start_time = global_track_start_time
		self.class_id = class_id
		self.start_frame = start_frame
		self.im_index = 0 # TODO: image index should start from self.start_frame
		self.results = {}
		self.reid_rads_patience = reid_rads_patience['patience']
		self.reid_radious = reid_rads_patience['rads']
		self.isSCT_ReID = isSCT_ReID

	def reset(self, hard=True):
		self.tracks = []
		self.inactive_tracks = []

		if hard:
			self.track_num = 0
			self.results = {}
			self.im_index = 0

	def tracks_to_inactive(self, tracks):
		self.tracks = [t for t in self.tracks if t not in tracks]
		for t in tracks:
			t.pos = t.last_pos[-1]
		self.inactive_tracks += tracks

	def add(self, new_det_pos, new_det_scores, new_det_features):
		"""Initializes new Track objects and saves them."""
		num_new = new_det_pos.size(0)
		for i in range(num_new):
			#increment track_num if its already in global_track_num list
			while self.track_num + i in self.global_track_num:
				self.track_num = self.track_num+1
			#update the global_track_num list with new id
			self.global_track_num.append(self.track_num + i)
			#global track start time starts from zero in tracktor
			self.global_track_start_time[self.track_num + i] = self.im_index+self.start_frame
			#print(self.global_track_num)
			self.tracks.append(Track(
				new_det_pos[i].view(1, -1),
				new_det_scores[i],
				self.track_num + i,
				new_det_features[i].view(1, -1),
				self.inactive_patience,
				self.max_features_num,
				self.motion_model_cfg['n_steps'] if self.motion_model_cfg['n_steps'] > 0 else 1
			))

			# initialize patience and rads for inactive tracks
			self.reid_rads_patience[self.track_num + i] = self.inactive_patience
			self.reid_radious[self.track_num + i] = [60, self.inactive_patience]

		self.track_num += num_new

	def regress_tracks(self, blob):
		"""Regress the position of the tracks and also checks their scores."""
		pos = self.get_pos()

		# regress
		#self.obj_detect.load_image(blob['img'])

		boxes, scores = self.obj_detect.predict_boxes(pos, features=blob['features'])
		#filter for person classes
		#boxes = boxes[labels==1]
		#scores = scores[labels==1]
		pos = clip_boxes_to_image(boxes, blob['img'].shape[-2:])

		s = []
		for i in range(len(self.tracks) - 1, -1, -1):
			t = self.tracks[i]
			t.score = scores[i]
			if scores[i] <= self.regression_person_thresh:
				self.tracks_to_inactive([t])
			else:
				s.append(scores[i])
				# t.prev_pos = t.pos
				t.pos = pos[i].view(1, -1)

		return torch.Tensor(s[::-1]).cuda()

	def get_pos(self):
		"""Get the positions of all active tracks."""
		if len(self.tracks) == 1:
			pos = self.tracks[0].pos
		elif len(self.tracks) > 1:
			pos = torch.cat([t.pos for t in self.tracks], 0)
		else:
			pos = torch.zeros(0).cuda()
		return pos

	def get_features(self):
		"""Get the features of all active tracks."""
		if len(self.tracks) == 1:
			features = self.tracks[0].features
		elif len(self.tracks) > 1:
			features = torch.cat([t.features for t in self.tracks], 0)
		else:
			features = torch.zeros(0).cuda()
		return features

	def get_inactive_features(self):
		"""Get the features of all inactive tracks."""
		if len(self.inactive_tracks) == 1:
			features = self.inactive_tracks[0].features
		elif len(self.inactive_tracks) > 1:
			features = torch.cat([t.features for t in self.inactive_tracks], 0)
		else:
			features = torch.zeros(0).cuda()
		return features

	def reid(self, blob, new_det_pos, new_det_scores):
		"""Tries to ReID inactive tracks with provided detections."""
		new_det_features = [torch.zeros(0).cuda() for _ in range(len(new_det_pos))]

		if self.do_reid:
			new_det_features = self.reid_network.test_rois(
				blob['img'], new_det_pos).data

			if len(self.inactive_tracks) >= 1:
				# calculate appearance distances
				dist_mat, pos = [], []
				for t in self.inactive_tracks:
					dist_mat.append(torch.cat([t.test_features(feat.view(1, -1))
					                           for feat in new_det_features], dim=1))
					pos.append(t.pos)
				if len(dist_mat) > 1:
					dist_mat = torch.cat(dist_mat, 0)
					pos = torch.cat(pos, 0)
				else:
					dist_mat = dist_mat[0]
					pos = pos[0]

				# calculate IoU distances
				iou = bbox_overlaps(pos, new_det_pos)
				iou_mask = torch.ge(iou, self.reid_iou_threshold)
				#iou = center_distances(pos, new_det_pos)
				#iou_mask = torch.le(iou, 50)
				iou_neg_mask = ~iou_mask
				# make all impossible assignments to the same add big value
				dist_mat = dist_mat * iou_mask.float() + iou_neg_mask.float() * 1000
				dist_mat = dist_mat.cpu().numpy()

				row_ind, col_ind = linear_sum_assignment(dist_mat)

				assigned = []
				remove_inactive = []
				for r, c in zip(row_ind, col_ind):
					if dist_mat[r, c] <= self.reid_sim_threshold:
						t = self.inactive_tracks[r]
						self.tracks.append(t)
						t.count_inactive = 0
						t.pos = new_det_pos[c].view(1, -1)
						t.reset_last_pos()
						t.add_features(new_det_features[c].view(1, -1))
						assigned.append(c)
						remove_inactive.append(t)
						#init rads and patience when inactive track changes to active state
						self.reid_rads_patience[int(t.id)] = self.inactive_patience
						self.reid_radious[int(t.id)] = [60, self.inactive_patience]

				for t in remove_inactive:
					self.inactive_tracks.remove(t)

				keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().cuda()
				if keep.nelement() > 0:
					new_det_pos = new_det_pos[keep]
					new_det_scores = new_det_scores[keep]
					new_det_features = new_det_features[keep]
				else:
					new_det_pos = torch.zeros(0).cuda()
					new_det_scores = torch.zeros(0).cuda()
					new_det_features = torch.zeros(0).cuda()

		return new_det_pos, new_det_scores, new_det_features

	def update_reid_rads(self, max_rads=250, max_patience=200, cam=None):
		"Update inactive tracks reid radious and patience"
		if len(self.inactive_tracks) >= 1:
			# track features
			pos, ids, inact_cnts = [], [], []
			for t in self.inactive_tracks:
				pos.append(t.pos)
				ids.append(t.id)
				inact_cnts.append(0.5*t.count_inactive)

			if len(self.inactive_tracks) > 1:
				pos = torch.cat(pos, 0)
			else:
				pos = pos[0]
		reid_rads, reid_patiences = ReID_search_distances(pos, max_dist=max_rads,
														  max_patience=max_patience)
		# inact_cnts = torch.tensor(inact_cnts, dtype=torch.float).cuda()
		# reid_rads += inact_cnts.reshape(inact_cnts.shape[0], 1)
		assert len(ids) == len(reid_rads) == len(reid_patiences.cpu())
		i = 0
		for ID, patience, rad in zip(ids, reid_patiences.cpu().numpy(), reid_rads.cpu().numpy()):
			# body scanner mask
			if cam in [3]:
				cx = pos[i][0] + (pos[i][2] - pos[i][0]) / 2.
				ymax = pos[i][3]
				if 1100 > cx > 650 and 100 < ymax < 460:
					self.reid_rads_patience[int(ID)] += 2  # inact_cnts[i]
					# rad += inact_cnts[i] * 20 #velocity should be used
					rad = 300  # fixed
					reid_rads[i] = torch.tensor([rad], dtype=torch.float).cuda()
					print('applied occlusion mask on {}'.format(ID))
					reid_thr = rad
				# AIT mask
				if 720 > cx > 50 and 330 < ymax < 800:
					self.reid_rads_patience[int(ID)] += 2  # inact_cnts[i]
					# rad += inact_cnts[i] * 20
					rad = 400
					reid_rads[i] = torch.tensor([rad], dtype=torch.float).cuda()
					reid_thr = rad

			self.reid_rads_patience[int(ID)] += 1
			self.reid_radious[int(ID)] = [rad, self.reid_rads_patience[int(ID)]]
			#print('#############################################')
			#print('# ID: {}, Patience: {}, rads: {} #'.format(ID, self.reid_rads_patience[int(ID)], rad))
			#print('#############################################')
			i += 1
		# print(self.reid_radious)
		# print(rad)
		# print('lost track: {}'.format(pos.shape))
		# print('reid_rads: {}'.format(reid_rads))
		#print('reid_patiences: {} at frame: {}'.format(self.reid_rads_patience, self.im_index))

		return reid_rads


	def sct_reid(self, blob, new_det_pos, new_det_scores, reid_rads=None,
				 cam=None, reid_thr=2.0, obj_reid_rads=300):
		"""Tries to ReID inactive tracks with provided detections."""
		new_det_features = [torch.zeros(0).cuda() for _ in range(len(new_det_pos))]

		if self.do_reid:
			new_det_features = self.reid_network.test_rois(
				blob['img'], new_det_pos).data

			if len(self.inactive_tracks) >= 1:
				# calculate appearance distances
				dist_mat, pos, ids, inact_cnts = [], [], [], []
				for t in self.inactive_tracks:
					dist_mat.append(torch.cat([t.test_features(feat.view(1, -1))
					                           for feat in new_det_features], dim=1))
					pos.append(t.pos)
					ids.append(t.id)
					inact_cnts.append(0.5*t.count_inactive)

				if len(dist_mat) > 1:
					dist_mat = torch.cat(dist_mat, 0)
					pos = torch.cat(pos, 0)
				else:
					dist_mat = dist_mat[0]
					pos = pos[0]

				#pdb.set_trace()
				#print('pos: {}'.format(pos))
				#print('new_dets: {}'.format( new_det_pos))
				cdist = center_distances(pos, new_det_pos)

				#compute ReID Euclidean search space using d(new_dets_pose, lost_pose)
				#cdist_mask = torch.le(cdist, 250)
				#TODO: verify this elementwise comparisons
				cdist_mask = torch.le(cdist, reid_rads)

				cdist_neg_mask = ~cdist_mask
				# make all impossible assignments to the same add big value
				if cam in [3]:
					dist_mat = cdist #avoid appearance
					reid_thr = obj_reid_rads # max(reid_rads)

				dist_mat = dist_mat * cdist_mask.float() + cdist_neg_mask.float() * 10000
				dist_mat = dist_mat.cpu().numpy()

				row_ind, col_ind = linear_sum_assignment(dist_mat)

				assigned = []
				remove_inactive = []
				for r, c in zip(row_ind, col_ind):
					if dist_mat[r, c] <= reid_thr: #self.reid_sim_threshold: train reid model to improve app
						t = self.inactive_tracks[r]
						self.tracks.append(t)
						t.count_inactive = 0
						t.pos = new_det_pos[c].view(1, -1)
						t.reset_last_pos()
						t.add_features(new_det_features[c].view(1, -1))
						assigned.append(c)
						remove_inactive.append(t)
						#update patience and rads
						#TODO: can we remove the rads and patience after moving from inactive to active
						self.reid_rads_patience[int(t.id)] = self.inactive_patience
						self.reid_radious[int(t.id)] = [60, self.inactive_patience]

				for t in remove_inactive:
					self.inactive_tracks.remove(t)
					#self.reid_rads_patience.pop(int(t.id))
					#self.reid_radious.pop(int(t.id))
					#remove patience for inactive
					#if t.id in self.reid_rads_patience:
						#self.reid_rads_patience.pop(t.id)

				keep = torch.Tensor([i for i in range(new_det_pos.size(0)) if i not in assigned]).long().cuda()
				if keep.nelement() > 0:
					new_det_pos = new_det_pos[keep]
					new_det_scores = new_det_scores[keep]
					new_det_features = new_det_features[keep]
				else:
					new_det_pos = torch.zeros(0).cuda()
					new_det_scores = torch.zeros(0).cuda()
					new_det_features = torch.zeros(0).cuda()

		return new_det_pos, new_det_scores, new_det_features

	def get_appearances(self, blob):
		"""Uses the siamese CNN to get the features for all active tracks."""
		new_features = self.reid_network.test_rois(blob['img'], self.get_pos()).data
		return new_features

	def add_features(self, new_features):
		"""Adds new appearance features to active tracks."""
		for t, f in zip(self.tracks, new_features):
			t.add_features(f.view(1, -1))

	def align(self, blob):
		"""Aligns the positions of active and inactive tracks depending on camera motion."""
		if self.im_index > 0:
			im1 = np.transpose(self.last_image.cpu().numpy(), (1, 2, 0))
			im2 = np.transpose(blob['img'][0].cpu().numpy(), (1, 2, 0))
			im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
			im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
			warp_matrix = np.eye(2, 3, dtype=np.float32)
			criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, self.number_of_iterations,  self.termination_eps)
			#https://kite.com/python/docs/cv2.findTransformECC
			#im2 warped to im1 based on similarity in image intensity
			cc, warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, self.warp_mode, criteria)
			warp_matrix = torch.from_numpy(warp_matrix)

			for t in self.tracks:
				#if self.im_index>10:
					#pdb.set_trace()
				t.pos = warp_pos(t.pos, warp_matrix)
				# t.pos = clip_boxes(Variable(pos), blob['im_info'][0][:2]).data

			if self.do_reid:
				for t in self.inactive_tracks:
					t.pos = warp_pos(t.pos, warp_matrix)

			if self.motion_model_cfg['enabled']:
				for t in self.tracks:
					for i in range(len(t.last_pos)):
						t.last_pos[i] = warp_pos(t.last_pos[i], warp_matrix)

	def motion_step(self, track):
		"""Updates the given track's position by one step based on track.last_v"""
		if self.motion_model_cfg['center_only']:
			center_new = get_center(track.pos) + track.last_v
			track.pos = make_pos(*center_new, get_width(track.pos), get_height(track.pos))
		else:
			track.pos = track.pos + track.last_v

	def motion(self):
		"""Applies a simple linear motion model that considers the last n_steps steps."""
		for t in self.tracks:
			last_pos = list(t.last_pos)

			# avg velocity between each pair of consecutive positions in t.last_pos
			if self.motion_model_cfg['center_only']:
				vs = [get_center(p2) - get_center(p1) for p1, p2 in zip(last_pos, last_pos[1:])]
			else:
				vs = [p2 - p1 for p1, p2 in zip(last_pos, last_pos[1:])]

			t.last_v = torch.stack(vs).mean(dim=0)
			self.motion_step(t)

		if self.do_reid:
			for t in self.inactive_tracks:
				if t.last_v.nelement() > 0:
					self.motion_step(t)

	def step(self, blob, detections, cam=None):
		"""This function should be called every timestep to perform tracking with a blob
		containing the image information.
		"""
		for t in self.tracks:
			# add current position to last_pos list
			t.last_pos.append(t.pos.clone())

		###########################
		# Look for new detections #
		###########################
		#images = torch.cat((blob['img'], blob['img']))
		#self.obj_detect.load_image(images)
		#det_start = time.time()
		#self.obj_detect.load_image(blob['img'])
		#pdb.set_trace()


		#boxes, scores, labels = self.obj_detect.detect_clasp1(blob['img'])
		boxes, scores, labels = detections['boxes'], detections['scores'], detections['labels']
		# filter person class
		boxes = boxes[labels == self.class_id]
		scores = scores[labels == self.class_id]
		#print('Predicted box: ', boxes)
		#print('Predicted scores: ', scores)
		# TODO: visualize online detection results to look into the performance

		if boxes.nelement() > 0:

			boxes = clip_boxes_to_image(boxes, blob['img'].shape[-2:])

			# Filter out tracks that have too low person score
			#Computes input>other\text{input} > \text{other}input>other element-wise.
			inds = torch.gt(scores, self.detection_person_thresh).nonzero().view(-1)
		else:
			inds = torch.zeros(0).cuda()

		if inds.nelement() > 0:
			det_pos = boxes[inds]

			det_scores = scores[inds]
		else:
			det_pos = torch.zeros(0).cuda()
			det_scores = torch.zeros(0).cuda()

		##################
		# Predict tracks #
		##################

		num_tracks = 0
		nms_inp_reg = torch.zeros(0).cuda()
		#pdb.set_trace()
		if len(self.tracks):
			# align
			if self.do_align:
				self.align(blob)

			# apply motion model
			# TODO: tracks should contain position and corresponding center motion: vx, vy
			if self.motion_model_cfg['enabled']:
				self.motion()
				self.tracks = [t for t in self.tracks if t.has_positive_area()]

			# regress
			person_scores = self.regress_tracks(blob)

			if len(self.tracks):
				# create nms input

				# nms here if tracks overlap
				keep = nms(self.get_pos(), person_scores, self.regression_nms_thresh)

				self.tracks_to_inactive([self.tracks[i] for i in list(range(len(self.tracks))) if i not in keep])

				if keep.nelement() > 0 and self.do_reid:
						new_features = self.get_appearances(blob)
						self.add_features(new_features)

		#####################
		# Create new tracks #
		#####################

		# !!! Here NMS is used to filter out detections that are already covered by tracks. This is
		# !!! done by iterating through the active tracks one by one, assigning them a bigger score
		# !!! than 1 (maximum score for detections) and then filtering the detections with NMS.
		# !!! In the paper this is done by calculating the overlap with existing tracks, but the
		# !!! result stays the same.
		if det_pos.nelement() > 0:
			keep = nms(det_pos, det_scores, self.detection_nms_thresh)
			det_pos = det_pos[keep]
			det_scores = det_scores[keep]

			# check with every track in a single run (problem if tracks delete each other)
			for t in self.tracks:
				nms_track_pos = torch.cat([t.pos, det_pos])
				nms_track_scores = torch.cat(
					[torch.tensor([2.0]).to(det_scores.device), det_scores])
				keep = nms(nms_track_pos, nms_track_scores, self.detection_nms_thresh)

				keep = keep[torch.ge(keep, 1)] - 1

				det_pos = det_pos[keep]
				det_scores = det_scores[keep]
				if keep.nelement() == 0:
					break

		if det_pos.nelement() > 0:
			new_det_pos = det_pos
			new_det_scores = det_scores

			# try to reidentify tracks
			# TODO: tracks should contain position and corresponding center motion: vx, vy and appearance feature descriptor
			start_reid = time.time()
			new_det_pos, new_det_scores, new_det_features = self.reid(blob, new_det_pos, new_det_scores)

			#update patience and rads first
			if len(self.inactive_tracks)>=1:
				if cam == 3:
					max_radious = 300
					max_patience = 80
				if cam == 1:
					max_radious = 250
					max_patience = 60
				else:
					max_radious = 250
					max_patience = 50

				reid_rads = self.update_reid_rads(max_rads=max_radious,
												  max_patience=max_patience,
												  cam=cam)

				if new_det_pos.nelement() > 0 and cam in [3,5,1, 360, 340, 440, 300] and self.isSCT_ReID:
					#assert len(new_det_pos)==1, 'found {} new dets at {}'.format(len(new_det_pos), self.im_index)

					new_det_pos, \
					new_det_scores, \
					new_det_features = self.sct_reid(blob,
													 new_det_pos,
													 new_det_scores,
													 reid_rads=reid_rads,
													 cam=cam,
													 reid_thr=3.0,
													 obj_reid_rads=max_radious)
					print('double reid time: {} sec'.format(time.time() - start_reid))

			# add new: use global track increment for multi-cameras: same id in different camera is not allowed
			#add new track starting time
			if new_det_pos.nelement() > 0:
				self.add(new_det_pos, new_det_scores, new_det_features)

		####################
		# Generate Results #
		# TODO: Results should contain pos
		# TODO: Results should contain motion v_cx, v_cy
		# TODO: Results should contain appearance descriptor
		####################

		for t in self.tracks:
			if t.id not in self.results.keys():
				self.results[t.id] = {}
			#self.results[t.id][self.im_index] = np.concatenate([t.pos[0].cpu().numpy(), np.array([t.score])])
			#save track state with camera index: without velocity and appearance feature
			if cam is not None:
				self.results[t.id][self.im_index] = np.concatenate([t.pos[0].cpu().numpy(), np.array([t.score]),  np.array([cam])])#, velocity.reshape(2), t.features[0].cpu().numpy().reshape(128)])
			else:
				if len(t.last_v.cpu().numpy()) == 0:
					# motion is not estimated yet
					velocity = np.array([0., 0.])
				else:
					velocity = t.last_v.cpu().numpy()
				self.results[t.id][self.im_index] = np.concatenate([t.pos[0].cpu().numpy(), np.array([t.score])])  # , velocity.reshape(2), t.features[0].cpu().numpy().reshape(128)])

		for t in self.inactive_tracks:
			t.count_inactive += 1

		#self.inactive_tracks = [
			#t for t in self.inactive_tracks if t.has_positive_area() and t.count_inactive <= self.inactive_patience
		#]
		#keep inactive tracks until inactive patience
		self.inactive_tracks = [
			t for t in self.inactive_tracks if t.has_positive_area()
											   and t.count_inactive <= self.reid_rads_patience[int(t.id)]
											   and self.reid_rads_patience[int(t.id)]<60 #max wait time
		]

		self.im_index += 1
		self.last_image = blob['img'][0]

	def get_results(self):
		return self.results


class Track(object):
	"""This class contains all necessary for every individual track."""

	def __init__(self, pos, score, track_id, features, inactive_patience, max_features_num, mm_steps):
		self.id = track_id
		self.pos = pos
		self.score = score
		self.features = deque([features])
		self.ims = deque([])
		self.count_inactive = 0
		self.inactive_patience = inactive_patience
		self.max_features_num = max_features_num
		self.last_pos = deque([pos.clone()], maxlen=mm_steps + 1)
		self.last_v = torch.Tensor([])
		self.gt_id = None

	def has_positive_area(self):
		# is x2>x1 and y2>y1
		# TODO: thresholded by target area
		return self.pos[0, 2] > self.pos[0, 0] and self.pos[0, 3] > self.pos[0, 1]

	def add_features(self, features):
		"""Adds new appearance features to the object."""
		self.features.append(features)
		if len(self.features) > self.max_features_num:
			self.features.popleft()

	def test_features(self, test_features):
		"""Compares test_features to features of this Track object"""
		if len(self.features) > 1:
			features = torch.cat(list(self.features), dim=0)
		else:
			features = self.features[0]
		features = features.mean(0, keepdim=True)
		dist = F.pairwise_distance(features, test_features, keepdim=True)
		return dist

	def reset_last_pos(self):
		self.last_pos.clear()
		self.last_pos.append(self.pos.clone())
