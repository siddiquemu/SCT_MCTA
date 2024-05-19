import configparser
import csv
import os
import os.path as osp

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

import cv2

from ..config import cfg
from torchvision.transforms import ToTensor
import pdb

class MOT17Sequence(Dataset):
    """Multiple Object Tracking Dataset.

    This dataloader is designed so that it can handle only one sequence, if more have to be
    handled one should inherit from this class.
    """

    def __init__(self, seq_name=None, dets='', vis_threshold=0.0,
                 normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225]):
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
        """
        self._seq_name = seq_name
        self._dets = dets
        self._vis_threshold = vis_threshold

        self._mot_dir = osp.join(cfg.DATA_DIR, 'MOT17Det')
        self._label_dir = osp.join(cfg.DATA_DIR, 'MOT16Labels')
        self._raw_label_dir = osp.join(cfg.DATA_DIR, 'MOT16-det-dpm-raw')
        self._mot17_label_dir = osp.join(cfg.DATA_DIR, 'MOT17Labels')

        self._train_folders = os.listdir(os.path.join(self._mot_dir, 'train_gt'))
        self._test_folders = os.listdir(os.path.join(self._mot_dir, 'test'))

        self.transforms = ToTensor()

        if seq_name is not None:
            assert seq_name in self._train_folders or seq_name in self._test_folders, \
                'Image set does not exist: {}'.format(seq_name)

            self.data, self.no_gt = self._sequence()
        else:
            self.data = []
            self.no_gt = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return the ith image converted to blob"""
        data = self.data[idx]
        img = Image.open(data['im_path']).convert("RGB")
        img = self.transforms(img)

        sample = {}
        sample['img'] = img
        sample['dets'] = torch.tensor([det[:4] for det in data['dets']])
        sample['img_path'] = data['im_path']
        sample['gt'] = data['gt']
        sample['vis'] = data['vis']

        return sample
    # function used for clasp data loader
    def _sequence(self):
        seq_name = self._seq_name
        if seq_name in self._train_folders:
            #seq_path = osp.join(self._mot_dir, 'train_gt', seq_name)
            #clasp: train when use PANet dets as GT, train_gt: CLASP2 gt at each 100th frame (30FPS)
            seq_path = osp.join(self._mot_dir, 'train_gt', seq_name)
            #label_path = osp.join(self._label_dir, 'train', 'MOT16-'+seq_name[-2:])
            label_path = osp.join(self._label_dir, seq_name)
            mot17_label_path = self._mot17_label_dir
        else:
            seq_path = osp.join(self._mot_dir, 'test', seq_name)
            label_path = osp.join(self._label_dir, 'test', 'MOT16-'+seq_name[-2:])
            mot17_label_path = osp.join(self._mot17_label_dir, 'test')
        raw_label_path = osp.join(self._raw_label_dir, 'MOT16-'+seq_name[-2:])

        config_file = osp.join(seq_path, 'seqinfo.ini')

        assert osp.exists(config_file), \
            'Config file does not exist: {}'.format(config_file)

        config = configparser.ConfigParser()
        config.read(config_file)
        seqLength = int(config['Sequence']['seqLength'])
        imDir = config['Sequence']['imDir']

        imDir = osp.join(seq_path, imDir)
        gt_file = osp.join(seq_path, 'gt', 'gt.txt')

        total = []
        train = []
        val = []

        visibility = {}
        boxes = {}
        dets = {}

        for i in range(1, seqLength+1):#
            boxes[i] = {}
            visibility[i] = {}
            dets[i] = []
        #we are not using clasp gt
        no_gt = False
        if osp.exists(gt_file): # check the gt file is exist
            with open(gt_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    #pdb.set_trace()
                    # class person, certainity 1, visibility >= 0.25
                    if int(float(row[6])) == 1 and int(float(row[7])) == 1 and float(row[8]) >= self._vis_threshold:
                        # Make pixel indexes 0-based, should already be 0-based (or not)
                        x1 = int(float(row[2])) - 1
                        y1 = int(float(row[3])) - 1
                        # This -1 accounts for the width (width of 1 x1=x2)
                        x2 = x1 + int(float(row[4])) - 1
                        y2 = y1 + int(float(row[5])) - 1
                        bb = np.array([x1,y1,x2,y2], dtype=np.float32)
                        boxes[int(float(row[0]))][int(float(row[1]))] = bb
                        visibility[int(float(row[0]))][int(float(row[1]))] = float(row[8])
        else:
            no_gt = True

        det_file = self.get_det_file(label_path, raw_label_path, mot17_label_path)

        if osp.exists(det_file) and no_gt: # check the det file is exist to use as gt
            with open(det_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')

                for row in reader:
                    if seqLength>=float(row[0]):
                        x1 = float(row[2]) - 1
                        y1 = float(row[3]) - 1
                        # This -1 accounts for the width (width of 1 x1=x2)
                        x2 = x1 + float(row[4]) - 1
                        y2 = y1 + float(row[5]) - 1
                        score = float(row[6])
                        bb = np.array([x1,y1,x2,y2, score], dtype=np.float32)
                        dets[int(float(row[0]))].append(bb)
        #pdb.set_trace()
        for i in range(1,seqLength+1): # for mot jpg, for clasp png
            #TODO: use gt image frames from neu server to ignore this delay
            #boxes are available only for gt frames: boxes[2200], boxes[2300]
            im_path = osp.join(imDir,"{:06d}.png".format(i))#i+5, 05d.jpg
            print(im_path)
            assert os.path.exists(img_path), \
                'Path does not exist: {img_path}'
            #save image path if image has detection (some frames in the video may have no detection)
            if i in gt_frameset:
                self._img_paths.append(img_path)
                print(img_path)
                print(boxes[i])
                sample = {'gt': boxes[i],
                          'im_path': im_path,
                          'vis': visibility[i],
                          'dets': dets[i], }

                total.append(sample)
            else:
                continue


        return total, no_gt


    def get_det_file(self, label_path, raw_label_path, mot17_label_path):
        #pdb.set_trace()
        if self._dets == "DPM":
            det_file = osp.join(label_path, 'det', 'det.txt')
        elif self._dets == "DPM_RAW16":
            det_file = osp.join(raw_label_path, 'det', 'det-dpm-raw.txt')
        elif "17" in self._seq_name:
            det_file = osp.join(
                mot17_label_path,
                f"{self._seq_name}-{self._dets[:-2]}",
                'det',
                'det.txt')
        # get det file for clasp videos
        elif "cam02exp2.mp4" in self._seq_name:
            det_file = osp.join(
                mot17_label_path,
                self._seq_name,
                'det',self._seq_name+'.txt')
        else:
            det_file = ""
        return det_file

    def __str__(self):
        return f"{self._seq_name}-{self._dets[:-2]}"

    def write_results(self, all_tracks, output_dir):
        """Write the tracks in the format for MOT16/MOT17 sumbission

        all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        Files to sumbit:
        ./MOT16-01.txt
        ./MOT16-02.txt
        ./MOT16-03.txt
        ./MOT16-04.txt
        ./MOT16-05.txt
        ./MOT16-06.txt
        ./MOT16-07.txt
        ./MOT16-08.txt
        ./MOT16-09.txt
        ./MOT16-10.txt
        ./MOT16-11.txt
        ./MOT16-12.txt
        ./MOT16-13.txt
        ./MOT16-14.txt
        """

        #format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

        assert self._seq_name is not None, "[!] No seq_name, probably using combined database"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if "17" in self._dets:
            file = osp.join(output_dir, 'MOT17-'+self._seq_name[6:8]+"-"+self._dets[:-2]+'.txt')
        else:
            file = osp.join(output_dir, self._seq_name+'.txt')

        with open(file, "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    #writer.writerow([frame+1, i+1, x1+1, y1+1, x2-x1+1, y2-y1+1, -1, -1, -1, -1])
                    writer.writerow([frame + 1, i + 1, x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, -1, -1, -1, -1])

    def write_results_clasp(self, all_tracks, output_dir, cam=None):
        """Write the tracks in the format for MOT16/MOT17 sumbission

        all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        Files to sumbit:
        ./MOT16-01.txt
        ./MOT16-02.txt
        ./MOT16-03.txt
        ./MOT16-04.txt
        ./MOT16-05.txt
        ./MOT16-06.txt
        ./MOT16-07.txt
        ./MOT16-08.txt
        ./MOT16-09.txt
        ./MOT16-10.txt
        ./MOT16-11.txt
        ./MOT16-12.txt
        ./MOT16-13.txt
        ./MOT16-14.txt
        """

        #format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

        assert self._seq_name is not None, "[!] No seq_name, probably using combined database"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if "17" in self._dets:
            file = osp.join(output_dir, 'MOT17-'+self._seq_name[6:8]+"-"+self._dets[:-2]+'.txt')
        else:
            file = osp.join(output_dir, self._seq_name+'.txt')

        with open(file, "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    score = bb[4]
                    v_cx = bb[5]
                    v_cy = bb[6]
                    app = bb[7::]
                    #writer.writerow([frame+1, i+1, x1+1, y1+1, x2-x1+1, y2-y1+1, -1, -1, -1, -1])
                    #f.write("\n".join(" ".join(map(str, x)) for x in (a, b)))
                    if cam is not None:
                        #[fr,id,x,y,w,h]
                        t_feature = [frame + 1, i + 1, x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, cam]
                    else:
                        t_feature = [frame + 1, i + 1, x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, -1, -1, -1, -1, v_cx, v_cy]
                    writer.writerow(t_feature+app.tolist())


class CLASPSequence(MOT17Sequence):

    def __init__(self, seq_name=None, dets='', vis_threshold=0.0,
                 normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225]):
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
        """
        self._seq_name = seq_name
        self._dets = dets
        self._vis_threshold = vis_threshold
        #image dir
        self._mot_dir = osp.join(cfg.DATA_DIR, 'CLASP')
        #det dir
        self._mot17_label_dir = osp.join(cfg.DATA_DIR, 'train')

        # TODO: refactor code of both classes to consider 16,17 and 19
        self._label_dir = osp.join(cfg.DATA_DIR, 'MOT16Labels')
        self._raw_label_dir = osp.join(cfg.DATA_DIR, 'MOT16-det-dpm-raw')
        # sequence folders
        self._train_folders = os.listdir(os.path.join(self._mot_dir, 'train'))
        self._test_folders = os.listdir(os.path.join(self._mot_dir, 'test'))

        self.transforms = ToTensor()

        if seq_name is not None:
            assert seq_name in self._train_folders or seq_name in self._test_folders, \
                'Image set does not exist: {}'.format(seq_name)

            self.data, self.no_gt = self._sequence()
        else:
            self.data = []
            self.no_gt = True


class CLASPDataloader(Dataset):
    """Multiple Object Tracking CLASP Dataset.

    This dataloader is designed so that it can handle only one sequence, if more have to be
    handled one should inherit from this class.
    """

    def __init__(self, seq_name=None, dets='', det_score=0.9,
                 frame_offset=0, vis_threshold=0.0, data_name='CLASP',
                 normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225]):
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
        """
        self._seq_name = seq_name
        self._dets = dets
        self._det_score = det_score
        self._vis_threshold = vis_threshold
        self._frame_offset = frame_offset
        self.data_name = data_name

        #self._clasp_dir = osp.join(cfg.DATA_DIR, 'CLASP') #CLASP2
        #self._clasp_dir = osp.join(cfg.DATA_DIR, 'CLASP1')  # CLASP1
        self._clasp_dir = osp.join(cfg.DATA_DIR, 'PVD/HDPVD_new')

        #self._train_folders = os.listdir(os.path.join(self._clasp_dir, 'train_gt'))
        #self._train_folders = os.listdir(os.path.join(self._clasp_dir, 'train_gt_all/PB_gt'))
        self._train_folders = os.listdir(os.path.join(self._clasp_dir, 'train_gt'))
        #self._test_folders = os.listdir(os.path.join(self._clasp_dir, 'test'))

        self.transforms = ToTensor()

        if seq_name is not None:
            assert seq_name in self._train_folders or seq_name in self._test_folders, \
                'Image set does not exist: {}'.format(seq_name)

            self.data, self.no_gt = self._sequence()
        else:
            self.data = []
            self.no_gt = True

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """Return the ith image converted to blob"""
        data = self.data[idx]
        img = Image.open(data['im_path']).convert("RGB")
        img = self.transforms(img)

        sample = {}
        sample['img'] = img
        sample['dets'] = torch.tensor([det[:4] for det in data['dets']])
        sample['img_path'] = data['im_path']
        sample['gt'] = data['gt']
        sample['vis'] = data['vis']

        return sample

    # function used for clasp data loader
    def _sequence(self):
        seq_name = self._seq_name
        if seq_name in self._train_folders:
            seq_path = osp.join(self._clasp_dir, 'train_gt', seq_name)
            #label_path = osp.join(self._clasp_dir, seq_name)

            #mot17_label_path = self._mot17_label_dir
        else:
            print('{} not found'.format(seq_name))
            #seq_path = osp.join(self._mot_dir, 'test', seq_name)
            #label_path = osp.join(self._label_dir, 'test', 'MOT16-' + seq_name[-2:])
            #mot17_label_path = osp.join(self._mot17_label_dir, 'test')
        #raw_label_path = osp.join(self._raw_label_dir, 'MOT16-' + seq_name[-2:])

        config_file = osp.join(seq_path, 'seqinfo.ini')

        assert osp.exists(config_file), \
            'Config file does not exist: {}'.format(config_file)

        config = configparser.ConfigParser()
        config.read(config_file)
        seqLength = int(config['Sequence']['seqLength'])
        imDir = config['Sequence']['imDir']
        im_ext = config['Sequence']['imExt']
        #read all the gt dets
        gt_file = osp.join(seq_path, 'gt', 'gt.txt')
        GT = np.loadtxt(gt_file, delimiter=',')
        #separate frames for person or bag class in clasp2 - 1:person, 2: bag
        GT = GT[GT[:,-2]==1]
        gt_frameset = GT[:, 0].astype(
            'int') + self._frame_offset  # score thresholding is necessary when dets are used to train
        print('Frames: {}'.format(np.unique(gt_frameset)))
        print('class id: {}'.format(GT[:,-2]))

        imDir = osp.join(seq_path, imDir)
        img_paths = {}
        for i in np.unique(gt_frameset):#range(1, seqLength + 1):
            if seq_name in ['G_9', 'G_11']:
                img_path = os.path.join(imDir, f"{i:05d}{im_ext}") #clasp1
            else:
                img_path = os.path.join(imDir, f"{i:06d}{im_ext}")
            assert os.path.exists(img_path), \
                'Path does not exist: {}'.format(img_path)
            # self._img_paths.append((img_path, im_width, im_height))
            # save image path if image has detection (some frames in the video may have no detection)

            img_paths[i] = img_path


        total = []
        train = []
        val = []

        visibility = {}
        boxes = {}
        dets = {}

        for i in np.unique(gt_frameset):#range(1, seqLength + 1):  #
            boxes[i] = {}
            visibility[i] = {}
            dets[i] = []
        # we are not using clasp gt
        no_gt = False
        if osp.exists(gt_file):  # check the gt file is exist
            with open(gt_file, "r") as inf:
                reader = csv.reader(inf, delimiter=',')
                for row in reader:
                    # pdb.set_trace()
                    # class person, certainity 1, visibility >= 0.25
                    if int(float(row[6])) == 1 and int(float(row[7])) == 1 and int(float(row[-2]))==1 \
                            and float(row[8]) >= self._vis_threshold:
                        # Make pixel indexes 0-based, should already be 0-based (or not)
                        x1 = int(float(row[2])) - 1
                        y1 = int(float(row[3])) - 1
                        # This -1 accounts for the width (width of 1 x1=x2)
                        x2 = x1 + int(float(row[4])) - 1
                        y2 = y1 + int(float(row[5])) - 1
                        bb = np.array([x1, y1, x2, y2], dtype=np.float32)
                        boxes[int(float(row[0])+self._frame_offset)][int(float(row[1]))] = bb #boxes['frame'+'offset']['ID'] = bb
                        visibility[int(float(row[0])+self._frame_offset)][int(float(row[1]))] = float(row[8])
        else:
            no_gt = True

        # pdb.set_trace()
        #for i in range(1, seqLength + 1):  # for mot jpg, for clasp png
            # TODO: use gt image frames from neu server to ignore this delay
            # boxes are available only for gt frames: boxes[2200], boxes[2300]
            # save image path if image has detection (some frames in the video may have no detection)
        for i in np.unique(gt_frameset):
            img_path = img_paths[i]
            print(img_path)
            print(boxes[i])
            sample = {'gt': boxes[i],
                      'im_path': img_path,
                      'vis': visibility[i],
                      'dets': dets[i], }

            total.append(sample)
        #else:
            #continue

        return total, no_gt

    def __str__(self):
        return f"{self._seq_name}-{self._dets[:-2]}"

    def write_results_clasp(self, all_tracks, output_dir, cam=None):
        """Write the tracks in the format for MOT16/MOT17 sumbission

        all_tracks: dictionary with 1 dictionary for every track with {..., i:np.array([x1,y1,x2,y2]), ...} at key track_num

        Each file contains these lines:
        <frame>, <id>, <bb_left>, <bb_top>, <bb_width>, <bb_height>, <conf>, <x>, <y>, <z>

        Files to sumbit:
        ./MOT16-01.txt
        ./MOT16-02.txt
        ./MOT16-03.txt
        ./MOT16-04.txt
        ./MOT16-05.txt
        ./MOT16-06.txt
        ./MOT16-07.txt
        ./MOT16-08.txt
        ./MOT16-09.txt
        ./MOT16-10.txt
        ./MOT16-11.txt
        ./MOT16-12.txt
        ./MOT16-13.txt
        ./MOT16-14.txt
        """

        # format_str = "{}, -1, {}, {}, {}, {}, {}, -1, -1, -1"

        assert self._seq_name is not None, "[!] No seq_name, probably using combined database"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if "17" in self._dets:
            file = osp.join(output_dir, 'MOT17-' + self._seq_name[6:8] + "-" + self._dets[:-2] + '.txt')
        else:
            file = osp.join(output_dir, self._seq_name + '.txt')

        with open(file, "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    score = bb[4]
                    v_cx = bb[5]
                    v_cy = bb[6]
                    app = bb[7::]
                    # writer.writerow([frame+1, i+1, x1+1, y1+1, x2-x1+1, y2-y1+1, -1, -1, -1, -1])
                    # f.write("\n".join(" ".join(map(str, x)) for x in (a, b)))
                    if cam is not None:
                        # [fr,id,x,y,w,h]
                        t_feature = [frame + 1, i + 1, x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, cam]
                    else:
                        t_feature = [frame + 1, i + 1, x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, -1, -1, -1, -1, v_cx,
                                     v_cy]
                    writer.writerow(t_feature + app.tolist())

class CLASPSGTsequence(CLASPDataloader):

    def __init__(self, seq_name=None, dets='', det_score=0.9, frame_offset=0, vis_threshold=0.0,
                 normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225]):
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
        """
        self._seq_name = seq_name
        self._dets = dets
        self._det_score = det_score
        self._vis_threshold = vis_threshold
        self._frame_offset = frame_offset
        # invoking the constructor of
        # the parent class
        #super(CLASPSGTsequence, self).__init__()
        #self._clasp_dir = osp.join(cfg.DATA_DIR, 'CLASP') #CLASP2
        #self._clasp_dir = osp.join(cfg.DATA_DIR, 'CLASP1')  # CLASP1
        self._clasp_dir = osp.join(cfg.DATA_DIR, 'PVD/HDPVD_new')

        #self._train_folders = os.listdir(os.path.join(self._clasp_dir, 'train_gt'))
        #self._train_folders = os.listdir(os.path.join(self._clasp_dir, 'train_gt_all/PB_gt'))
        self._train_folders = os.listdir(os.path.join(self._clasp_dir, 'train_gt'))
        #self._test_folders = os.listdir(os.path.join(self._clasp_dir, 'test'))

        self.transforms = ToTensor()

        if seq_name is not None:
            assert seq_name in self._train_folders or seq_name in self._test_folders, \
                'Image set does not exist: {}'.format(seq_name)

            self.data, self.no_gt = self._sequence()
        else:
            self.data = []
            self.no_gt = True

class MOT19Sequence(MOT17Sequence):

    def __init__(self, seq_name=None, dets='', vis_threshold=0.0,
                 normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225]):
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
        """
        self._seq_name = seq_name
        self._dets = dets
        self._vis_threshold = vis_threshold

        self._mot_dir = osp.join(cfg.DATA_DIR, 'MOT19')
        self._mot17_label_dir = osp.join(cfg.DATA_DIR, 'MOT19')

        # TODO: refactor code of both classes to consider 16,17 and 19
        self._label_dir = osp.join(cfg.DATA_DIR, 'MOT16Labels')
        self._raw_label_dir = osp.join(cfg.DATA_DIR, 'MOT16-det-dpm-raw')

        self._train_folders = os.listdir(os.path.join(self._mot_dir, 'train'))
        self._test_folders = os.listdir(os.path.join(self._mot_dir, 'test'))

        self.transforms = ToTensor()

        if seq_name is not None:
            assert seq_name in self._train_folders or seq_name in self._test_folders, \
                'Image set does not exist: {}'.format(seq_name)

            self.data, self.no_gt = self._sequence()
        else:
            self.data = []
            self.no_gt = True

    def get_det_file(self, label_path, raw_label_path, mot17_label_path):
        # FRCNN detections
        if "MOT19" in self._seq_name:
            det_file = osp.join(mot17_label_path, self._seq_name, 'det', 'det.txt')
        else:
            det_file = ""
        return det_file

    def write_results(self, all_tracks, output_dir):
        assert self._seq_name is not None, "[!] No seq_name, probably using combined database"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file = osp.join(output_dir, f'{self._seq_name}.txt')

        print("[*] Writing to: {}".format(file))

        with open(file, "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    writer.writerow(
                        [frame + 1, i + 1, x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, -1, -1, -1, -1])


class MOT20Sequence(MOT17Sequence):

    def __init__(self, seq_name=None, dets='', vis_threshold=0.0,
                 normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225]):
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
        """
        self._seq_name = seq_name
        self._dets = dets
        self._vis_threshold = vis_threshold

        self._mot_dir = osp.join(cfg.DATA_DIR, 'MOT20')
        self._mot17_label_dir = osp.join(cfg.DATA_DIR, 'MOT20')

        # TODO: refactor code of both classes to consider 16,17 and 19
        self._label_dir = osp.join(cfg.DATA_DIR, 'MOT16Labels')
        self._raw_label_dir = osp.join(cfg.DATA_DIR, 'MOT16-det-dpm-raw')

        self._train_folders = os.listdir(os.path.join(self._mot_dir, 'train'))
        self._test_folders = os.listdir(os.path.join(self._mot_dir, 'test'))

        self.transforms = ToTensor()

        if seq_name is not None:
            assert seq_name in self._train_folders or seq_name in self._test_folders, \
                'Image set does not exist: {}'.format(seq_name)

            self.data, self.no_gt = self._sequence()
        else:
            self.data = []
            self.no_gt = True

    def get_det_file(self, label_path, raw_label_path, mot17_label_path):
        # FRCNN detections
        if "MOT20" in self._seq_name:
            det_file = osp.join(mot17_label_path, self._seq_name, 'det', 'det.txt')
        else:
            det_file = ""
        return det_file

    def write_results(self, all_tracks, output_dir):
        assert self._seq_name is not None, "[!] No seq_name, probably using combined database"

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        file = osp.join(output_dir, f'{self._seq_name}.txt')

        print("[*] Writing to: {}".format(file))

        with open(file, "w") as of:
            writer = csv.writer(of, delimiter=',')
            for i, track in all_tracks.items():
                for frame, bb in track.items():
                    x1 = bb[0]
                    y1 = bb[1]
                    x2 = bb[2]
                    y2 = bb[3]
                    writer.writerow(
                        [frame + 1, i + 1, x1 + 1, y1 + 1, x2 - x1 + 1, y2 - y1 + 1, -1, -1, -1, -1])


class MOT17LOWFPSSequence(MOT17Sequence):

    def __init__(self, split, seq_name=None, dets='', vis_threshold=0.0,
                 normalize_mean=[0.485, 0.456, 0.406],
                 normalize_std=[0.229, 0.224, 0.225]):
        """
        Args:
            seq_name (string): Sequence to take
            vis_threshold (float): Threshold of visibility of persons above which they are selected
        """
        self._seq_name = seq_name
        self._dets = dets
        self._vis_threshold = vis_threshold

        self._mot_dir = osp.join(cfg.DATA_DIR, 'MOT17_LOW_FPS', f'MOT17_{split}_FPS')
        self._mot17_label_dir = osp.join(cfg.DATA_DIR, 'MOT17_LOW_FPS', f'MOT17_{split}_FPS')

        # TODO: refactor code of both classes to consider 16,17 and 19
        self._label_dir = osp.join(cfg.DATA_DIR, 'MOT16Labels')
        self._raw_label_dir = osp.join(cfg.DATA_DIR, 'MOT16-det-dpm-raw')

        self._train_folders = os.listdir(os.path.join(self._mot_dir, 'train'))
        self._test_folders = os.listdir(os.path.join(self._mot_dir, 'test'))

        self.transforms = Compose([ToTensor(), Normalize(normalize_mean,
                                                         normalize_std)])

        if seq_name is not None:
            assert seq_name in self._train_folders or seq_name in self._test_folders, \
                'Image set does not exist: {}'.format(seq_name)

            self.data, self.no_gt = self._sequence()
        else:
            self.data = []
            self.no_gt = True
