import configparser
import csv
import os
import os.path as osp
import pickle

from PIL import Image
import numpy as np
import scipy
import torch

class CLASP2ObjDetect(torch.utils.data.Dataset):
    """ Data class for the CLASP Multi-View Datasets:
    """

    def __init__(self, annotations=None, num_classes=None, imgs=None, transforms=None,
                 cam_data_split=0.8, data_type='train', vis_threshold=0.0):
        self.ann = annotations
        self.transforms = transforms
        self._vis_threshold = vis_threshold
        self._classes = ('background', 'passenger')
        if num_classes==3:
            self._classes = ('background', 'passenger', 'tso')
        self._img_paths = imgs

        #TODO: use augnebted det set as GT
        GT = self.ann
        assert len(GT)>0
        # TODO: semi-supervised approach: use few percent of total annotations
        '''
        if self.num_classes==2:
            GTp = [gt for gt in GT if gt[9]==1]
        else:
            GTb = [gt for gt in GT if gt[9]==2]
        gt_frameset = np.unique(GT[:,0].astype('int'))
        gt_len = len(gt_frameset)
        print('full set {}: Nfr {}, person {} bag {}'.format(seq_name, gt_len, len(GTp), len(GTb)))

        #Apply split 80% for train and 20% for test
        #random split: keep split similar for training and testing forever
        random.seed(42)
        train_subset = random.sample(list(gt_frameset), int(gt_len*cam_data_split))
        print('random sample {}'.format(train_subset[2]))

        train_GTp = [gt for gt in GT if gt[0] in train_subset and gt[9]==1]
        train_GTb = [gt for gt in GT if gt[0] in train_subset and gt[9]==2]
        print('train split {}: Nfr {}, person {} bag {}'.format(seq_name, len(train_subset), len(train_GTp), len(train_GTb)))

        test_subset = np.array([t for t in gt_frameset if t not in train_subset])
        test_GTp = [gt for gt in GT if gt[0] not in train_subset and gt[9]==1]
        test_GTb = [gt for gt in GT if gt[0] not in train_subset and gt[9]==2]
        print('test split {}: Nfr {}, person {} bag {}'.format(seq_name, len(test_subset), len(test_GTp), len(test_GTb)))
        print('-------------------------------------------------------')
        assert len(gt_frameset)== len(test_subset)+len(train_subset),'split is not correct'
        '''



    @property
    def num_classes(self):
        return len(self._classes)

    def _get_annotation(self, idx):
        """
        """
        img_path = self._img_paths[idx]
        file_index = int(os.path.basename(img_path).split('.')[0])

        if self.ann is None:
            gt_file = os.path.join(os.path.dirname(
                os.path.dirname(img_path)), 'gt', 'gt.txt')

            assert os.path.exists(gt_file), \
                'GT file does not exist: {}'.format(gt_file)

        bounding_boxes = []
        for row in self.ann:
            #print(row)
            if int(float(row[0])) == file_index: #and float(row[6]) >= 0.9 and int(float(row[7])) == 1:
                bb = {}
                #print(row)
                bb['bb_left'] = int(row[2])
                bb['bb_top'] = int(row[3])
                bb['bb_width'] = int(row[4])
                bb['bb_height'] = int(row[5])
                bb['visibility'] = 1.0
                bb['labels'] = int(row[9]) # based on the available class in the dataset
                #bb['cam'] = int(row[10]) # not used in monocular detector model
                bounding_boxes.append(bb)

        num_objs = len(bounding_boxes)

        boxes = torch.zeros((num_objs, 4), dtype=torch.float32)
        labels = []
        visibilities = torch.zeros((num_objs), dtype=torch.float32)
        
        for i, bb in enumerate(bounding_boxes):
            # Make pixel indexes 0-based, should already be 0-based (or not)
            x1 = bb['bb_left'] - 1
            y1 = bb['bb_top'] - 1
            # This -1 accounts for the width (width of 1 x1=x2)
            x2 = x1 + bb['bb_width'] - 1
            y2 = y1 + bb['bb_height'] - 1

            boxes[i, 0] = x1
            boxes[i, 1] = y1
            boxes[i, 2] = x2
            boxes[i, 3] = y2
            visibilities[i] = bb['visibility']
            labels.append(bb['labels'])
        #when return annotations for one image: make sure the labels for multi-class dataset   
        return {'boxes': boxes,
                'labels': torch.tensor(labels, dtype=torch.int64),
                'image_id': torch.tensor([idx]),
                'area': (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]),
                'iscrowd': torch.zeros((num_objs,), dtype=torch.int64),
                'visibilities': visibilities,}

    def __getitem__(self, idx):
        # load images ad masks
        img_path = self._img_paths[idx]

        # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")

        target = self._get_annotation(idx)
        
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self._img_paths)
