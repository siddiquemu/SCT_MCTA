import configparser
import csv
import os
import os.path as osp
import pickle

from PIL import Image
import numpy as np
import scipy
import torch

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import utils


def get_detection_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.roi_heads.nms_thresh = 0.3

    return model

def get_transform(train):
    transforms = []
    # converts the image, a PIL image, into a PyTorch Tensor
    transforms.append(T.ToTensor())
    if train:
        # during training, randomly flip the training images
        # and ground-truth for data augmentation
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

# use our dataset and defined transformations
#dataset = MOT17ObjDetect(codeBase+'/data/CLASP/train_gt', get_transform(train=True))
#dataset_no_random = MOT17ObjDetect(codeBase+'/data/CLASP/train_gt', get_transform(train=False))

#dataset_test = MOT17ObjDetect(codeBase+'/data/CLASP/test', get_transform(train=False))

#wild-track dataset
dataset = MulticamObjDetect(data_path+'/train_gt', get_transform(train=True))
dataset_no_random = MulticamObjDetect(data_path+'/train_gt', get_transform(train=False))

# split the dataset in train and test set
torch.manual_seed(1)
#indices = torch.randperm(len(dataset)).tolist()
#dataset = torch.utils.data.Subset(dataset, indices[:-800])
#dataset_test = torch.utils.data.Subset(dataset_test, indices[-800:])

# define training and validation data loaders

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=2, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)
data_loader_no_random = torch.utils.data.DataLoader(
    dataset_no_random, batch_size=1, shuffle=False, num_workers=4,
    collate_fn=utils.collate_fn)

#data_loader_test = torch.utils.data.DataLoader(
    #dataset_test, batch_size=1, shuffle=False, num_workers=4,
    #collate_fn=utils.collate_fn)

#initialize model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# get the model using our helper function
model = get_detection_model(dataset.num_classes)
# move model to the right device
model.to(device)

model_state_dict = torch.load("/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d6/tracking_wo_bnw/output/faster_rcnn_fpn_training_mot_17/model_epoch_27.model")
#model_state_dict = torch.load("/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d6/tracking_wo_bnw/output/faster_rcnn_fpn_training_clasp/model_epoch_100.model")

model.load_state_dict(model_state_dict)

# construct an optimizer
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.00001,
                            momentum=0.9, weight_decay=0.0005)

# and a learning rate scheduler which decreases the learning rate by
# 10x every 3 epochs
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=10,
                                               gamma=0.1)
#train
num_epochs = 100
import os

# model_save_path = "/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d6/tracking_wo_bnw/output/faster_rcnn_fpn_training_clasp_gt_panet/"
model_save_path = "/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d6/tracking_wo_bnw/output/training_wild_track/"
# model_save_path = "/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d6/tracking_wo_bnw/output/train_logan_data/"

if not os.path.exists(model_save_path):
    os.makedirs(model_save_path)

for epoch in range(1, num_epochs + 1):
    print('epoch {}'.format(epoch))
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)
    # update the learning rate
    # lr_scheduler.step()
    # evaluate on the test dataset
    if epoch % 10 == 0:
        # evaluate_and_write_result_files(model, data_loader_no_random)
        torch.save(model.state_dict(), model_save_path + "model_epoch_{}.model".format(epoch))