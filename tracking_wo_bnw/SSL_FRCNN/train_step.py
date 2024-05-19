import torchvision
import torch
import sys
from pathlib import Path
import os
FILE = Path(__file__).resolve()
SSL_ROOT = FILE.parents[0]
if str(SSL_ROOT) not in sys.path:
    sys.path.append(str(SSL_ROOT))
ROOT = Path(os.path.relpath(SSL_ROOT, Path.cwd()))
sys.path.insert(0, f'{str(SSL_ROOT.parent)}/tracking_wo_bnw')
sys.path.insert(1, f'{str(SSL_ROOT.parent)}/tracking_wo_bnw/src/tracktor')
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import transforms as T
import utils
from clasp2dataloader import *
import numpy as np
import random
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection import FasterRCNN
from torch.hub import load_state_dict_from_url

class SST_Model(object):
    def __init__(self, num_classes=2, nms_thresh=0.3, load_iter_checkpoint=0, train_gpu=0,
                 backbone='ResNet50FPN', pretrained_model_path=None,
                 model_save_path=None, train_data=None, train_imgs=None, num_epochs=100, cam=None):
        self.num_classes = num_classes
        self.nms_thresh = nms_thresh
        self.train_gpu = train_gpu
        torch.cuda.set_device(self.train_gpu)
        self.backbone = backbone
        self.pretrained_weights = pretrained_model_path
        self.train_data = train_data
        self.train_imgs = train_imgs
        assert len(train_imgs)==len(np.unique(self.train_data[:,0])) and len(train_imgs)>0,\
            'no training samples are generated due to bad training'
        self.model_save_path = model_save_path
        self.cam= cam
        self.SSL_iter = load_iter_checkpoint
        self.num_epochs = num_epochs
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)

    @staticmethod
    def show_net():
        #torchvision.models.detection.fasterrcnn_resnet101_fpn
        #model = torchvision.models.detection.fasterrcnn_resnet101_fpn(pretrained=False)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        i=0
        for p_name in model.state_dict():
            print(i, p_name)
            i+=1
            
    @staticmethod        
    def fasterrcnn_resnet34_fpn(num_classes=91, pretrained_backbone=True, **kwargs):
        
        backbone = resnet_fpn_backbone('resnet34', pretrained_backbone)
        model = FasterRCNN(backbone, num_classes, **kwargs)
        # if pretrained:
        #     state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'],
        #                                         progress=progress)
        #     model.load_state_dict(state_dict)
        return model

    def get_detection_model(self):
        # load detection model pre-trained on COCO
        #pretrained_model = torch.load(self.pretrained_weights)
        if self.backbone=='ResNet50FPN':
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            i=0
            for param in model.parameters():
                if i<=23: #23 upto conv2
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                print(i, param.requires_grad)
                i+=1

        if self.backbone=='ResNet101FPN':
            model = torchvision.models.detection.fasterrcnn_resnet101_fpn(pretrained=False)
            i=0
            for param in model.parameters():
                if i<=23: #clasp gt: 2nd stage: freeze upto conv3- 93: #1st stage: 125 only box head, 2nd stage: 23 upto conv2
                    param.requires_grad = False
                else:
                    param.requires_grad = True
                print(i, param.requires_grad)
                i+=1
                
        if self.backbone=='ResNet34FPN':
            model = self.fasterrcnn_resnet34_fpn(pretrained_backbone=True)
            i=0
            for param in model.parameters():
                # if i<=23:
                #     param.requires_grad = False
                # else:
                #     param.requires_grad = True
                print(i, param.requires_grad)
                i+=1


        # if self.num_classes>2:
        #     model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        #     model.roi_heads.nms_thresh = self.nms_thresh
        #     #model.load_state_dict(pretrained_model, strict=False)
        if self.SSL_iter==0:
            #load model before initializing the new class layers
            # Here pretrained weights belongs to different datasets/modality
            print(f'Iter {self.SSL_iter}, load pretrained weights: {self.pretrained_weights} with backbone: {self.backbone}')
            if self.pretrained_weights is not None:
                try:
                    model.load_state_dict(torch.load(self.pretrained_weights), strict=False)
                except Exception as e:
                    print(f"{self.pretrained_weights} loading failed due to {e}. COCO model will be used as initial model")
                    
            # get the number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
            model.roi_heads.nms_thresh = self.nms_thresh

        else:
            #load pretrained weights after initializing the new last layers
            #Here model weights are already trained on own classes
            #model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
            try:
                model.roi_heads.nms_thresh = self.nms_thresh
                print('load pretrained weights from: {}'.format(self.pretrained_weights))
                model.load_state_dict(torch.load(self.pretrained_weights), strict=False)
            except:
                in_features = model.roi_heads.box_predictor.cls_score.in_features
                #modify last layer for new classes
                model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
                model.roi_heads.nms_thresh = self.nms_thresh
        return model

    def get_transform(self, train=False):
        transforms = []
        # converts the image, a PIL image, into a PyTorch Tensor
        transforms.append(T.ToTensor())
        if train:
            # during training, randomly flip the training images
            # and ground-truth for data augmentation
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)

    def load_dataset(self):
        # use our dataset and defined transformations
        dataset = CLASP2ObjDetect(annotations=self.train_data,
                                  num_classes=self.num_classes,
                                  imgs=self.train_imgs,
                                  transforms=self.get_transform(train=True))

        dataset_no_random = CLASP2ObjDetect(annotations=self.train_data,
                                            num_classes=self.num_classes,
                                            imgs=self.train_imgs,
                                            transforms=self.get_transform(train=False))
        # split the dataset in train and test set
        #torch.manual_seed(1)
        #indices = torch.randperm(len(dataset)).tolist()
        #dataset = torch.utils.data.Subset(dataset, indices[:-701])
        #dataset_test = torch.utils.data.Subset(dataset_test, indices[-701:])
        # define training and validation data loaders
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)
        data_loader_no_random = torch.utils.data.DataLoader(
            dataset_no_random, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=utils.collate_fn)

        print('total annotated frames in cam/cams {} for training {}'.format(self.cam, len(dataset)))
        print('number of classes {}'.format(dataset.num_classes))
        return data_loader, dataset

    def train_epochs(self):
        torch.cuda.set_device(self.train_gpu)
        device = torch.device(f"cuda:{self.train_gpu}") if torch.cuda.is_available() else torch.device('cpu')
        #load dataset
        data_loader, dataset = self.load_dataset()
        # get the model using our helper function
        model = self.get_detection_model()
        # move model to the right device
        model.to(device)

        #full: 0.00001, roi head: 0.0001
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.0001,
                                    momentum=0.9, weight_decay=0.0005)

        # and a learning rate scheduler which decreases the learning rate by
        # 10x every 3 epochs
        # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
        #                                                step_size=20,
        #                                                gamma=0.1)
        #TODO: implement multiple GPUs training
        for epoch in range(1, self.num_epochs + 1):
            print('epoch {}'.format(epoch))
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)
            # update the learning rate
            #lr_scheduler.step()
            # evaluate on the test dataset
            if epoch % 10 == 0:
              #evaluate_and_write_result_files(model, data_loader_no_random)
              torch.save(model.state_dict(), os.path.join(self.model_save_path,"model_epoch_{}.model".format(epoch)))
              #evaluate(model, data_loader_test, device=device)
