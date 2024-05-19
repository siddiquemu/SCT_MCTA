import torch
import torchvision
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import PIL
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import horovod.torch as hvd

import torchvision
import torch
import sys

sys.path.insert(0, '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw')
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from engine import train_one_epoch, evaluate
import transforms as T
import utils
from clasp2dataloader import *
import os
import numpy as np
import random


class SST_Model(object):
    def __init__(self, num_classes=2, nms_thresh=0.3, load_iter_checkpoint=0, train_gpu=1,
                 backbone='ResNet50FPN', pretrained_torch_model=None,
                 model_save_path=None, train_data=None, train_imgs=None, num_epochs=100, cam=None):
        self.num_classes = num_classes
        self.nms_thresh = nms_thresh
        self.train_gpu = train_gpu
        torch.cuda.set_device(self.train_gpu)
        self.backbone = backbone
        self.pretrained_weights = pretrained_torch_model
        self.train_data = train_data
        self.train_imgs = train_imgs
        assert len(train_imgs) == len(np.unique(self.train_data[:, 0])) and len(train_imgs) > 0, \
            'no training samples are generated due to bad training'
        self.model_save_path = model_save_path
        self.cam = cam
        self.SSL_iter = load_iter_checkpoint
        self.num_epochs = num_epochs
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        self.hvd = hvd

    @staticmethod
    def show_net():
        # torchvision.models.detection.fasterrcnn_resnet101_fpn
        # model = torchvision.models.detection.fasterrcnn_resnet101_fpn(pretrained=False)
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        i = 0
        for p_name in model.state_dict():
            print(i, p_name)
            i += 1

    def get_detection_model(self):
        # load an instance segmentation model pre-trained on COCO
        if self.backbone == 'ResNet50FPN':
            model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
            i = 0
            for param in model.parameters():
                if i <= 23:  # 23 upto conv2
                    param.requires_grad = False
                print(i, param.requires_grad)
                i += 1

        if self.backbone == 'ResNet101FPN':
            model = torchvision.models.detection.fasterrcnn_resnet101_fpn(pretrained=False)
            i = 0
            for param in model.parameters():
                if i <= 23:  # clasp gt: 2nd stage: freeze upto conv3- 93: #1st stage: 125 only box head, 2nd stage: 23 upto conv2
                    param.requires_grad = False
                print(i, param.requires_grad)
                i += 1
            if self.pretrained_weights:
                if self.backbone == 'ResNet101FPN': pretrained_model = torch.load(self.pretrained_weights)

                if self.backbone == 'ResNet50FPN': pretrained_model = torch.load(self.pretrained_weights)

                model.load_state_dict(pretrained_model, strict=False)
        if self.SSL_iter>0:
            # get the number of input features for the classifier
            in_features = model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            model.roi_heads.box_predictor = FastRCNNPredictor(in_features, self.num_classes)
        model.roi_heads.nms_thresh = self.nms_thresh
        model.cuda()
        model.train()
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
        dataset = CLASP2ObjDetect(annotations=self.train_data, imgs=self.train_imgs,
                                  transforms=self.get_transform(train=True))

        dataset_no_random = CLASP2ObjDetect(annotations=self.train_data, imgs=self.train_imgs,
                                            transforms=self.get_transform(train=False))

        # split the dataset in train and test set
        # torch.manual_seed(1)
        # indices = torch.randperm(len(dataset)).tolist()
        # dataset = torch.utils.data.Subset(dataset, indices[:-701])
        # dataset_test = torch.utils.data.Subset(dataset_test, indices[-701:])

        # define training and validation data loaders

        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)
        data_loader_no_random = torch.utils.data.DataLoader(
            dataset_no_random, batch_size=1, shuffle=False, num_workers=4,
            collate_fn=utils.collate_fn)

        print('total frames from each camera {} for training {}'.format(self.cam, len(dataset)))
        print('number of classes {}'.format(dataset.num_classes))

        # NEW
        # Distributed sampler.
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, num_replicas=hvd.size(), rank=hvd.rank()
        )
        dataloader = DataLoader(
            dataset, batch_size=8, shuffle=False, sampler=sampler
        )

        return dataloader, sampler

    def train_epochs(self):
        torch.cuda.set_device(self.train_gpu)
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        # load dataset
        data_loader, dataset = self.load_dataset()
        # get the model using our helper function
        model = self.get_detection_model()
        # move model to the right device
        model.to(device)
        # clasp stage1 resnet101fpn
        if self.SSL_iter > 0:
            # load model from previous iter
            if self.cam == 1:
                last_cam = 1
            if self.cam > 1:
                last_cam = self.cam - 1
            # last_cam = random.sample([1,2,3,4,5], 1)[0]
            print('load iter{}_C{} model for training C{} at current iter {}'.format(
                self.SSL_iter - 1, last_cam, self.cam, self.SSL_iter))
            model_state_dict = torch.load(
                os.path.join(self.model_save_path.split(self.model_save_path.split('/')[-2])[0],
                             "C{}/iter{}_C{}/model_epoch_{}.model".format(last_cam, self.SSL_iter - 1, last_cam,
                                                                          self.num_epochs)))
            model.load_state_dict(model_state_dict)
        # full: 0.00001, roi head: 0.0001
        # construct an optimizer
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.0001,
                                    momentum=0.9, weight_decay=0.0005)

        # and a learning rate scheduler which decreases the learning rate by
        # 10x every 3 epochs
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=10,
                                                       gamma=0.1)

        for epoch in range(1, self.num_epochs + 1):
            print('epoch {}'.format(epoch))
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50)
            # update the learning rate
            # lr_scheduler.step()
            # evaluate on the test dataset
            if epoch % 10 == 0:
                # evaluate_and_write_result_files(model, data_loader_no_random)
                torch.save(model.state_dict(), os.path.join(self.model_save_path, "model_epoch_{}.model".format(epoch)))
                # evaluate(model, data_loader_test, device=device)
    def train_epochs_nGPU(self):
        for epoch in range(1, self.num_epochs + 1):
            # NEW:
            # set epoch to sampler for shuffling.
            sampler.set_epoch(epoch)

            losses = []

            for i, (batch, segmap) in enumerate(dataloader):
                optimizer.zero_grad()

                batch = batch.cuda()
                segmap = segmap.cuda()

                output = model(batch)['out']
                loss = criterion(output, segmap.type(torch.int64))
                loss.backward()
                optimizer.step()

                curr_loss = loss.item()
                # if i % 10 == 0:
                #     print(
                #         f'Finished epoch {epoch}, batch {i}. Loss: {curr_loss:.3f}.'
                #     )

                if hvd.rank() == 0:
                    writer.add_scalar('training loss', curr_loss)
                losses.append(curr_loss)

            # print(
            #     f'Finished epoch {epoch}. '
            #     f'avg loss: {np.mean(losses)}; median loss: {np.min(losses)}'
            # )
            if hvd.rank() == 0 and epoch % 5 == 0:
                if not os.path.exists('/spell/checkpoints/'):
                    os.mkdir('/spell/checkpoints/')
                torch.save(model.state_dict(), f'/spell/checkpoints/model_{epoch}.pth')
        torch.save(model.state_dict(), f'/spell/checkpoints/model_final.pth')


# VOCSegmentation returns a raw dataset: images are non-resized and in the PIL format. To transform them
# to something suitable for input to PyTorch, we need to wrap the output in our own dataset class.
class PascalVOCSegmentationDataset(Dataset):
    def __init__(self, raw):
        super().__init__()
        self._dataset = raw
        self.resize_img = torchvision.transforms.Resize((256, 256), interpolation=PIL.Image.BILINEAR)
        self.resize_segmap = torchvision.transforms.Resize((256, 256), interpolation=PIL.Image.NEAREST)

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        img, segmap = self._dataset[idx]
        img, segmap = self.resize_img(img), self.resize_segmap(segmap)
        img, segmap = np.array(img), np.array(segmap)
        img, segmap = (img / 255).astype('float32'), segmap.astype('int32')
        img = np.transpose(img, (-1, 0, 1))

        # The PASCAL VOC dataset PyTorch provides labels the edges surrounding classes in 255-valued
        # pixels in the segmentation map. However, PyTorch requires class values to be contiguous
        # in range 0 through n_classes, so we must relabel these pixels to 21.
        segmap[segmap == 255] = 21

        return img, segmap


def get_dataloader():
    _PascalVOCSegmentationDataset = torchvision.datasets.VOCSegmentation(
        '/mnt/pascal_voc_segmentation/', year='2012', image_set='train', download=True,
        transform=None, target_transform=None, transforms=None
    )
    dataset = PascalVOCSegmentationDataset(_PascalVOCSegmentationDataset)
    # NEW
    # Distributed sampler.
    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset, num_replicas=hvd.size(), rank=hvd.rank()
    )
    dataloader = DataLoader(
        dataset, batch_size=8, shuffle=False, sampler=sampler
    )

    return dataloader, sampler


def get_model():
    # num_classes is 22. PASCAL VOC includes 20 classes of interest, 1 background class, and the 1
    # special border class mentioned in the previous comment. 20 + 1 + 1 = 22.
    DeepLabV3 = torchvision.models.segmentation.deeplabv3_resnet101(
        pretrained=False, progress=True, num_classes=22, aux_loss=None
    )
    model = DeepLabV3
    model.cuda()
    model.train()

    return model


def train(NUM_EPOCHS):
    for epoch in range(1, NUM_EPOCHS + 1):
        # NEW:
        # set epoch to sampler for shuffling.
        sampler.set_epoch(epoch)

        losses = []

        for i, (batch, segmap) in enumerate(dataloader):
            optimizer.zero_grad()

            batch = batch.cuda()
            segmap = segmap.cuda()

            output = model(batch)['out']
            loss = criterion(output, segmap.type(torch.int64))
            loss.backward()
            optimizer.step()

            curr_loss = loss.item()
            # if i % 10 == 0:
            #     print(
            #         f'Finished epoch {epoch}, batch {i}. Loss: {curr_loss:.3f}.'
            #     )

            if hvd.rank() == 0:
                writer.add_scalar('training loss', curr_loss)
            losses.append(curr_loss)

        # print(
        #     f'Finished epoch {epoch}. '
        #     f'avg loss: {np.mean(losses)}; median loss: {np.min(losses)}'
        # )
        if hvd.rank() == 0 and epoch % 5 == 0:
            if not os.path.exists('/spell/checkpoints/'):
                os.mkdir('/spell/checkpoints/')
            torch.save(model.state_dict(), f'/spell/checkpoints/model_{epoch}.pth')
    torch.save(model.state_dict(), f'/spell/checkpoints/model_final.pth')


if __name__ == '__main__':
    # NEW:
    # Init horovod
    hvd.init()
    torch.cuda.set_device(hvd.local_rank())
    torch.set_num_threads(1)

    writer = SummaryWriter(f'/spell/tensorboards/model_4')

    # since the background class doesn't matter nearly as much as the classes of interest to the
    # overall task a more selective loss would be more appropriate, however this training script
    # is merely a benchmark so we'll just use simple cross-entropy loss
    criterion = nn.CrossEntropyLoss()

    # NEW:
    # Download the data on only one thread. Have the rest wait until the download finishes.
    if hvd.local_rank() == 0:
        get_model()
        get_dataloader()
    hvd.join()
    print(f"Rank {hvd.rank() + 1}/{hvd.size()} process cleared download barrier.")

    model = get_model()
    dataloader, sampler = get_dataloader()

    # NEW:
    # Scale learning learning rate by size.
    optimizer = Adam(model.parameters(), lr=1e-3 * hvd.size())

    # New:
    # Broadcast parameters & optimizer state.
    hvd.broadcast_parameters(model.state_dict(), root_rank=0)
    hvd.broadcast_optimizer_state(optimizer, root_rank=0)

    # NEW:
    # (optional) Free-ish compression (reduces over-the-wire size -> increases speed).
    compression = hvd.Compression.fp16

    # NEW:
    # Wrap optimizer with DistributedOptimizer.
    optimizer = hvd.DistributedOptimizer(optimizer,
                                         named_parameters=model.named_parameters(),
                                         compression=compression,
                                         op=hvd.Average)
    train(20)