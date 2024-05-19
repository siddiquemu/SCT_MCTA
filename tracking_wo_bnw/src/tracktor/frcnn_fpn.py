from collections import OrderedDict

import torch
import torch.nn.functional as F

from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
from torchvision.models.detection.transform import resize_boxes
import pdb
import pdb
class FRCNN_FPN(FasterRCNN):

    def __init__(self, num_classes, box_nms_thresh=0.5, backbone_type='ResNet50FPN', class_id=None):
        if backbone_type=='ResNet34FPN':
            backbone = resnet_fpn_backbone('resnet34', False)
        if backbone_type=='ResNet50FPN':
            backbone = resnet_fpn_backbone('resnet50', False)
        if backbone_type=='ResNet101FPN':
            backbone = resnet_fpn_backbone('resnet101', False)
        super(FRCNN_FPN, self).__init__(backbone, num_classes, box_nms_thresh=0.5)
        # these values are cached to allow for feature reuse
        self.original_image_sizes = None
        self.preprocessed_images = None
        self.features = None
        self.class_id = class_id

    def detect(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]

        return detections['boxes'].detach(), detections['scores'].detach()

    def detect_clasp1(self, img):
        device = list(self.parameters())[0].device
        img = img.to(device)

        detections = self(img)[0]

        return detections['boxes'].detach(), detections['scores'].detach(), detections['labels'].detach()

    def detect_batches(self, imgs):
        device = list(self.parameters())[0].device
        imgs = imgs.to(device)

        detections = self(imgs)

        return detections

    def predict_boxes(self, boxes, class_id=None):
        device = list(self.parameters())[0].device
        boxes = boxes.to(device)


        boxes = resize_boxes(boxes, self.original_image_sizes[0], self.preprocessed_images.image_sizes[0])
        proposals = [boxes]
        # backbone features are computed when detector load image
        # and the new proposals features are computed based on that
        box_features = self.roi_heads.box_roi_pool(self.features, proposals, self.preprocessed_images.image_sizes)
        box_features = self.roi_heads.box_head(box_features)
        class_logits, box_regression = self.roi_heads.box_predictor(box_features)
        num_classes = class_logits.shape[-1]

        #boxes, scores, labels = self.roi_heads.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)
        pred_scores = F.softmax(class_logits, -1)
        # create labels for each prediction
        pred_labels = torch.arange(num_classes, device=device)
        pred_labels = pred_labels.view(1, -1).expand_as(pred_scores)

        # score_thresh = self.roi_heads.score_thresh
        # nms_thresh = self.roi_heads.nms_thresh

        # self.roi_heads.score_thresh = self.roi_heads.nms_thresh = 1.0
        # self.roi_heads.score_thresh = 0.0
        # self.roi_heads.nms_thresh = 1.0
        # detections, detector_losses = self.roi_heads(
        #     features, [boxes.squeeze(dim=0)], images.image_sizes, targets)

        # self.roi_heads.score_thresh = score_thresh
        # self.roi_heads.nms_thresh = nms_thresh

        # detections = self.transform.postprocess(
        #     detections, images.image_sizes, original_image_sizes)

        # detections = detections[0]
        # return detections['boxes'].detach().cpu(), detections['scores'].detach().cpu()

        # remove predictions with the background label
        pred_boxes = pred_boxes[:, 1:].squeeze(dim=1).detach()
        pred_labels = pred_labels[:, 1:].squeeze(dim=1).detach()
        pred_scores = pred_scores[:, 1:].squeeze(dim=1).detach()
        # if dataset=='clasp1' #for person class pred_boxes[0]
        if self.class_id:
            pred_boxes = pred_boxes[pred_labels == self.class_id]
            pred_scores = pred_scores[pred_labels == self.class_id]
        else:
            pred_boxes = pred_boxes[pred_labels == 1]
            pred_scores = pred_scores[pred_labels == 1]

        pred_boxes = resize_boxes(pred_boxes, self.preprocessed_images.image_sizes[0], self.original_image_sizes[0])

        return pred_boxes, pred_scores

    #use this function to load cameras image and corresponding backbone features,
    # original and transformed image parameters which will be used for regression again
    def load_image(self, images):
        device = list(self.parameters())[0].device
        images = images.to(device)

        self.original_image_sizes = [img.shape[-2:] for img in images]

        preprocessed_images, _ = self.transform(images, None)
        self.preprocessed_images = preprocessed_images

        self.features = self.backbone(preprocessed_images.tensors)
        if isinstance(self.features, torch.Tensor):
            self.features = OrderedDict([(0, self.features)])
