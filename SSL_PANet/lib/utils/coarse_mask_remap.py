from __future__ import division
import numpy as np
import imutils
import cv2
def mask_remap(coarse_mask,angle):
    # to remap coarse segmentation mask
    # TODO: place rotated coarse mask on rotated mask image, re-rotate the mask image and crop the remapped mask
    imgcrp = np.zeros((coarse_mask.shape[0], coarse_mask.shape[1]), dtype='float')
    imgrerot = imutils.rotate_bound(coarse_mask, -angle)  # mask_image
    coarse_org = imgrerot[np.int(imgrerot.shape[0] / 2.0) - np.int(imgcrp.shape[0] / 2.0): np.int(imgrerot.shape[0] / 2.0) + np.int(imgcrp.shape[0] / 2.0), \
              np.int(imgrerot.shape[1] / 2.0) - np.int(imgcrp.shape[1] / 2.0): np.int(imgrerot.shape[1] / 2.0) + np.int(imgcrp.shape[1] / 2.0)]
    # save masks at multiple inference
    return coarse_org

def mask_image_remap(mask_image,img_org,angle):
    # to remap coarse segmentation mask
    # TODO: place rotated coarse mask on rotated mask image, re-rotate the mask image and crop the remapped mask
    imgrerot = imutils.rotate_bound(mask_image, -angle)  # mask_image
    Hrot,Wrot = imgrerot.shape[0]//2,imgrerot.shape[1]//2
    H,W = img_org.shape[0]//2, img_org.shape[1]//2
    mask_org = imgrerot[Hrot - H: Hrot + H,Wrot - W:Wrot + W]
    # save masks at multiple inference
    assert (2*H,2*W)==mask_org.shape
    return mask_org

def box_image_remap(box, img_rot, img_org, angle):
    # to remap coarse segmentation mask
    # TODO: place rotated coarse mask on rotated mask image, re-rotate the mask image and crop the remapped mask
    box = np.array(box)
    bb = box.astype('int')
    bImg = np.zeros(shape=[img_rot.shape[0], img_rot.shape[1]], dtype=np.uint8)
    bImg[bb[1]:bb[3],bb[0]:bb[2]] = 255
    imgrerot = imutils.rotate_bound(bImg, -angle)  # mask_image
    Hrot,Wrot = imgrerot.shape[0]//2,imgrerot.shape[1]//2
    H,W = img_org.shape[0]//2, img_org.shape[1]//2
    mask_org = imgrerot[Hrot - H: Hrot + H,Wrot - W:Wrot + W]
    [x, y, w, h] = cv2.boundingRect(mask_org)
    #get the center of the rotated box contour
    contours = cv2.findContours(mask_org, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours[0]:
        # calculate moments for each contour
        M = cv2.moments(c)
        # calculate x,y coordinate of center
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
    # save masks at multiple inference
    assert (2*H,2*W)==mask_org.shape
    return x, y, w, h