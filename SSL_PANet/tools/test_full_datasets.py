import copy
import glob
import os.path

import cv2
import time

import matplotlib.pyplot as plt
import numpy as np
from panet_flower_model import panet

class sliding_windows(object):
    def __init__(self, img, step_size=None, window_size=None):
        self.img=img
        self.step_size=step_size # step_size = (w_step, h_step)
        self.window_size=window_size #window_size = (w, h)

    @staticmethod
    def pad_img(img, step_size):
        return cv2.copyMakeBorder(img, step_size[1], step_size[1],
                                      step_size[0], step_size[0],
                                      cv2.BORDER_CONSTANT, None, value = 0)

    def sliding_window(self):
        # slide a window across the image
        self.img = self.pad_img(self.img, self.step_size)
        for y in range(0, self.img.shape[0], self.step_size[1]):
            for x in range(0, self.img.shape[1], self.step_size[0]):
                # yield the current window
                yield (x, y, self.img[y:y + self.window_size[1], x:x + self.window_size[0]])

    def get_padded_img(self):
        return self.pad_img(self.img, self.step_size).copy()

    def remove_pad(self, pad_img, img_shape):
        return pad_img[self.step_size[1]:img_shape[0]+self.step_size[1],
               self.step_size[0]:img_shape[1]+self.step_size[0]]

    def get_padded_mask(self):
        img = self.pad_img(self.img, self.step_size).copy()
        return np.zeros((img.shape[0], img.shape[1]), dtype='float')

    def show_input_wndows(self, x,y):
        clone = self.get_padded_img().copy()
        clone = cv2.rectangle(clone, (x, y), (x + self.window_size[0], y + self.window_size[1]), (0, 255, 0), 2)
        final_img = cv2.resize(clone, (self.img.shape[1]//4, self.img.shape[0]//3))
        cv2.imshow('image',final_img)
        cv2.waitKey(1)
        time.sleep(2)

    def print_w_stats(self):
        print('window_size: {}'.format(self.window_size))
        print('step size: {}'.format(self.step_size))
        print('image shape: {}'.format(self.img.shape))

    def save_final_pred(self, save_path, final_pred):
        #TODO: Why saved binary mask has more than two unique values?
        final_pred[final_pred>=0.3] = 255
        final_pred[final_pred < 0.3] = 0
        print('final pred unique values: {}'.format(np.unique(final_pred)))
        cv2.imwrite(save_path, final_pred)

if __name__ == '__main__':
    #get images
    vis_window = 0
    data_type = 'AppleB'

    if data_type=='AppleA':
        data_dir = '/media/siddique/RemoteServer/LabFiles/Walden/trainTestSplit/test/rawData/flowers'
        img_fmt = 'bmp'

    elif data_type in ['AppleB', 'Peach', 'Pear']:
        data_dir = '/media/siddique/RemoteServer/LabFiles/Walden/otherFlowerDatasets/{}/rawData/rawFlowers'.format(data_type)
        img_fmt = 'bmp'

    folder = glob.glob(data_dir+'/*')
    folder.sort(key=lambda f: int(''.join(filter(str.isdigit, str(f)))))
    out_dir = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/flower/{}/panet_pred'.format(data_type)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    #initialize detector
    pred_scores = np.arange(0.1, 1, 0.05)
    for pred_score in pred_scores:
        flower_detector = panet(data='AppleA_train', pred_score=0.5,
                                nms_thr=0.3, gpu_id=1, iter_i=2)
        flower_model, dataset = flower_detector.configure_detector()

        for img_path in folder:
            img=cv2.imread(img_path)

            if data_type=='AppleA':
                fr = float(os.path.basename(img_path).split('.')[0].split('IMG_')[-1])
                step_size = [img.shape[1] // 8, img.shape[0] // 6]  # [step_w, step_h]
                window_size = [img.shape[1] // 4, img.shape[0] // 3]  # [w_W, w_H]

            elif data_type in ['AppleB', 'Peach']:
                fr = float(os.path.basename(img_path).split('.')[0])
                step_size = [img.shape[1] // 8, img.shape[0] // 8]  # [step_w, step_h]
                window_size = [img.shape[1] // 4, img.shape[0] // 4]  # [w_W, w_H]

            elif data_type in ['Pear']:
                fr = float(os.path.basename(img_path).split('.')[0].split('_')[-1])
                step_size = [img.shape[1] // 8, img.shape[0] // 8]  # [step_w, step_h]
                window_size = [img.shape[1] // 4, img.shape[0] // 4]  # [w_W, w_H]

            if vis_window:
                cv2.namedWindow('image', cv2.WINDOW_NORMAL)
                cv2.resizeWindow("image", img.shape[1] // 4, img.shape[0] // 3)
            #prepare sliding windows
            SWs = sliding_windows(img=img, step_size=step_size, window_size=window_size)
            SWs.print_w_stats()
            window_id = 1
            #utput mask having actual resolution with padding
            score_mask = SWs.get_padded_mask()
            pix_count_mask = copy.deepcopy(score_mask)
            #iterate over window images
            for (x, y, window) in SWs.sliding_window():
                if window.shape[0] != window_size[1] or window.shape[1] != window_size[0]:
                    continue
                print('window size: {}'.format(window.shape))
                print('window id: {}'.format(window_id))
                #[x1,y1,x2,y2]
                x1,y1,x2,y2 = x, y, x + window_size[0], y + window_size[1]
                print([x, y, x + window_size[0], y + window_size[1]])
                #get predictions from SSL model
                pred_mask = flower_detector.predict(flower_model, dataset, window, window_id, angle=0)
                if pred_mask is not None:
                    #print(pred_mask.shape)
                    #print(score_mask[y1:y2,x1:x2].shape)
                    score_mask[y1:y2, x1:x2] += pred_mask
                    #count predicted pixels: to compute average for overlapped pixels
                white_mask = np.ones((window.shape[0], window.shape[1]), dtype='float')
                #print(white_mask.shape)
                #print(pix_count_mask[y1:y2,x1:x2].shape)
                pix_count_mask[y1:y2,x1:x2] += white_mask
                window_id+=1
            print('max values: {}, {}'.format(score_mask.max(), pix_count_mask.max()))
            final_pred = score_mask/pix_count_mask
            final_pred = SWs.remove_pad(final_pred, img.shape)
            assert final_pred.shape[0]==img.shape[0]
            save_path = os.path.join(out_dir, os.path.basename(img_path))
            SWs.save_final_pred(save_path, final_pred)