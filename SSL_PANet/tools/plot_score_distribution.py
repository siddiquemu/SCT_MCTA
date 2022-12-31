import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
dataset='CLASP2'
if dataset in ['CLASP1']:
    exp = 0
    cam = 11
    dataset = '{}C{}'.format(exp, cam)
    #read det and cluster score
    data_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/CLASP1/test_augMS_gt'
    out_path = data_path + '/hist_plot'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    det_scores = pd.read_csv(data_path+'/det_scores_clasp1_test_aug1_{}.csv'.format(dataset), index_col=False)
    cluster_scores = pd.read_csv(data_path+'/cluster_scores_clasp1_test_aug1_{}.csv'.format(dataset), index_col=False)
if dataset in ['CLASP2']:
    exp = 0

    #read det and cluster score
    data_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/CLASP/test_augMS_gt_score'
    out_path = data_path + '/iter{}/hist_plot'.format(exp)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    det_scores = pd.read_csv(data_path+'/iter{}/det_scores_clasp2_test_aug_{}.csv'.format(exp, exp), index_col=False)
    cluster_scores = pd.read_csv(data_path+'/iter{}/cluster_scores_clasp2_test_aug_{}.csv'.format(exp, exp), index_col=False)
if dataset=='PVD':
    exp = 1
    cam = 1
    dataset = '{}C{}'.format(exp, cam)
    # read det and cluster score
    data_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/PVD/test_augMS_gt'
    out_path = data_path + '/hist_plot'
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    det_scores = pd.read_csv(data_path + '/det_scores_PVD_test_aug1_{}.csv'.format(dataset), index_col=False)
    cluster_scores = pd.read_csv(data_path + '/cluster_scores_PVD_test_aug1_{}.csv'.format(dataset), index_col=False)

#hist = np.histogram(det_scores, bins=50, range=(0, 1))
# max_dist = hist[1][np.argmax(hist[0])]
dscores = det_scores.values
cscores = cluster_scores.values
#person
scores = dscores[:,0][dscores[:,1]==1]
plt.hist(scores, bins=50, range=(0,1))
plt.savefig(out_path+'/person_det_{}.png'.format(dataset), dpi=300)
plt.close()

scores = cscores[:,0][cscores[:,1]==1]
plt.hist(scores, bins=50, range=(0,1))
plt.savefig(out_path+'/person_cluster_{}.png'.format(dataset), dpi=300)
plt.close()

#bag
scores = dscores[:,0][dscores[:,1]>1]
plt.hist(scores, bins=50, range=(0,1))
plt.savefig(out_path+'/bag_det_{}.png'.format(dataset), dpi=300)
plt.close()

scores = cscores[:,0][cscores[:,1]>1]
plt.hist(scores, bins=50, range=(0,1))
plt.savefig(out_path+'/bag_cluster_{}.png'.format(dataset), dpi=300)
plt.close()
