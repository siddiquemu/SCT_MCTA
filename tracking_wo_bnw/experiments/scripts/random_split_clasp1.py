#same random split
import random
import numpy as np
codeBase = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw'
for data in ['A', 'B', 'C', 'D', 'E']:
    for cam in [9, 11]:
        GT = np.loadtxt(codeBase+'/data/CLASP1/train_gt/{}_{}/gt/gt.txt'.format(data, cam),delimiter=',')
        GTp = [gt for gt in GT if gt[9]==1]
        GTb = [gt for gt in GT if gt[9]==2]
        gt_frameset = np.unique(GT[:,0].astype('int'))
        gt_len = len(gt_frameset)
        print('full set {}_{}: Nfr {}, person {} bag {}'.format(data, cam, gt_len, len(GTp), len(GTb)))

        #random split: keep split similar for training and testing forever
        random.seed(42)
        subset = random.sample(list(gt_frameset), int(gt_len*0.8))
        print('random sample {}'.format(subset[2]))
        #print(subset)
        train_GTp = [gt for gt in GT if gt[0] in subset and gt[9]==1]
        train_GTb = [gt for gt in GT if gt[0] in subset and gt[9]==2]
        print('train split {}_{}: Nfr {}, person {} bag {}'.format(data, cam, len(subset), len(train_GTp), len(train_GTb)))

        test_subset = np.array([t for t in gt_frameset if t not in subset])
        test_GTp = [gt for gt in GT if gt[0] not in subset and gt[9]==1]
        test_GTb = [gt for gt in GT if gt[0] not in subset and gt[9]==2]
        print('test split {}_{}: Nfr {}, person {} bag {}'.format(data, cam, len(test_subset), len(test_GTp), len(test_GTb)))
        print('-------------------------------------------------')