import scipy.io as io
import matplotlib.pyplot as plt
scoremap = io.loadmat('/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/flower/AppleB/panet_pred/scores/1.mat')
scoremap = scoremap['scoreMap_M']
plt.figure(1)
plt.imshow(scoremap[:,:,1])
bck = 1 - scoremap[:,:,1]
bck[bck==1] = 0.005
plt.figure(2)
plt.imshow(bck)

plt.show()