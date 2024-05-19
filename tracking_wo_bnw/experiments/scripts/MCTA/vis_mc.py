import os
import glob
import cv2
import numpy as np
img_HW = [800.0, 1280.0]
img_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/output/PVD/C'
demo_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/output/PVD/C/demo'
if not os.path.exists(demo_path): os.makedirs(demo_path)

cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.resizeWindow("image", 1280, 800)
for fr in range(5, 6002+1, 5):
    imgC1 = cv2.imread('{}/1_C{}/{:06d}.png'.format(img_path, 1, fr))
    imgC2 = cv2.imread('{}/1_C{}/{:06d}.png'.format(img_path, 2, fr))
    imgC3 = cv2.imread('{}/1_C{}/{:06d}.png'.format(img_path,3, fr))
    imgC4 = cv2.imread('{}/1_C{}/{:06d}.png'.format(img_path, 4, fr))
    imgC5 = cv2.imread('{}/1_C{}/{:06d}.png'.format(img_path, 5, fr))
    imgC6 = cv2.imread('{}/1_C{}/{:06d}.png'.format(img_path, 6, fr))

    blankImg = np.zeros(shape=[int(img_HW[0]), int(img_HW[1]), 3], dtype=np.uint8)
    # imgC2 = cv2.copyMakeBorder(imgC2, 0, 0, 0, 250, cv2.BORDER_CONSTANT, value=0)
    # imgC9 = cv2.copyMakeBorder(imgC9, 0, 0, 250, 0, cv2.BORDER_CONSTANT, value=0)
    # imgC13 = cv2.copyMakeBorder(imgC13, 0, 0, 700, 0, cv2.BORDER_CONSTANT, value=0)
    # imgC14 = cv2.copyMakeBorder(imgC14, 0, 0, 0, 700, cv2.BORDER_CONSTANT, value=0)

    imgC1C2C3 = cv2.hconcat([imgC3, imgC2, imgC1])  # blankImg
    imgC4C5C6 = cv2.hconcat([imgC5, imgC4, imgC6])  # imgC4
    final_img = cv2.vconcat([imgC1C2C3, imgC4C5C6])  # img: c4, img2: c2

    # blankImg = np.zeros(shape=[1080, 1920, 3], dtype=np.uint8)
    # img13Blnk = cv2.vconcat([imgC13, blankImg])
    # img1314 = cv2.vconcat([imgC14, imgC13])
    # final_img = cv2.hconcat([img1314, final_img])

    final_img = cv2.resize(final_img, (int(3 * img_HW[1]), int(2 * img_HW[0])),
                           interpolation=cv2.INTER_AREA)  # 1.5

    # img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    cv2.imwrite(demo_path + '/{:06d}.jpg'.format(fr), final_img)
    cv2.imshow("image", final_img)
    cv2.waitKey(5)

cv2.destroyAllWindows()