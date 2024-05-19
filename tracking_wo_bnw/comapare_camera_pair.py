import glob
import cv2
import os
def delete_all(demo_path, fmt='png'):
    import glob
    filelist = glob.glob(os.path.join(demo_path, '*.' + fmt))
    if len(filelist) > 0:
        for f in filelist:
            os.remove(f)

cam1Path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d6/tracking_wo_bnw/output/tracktor/MOT17/Tracktor++/CLASP_test/demo_cam_compare/cam13/*.png'
cam2Path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d6/tracking_wo_bnw/output/tracktor/MOT17/Tracktor++/CLASP_test/demo_cam_compare/cam14/*.png'
out_dir =  '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d6/tracking_wo_bnw/output/tracktor/MOT17/Tracktor++/CLASP_test/demo_cam_compare/'

cam1_set = glob.glob(cam1Path)
cam1_set.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

cam2_set = glob.glob(cam2Path)
cam2_set.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
assert len(cam1_set)==len(cam2_set)
#cv2.namedWindow('image', cv2.WINDOW_NORMAL)
#cv2.resizeWindow("image", 640, 480)

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(os.path.join(out_dir,'C13_C11.avi'), fourcc, 1, (1920, 540))
for P1,P2 in zip(cam1_set,cam2_set):
    im1 = cv2.imread(P1)
    im2 = cv2.imread(P2)
    img = cv2.hconcat([im1,im2])
    img = cv2.resize(img,(1920,540))
    out.write(img)
out.release()
cv2.destroyAllWindows()
