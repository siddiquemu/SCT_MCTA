from __future__ import division
import numpy as np
# import imutils
import cv2
import matplotlib
import matplotlib.pyplot as plt
import glob

import pdb
from shapely.geometry import Polygon
# from scipy.misc import imsave
# from skimage import exposure

def plot_lines(img, points2plane, color, W=2910, H=1080):
    # Correct order of points to draw a box
    for (i, j) in [(0, 1), (0, 2), (1, 3), (2, 3)]:
        (x0, y0) = points2plane[:, i].astype('int').flatten()
        x0 = max(min(x0, W), 0)
        y0 = max(min(y0, H), 0)
        (x1, y1) = points2plane[:, j].astype('int').flatten()
        x1 = max(min(x1, W), 0)
        y1 = max(min(y1, H), 0)
        img = cv2.line(img, (x0, y0), (x1, y1), color, thickness=3, lineType=8)
    return img


def projection_map_size(shape, H):
    minx = shape[0]
    miny = shape[1]
    maxx = 0
    maxy = 0
    for (i, j) in ((0, 0), (shape[1], 0), (0, shape[0]), (shape[1], shape[0])):
        point = np.matmul(H, [i, j, 1])
        x = point[0] / point[2]
        y = point[1] / point[2]
        if x < minx:
            minx = x
        if y < miny:
            miny = y
        if x > maxx:
            maxx = x
        if y > maxy:
            maxy = y
    H_o = np.linalg.inv(np.column_stack(([1, 0, 0], [0, 1, 0], [minx, miny, 1])))
    size = (int(maxx - minx), int(maxy - miny))
    return (H_o, size)


def showWarpedImage(image, H):
    (H_o, size) = projection_map_size(image.shape, H)
    result = cv2.warpPerspective(image, np.matmul(H_o, H), size, flags=cv2.INTER_LINEAR)
    plt.imshow(result)


# TODO: Use this to filter the projected bounding boxes that fall outside the image
def checkcolinear(points2plane):
    # Correct order of points to draw a box
    (x0, y0) = points2plane[:, 0].astype('int').flatten()
    (x1, y1) = points2plane[:, 1].astype('int').flatten()
    (x2, y2) = points2plane[:, 2].astype('int').flatten()
    return (x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1)) > 0.01


def applyTransform(source_corners, H):
    # **source_corners - (n,2)
    dest_corners = np.empty(source_corners.shape)
    for i in range(len(dest_corners)):
        #iterate for each corner
        w = H[2][0] * source_corners[i][0] + H[2][1] * source_corners[i][1] + H[2][2] * 1
        dest_corners[i][0] = (H[0][0] * source_corners[i][0] + H[0][1] * source_corners[i][1] + H[0][2] * 1) / w
        dest_corners[i][1] = (H[1][0] * source_corners[i][0] + H[1][1] * source_corners[i][1] + H[1][2] * 1) / w
    return dest_corners

def camera_intrinsics(cam):
    # currently used camera: 2,4,5,9,11,13
    # A: 3*3 camera matrix
    # dist_coeff: distortion coefficient k1, k2
    if cam == 9:
        A = np.array([[1217.6, 0.0, 972.3],
                      [0, 1217.8, 550.9],
                      [0.0, 0.0, 1.0]])

        dist_coeff = np.array([-0.3235, 0.0887, 0, 0])  # why 4 or 5 coefficients are used in opencv????
    if cam == 11:  # assume that 9 and 11 have similar distortion
        A = np.array([[1217.6, 0.0, 972.3],
                      [0, 1217.8, 550.9],
                      [0.0, 0.0, 1.0]])
        dist_coeff = np.array([-0.3235, 0.0887, 0, 0])  # why 4 or 5 coefficients are used in opencv????
    if cam == 13:  # assume that 9 and 11 have similar distortion
        A = np.array([[1217.6, 0.0, 972.3],
                      [0, 1217.8, 550.9],
                      [0.0, 0.0, 1.0]])
        dist_coeff = np.array([-0.3235, 0.0887, 0, 0])  # why 4 or 5 coefficients are used in opencv????

    if cam == 14:  # assume that 9 and 11 have similar distortion
        A = np.array([[1217.6, 0.0, 972.3],
                      [0, 1217.8, 550.9],
                      [0.0, 0.0, 1.0]])
        dist_coeff = np.array([-0.3235, 0.0887, 0, 0])  # why 4 or 5 coefficients are used in opencv????

    if cam == 2:#989.6,623.4
        A = np.array([[1315.7, 0.0, 960.0],
                       [0.0, 1313.0, 540.0],
                       [0.0, 0.0, 1.0]])
        dist_coeff = np.array([-0.415, 0.17,0.0, 0.0])  # why 4 or 5 coefficients are used in opencv????
    if cam == 5:  # assume that c2 and c5 have similar distortion
        A = np.array([[1315.7, 0.0, 989.6],
                       [0.0, 1313.0, 623.4],
                       [0.0, 0.0, 1.0]])
        dist_coeff = np.array([-0.415, 0.17, 0.0, 0.0])  # why 4 or 5 coefficients are used in opencv????
    if cam == 4:  # assume that c2 and c5 have similar distortion
        A = np.array([[1315.7, 0.0, 989.6],
                       [0.0, 1313.0, 623.4],
                       [0.0, 0.0, 1.0]])
        dist_coeff = np.array([-0.415, 0.17, 0.0, 0.0]) # why 4 or 5 coefficients are used in opencv????

    return dist_coeff, A


def undistorted_coords(trklt, dist_coeff, A):
    # use a copy of traclet centroids to convert into undistorted format
    # ***A: Camera Intrinsic Matrix
    # https://docs.opencv.org/3.4/db/d58/group__calib3d__fisheye.html
    # https: // www.mathworks.com / help / vision / ref / estimatecameraparameters.html
    # new camMatrix
    im_shape = (1920, 1080)  # img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(A, dist_coeff, im_shape, 1, im_shape)
    trklt = cv2.undistortPoints(trklt.reshape(trklt.shape[0], 1, 2), A, dist_coeff, 0, newcameramtx)
    return trklt.reshape(trklt.shape[0], 2)

def undistort_image(img, camMatrix, distCoeff):
    """
    This can straighten out an image given the intrinsic matrix (camera
    matrix) and the distortion coefficients.

    Parameters
    ----------
    img : image
        This is the image we wish to undistort
    camMatrix : 2dmatrix
        This is the intrinsic matrix of the camera (3x3)
    distCoeff : 1darray
        This is the distortion coefficient of the camera (1x5)

    """
    h,  w = img.shape[:2]
    print('h {} w {}'.format(h,w))
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camMatrix,
                                                      distCoeff,
                                                      (w, h),
                                                      1,
                                                      (w, h)
                                                      )
    # undistort
    dst = cv2.undistort(img, camMatrix, distCoeff, None, newcameramtx)
    x, y, w, h = roi
    #dst = dst[y:y+h, x:x+w]
    return dst

def getProjectedBoxes(H, dets):
    point2plane = []
    centers = []
    for i in range(dets.shape[0]):
        box = dets[i]
        print('angle and class', box[8], box[7])
        #[cx,cy]
        cxcy =  np.array([box[2]+box[4]/2., box[3]+box[5]/2.])
        dist_coeff, A = camera_intrinsics(9)
        cxcy = undistorted_coords(cxcy.reshape(1,2), dist_coeff, A)
        cxcy = applyTransform(cxcy, H)
        # [x1,y1],[x2,y1],[x1,y2], [x2,y2]
        box_corner_points = np.array(([box[2], box[3]], [box[2] + box[4], box[3]], [box[2], box[3] + box[5]],
                                      [box[2] + box[4], box[3] + box[5]]), dtype='float')

        box_corner_points = undistorted_coords(box_corner_points, dist_coeff, A)

        box_corner_points = np.array([box_corner_points])
        point2plane.append(cv2.perspectiveTransform(box_corner_points, H))

        centers.append(cxcy[0])
    return point2plane, centers

def getUndistortedBoxes(dist_coeff, A,  dets):
    points = []
    for i in range(dets.shape[0]):
        box = dets[i]
        # [x1,y1],[x2,y1],[x1,y2], [x2,y2]
        box_corner_points = np.array(([box[2], box[3]], [box[2] + box[4], box[3]], [box[2], box[3] + box[5]],
                                      [box[2] + box[4], box[3] + box[5]]), dtype='float')

        box_corner_points = undistorted_coords(box_corner_points, dist_coeff, A)
        points.append(np.array([box_corner_points]))

    return points


def drawProjectedBoxes(image, point2plane, color=(255, 0, 0)):
    for point in point2plane:
        image = plot_lines(image, point, color)
    return image

def drawBoxes(img, dets, isDistorted=False):
    for i in range(dets.shape[0]):
        bb = dets[i]
        if not isDistorted:
            print('angle and class', bb[8], bb[7])
            cv2.rectangle(img, (int(bb[2]), int(bb[3])),
                      (int(bb[2] + bb[4]), int(bb[3] + bb[4])),
                      (0, 255, 0), 5, cv2.LINE_AA)
        else:
            cv2.rectangle(img, (int(bb[0]), int(bb[1])),
                      (int(bb[0] + bb[2]), int(bb[1] + bb[3])),
                      (0, 255, 0), 5, cv2.LINE_AA)
    return img

def getH(R, T):
    H = np.append(R, T.reshape(3, 1), 1)
    H = np.append(H, [[0, 0, 0, 1]], 0)
    return H


def H1to9():
    R_1to9 = np.array(([0.93089, -0.35805, 0.07235], [0.22988, 0.72813, 0.64575], [-0.28389, -0.58448, 0.76011]),
                      dtype='float')
    T_1to9 = np.array(([1470.616, 2478.045, -121.717]), dtype='float')
    H_1to9 = getH(R_1to9, T_1to9)
    return H_1to9


def H1to2():
    R_1to2 = np.array(([0.89316, -0.44761, 0.04366], [0.30741, 0.67849, 0.6672], [-0.32826, -0.58249, 0.7436]),
                      dtype='float')
    T_1to2 = np.array(([2539.67, 210.50, -119.50]), dtype='float')
    H_1to2 = getH(R_1to2, T_1to2)
    return H_1to2


def getHomography(K_dest, K_source, H_source, inv, d_plane=2000, n=[0, 0, 1]):
    R = H_source[0:3, 0:3]
    t = H_source[0:3, 3]
    # This rotation is only necessary in the inverse homographies. It shouldn't be needed if we go from the
    # reference camera to the target camera
    if inv:
        t = np.matmul(np.transpose(R), t)
    Rtnd = R - np.outer(t, n) / d_plane
    return np.matmul(K_source, np.matmul(Rtnd, np.linalg.inv(K_dest)))


def drawVolume(image, proj):
    boxes = np.stack([np.reshape(np.asarray(np.transpose(proj[0:4, :])), (2, 4)),
                      np.reshape(np.asarray(np.transpose(proj[4:8, :])), (2, 4))])

    image = drawProjectedBoxes(image, boxes)

    # TODO: Correct the order of these points
    side1 = np.column_stack([boxes[0, 0:2, 0], boxes[1, 0:2, 0], boxes[0, 0:2, 2], boxes[1, 0:2, 2]])
    side2 = np.column_stack([boxes[0, 0:2, 1], boxes[1, 0:2, 1], boxes[0, 0:2, 3], boxes[1, 0:2, 3]])
    sides = np.array([side1, side2])

    image = drawProjectedBoxes(image, sides)

    return image

def groundProjectPoint(H, rotMat, tvec, camera_matrix, image_point, z=0.0):
    #https://github.com/balcilar/Stereo-Camera-Calibration-Orthogonal-Planes

    camMat = np.asarray(camera_matrix)
    iRot = np.linalg.inv(rotMat)
    iCam = np.linalg.inv(camMat)

    uvPoint = np.ones((3, 1))


    # Image point
    uvPoint[0, 0] = image_point[0]
    uvPoint[1, 0] = image_point[1]

    tempMat = np.matmul(np.matmul(iRot, iCam), uvPoint)
    tempMat2 = np.matmul(iRot, tvec)
    #pdb.set_trace()
    s = (z + tempMat2[2, 0]) / tempMat[2, 0]
    wcPoint = np.matmul(iRot, (np.matmul(s*iCam, uvPoint) - tvec))
    # wcPoint[2] will not be exactly equal to z, but very close to it
    assert int(abs(wcPoint[2] - z) * (10 ** 8)) == 0
    wcPoint[2] = z

    return wcPoint, s

def projectPrimaryCentroid(center, H, K, dist):
    # **points - four corner points of the projection from auxiliary
    #centroid = [points[0][0] + (points[1][0] - points[0][0]) / 2.,
                #points[0][1] + (points[3][1] - points[0][1]) / 2.]
    centroid = center
    wpoint_center, scale = groundProjectPoint(H, H[:-1, :-1], H[:-1, 3:], K, centroid, 1) #-850
    wpoints_cent = np.reshape(wpoint_center, (3,)) +  np.array([0, 0, 1])

    #convert rotation matrices to angle axis
    rvec, _ = cv2.Rodrigues(H[:-1, :-1])
    proj_cent, _ = cv2.projectPoints(wpoints_cent.astype('float'), rvec, H[:-1, 3:], K, dist)
    return proj_cent[0][0], wpoint_center[:2,0]

def projectCentroid(center, H, K, dist):
    #https://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point
    #currently used for auxiliary
    # **points - four corner points of the projection from auxiliary
    #centroid = [points[0][0] + (points[1][0] - points[0][0]) / 2.,
                #points[0][1] + (points[3][1] - points[0][1]) / 2.]
    centroid = center
    #groundProjectPoint(rotMat, tvec, camera_matrix, image_point, z=0.0)
    #pdb.set_trace()
    wpoint, scale = groundProjectPoint(H, H[:-1, :-1], H[:-1, 3:], K, centroid, 450.0) #-850
    wpoint_center, scale = groundProjectPoint(H, H[:-1, :-1], H[:-1, 3:], K, centroid, 450.0) #-850
    #center_plane = np.array([[200, -200, 0], [200, 200, 0], [-200, -200, 0], [-200, 200, 0]])
    box = np.array([[200, -200, 850],
                    [200, 200, 850],
                    [-200, -200, 850],
                    [-200, 200, 850],

                    [200, -200, -850],
                    [200, 200, -850],
                    [-200, -200, -850],
                    [-200, 200, -850]])
    wpoints_cent = np.reshape(wpoint_center, (3,)) +  np.array([0, 0, 0])
    wpoints = np.reshape(wpoint, (3,)) + box

    #convert rotation matrices to angle axis
    rvec, _ = cv2.Rodrigues(H[:-1, :-1])
    proj, _ = cv2.projectPoints(wpoints.astype('float'), rvec, H[:-1, 3:], K, dist)
    #proj_cplane, _ = cv2.projectPoints(wpoints_cplane.astype('float'), rvec, H[:-1, 3:], K, dist)
    proj_cent, _ = cv2.projectPoints(wpoints_cent.astype('float'), rvec, H[:-1, 3:], K, dist)

    #Project C2 world points into image plane
    dist_coeff2,K2 = camera_intrinsics(2)
    img2Points = np.ones((1,2),dtype='float')
    img2Points[0,0] = (1 / scale) * K2[0, 0] * wpoint_center[0, 0] / 1.0 + K2[0, 2]
    img2Points[0,1] = (1 / scale) * K2[1, 1] * wpoint_center[1, 0] / 1.0 + K2[1, 2]
    #pdb.set_trace()
    img2Points = undistorted_coords(img2Points, dist_coeff2, A2)
    #shifted_center = np.sum(proj_cent[:,0,:],axis=0)/4.0
    # wpoint_center[:2,0]
    return proj, proj_cent[0][0], img2Points.reshape(2,)

def drawCenters(img, centers, color=None):
    for center in centers:
        img = cv2.circle(img, (int(center[0]), int(center[1])), 10, color, -10)
    return img

path9 = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/cam09exp2.mp4/*.png'
files9 = glob.glob(path9)
files9.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

path2 = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/imgs/cam02exp2.mp4/*.png'
files2 = glob.glob(path2)
files2.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

# read detections
# det_path9 = '/media/RemoteServer/labfiles/CLASP2/2019_04_16/exp2/camera_projection/cam9/*.npy'
det_path9 = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results_train/cam09exp2.mp4img.txt'
dets9 = np.loadtxt(det_path9,delimiter=',')

# det_path2 = '/media/RemoteServer/labfiles/CLASP2/2019_04_16/exp2/camera_projection/cam2/*.npy'
det_path2 = '/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/exp2/Results_train/cam02exp2.mp4img.txt'
dets2 = np.loadtxt(det_path2,delimiter=',')

out_path = '/home/siddique/multi-camera-association/3d_box/'

for j in range(3901, len(files2)):
    # Image
    image2 = cv2.imread(files2[j+46])
    image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    image9 = cv2.imread(files9[j])
    image9 = cv2.cvtColor(image9, cv2.COLOR_BGR2RGB)
    # detections
    det_2_j = dets2[dets2[:,0]==j+46]
    det_9_j = dets9[dets9[:, 0] == j]

    #H9to2: pixel unit 463.84, -782.41 268 -569 pixels
    H_box = [[0.988, 0.149, 0.00],
        [-0.149, 0.988, 0.00],
        [463.0, -782.0, 1.00]]
    H_box = np.transpose(H_box)

    #H for ground projection: mm unit tx=660, ty=-1402.0
    H = np.transpose(np.array([[0.988, -0.149, 0.0, 0.0],
                               [0.149, 0.988, 0.0, 0.0],
                               [0.0, 0.0, 1.0, 0.0],
                              [660.0, -1402.0, -3000.0, 1.0]]))

    #Current C9-magenta and C2-yellow centroids on C2 for distance computation
    # get projected centers or all corner points??: all corner points
    #points, centers = getProjectedBoxes(H_box, det_9_j) #np.eye(3)
    #image2 = drawProjectedBoxes(image2, points, (0, 0, 255))
    #image2 = drawCenters(image2, centers, color=(255, 0, 255))
    # draw the undistorted primary centers

    # get undistorted box and center C9
    dist_coeff9, A9 = camera_intrinsics(9)
    # C9:[cx,cy]- green
    cxcy9 = np.transpose(np.array([det_9_j[:,2] +det_9_j[:,4] / 2., det_9_j[:,3] + det_9_j[:,5] / 2.]))
    C9_undist_boxs = getUndistortedBoxes(dist_coeff9, A9, det_9_j)
    cxcy9 = undistorted_coords(cxcy9, dist_coeff9, A9)
    image9 = undistort_image(image9, A9, dist_coeff9)
    image9 = drawProjectedBoxes(image9,  C9_undist_boxs, (0, 255, 0))
    image9 = drawCenters(image9, cxcy9, color=(0, 255, 0))
    # get undistorted center C2
    # C2:[cx,cy]- yellow
    cxcy2 = np.transpose(np.array([det_2_j[:,2] +det_2_j[:,4] / 2., det_2_j[:,3] + det_2_j[:,5] / 2.]))
    dist_coeff2, A2 = camera_intrinsics(2)
    cxcy2 = undistorted_coords(cxcy2, dist_coeff2, A2)
    image2 = undistort_image(image2, A2, dist_coeff2)
    image2 = drawCenters(image2, cxcy2, color=(255, 255, 0))



    # When centroids are projected on common ground plane in C2
    #centroids>world>projected on ground>change ground to 1.5m plane

    #use C9 params for ground projection
    #since points are already undistorted
    dist_coeff9 = np.array([0.0, 0.0, 0.0, 0.0])
    dist_coeff2 = np.array([0.0, 0.0, 0.0, 0.0])
    new_centers = []
    for center in cxcy9:
        proj, new_center, proj_center = projectCentroid(center, H, A9, dist_coeff9)
        #image2 = drawVolume(image2, proj)
        new_centers.append(proj_center)
        # draw shifted centers in cube plane
    image2 = drawCenters(image2, new_centers, color=(0, 255, 0))

    #use C2 params for ground projection
    H2=np.eye(4)
    H2[2,3]=H[2,3]
    projPrimaryCenter = []
    for i, pt in enumerate(cxcy2):
        proj, center, proj_center = projectCentroid(pt, H2, A2, dist_coeff2)
        i#mage2 = drawVolume(image2, proj)
        projPrimaryCenter.append(proj_center)
    image2 = drawCenters(image2, projPrimaryCenter, color=(0, 0, 255))



    # draw box on image9
    #cxcy9_undist = np.transpose(np.array([det_9_j[:,2] , det_9_j[:,3] , det_9_j[:,4], det_9_j[:,5]]))
    #image9 = drawBoxes(image9,det_9_j)
    #image2 = drawBoxes(image2, det_2_j)
    final_img = cv2.vconcat([image9, image2])
    #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    #cv2.resizeWindow("image", 640, 480)
    #cv2.imshow("image", cv2.undistort(image2, K, dist))
    #cv2.waitKey(0)

    plt.imsave(out_path + str(j) + '.png', final_img, dpi=200)