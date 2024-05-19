###### For new MCTA algorthm paper
import numpy as np
import cv2
import os
import glob
import matplotlib.pyplot as plt
class Projection(object):
    def __init__(self, gt_path=None, out_path=None, images_path=None):
        self.gt_path = gt_path
        self.out_path = out_path
        self.images_path = images_path
        self.image_ext = 'png'

    def cam_params_getter(self, cam=None, isDistorted=False):
        if cam == 1:
            A = np.array([[1743.447, 0.0, 934.520],
                          [0.0, 1735.156, 444.398],
                          [0.0, 0.0, 1.0]])
            Rvec = np.array([1.759, 0.467, -0.331])
            Tvec = np.array([[-525.894], [45.407], [986.723]])
            if isDistorted:
                dist_coeff = np.array([-0.43, 0.61, 0.008, 0.0018, -0.69])
            else:
                dist_coeff = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        if cam == 2:
            A = np.array([[1707.266, 0.0, 978.131], # 1707.266845703125 0.0 978.1306762695312 0.0 1719.0408935546875 417.01922607421875 0.0 0.0 1.0
                          [0.0, 1719.041, 417.019],
                          [0.0, 0.0, 1.0]])
            Rvec = np.array([0.616, -2.145, 1.657])  # 0.6167870163917542 -2.14595890045166 1.6577140092849731
            Tvec = np.array([[1195.231], [-336.514], [2040.539]])  # 1195.231201171875 -336.5144958496094 2040.53955078125
            if isDistorted:
                dist_coeff = np.array([-0.22, -0.31, 0.0087, 0.00036, 0.61])

            else:
                dist_coeff = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        if cam == 4:
            A = np.array([[1725.277, 0.0, 995.014],
                          [0.0, 1720.581, 520.419],
                          [0.0, 0.0, 1.0]])
            Rvec = np.array([1.664, 0.967, -0.694])
            Tvec = np.array([[42.362], [-45.361], [1106.857]])
            if isDistorted:
                dist_coeff = np.array([-0.504, 0.76, 0.001, 0.003, -0.59])
            else:
                dist_coeff = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        if cam == 5:  # 1708.6573486328125 0.0 936.0921630859375 0.0 1737.1904296875 465.18243408203125 0.0 0.0 1.0
            A = np.array([[1708.657, 0.0, 936.092],
                          [0.0, 1737.190, 465.182],
                          [0.0, 0.0, 1.0]])
            Rvec = np.array([1.213, -1.477, 1.277])  # 1.2132920026779175 -1.4771349430084229 1.2775369882583618
            Tvec = np.array([[836.662], [85.868], [600.288]])  # 836.6625366210938 85.86837005615234 600.2880859375
            if isDistorted:
                dist_coeff = np.array([-0.31, 0.24, 0.01, 0.00007, -0.16])

            else:
                dist_coeff = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        if cam == 6:  # 1742.977783203125 0.0 1001.0738525390625 0.0 1746.0140380859375 362.4325866699219 0.0 0.0 1.0
            A = np.array([[1742.977, 0.0, 1001.073],
                          [0.0, 1746.014, 362.432],
                          [0.0, 0.0, 1.0]])
            Rvec = np.array([1.691, -0.396, 0.355])  # 1.6907379627227783 -0.3968360126018524 0.355197012424469
            Tvec = np.array(
                [[-338.553], [62.876], [1044.094]])  # -338.5532531738281 62.87659454345703 1044.094482421875
            if isDistorted:
                dist_coeff = np.array([-0.34, 0.45, 0.028, -0.001, -0.55])
            else:
                dist_coeff = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        if cam == 7:  # 1732.4674072265625 0.0 931.2559204101562 0.0 1757.58203125 459.43389892578125 0.0 0.0 1.0
            A = np.array([[1732.467, 0.0, 931.256],
                          [0.0, 1757.582, 459.433],
                          [0.0, 0.0, 1.0]])
            Rvec = np.array([1.644, 1.126, -0.727])  # 1.6439390182495117 1.126188039779663 -0.7273139953613281
            Tvec = np.array(
                [[-648.945], [-57.225], [1052.767]])  # -648.9456787109375 -57.225215911865234 1052.767578125
            if isDistorted:
                dist_coeff = np.array([-0.306, 0.103, 0.009, 0.00005, 0.16])
            else:
                dist_coeff = np.array([0.0, 0.0, 0.0, 0.0, 0.0])  # -
        if cam == 3:  # 1738.7144775390625 0.0 906.56689453125 0.0 1752.8876953125 462.0346374511719 0.0 0.0 1.0
            A = np.array([[1738.714, 0.0, 906.567],
                          [0.0, 1752.887, 462.034],
                          [0.0, 0.0, 1.0]])
            Rvec = np.array([0.551, 2.229, -1.772])  # 0.5511789917945862 2.229501962661743 -1.7721869945526123
            Tvec = np.array(
                [[55.071], [-213.244], [1992.845]])  # 55.07157897949219 -213.2444610595703 1992.845703125
            if isDistorted:
                dist_coeff = np.array([-0.295, -0.926, 0.016, -0.001, 4.33])
            else:
                dist_coeff = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        return A, Rvec, Tvec, dist_coeff


    def images_getter(self, cam):
        if os.path.isdir(self.images_path):
            im_list = sorted(glob.iglob(self.images_path + '/C{}/*.'.format(cam) + self.image_ext))
        return im_list


    def image_saver(self, img, basename):
        if os.path.isdir(self.out_path):
            plt.imsave(self.out_path + '/results/' + basename, img, dpi=300)


    @staticmethod
    def groundProjectPoint(rotMat, tvec, camera_matrix, image_point, z=0.0):
        # https://github.com/balcilar/Stereo-Camera-Calibration-Orthogonal-Planes
        camMat = np.asarray(camera_matrix)
        iRot = np.linalg.inv(rotMat)
        iCam = np.linalg.inv(camMat)

        uvPoint = np.ones((3, 1))

        # Image point
        uvPoint[0, 0] = image_point[0]
        uvPoint[1, 0] = image_point[1]

        tempMat = np.matmul(np.matmul(iRot, iCam), uvPoint)
        tempMat2 = np.matmul(iRot, tvec)
        # pdb.set_trace()
        s = (z + tempMat2[2,0]) / tempMat[2, 0]
        wcPoint = np.matmul(iRot, (np.matmul(s * iCam, uvPoint) - tvec))
        # wcPoint[2] will not be exactly equal to z, but very close to it
        assert int(abs(wcPoint[2] - z) * (10 ** 8)) == 0
        wcPoint[2] = z

        return wcPoint, s

    def plot_lines(self, img, points2plane, color, W=2910, H=1080):
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


    def drawProjectedBoxes(self, image, point2plane, color=(255, 0, 0)):
        for point in point2plane:
            image = self.plot_lines(image, point, color)
        return image

    def drawBoxes(self, img, dets, color=None):
        for i in range(dets.shape[0]):
            bb = dets[i][2:6]

            cv2.rectangle(img, (int(bb[0]), int(bb[1])),
                          (int(bb[0] + bb[2]), int(bb[1] + bb[3])),
                          (0, 255, 0), 5, cv2.LINE_AA)
        return img


    def drawVolume(self, image, proj, color=(255, 0, 0)):
        boxes = np.stack([np.reshape(np.asarray(np.transpose(proj[0:4, :])), (2, 4)),
                          np.reshape(np.asarray(np.transpose(proj[4:8, :])), (2, 4))])

        image = self.drawProjectedBoxes(image, boxes, color)

        # TODO: Correct the order of these points
        side1 = np.column_stack([boxes[0, 0:2, 0], boxes[1, 0:2, 0], boxes[0, 0:2, 2], boxes[1, 0:2, 2]])
        side2 = np.column_stack([boxes[0, 0:2, 1], boxes[1, 0:2, 1], boxes[0, 0:2, 3], boxes[1, 0:2, 3]])
        sides = np.array([side1, side2])

        image = self.drawProjectedBoxes(image, sides, color)

        return image

    def project_to_world_grid(self, _origin=[-300, -90, 0], _size=[1440, 480],
                              _offset=2.5, rvec=None, tvec=None, camera_matrices=None, dist_coef=None):
        """
        -- collected from wildtrack annotation tool and modified by siddique
        Generates 3D points on a grid & projects them into all the views,
        using the given extrinsic and intrinsic calibration parameters.
        :param _origin: [tuple] of the grid origin (x, y, z)
        :param _size: [tuple] of the size (width, height) of the grid
        :param _offset: [float] step for the grid density
        :param rvec: [list] extrinsic parameters
        :param tvec: [list] extrinsic parameters
        :param camera_matrices: [list] intrinsic parameters
        :param dist_coef: [list] intrinsic parameters
        :return:
        """
        points = []
        for i in range(_size[0] * _size[1]):
            x = _origin[0] + _offset * (i % 480)
            y = _origin[1] + _offset * (i / 480)
            points.append(np.float32([[x, y, 0]]))  # ground points, z-axis is 0
        #collect projected points when they are in the world grid
        projected = []
        for c in range(len(camera_matrices)):
            imgpts, _ = cv2.projectPoints(np.asarray(points),  # 3D points
                                          np.asarray(rvec[c]),  # rotation rvec
                                          np.asarray(tvec[c]),  # translation tvec
                                          camera_matrices[c],  # camera matrix
                                          dist_coef[c])  # distortion coefficients
            projected.append(imgpts)
        return projected

    def project3D(self, center, cam_params, cam=2):
        # https://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point
        centroid = center
        box_height = 200
        bos_pos = 30
        params = cam_params['C{}'.format(cam)]
        #if cam in [2,4,7,6,5,3]:
        rotMat, _ = cv2.Rodrigues(params['rvec'])
        wpoint, scale = self.groundProjectPoint(rotMat, params['tvec'], params['A'], centroid, 0.0)  # -850
        wpoint_center, scale = self.groundProjectPoint(rotMat, params['tvec'], params['A'], centroid, 0.0)  # -850
        # center_plane = np.array([[200, -200, 0], [200, 200, 0], [-200, -200, 0], [-200, 200, 0]])
        box = np.array([[0, 4*bos_pos, 0],
                        [0, 0, 0],
                        [-2*bos_pos, 4*bos_pos, 0],
                        [-2*bos_pos, 0, 0],

                        [0, 4*bos_pos, box_height],
                        [0, 0, box_height],
                        [-2*bos_pos, 4*bos_pos, box_height],
                        [-2*bos_pos, 0, box_height]])
        wpoints_cent = np.reshape(wpoint_center, (3,)) + np.array([0, 0, 0])
        return wpoints_cent[0:2]

    def projectCentroid(self, center, cam_params, ca=None, cp=None):
        # https://stackoverflow.com/questions/12299870/computing-x-y-coordinate-3d-from-image-point
        centroid = center
        box_height = 200
        bos_pos = 30
        params = cam_params['C{}'.format(ca)]
        #if cam in [2,4,7,6,5,3]:
        rotMat, _ = cv2.Rodrigues(params['rvec'])
        wpoint, scale = self.groundProjectPoint(rotMat, params['tvec'], params['A'], centroid, 0.0)  # -850
        wpoint_center, scale = self.groundProjectPoint(rotMat, params['tvec'], params['A'], centroid, 0.0)  # -850
        # center_plane = np.array([[200, -200, 0], [200, 200, 0], [-200, -200, 0], [-200, 200, 0]])
        box = np.array([[0, 4*bos_pos, 0],
                        [0, 0, 0],
                        [-2*bos_pos, 4*bos_pos, 0],
                        [-2*bos_pos, 0, 0],

                        [0, 4*bos_pos, box_height],
                        [0, 0, box_height],
                        [-2*bos_pos, 4*bos_pos, box_height],
                        [-2*bos_pos, 0, box_height]])
        wpoints_cent = np.reshape(wpoint_center, (3,)) + np.array([0, 0, 0])
        wpoints = np.reshape(wpoint, (3,)) + box
        '''
                if cam==1:
            rotMat, _ = cv2.Rodrigues(cam_params['C1']['rvec'])
            wpoint, scale = self.groundProjectPoint(rotMat, cam_params['C1']['tvec'], cam_params['C1']['A'], centroid,
                                                    0.0)  # -850
            wpoint_center, scale = self.groundProjectPoint(rotMat, cam_params['C1']['tvec'], cam_params['C1']['A'], centroid,
                                                 0.0)
            box = np.array([[100, -100, -box_height],
                            [100, 100, -box_height],
                            [-100, -100, -box_height],
                            [-100, 100, -box_height],

                            [100, -100, box_height],
                            [100, 100, box_height],
                            [-100, -100, box_height],
                            [-100, 100, box_height]])
            wpoints_cent = np.reshape(wpoint_center, (3,)) + np.array([0, 0, 0])
            wpoints = np.reshape(wpoint, (3,)) + box
        '''

        #project world point to reference camera C1
        projb, _ = cv2.projectPoints(wpoints.astype('float'),
                                     cam_params['C'+str(cp)]['rvec'],
                                     cam_params['C'+str(cp)]['tvec'],
                                     cam_params['C'+str(cp)]['A'],
                                     cam_params['C'+str(cp)]['dist_coeff'])

        # proj_cplane, _ = cv2.projectPoints(wpoints_cplane.astype('float'), rvec, H[:-1, 3:], K, dist)

        # Project C2 world points into image plane
        #img2Points = np.ones((1, 2), dtype='float')
        #img2Points[0, 0] = (1 / scale) * cam_params['C1']['A'][0, 0] * wpoint_center[0, 0] / 1.0 + cam_params['C1']['A'][0, 2]
        #img2Points[0, 1] = (1 / scale) * cam_params['C1']['A'][1, 1] * wpoint_center[1, 0] / 1.0 + cam_params['C1']['A'][1, 2]
        proj, _ = cv2.projectPoints(wpoint_center.reshape(1,3),
                                     cam_params['C'+str(cp)]['rvec'],
                                     cam_params['C'+str(cp)]['tvec'],
                                     cam_params['C'+str(cp)]['A'],
                                     cam_params['C'+str(cp)]['dist_coeff'])
        # pdb.set_trace()
        # if cam in [9,2]:
        # img2Points = undistorted_coords(img2Points, dist_coeff2, A2)

        # shifted_center = np.sum(proj_cent[:,0,:],axis=0)/4.0
        # wpoint_center[:2,0]
        return projb, proj.reshape(2,)#img2Points.reshape(2, )

    def drawCenters(self, img, centers, color=None):
        for center in centers:
            img = cv2.circle(img, (int(center[0]), int(center[1])), 5, color, -10)
        return img

    def world2img(self, gt_world, cam_params=None):
        img_points = []
        for wpoint_center in gt_world:
            proj, _ = cv2.projectPoints(wpoint_center[2:5].reshape(1,3), cam_params['C1']['rvec'], cam_params['C1']['tvec'],
                                    cam_params['C1']['A'], cam_params['C1']['dist_coeff'])
            img_points.append(proj.reshape(2,))
        return img_points

if __name__ == '__main__':
    #read two camera GT
    data_path = '/media/siddique/RemoteServer/LabFiles/CLASP/CLASP_Data/Data_GT/multicamera_wildtrack/wildtrack/Wildtrack_dataset'
    #select camera pair
    cam1 = 1
    cam2= 5

    gt_c1 = np.loadtxt(os.path.join(data_path, 'annotaions_mot/C{}.txt'.format(cam1)),delimiter=',')
    gt_c2 = np.loadtxt(os.path.join(data_path, 'annotaions_mot/C{}.txt'.format(cam2)),delimiter=',')
    img_path = os.path.join(data_path,'Image_subsets')
    cam_projections = Projection(gt_path=None,out_path=data_path,images_path=img_path)



    im1_list = cam_projections.images_getter(cam=cam1)
    im2_list = cam_projections.images_getter(cam=cam2)

    cam_params={}
    cam_params['C{}'.format(cam1)]={}
    cam_params['C{}'.format(cam2)]={}
    draw_3dbox = True
    for j, im_name in enumerate(im1_list):
        print(im_name)
        # Image
        image1 = cv2.imread(im_name)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.imread(im2_list[j])
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        # detections
        det_1_j = gt_c1[gt_c1[:,0]==j*5]
        det_2_j = gt_c2[gt_c2[:,0]==j*5]

        # get cam prams
        Cparam = cam_params['C{}'.format(cam1)]
        Cparam['A'],Cparam['rvec'],\
        Cparam['tvec'], Cparam['dist_coeff'] = cam_projections.cam_params_getter(cam=cam1)

        Cparam = cam_params['C{}'.format(cam2)]
        Cparam['A'],Cparam['rvec'],\
        Cparam['tvec'], Cparam['dist_coeff'] = cam_projections.cam_params_getter(cam=cam2)

        # get undistorted box and center: images are already distorted

        #cxcy2 = np.transpose(np.array([det_2_j[:,2] +det_2_j[:,4] / 2., det_2_j[:,3] + det_2_j[:,5] / 2.]))
        cxcy2 = np.transpose(np.array([det_2_j[:,2]+det_2_j[:,4] / 2., det_2_j[:, 3] + det_2_j[:, 5]]))
        if not draw_3dbox:
            image2 = cam_projections.drawBoxes(image2,  det_2_j, (0, 255, 0))

        image2 = cam_projections.drawCenters(image2, cxcy2, color=(0, 255, 0))

        #cxcy1 = np.transpose(np.array([det_1_j[:,2] +det_1_j[:,4] / 2., det_1_j[:,3] + det_1_j[:,5] / 2.]))
        cxcy1 = np.transpose(np.array([det_1_j[:,2]+det_1_j[:,4] / 2., det_1_j[:, 3] + det_1_j[:, 5]]))
        if not draw_3dbox:
            image1 = cam_projections.drawBoxes(image1,  det_1_j, (255, 255, 0))
        image1 = cam_projections.drawCenters(image1, cxcy1, color=(255, 255, 0))

        # When centroids are projected on common ground plane in C2
        #centroids>world>projected on ground>change ground to 1.5m plane

        #use C9 params for ground projection
        #since points are already undistorted
        new_centers = []
        for center in cxcy2:
            proj, proj_center = cam_projections.projectCentroid(center, cam_params,cam=cam2)
            if draw_3dbox:
                image1 = cam_projections.drawVolume(image1, proj, color=(0, 255, 0))
            new_centers.append(proj_center)
            # draw shifted centers in cube plane
        image1 = cam_projections.drawCenters(image1, new_centers, color=(0, 255, 0))

        #use C2 params for ground projection
        '''
            projPrimaryCenter = []
        for i, pt in enumerate(cxcy1):
            proj, proj_center = cam_projections.projectCentroid(pt, cam_params,cam=1)
            if draw_3dbox:
                image2 = cam_projections.drawVolume(image1, proj, color=(255, 0, 0))
            projPrimaryCenter.append(proj_center)
        image1 = cam_projections.drawCenters(image1, projPrimaryCenter, color=(255, 0, 0))
        '''




        # draw box on image9
        #cxcy9_undist = np.transpose(np.array([det_9_j[:,2] , det_9_j[:,3] , det_9_j[:,4], det_9_j[:,5]]))
        #image9 = drawBoxes(image9,det_9_j)
        #image2 = drawBoxes(image2, det_2_j)
        final_img = cv2.vconcat([image2, image1])
        #cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        #cv2.resizeWindow("image", 640, 480)
        #cv2.imshow("image", cv2.undistort(image2, K, dist))
        #cv2.waitKey(0)
        cam_projections.image_saver(final_img, os.path.basename(im_name))
