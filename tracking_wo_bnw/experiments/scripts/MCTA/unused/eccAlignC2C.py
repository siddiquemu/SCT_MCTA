import numpy as np
import cv2
class AlignC2C(object):
    def __init__(self, img1, img2,  warp_mode = cv2.MOTION_HOMOGRAPHY, number_of_iterations=1000, termination_eps=0.000000001):
        self.img_primary = img1
        self.img_auxiliary = img2
        self.number_of_iterations = number_of_iterations
        self.termination_eps = termination_eps
        self.warp_mode = warp_mode #cv2.MOTION_AFFINE

    def eccAlign(self):

        # Convert images to grayscale
        im1_gray = cv2.cvtColor(self.img_primary, cv2.COLOR_BGR2GRAY)
        im2_gray = cv2.cvtColor(self.img_auxiliary, cv2.COLOR_BGR2GRAY)

        # Find size of image1
        sz = self.img_primary.shape

        # Define 2x3 or 3x3 matrices and initialize the matrix to identity
        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            warp_matrix = np.eye(3, 3, dtype=np.float32)
        else:
            warp_matrix = np.eye(2, 3, dtype=np.float32)

        # Define termination criteria
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                    self.number_of_iterations, self.termination_eps)

        # Run the ECC algorithm. The results are stored in warp_matrix.
        cc, warp_matrix = cv2.findTransformECC(im1_gray, im2_gray, warp_matrix, self.warp_mode, criteria, inputMask=None, gaussFiltSize=1)

        if self.warp_mode == cv2.MOTION_HOMOGRAPHY:
            # Use warpPerspective for Homography
            im2_aligned = cv2.warpPerspective(self.img_auxiliary, warp_matrix, (sz[1], sz[0]),
                                              flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
        else:
            # Use warpAffine for Translation, Euclidean and Affine
            im2_aligned = cv2.warpAffine(self.img_auxiliary, warp_matrix, (sz[1], sz[0]),
                                         flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        return im2_aligned, warp_matrix

    def vis_align(self, im2_aligned):
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.resizeWindow("image", 640, 480)
        final_img = cv2.hconcat([ im2_aligned,self.img_primary])
        cv2.imshow("image", final_img)
        cv2.waitKey(200)
        # (ORB) feature based alignment

def applyTransform(source_corners, H):
    dest_corners = np.empty(2)
    w = H[2][0] * source_corners[0] + H[2][1] * source_corners[1] + H[2][2] * 1
    dest_corners[0] = (H[0][0] * source_corners[0] + H[0][1] * source_corners[1] + H[0][2] * 1) / w
    dest_corners[1] = (H[1][0] * source_corners[0] + H[1][1] * source_corners[1] + H[1][2] * 1) / w
    return dest_corners

import matplotlib.pyplot as plt
if __name__ == '__main__':
    img1 = cv2.imread('/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/test_data/exp1/imgs/cam11exp1.mp4/000098.png')
    img2 = cv2.imread('/media/siddique/RemoteServer/LabFiles/CLASP2/2019_10_24/test_data/exp1/imgs/cam13exp1.mp4/000001.png')
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2= cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    #im2_aligned, warp_matrix = AlignC2C(img1, img2).eccAlign()
    #print warp_matrix
    warp_matrix = [[5.3732557e+00 ,- 3.0405042e-01, - 1.1991694e+03],
                    [8.4025323e-01,  3.8106701e+00, - 8.6227631e+02],
                    [1.2132257e-03, 1.4179787e-03, 1.0000000e+00]]
    img1_pos =[1842,688]
    img1 = cv2.circle(img1, (img1_pos[0], img1_pos[1]), 10, (255, 255, 0), -10)
    #pos_project = applyTransform(img1_pos, warp_matrix)
    pos_project = cv2.warpPerspective(img1_pos, warp_matrix, (img1.shape[1], img1.shape[0]),
                                      flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)
    img2 = cv2.circle(img2, (int(pos_project[0]), int(pos_project[1])), 10, (255, 255, 0), -10)
    #AlignC2C(img1, img2).vis_align(im2_aligned)
    final_img = cv2.hconcat([ img2,img1])
    plt.imshow(final_img)
    plt.show()