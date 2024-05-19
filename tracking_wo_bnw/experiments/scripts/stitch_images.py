# import the necessary packages
import numpy as np
import imutils
import cv2
import os
class Stitcher:
	def __init__(self):
		# determine if we are using OpenCV v4.X
		self.isv4 = True #imutils.is_cv4(or_better=True)

	def stitch(self, images, ratio=0.8, reprojThresh=10.0,
		showMatches=False):
		# unpack the images, then detect keypoints and extract
		# local invariant descriptors from them
		(imageB, imageA) = images
		(kpsA, featuresA) = self.detectAndDescribe(imageA)
		(kpsB, featuresB) = self.detectAndDescribe(imageB)
		# match features between the two images
		M = self.matchKeypoints(kpsA, kpsB,
			featuresA, featuresB, ratio, reprojThresh)
		# if the match is None, then there aren't enough matched
		# keypoints to create a panorama
		if M is None:
			return None

		# otherwise, apply a perspective warp to stitch the images
		# together
		(matches, H, status) = M
		print('H: {}'.format(H))
		#imageB = imutils.rotate_bound(imageB, -90)
		result = cv2.warpPerspective(imageA, H,
			(imageA.shape[1] + imageB.shape[1], imageA.shape[0])) #+imageB.shape[0]
		result[0:imageB.shape[0], 0:imageB.shape[1]] = imageB
		# check to see if the keypoint matches should be visualized
		if showMatches:
			vis = self.drawMatches(imageA, imageB, kpsA, kpsB, matches,
				status)
			# return a tuple of the stitched image and the
			# visualization
			return (result, vis)
		# return the stitched image
		return result

	def detectAndDescribe(self, image):
		# convert the image to grayscale
		gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
		# check to see if we are using OpenCV 3.X
		if self.isv4:
			# detect and extract features from the image
			descriptor = cv2.xfeatures2d.SIFT_create()
			(kps, features) = descriptor.detectAndCompute(image, None)
		# otherwise, we are using OpenCV 2.4.X
		else:
			# detect keypoints in the image
			detector = cv2.FeatureDetector_create("SIFT")
			kps = detector.detect(gray)
			# extract features from the image
			extractor = cv2.DescriptorExtractor_create("SIFT")
			(kps, features) = extractor.compute(gray, kps)
		# convert the keypoints from KeyPoint objects to NumPy
		# arrays
		kps = np.float32([kp.pt for kp in kps])
		# return a tuple of keypoints and features
		return (kps, features)

	def matchKeypoints(self, kpsA, kpsB, featuresA, featuresB,
		ratio, reprojThresh):
		# compute the raw matches and initialize the list of actual
		# matches
		#matcher = cv2.DescriptorMatcher_create("BruteForce")
		index_params = dict(algorithm=0, trees=5)
		search_params = dict()
		matcher = cv2.FlannBasedMatcher(index_params, search_params)
		rawMatches = matcher.knnMatch(featuresA, featuresB, k=2)
		matches = []
		# loop over the raw matches
		for m in rawMatches:
			# ensure the distance is within a certain ratio of each
			# other (i.e. Lowe's ratio test)
			if len(m) == 2 and m[0].distance < m[1].distance * ratio:
				matches.append((m[0].trainIdx, m[0].queryIdx))

		# computing a homography requires at least 4 matches
		if len(matches) > 4:
			print('found total matches: {}'.format(len(matches)))
			# construct the two sets of points
			ptsA = np.float32([kpsA[i] for (_, i) in matches])
			ptsB = np.float32([kpsB[i] for (i, _) in matches])
			#manual points
			#ptsA = np.array([[1167, 457],[1192,370],[1039,418],[1046,636]], dtype='float32') #,[556,736], [609, 636], [1063,306]
			#ptsB = np.array([[489, 630], [799, 677], [507, 437], [182, 515]],
							#dtype='float32') #, [445, 124], [590, 73], [1157, 385]
			# compute the homography between the two sets of points
			(H, status) = cv2.findHomography(ptsA, ptsB, cv2.RANSAC,
				reprojThresh)
			# return the matches along with the homograpy matrix
			# and status of each matched point
			return (matches, H, status)
		# otherwise, no homograpy could be computed
		return None

	def drawMatches(self, imageA, imageB, kpsA, kpsB, matches, status):
		# initialize the output visualization image
		(hA, wA) = imageA.shape[:2]
		(hB, wB) = imageB.shape[:2]
		vis = np.zeros((max(hA, hB), wA + wB, 3), dtype="uint8")
		vis[0:hA, 0:wA] = imageA
		vis[0:hB, wA:] = imageB
		# loop over the matches
		for ((trainIdx, queryIdx), s) in zip(matches, status):
			# only process the match if the keypoint was successfully
			# matched
			if s in [0, 1]:
				# draw the match
				ptA = (int(kpsA[queryIdx][0]), int(kpsA[queryIdx][1]))
				ptB = (int(kpsB[trainIdx][0]) + wA, int(kpsB[trainIdx][1]))
				cv2.line(vis, ptA, ptB, (0, 255, 0), 1)
		# return the visualization
		return cv2.resize(vis,(vis.shape[1]//2, vis.shape[0]//2))

if __name__ == '__main__':
	fr_num = 1
	img_path = '/media/siddique/464a1d5c-f3c4-46f5-9dbb-bf729e5df6d61/tracking_wo_bnw/data/PVD/HDPVD_new/train_gt'
	imgB = cv2.imread(os.path.join(img_path, 'C{}/img1/{:06d}.png'.format(330, fr_num + 33)))
	#imgB = cv2.rotate(imgB, cv2.ROTATE_90_COUNTERCLOCKWISE)
	#imgB = imutils.rotate_bound(imgB, angle=-90)
	imgA = cv2.imread(os.path.join(img_path, 'C{}/img1/{:06d}.png'.format(360, fr_num + 6)))
	stitcher = Stitcher()
	(result, vis) = stitcher.stitch([imgA, imgB], showMatches=True)
	#cv2.imshow("Image A", imgA)
	#cv2.imshow("Image B", imgB)
	cv2.imshow("Keypoint Matches", vis)
	cv2.imshow("Result", result)
	cv2.waitKey(0)
