# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# Apply a distortion correction to raw images.
# Use color transforms, gradients, etc., to create a thresholded binary image.
# Apply a perspective transform to rectify binary image ("birds-eye view").
# Detect lane pixels and fit to find the lane boundary.
# Determine the curvature of the lane and vehicle position with respect to center.
# Warp the detected lane boundaries back onto the original image.
# Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

import numpy as np
import cv2
import glob
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def warpImage(img, src, dst):
    '''
    Method to return a warped image given frour pairs of corresponding points.
    '''
    #img_size = (img.shape[1], img.shape[0])
    #M = cv2.getPerspectiveTransform(src, dst)
    #warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_NEAREST)  # keep same size as input image

    #return warped

def getDistortion(images, model_shape=(9,6), visualize=False):
    '''
    Method to generate the calibration parameters from a group of calibration images and
    a given model shape.
    Adapted from: http://docs.opencv.org/3.2.0/dc/dbb/tutorial_py_calibration.html and
    https://github.com/udacity/CarND-Camera-Calibration/blob/master/camera_calibration.ipynb
    '''
    # Termination criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Arrays to store object points and image points from all the images.
    object_points = [] # 3D
    image_points = [] # 2D
    
    # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(9,6,0)
    objp = np.zeros((model_shape[0] * model_shape[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:model_shape[0],0:model_shape[1]].T.reshape(-1,2)

    # Get the object points and image points
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, model_shape, None)

        # If found, add object points, image points (after refining them)
        if ret == True:
            object_points.append(objp)
            corners2=cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            image_points.append(corners)
        
            # Draw the corners
            cv2.drawChessboardCorners(img, (model_shape[0],model_shape[1]), corners2, ret)
            write_name = 'camera_cal/corners_found'+str(idx)+'.jpg'
            cv2.imwrite(write_name, img)
            if visualize:
                cv2.imshow('img', img)
                cv2.waitKey(500)

    # Get image shape from the first image
    img = cv2.imread(images[0])
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, img_size, None, None)

    return mtx, dist

# Define a class to receive the characteristics of each line detection
class Line():
    def __init__(self):
        # was the line detected in the last iteration?
        self.detected = False
        # x values of the last n fits of the line
        self.recent_xfitted = []
        #average x values of the fitted line over the last n iterations
        self.bestx = None
        #polynomial coefficients averaged over the last n iterations
        self.best_fit = None
        #polynomial coefficients for the most recent fit
        self.current_fit = [np.array([False])]
        #radius of curvature of the line in some units
        self.radius_of_curvature = None
        #distance in meters of vehicle center from the line
        self.line_base_pos = None
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float')
        #x values for detected line pixels
        self.allx = None
        #y values for detected line pixels
        self.ally = None

def saveDistortion(fname, mtx, dist):
    '''
    Method to save the distortion coeffecients and camera matrix to file.
    '''
    dist_pickle = {}
    dist_pickle["mtx"] = mtx
    dist_pickle["dist"] = dist
    pickle.dump(dist_pickle, open( fname, "wb" ))

def loadDistortion(fname):
    '''
    Method to load the distortion coeffecients and camera matrix from file.
    '''
    if os.path.isfile(fname):
        dist_pickle = pickle.load(open( fname, "rb" ))
        return dist_pickle["mtx"], dist_pickle["dist"]
    else:
        return 0,0

def visualizeDistortion(image, mtx, dist, fname, visualize=False):
    '''
    Method to save an example of pre and post image distortion correction.
    Adapted from: https://github.com/udacity/CarND-Camera-Calibration/blob/master/camera_calibration.ipynb
    '''
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.savefig(fname)
    if visualize == True:
        plt.show()

def showImage(image):
    '''
    Utility to show an image
    '''
    # convert to opencv BGR
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imshow('img',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def writeImage(fname, image):
    '''
    Utility to write an image
    '''
    # convert to opencv BGR
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(fname, image)

def readWriteUndistort(fread, fsave, mtx, dist):
    '''
    Utility to read and image, perform the undistortion, and save it.
    '''
    image = cv2.imread(fread)
    image = cv2.undistort(image, mtx, dist, None, mtx)
    cv2.imwrite(fsave, image)

def thresholdImage(image, visualize=False):

    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]

    # Grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Sobel x
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    thresh_min = 20
    thresh_max = 100
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= thresh_min) & (scaled_sobel <= thresh_max)] = 1

    # Threshold color channel
    s_thresh_min = 170
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1

    if visualize is True:
        # Plotting thresholded images
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
        ax1.set_title('Stacked thresholds')
        ax1.imshow(color_binary)

        ax2.set_title('Combined S channel and gradient thresholds')
        ax2.imshow(combined_binary, cmap='gray')
        plt.show()

    return combined_binary

def getBirdsEyeView(image):
    src = np.float32(
        [[252,680],
         [1053,680],
         [689,450],
         [593,450]])
    dst = np.float32(
        [[252,720],
         [1053,720],
         [1053,450],
         [252,450]])
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (image.shape[1], image.shape[0])
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped

def processImage(image, mtx, dist):

    # Apply a distortion correction to raw images.
    undist_img = cv2.undistort(image, mtx, dist, None, mtx)

    # Use color transforms, gradients, etc., to create a thresholded binary image.
    bin_img = thresholdImage(undist_img, visualize=False)

    # Apply a perspective transform to rectify binary image ("birds-eye view").
    bird_img = getBirdsEyeView(bin_img)



    # Detect lane pixels and fit to find the lane boundary.
    # Determine the curvature of the lane and vehicle position with respect to center.
    # Warp the detected lane boundaries back onto the original image.
    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    return undist_img

# Main

# Get camera matrix and distortion coeffecients (only do this once)
mtx, dist = loadDistortion("dist_pickle.p")
if mtx is 0:
    images = glob.glob('camera_cal/calibration*.jpg')
    mtx, dist = getDistortion(images, model_shape=(9,6), visualize=False)
    saveDistortion("dist_pickle.p", mtx, dist)
    visualizeDistortion(images[0], mtx, dist, "output_images/undistort.png", visualize=False)

readWriteUndistort("test_images/straight_lines2.jpg", "straight_sample.png", mtx, dist)
image = cv2.imread("straight_sample.png")
bird_img = getBirdsEyeView(image)
cv2.imshow('img',bird_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("straight_sample_birds_eye.png", bird_img)
exit()

dirs = os.listdir("test_images/")
for filename in dirs:
    image = mpimg.imread("test_images/" + filename)
    final_image = processImage(image, mtx, dist)
