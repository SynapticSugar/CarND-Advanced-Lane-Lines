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

# Top down image transform coordinates and pixel to meter conversion constants.
# Calculated from an undistorted 'test_images/straight_lines2.jpg'.
# pix2meters_x = 39/720.0
# pix2meters_y = 3.7/741.0
# src_pts = np.float32(
#         [[288,670],
#          [1029,670],
#          [669,438],
#          [613,438]])
# # dst_pts = np.float32(
# #         [[288,720],
# #          [1029,720],
# #          [1029,0],
# #          [288,0]])
# dst_pts = np.float32(
#         [[320,720],
#          [961,720],
#          [961,0],
#          [320,0]])

# less height
src_pts = np.float32(
        [[278,670],
         [1019,670],
         [680,444],
         [603,444]])
# centered and top-down
dst_pts = np.float32(
        [[270,670],
         [1011,670],
         [1011,0],
         [270,0]])
pix2meters_y = 33/720.0
pix2meters_x = 3.7/741.0

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
    # undistort image
    img = cv2.imread(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    # generate before and after plot
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)

    # save plot
    plt.savefig(fname)

    # display to screen
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
    cv2.imwrite(fname, img)

def readWriteUndistort(fread, fsave, mtx, dist):
    '''
    Utility to read and image, perform the undistortion, and save it.
    '''
    image = cv2.imread(fread)
    image = cv2.undistort(image, mtx, dist, None, mtx)
    cv2.imwrite(fsave, image)

def abs_sobel_thresh(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    '''
    Method to apply a sobel filter with threshold to a grayscale image.
    Returns binary image.
    '''
    # Calculate directional gradient
    if orient is'x':
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
    else:
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # Take the derivative in y
    abs_sobel = np.absolute(sobel) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))

    # Apply threshold
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return grad_binary

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    '''
    '''
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel) # Take the derivative in y
    
    # Calculate gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)

    # Rescale to 8 bit
    #scale_factor = np.max(gradmag)/255 
    #gradmag = (gradmag/scale_factor).astype(np.uint8) 
    scaled_gradmag = np.uint8(255*gradmag/np.max(gradmag))

    # Create a binary image of ones where threshold is met, zeros otherwise
    mag_binary = np.zeros_like(scaled_gradmag)
    mag_binary[(scaled_gradmag >= mag_thresh[0]) & (scaled_gradmag <= mag_thresh[1])] = 1

    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    '''
    '''
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    
    # Calculate gradient direction
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))

    # Apply threshold
    dir_binary =  np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return dir_binary

def thresholdImage(image, angle=True, visualize=False):
    '''
    Method to threshold the image using the combination of a threshold sobel x edge detection 
    and a threshold color saturation image. Returns a binary image.
    Adapted from Udacity lecture notes in Chapter 30 of Advanced Lane Finding.
    '''
    # Convert to HLS color space and separate the S channel
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]

    # Grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    ksize = 3
    gradx = abs_sobel_thresh(gray, orient='x', sobel_kernel=ksize, thresh=(20, 255))
    grady = abs_sobel_thresh(gray, orient='y', sobel_kernel=ksize, thresh=(150, 255))
    mag_binary = mag_thresh(gray, sobel_kernel=ksize, mag_thresh=(100, 255))
    dir_binary = dir_threshold(gray, sobel_kernel=ksize, thresh=(np.pi/4, np.pi/2))

    sobel_combined = np.zeros_like(gradx)
    sobel_combined[((gradx == 1) | (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1

    # Threshold color channel
    s_thresh_min = 150
    s_thresh_max = 255
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh_min) & (s_channel <= s_thresh_max)] = 1
    
    # Threshold hue channel 0-179
    h_value_min = 20 # yellow
    h_value_max = 25 # yellow
    h_binary = np.zeros_like(h_channel)
    h_binary[(h_channel >= h_value_min) & (h_channel <= h_value_max)] = 1

    # Stack each channel to view their individual contributions in green and blue respectively
    # This returns a stack of the two binary images, whose components you can see as different colors
    color_binary = np.dstack(( h_binary, sobel_combined, s_binary)) * 255

    # Combine the two binary thresholds
    combined_binary = np.zeros_like(sobel_combined)
    combined_binary[(s_binary == 1) | (sobel_combined == 1) | (h_binary == 1)] = 1

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
    '''
    Method to perform the top down perspective transform on the undistorted image.
    '''
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    img_size = (image.shape[1], image.shape[0])
    warped = cv2.warpPerspective(image, M, img_size, flags=cv2.INTER_LINEAR)
    return warped


def visualizeBirdsEye(fname, idx, thresh=False):
    '''
    Utility to visualize the birds eye perspective transform. Saves a plot of undistorted
    image before and after warp as well as the raw undistorted image. Has options to show
    threshold and display output.
    '''
    # load and undistort image from file
    image = cv2.imread(fname)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    undst = cv2.undistort(image, mtx, dist, None, mtx)

    # threshold the image if needed
    tag = ''
    if thresh is True:
        undst = thresholdImage(undst)
        undst = np.dstack((undst, undst, undst))*255
        tag='_bin'

    # draw the points used for the transform on the image
    vector = np.array(src_pts, np.int32)
    vector = vector.reshape((-1,1,2))
    cv2.polylines(undst, [vector], True, [255, 0, 0])

    # generate the visualization plot
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20,10))
    ax1.imshow(undst)
    ax1.set_title('Undistorted image' + tag + ' with points drawn', fontsize=30)
    write_name = 'output_images/'+str(idx)+'_undistort' + tag + '.jpg'
    writeImage(write_name, undst)
    bird_img = getBirdsEyeView(undst)
    ax2.imshow(bird_img)
    ax2.set_title('Top down image', fontsize=30)
    plt.savefig('output_images/'+str(idx)+'_visualize_birds_eye' + tag + '.png')
    write_name = 'output_images/'+str(idx)+'_birds_eye' + tag + '.jpg'
    writeImage(write_name, bird_img)
    #plt.show()


def findLaneLines(image, index=-1, visualize=False):
    '''
    Assumes that the input is a top down warped binary image.
    Adapted from Udacity lecture notes in Chapter 33 of Advanced Lane Finding.
    '''
    # Take a histogram of the bottom half of the image
    histogram = np.sum(image[int(image.shape[0]/2):,:], axis=0)

    # Create an output image to draw on and  visualize the result
    out_img = np.dstack((image, image, image))*255
    
    # Find the peak of the left and right halves of the histogram
    # These will be the starting point for the left and right lines
    midpoint = np.int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9

    # Set height of windows
    window_height = np.int(image.shape[0]/nwindows)
    
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    # Set the width of the windows +/- margin
    margin = 100
    
    # Set minimum number of pixels found to recenter window
    minpix = 50
    
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = image.shape[0] - (window+1)*window_height
        win_y_high = image.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin

        # Draw the windows on the visualization image
        cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
        (0,255,0), 2)
        cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
        (0,255,0), 2)

        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]

        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Fit new polynomials to x,y in world space
    left_fit_cr = np.polyfit(lefty*pix2meters_y, leftx*pix2meters_x, 2)
    right_fit_cr = np.polyfit(righty*pix2meters_y, rightx*pix2meters_x, 2)

    if visualize is True:
        # Generate x and y values for plotting
        ploty = np.linspace(0, image.shape[0]-1, image.shape[0] )
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]
        fig = plt.figure()
        plt.imshow(out_img)
        plt.plot(left_fitx, ploty, color='yellow')
        plt.plot(right_fitx, ploty, color='yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)
        plt.savefig('output_images/'+str(idx)+'_visualize_find_lines.png')
        #plt.show()

    return left_fit, right_fit, left_fit_cr, right_fit_cr

def visualizeFindLaneLines(fname, idx):
    # load and undistort image from file
    image = cv2.imread(fname)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    undst = cv2.undistort(image, mtx, dist, None, mtx)
    bin_img = thresholdImage(undst, visualize=False)
    bird_img = getBirdsEyeView(bin_img)
    left_fit, right_fit, left_fit_cr, right_fit_cr = findLaneLines(bird_img, index=idx, visualize=True)

def measureCurvature(y_eval, left_fit, right_fit):
    '''
    Adapted from Udacity lecture notes in Chapter 36 of Advanced Lane Finding.
    '''
    # Calculate the new radii of curvature
    left_curverad = ((1 + (2*left_fit[0]*y_eval*pix2meters_y + left_fit[1])**2)**1.5) / np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval*pix2meters_y + right_fit[1])**2)**1.5) / np.absolute(2*right_fit[0])
    
    # Now our radius of curvature is in meters
    # print(left_curverad, 'm', right_curverad, 'm')

    return (left_curverad + right_curverad) / 2

def genPolyArrays(y_min, y_max, left_poly, right_poly, segments=9):
    y_val = []
    left_fitx = []
    right_fitx = []
    step = np.float32(y_max-y_min)/segments
    for i in range(segments):
        y = y_max - (step * i)
        left_fitx.append(np.uint32(np.polyval(left_poly, y)))
        right_fitx.append(np.uint32(np.polyval(right_poly, y)))
        y_val.append(np.uint32(y))
    return y_val, left_fitx, right_fitx

def drawLane(bin_image, image, left_poly, right_poly, visualize=False):
    '''
    Adapted from Udacity lecture notes in Chapter 36 of Advanced Lane Finding.
    '''
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(bin_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Generate the polynomial arrays
    ploty, left_fitx, right_fitx = genPolyArrays(0, bin_image.shape[0], left_poly, right_poly)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = cv2.getPerspectiveTransform(dst_pts, src_pts)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(image, 1, newwarp, 0.3, 0)

    if visualize is True:
        fig = plt.figure()
        plt.imshow(result)
        plt.show()

    return result

def addTextOverlay(image, curvature, left_poly, right_poly, visualize=False):
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 2;
    thickness = 3;
    y = image.shape[0]
    leftx = np.polyval(left_poly, y)
    rightx = np.polyval(right_poly, y)
    center = (rightx + leftx) / 2
    offset = (center - image.shape[1] / 2) * pix2meters_x
    text = 'Radius of Curvature = {:0.0f}(m)'.format(curvature)
    cv2.putText(image, text, (100,100), font_face, font_scale, [255,255,255], thickness, 8)
    text = 'Vehicle is {:02.2f}m {} of center'.format(np.abs(offset), 'left' if offset >= 0 else 'right')
    cv2.putText(image, text, (100,200), font_face, font_scale, [255,255,255], thickness, 8)
    if visualize:
        fig = plt.figure()
        plt.imshow(image)
        plt.show()
    return image

def processImage(image): # mtx, dist are global
    '''
    The main processing pipeline. Call this for each image.
    '''
    # Apply a distortion correction to raw images.
    udist_img = cv2.undistort(image, mtx, dist, None, mtx)

    # Use color transforms, gradients, etc., to create a thresholded binary image.
    bin_img = thresholdImage(udist_img)

    # Apply a perspective transform to rectify binary image ("birds-eye view").
    bird_img = getBirdsEyeView(bin_img)

    # Detect lane pixels and fit to find the lane boundary.
    left_fit, right_fit, left_fit_cr, right_fit_cr = findLaneLines(bird_img)
    
    # Determine the curvature of the lane and vehicle position with respect to center.
    curvature = measureCurvature(bird_img.shape[0], left_fit_cr, right_fit_cr)

    # Warp the detected lane boundaries back onto the original image.
    udist_img = drawLane(bird_img, udist_img, left_fit, right_fit)

    # Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
    udist_img = addTextOverlay(udist_img, curvature, left_fit, right_fit)

    return udist_img

# Main

# Get camera matrix and distortion coeffecients (only do this once)
mtx, dist = loadDistortion("dist_pickle.p")
if mtx is 0:
    images = glob.glob('camera_cal/calibration*.jpg')
    mtx, dist = getDistortion(images, model_shape=(9,6), visualize=False)
    saveDistortion("dist_pickle.p", mtx, dist)
    visualizeDistortion(images[0], mtx, dist, "output_images/undistort.png", visualize=False)

# Process the test images
dirs = os.listdir("test_images/")
for idx, filename in enumerate(dirs):
    # For visualization
    visualizeBirdsEye("test_images/" + filename, idx, thresh=False)
    visualizeFindLaneLines("test_images/" + filename, idx)
    image = mpimg.imread("test_images/" + filename)
    final_image = processImage(image)

# Process the videos
from moviepy.editor import VideoFileClip
from IPython.display import HTML


if 0:
    project_output = 'output_videos/project_video.mp4'
    clip1 = VideoFileClip("project_video.mp4")
    project_clip = clip1.fl_image(processImage) #NOTE: this function expects color images!!
    project_clip.write_videofile(project_output, audio=False)

if 0:
    project_output = 'output_videos/challenge_video.mp4'
    clip1 = VideoFileClip("challenge_video.mp4")
    project_clip = clip1.fl_image(processImage) #NOTE: this function expects color images!!
    project_clip.write_videofile(project_output, audio=False)

if 0:
    project_output = 'output_videos/harder_challenge_video.mp4'
    clip1 = VideoFileClip("harder_challenge_video.mp4")
    project_clip = clip1.fl_image(processImage) #NOTE: this function expects color images!!
    project_clip.write_videofile(project_output, audio=False)
