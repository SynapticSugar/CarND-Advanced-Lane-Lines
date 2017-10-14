## Writeup Report

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a threshold binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistort.png "Undistorted"
[image2]: ./output_images/7_undistort.jpg "Road Transformed"
[image3]: ./output_images/7_threshold.png "Binary Example"
[image4]: ./output_images/1_visualize_birds_eye.png "Warp Example"
[image5]: ./output_images/5_visualize_fit_lines.png "Fit Visual"
[image6]: ./output_images/5_lane_lines.jpg "Output"
[video1]: ./output_videos/project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.

---
### Submission Files

* `writeup_report.md`
* `lane_finder.py`
* `/output_images/undistort.png`
* `/output_images/7_undistort.jpg`
* `/output_images/7_threshold.png`
* `/output_images/1_visualize_birds_eye.png`
* `/output_images/5_visualize_fit_lines.png`
* `/output_images/5_lane_lines.jpg`
* `/output_videos/project_video.mp4`
* `/output_videos/challenge_video.mp4`
* `/output_videos/harder_challenge_video.mp4`
* `dist_pickel.p`

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code to perform the calibration is contained in a function called `getDistortion()` lines 78 through 124.

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image using `cv2.findChessboardCorners()`.  `image_points` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

I then used the output `object_points` and `imgage_points` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![alt text][image1]

Once the camera matrix `mtx` and distortion parameters `dist` are known, they no longer need to be re-calculated and can be used repeatedly for the remainder of the pipeline.

Additionally, the distortion parameters are saved to a pickle file `dist_pickle.p` and can simply be retrieved for each subsequent program run.  See code lines 702 to 707 in `lane_finder.py`.

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

By calling `cv2.undistort()` with the saved `mtx` and `dist` parameters, I created an undistorted image, like the one below.

![alt text][image2]

The undistortion function is the first step of the `processImage()` pipeline in code lines 653 to 679 in `lane_finder.py`.

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 204 through 348 in `lane_finder.py`).

The sobel x and y gradient, magnitude, and overall gradient direction are all combined to form a composite sobel image.  The saturation and hue from a HLS colorspace as well as the value channel from a HSV colorspace were combined to form a composite binary image.

These two composites were combined to form the final threshold image.  The biggest contributors to the threshold were the x sobel and saturation channel.  The HSV value played a role in allowing the saturation's lower bound to be lowered without adding too much noise. This helped recover lines that were otherwise too hard to see on the road surface.

Here's an example of my output for this step for test image `test_images/test6.jpg`.

![alt text][image3]

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a class called `Perspective()` on line 18 
and a function called `getBirdsEyeView()` on line 350 in `lane_finer.py`.

The class holds the `src_pts` and `dst_pts` transformation verticies, pixels to world meters coordinate conversions, and a function to scale forward or back the 'top' of the source rectangle.  The `getBirdsEyeView()` function takes an image for input(`image`) and returns a warped image. The `src_pts` and `dst_pts` are accessed from the global `Perspective()` object.

The source points were hand picked from `test_images/straight_lines2.jpg` using the lane lines as a guide. Below is the input to the `Perspective()` class.
```python
src_pts = np.float32(
    [[278,670],  # bottom left
     [1019,670], # bottom right
     [680,444],  # top right
     [603,444]]) # top left
```
Two excerpts from the `Perspective()` class, lines 40 to 45 and 57 to 58, illustrating how the calculations are made.

```python
def _calcDst(self):
    self.dst_pts = np.float32(
    [[self.src_pts[0,0]+self.ctr_off, self.src_pts[0,1]],
     [self.src_pts[1,0]+self.ctr_off, self.src_pts[1,1]],
     [self.src_pts[1,0]+self.ctr_off, 0],
     [self.src_pts[0,0]+self.ctr_off, 0]])

...

self.src_pts = src_pts
self.ctr_off = int(640 - (src_pts[1,0] + src_pts[0,0]) / 2)
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 278.0, 670.0  | 270.0. 670.0  |
| 1019.0, 670.0 | 1011.0. 670.0 |
| 680.0, 444.0  | 1011.0. 0.0   |
| 603.0, 444.0  | 270.0. 0.0    |

I verified that my perspective transform was working as expected by drawing the `src_pts` points on the undistorted image and the `dst_pts` points onto its warped counterpart to verify that the lines appear parallel in the warped image. Code lines 360 to 398.

An example is below:

![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

The lane lines were detected by calling the function `findLaneLines()` on lines 435 to 550 in `lane_finder.py`.  I chose the sliding window approach over the convolution method as it provided better results without too much extra work on the `project_video.mp4`.

This function takes in the image, calculates a histogram of the bottom half of the image to caluclate left and right maximum locations as starting points for the rest of the algorithm. It marches a window, in steps, for each lane upward, through the thresholded image, and calcualtes the average x pixel location to recenter the window.  The pixels in the windows are added to the left or right lane data set respectively.  A second order polynomial curve is then fit to each lane both in pixel space and world space.  These are all returned back from the function. I decided to perform the initial histogram fit at each step in the pipeline as it was more robust than using the previous frames windows as starting points.

An example of a visualization of the `findLanesFunction()` is provided below.

![alt text][image5]

Here the left lane pixel inliers are shown in red and the right lane pixel inliers are shown in blue.  The corresponding polynomial curves are overlayed in yellow.

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I calculated the radius of curvature and the vehicle's position relative to the center of the lane in the function `measureCurvature()` on line 576 and `addTextOverlay()` on line 634 in my code in `lane_finder.py`.  I used 3.7 meters for the lane width to calculate the conversion from pixels to meters in the x direction.  I counted 11 dashes at 3 meters each in the transformed space to calculate the pixels to meters in the y direction.  All of this is done automatically in the `Perspective()` class.  The world space lane curves are calculated in the `findLanesFunction()` function.

To calculate curvature, the formula from the Udacity lecture notes in Chapter 36 of Advanced Lane Finding was used.  This is also explained in the sugested tutorial here: https://www.intmath.com/applications-differentiation/8-radius-curvature.php.

The overall center lane curvature was averaged from both left and right lanes curvature results.

The center of the lane was calculated from evaluating the x location from each lane polynomial fit at the y location at the bottom of the image (pixel 720).
The vehicle offset was converted to world coordinates using the x world scale factor `pvt.pix2meters_x` from the global `Perspective()` object `pvt`.


#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The lane was then recast back to the original undistorted image by use of the `drawLane()` function, code lines 601 to 632, in `lane_finder.py`.  Here is an example of my result on a test image:

![alt text][image6]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_videos/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I will talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.

I implemented the pipeline step by step in the same order as the project goals using the material from the lecture notes.  

I did encounter issues getting the threshold to work well on all of the test videos.  I found that successively adding utility functions for each type of the threshold, such as absSobelThresh(), magThresh, dirThreshold, saturationThreshold, valueThreshold, and hueThreshold, allowed me to 'play' with many different combinations very easily until I found the right one.

I found that having a perspective transform that was too far in the distance led to problems detecting lines and fitting curvature.  Rather than hard code this for each image or video, I used a `Perspective()` class to generate different distances on the fly by scaling the top of the src_pts as needed.  I found that a reduction in depth of the src points by 20% helped with the `harder_challenge_video.mp4`.

I did not need to use the `Line()` class template that were  suggested in the project notes, or add filtering to perform reasonably well on the entire project video. However, it would be useful if I were to extend this project to maximize performance on the two challenge videos.
I would use the `Line()` class to make sure that the line detections are consitent in the following ways: 
1. If one lane was not detected in the current frame but was detected in the previous frame, I would use the old detection
2. If one lane was not detected in the current frame and not detected in the previous frame, I could rebuild it by duplicating the other lane.
3. I could low pass filter the current frame lane line in order to avoid spurious detections and reduce jitter.
4. I could use the expected lane width to test the current frame lane detection and rule out impossible lane widths. This problem is especially evident in the `harder_challenge_video.mp4`.
