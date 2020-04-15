## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./output_images/undistorted_calibration2.jpg "Undistorted"
[image2]: ./undistorted_images/undistorted_test1.jpg "Road Transformed"
[image3]: ./combined.png "Binary Example"
[image4]: ./warped_grey.png "Warp Example"
[image6]: ./output_images/finish_test1.jpg "Output"
[image7]: ./output_images/undistorted_calibration2.jpg "Undistorted calibration image"

[video1]: ./output_video/project_video_all.mp4 "Project Video"
[video2]: ./output_video/challenge_video_all.mp4 "Challenge Video"
[video3]: ./output_video/harder_challenge_video_all.mp4 "Harder Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in the function calibrateCamera and search_corners of the IPython notebook located [here](./CarND-Advanced-Lane-Lines/Advanced_FInding_Project.ipynb)


I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result in:

"Undistorted calibration image"
![alt text][image7]



### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

To demonstrate this step, I will describe how I apply the distortion correction, you can see some images [here](./CarND-Advanced-Lane-Lines/undistorted_images/)


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used the HLS model. I filtered for S using the threshol:(170, 255).
Also I filtered the image using gradient magnitude thresholds like (20,100), all in X.
I used a combination of color and gradient thresholds to generate a binary image, you can see a example.
![alt text][image3]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warper_image()`, defined in the file [file](./CarND-Advanced-Lane-Lines/examples/example.ipynb) ``.  The `warper()` function takes as inputs an image (`image, image_proc, M, show=False`). This function call to perspectiveTransform() and send the image. perspectiveTransform() has defined  source (`src`) and destination (`dst`) points.  I chose the hardcode the source and destination points in the following manner:

```python
    src = np.float32(
        [[585, 460],
        [203, 720],
        [1127, 720],
        [695, 460]])

    
    dst = np.float32(
        [[320, 0],
        [320, 720],
        [960, 720],
        [960, 0]])
```



I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.
![alt text][image4]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

I used 2 functions:
fit_polynomial(binary_warped, nwindows = 9, margin = 50, minpix = 50)
search_around_poly(binary_warped,left_fit, right_fit)
The first is used when the image is "new" a photo or frame in the video that not have other recognized frame previous that was similar to the present frame.
The second is used when we procesed a video and is a improve for the search algoritm.
The 2 functions return the values for left_fit, right_fit.


#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I used the function measure_curvature(binary_warped,left_fit,right_fit) described in the file "./examples/example.ipynb".
The maths used for calculate the radio are the same that was explained in class but with the correction for pass from pixels to meters.
The algorithm for calculate the position of the vehicle with respect to center take in count the postion of the road lines and calculate your middle in the y = 600 (first I triyed with y = 720 but I had worse results because the algorithm for calculate the rects, sometime lose this high.) With the midlle of the lines and the middle of the image (640, position where is the camera) the algorithm take the different and this is the position of the vehicle with respect to center.

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in the function "draw_on_image(undist, warped_img, left_fit, right_fit, M, left_curvature, right_curvature, center, show_values = False). You can see some examples in the folder ./output_images.



---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](video1)



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

You can see the videos [video2] and [video3], but the results are worse, in the first the algorirthm have problems with the colors of the road, and in the second have a lot of problems, with the light, the obstacles and the shorted distance, the algorithm is optimizated for long distances. 
