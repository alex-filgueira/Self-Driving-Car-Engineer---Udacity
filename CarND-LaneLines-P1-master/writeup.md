# **Finding Lane Lines on the Road** 


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./images_output/solidWhiteCurve.jpg "Solid white curve"
[image2]: ./images_output/solidWhiteRight.jpg "Solid white only in right"
[image3]: ./images_output/solidYellowCurve.jpg "Solid yellow curve"
[image4]: ./images_output/solidYellowCurve2.jpg "Solid yellow curve 2"
[image5]: ./images_output/solidYellowLeft.jpg "Solid yellow left"
[image6]: ./images_output/whiteCarLaneSwitch.jpg "Solid white lane switch"

[video1]: ./videos_output/solidWhiteRight.mp4 "Solid white right"
[video2]: ./videos_output/solidYellowLeft.mp4 "Solid yellow left"
[video3]: ./videos_output/challenge.mp4 "Video for the challenge"


---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps:

1- The function color_filter_remove() filter all colors that are not white and yellow using HSV values.

2- The func. canny() search edges.

3- The func. region_of_interest() cropped the region in fron of the vehicle, with size of pyramid.

4- Apply the func. hough_lines() for search the lines in the image cropped.

5- The func. test_large_line() search the lines that match with some parameters and calculate the long lane for paint the 2 proyections in the final image.

6- Paint the 2 final lines in the image.



## If you'd like to include images to show how the pipeline works, here is how to include an image: 

You can see the images in the correspondent directory.

./images_output

"Solid white curve"
![alt text][image1]
"Solid white only in right"
![alt text][image2]

![alt text][image3] ![alt text][image4] ![alt text][image5] ![alt text][image6]

Also you can see the out videos in:

./videos_output directory.

[video1], [video2], [video3]


### 2. Identify potential shortcomings with your current pipeline



One potential shortcoming would be what would happen when the lines in the road are very worn-out.
Other potential problem is if the road have curves.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to the segmentattion in the calcul of the final big line for the proyectattion, this perhaps can help in the roads with curves.
