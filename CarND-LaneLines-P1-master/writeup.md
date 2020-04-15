# **Finding Lane Lines on the Road** 


---

**Finding Lane Lines on the Road**

The goals / steps of this project are the following:
* Make a pipeline that finds lane lines on the road
* Reflect on your work in a written report


[//]: # (Image References)

[image1]: ./examples/grayscale.jpg "Grayscale"

---

### Reflection

### 1. Describe your pipeline. As part of the description, explain how you modified the draw_lines() function.

My pipeline consisted of 6 steps. 
1- The function color_filter_remove() filter all colors that are not white and yellow using HSV values.
2- The func. canny() search edges.
3- The func. region_of_interest() cropped the region in fron of the vehicle, with size of pyramid.
4- Apply the func. hough_lines() for search the lines in the image cropped.
5- The func. test_large_line() search the lines that match with some parameters and calculate the long lane for paint the 2 proyections in the final image.
6- Paint the 2 final lines in the image.



If you'd like to include images to show how the pipeline works, here is how to include an image: 

![alt text]
You can see the images in the correspondent directory.
Example:
https://view54dc5dc0.udacity-student-workspaces.com/view/CarND-LaneLines-P1/test_images_output/solidWhiteCurve.jpg


### 2. Identify potential shortcomings with your current pipeline


One potential shortcoming would be what would happen when the lines in the road are very worn-out.
Other potential problem is if the road have curves.


### 3. Suggest possible improvements to your pipeline

A possible improvement would be to the segmentattion in the calcul of the final big line for the proyectattion, this perhaps can help in the roads with curves.
