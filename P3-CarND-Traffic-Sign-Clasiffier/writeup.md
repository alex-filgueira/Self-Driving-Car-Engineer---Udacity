# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)


[image0]: ./CarND-Traffic-Sign-Classifier-Project/images_output/statistics.PNG "Statistics"
[image1]: ./CarND-Traffic-Sign-Classifier-Project/images_output/visualization_data.PNG "Visualization"
[image2]: ./CarND-Traffic-Sign-Classifier-Project/images_output/gray_escaling.PNG "Grayscaling"
[image3]: ./CarND-Traffic-Sign-Classifier-Project/images_output/gradients.PNG "Gradients"
[image4]: ./CarND-Traffic-Sign-Classifier-Project/my_signals/14.stop_2.PNG "My signal -> Stop"
[image5]: ./CarND-Traffic-Sign-Classifier-Project/my_signals/2.max50.PNG "My signal -> Max 50-1"
[image6]: ./CarND-Traffic-Sign-Classifier-Project/my_signals/2.max50_2.PNG "My signal -> Max 50-2"
[image7]: ./CarND-Traffic-Sign-Classifier-Project/my_signals/20.curva_derecha.PNG "My signal -> Right curve"
[image8]: ./CarND-Traffic-Sign-Classifier-Project/my_signals/33.obligatoriodecha.PNG "My signal -> Right now"
[image9]: ./CarND-Traffic-Sign-Classifier-Project/images_output/my_images_statistic.PNG "My signals Statistics"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](./CarND-Traffic-Sign-Classifier-Project/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The summary statistics of the traffic signs data set:

* The size of training set is 34799
* The size of the validation set is 4410
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43


#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed

![alt text][image0]
![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

I tried proced the images in the Neural Network using RGB color but the results was not enought good.

After I convert the image to grayscale and also found the edges using gradient x + y filter but the results was worse.
![alt text][image3]

When I tried to procesd the images in the Neural Network using only the images in gray the results was so good.

Mod: 02/04/20 -> After a feedback I added a normalize for all the images.

Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]




#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

Input: 32x32x1 Gray image
L1:
convolution layer: 32x32x1 -> 28x28x12
max_pool: 28x28x12 -> 14x14x12
L2:
convolution layer 14x14x12 -> 10x10x25
max_pool: 10x10x25 -> 5x5x25
L3:
flatten
dropout
linear 625 -> 300
L4:
linear 300 -> 100
RELU
L5:
linear 100 -> 43


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, After a lot of test, I decided that the better parameters are the next:

epoch = 10
batch_size = 128
rate = 0.001

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* training set accuracy of:
    [2.4s] epoch 1/10: validation = 0.866
    [1.8s] epoch 2/10: validation = 0.928
    [1.8s] epoch 3/10: validation = 0.965
    [1.8s] epoch 4/10: validation = 0.974
    [1.8s] epoch 5/10: validation = 0.976
    [1.8s] epoch 6/10: validation = 0.980
    [1.8s] epoch 7/10: validation = 0.989
    [1.8s] epoch 8/10: validation = 0.993
    [1.8s] epoch 9/10: validation = 0.990
    [1.8s] epoch 10/10: validation = 0.994
* validation set accuracy of  0.937
* test set accuracy of 0.923911321947

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
    I used LeNET modificated because is the proposed for make the project.
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
    For the final preprocesed images selected (only grey images), I needed change some times the Hyperparameters fir take the best results, here a sumary for the last change when the results was more acurracy:
epoch =      10 - 10 - 20 - 5 - 10
batch_size = 64 - 128 - 32 -128 - 256
rate =       0.001 - = ... (when I triyed to change this parameter the result was ever worse.)
validation = 0.928 - 0.937 - 0.933 - 0.92 - 0.933
I triyed to aument and reduce epoch and batch_size but the results was worse or not better.
    
* Which parameters were tuned? How were they adjusted and why?
    Sumarize above.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
   This is explained in a lot of papers that study the LeNET architecture.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

I think that the most important problem for classify this images is the "quality", the system was trained with a big set of images taked from far, with bad light and they was cropped. My set of images have high colors and are good focus.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			              | Prediction	            					| 
|:---------------------------:|:-------------------------------------------:| 
| Turn right ahead    	      | No entry   									| 
| Dangerous curve to the right| Dangerous curve to the right 				|
| Speed limit (50km/h)		  | Speed limit (60km/h)						|
| Stop	      		          | No entry					 				|
| Speed limit (50km/h)		  | No passing for vehicles over 3.5 metric tons|


You can see ![alt text][image9] the accuracy of this prediction between the images that I aported and the first 5 images that the system calculated that was more probability of match.


The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%.


#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the Ipython notebook.

For the first image, the model is sure that this is a No entry sign (probability of 1), and the image does contain a No entry sign.

| Probability         	| Prediction	        					  | 
|:---------------------:|:-------------------------------------------:| 
| 1         			|  No entry   							      | 






For the second image the model is sure that this is a Dangerous curve to the right sign (probability of 1), and the prediction is correct. 

| Probability         	| Prediction	        					  | 
|:---------------------:|:-------------------------------------------:| 
| 1     				| Dangerous curve to the right 		    	  |


For the 3 image the model is some sure that this is a Speed limit (60km/h) sign (probability of 0.67), the image is Speed limit (50km/h) that is very similar.

| Probability         	| Prediction	        					  | 
|:---------------------:|:-------------------------------------------:| 
| .67         			|  Speed limit (60km/h)   							      | 
| 0.29 				    |No passing for vehicles over 3.5 metric tons 		    	  |
| .05			        | Speed limit (80m/h)						  |



For the 4 image the model is some sure that this is a No entry sign (probability of 0.63), the image is Stop, you can see that  No Entry is red and have a white line in the middle that can confused with the letters of STOP.


| Probability         	| Prediction	        					  | 
|:---------------------:|:-------------------------------------------:| 
| .63         			|  No Entry   							      | 
| 0.37     				|  Stop 		    	  |


For the 5 image the model is very sure that this is aNo passing for vehicles over 3.5 metric tons  sign (probability of 0.91), the image is Speed limit (50m/h). The image proposed have a some of similitud but the second selected, Speed limit (80m/h) have more, I don´t understand because the neural network taked this decission.

| Probability         	| Prediction	        					  | 
|:---------------------:|:-------------------------------------------:| 
| .91         			|  No passing for vehicles over 3.5 metric tons   							      | 
| 0.9     				| Speed limit (80m/h) 		    	  |




