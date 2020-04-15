# **Behavioral Cloning** 

## Writeup Template

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./images_out/center.jpg "Center Image"
[image2]: ./images_out/left.jpg "Left Image"
[image3]: ./images_out/right.jpg "Right Image"
[image4]: ./images_out/estudio.PNG "Model Mean Squared Error Loss Epcoh = 10"
[image5]: ./images_out/estudio2_2.PNG "Model Mean Squared Error Loss Epcoh = 5"
[video1]: ./video.mp4 "Video with the final result"
[video2]: ./video_speed30.mp4 "Video with the final result and speed = 30 mph"


## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* video.mp4 video with the results.

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model is based in the NVIDIA network mode:
Input 3@ -> Normalized 3@ -> Convolutional 24@ -> Dropout(0.2) -> Conv 36@ -> Dropout(0.2) -> Conv 48@ -> Dropout(0.2) -> Conv 64@ -> Dropout(0.2) ->
-> Conv 64@ -> Dropout(0.2) -> Faltten -> Dense 100 -> Dense 50 -> Dense 10 -> Dense 1

The code is de above:

dropout_p = .2
# setup using NVIDIA architecture
model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Dropout(dropout_p))
model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Dropout(dropout_p))
model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
model.add(Dropout(dropout_p))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(dropout_p))
model.add(Convolution2D(64, 3, 3, activation="relu"))
model.add(Dropout(dropout_p))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))



#### 2. Attempts to reduce overfitting in the model

The architechture selected not include a droput layer, I added this type of layer to my architecture (lines 86,88,90,92,94).

The model was trained and validated on a data set plus data modificated to ensure that the model was not overfitting (code lines 23-51).
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 102).
model.compile(loss='mse', optimizer='adam')

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

I read some documentattion about the archictechtures for this type of scenarios.
I found NVIDIA architechture in wich I see that have good results and is easy to understand.

I tryed some times to introduce in the system, compile etc. Until it worked relative fine.

For improve the result I cropped the images for extract the intesrest areas in the pictures and tried the stack with differents parameters, for example the number of epoch.

In the first test I used the udacity example data, and I had I enough good results.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. [video1]

#### 2. Final Model Architecture

The final model architecture (model.py lines 85-99) consisted of a convolution neural network with the layers and layer size sumaryzed in the firts point.


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one clock side and other contra clock side. Here is an example image of center lane driving:

![][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center.:

![][image2]
![][image3]


Then I repeated this process on track two in order to get more data points.

Also I used the udacity examples images for get more and more data.

To augment the data sat, I also flipping the images to enrich data training.

After the collection process, I had 16072 number of data points. I then preprocessed this data by cropping (70,25) pixels.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 
"Train on 12857 samples, validate on 3215 samples"

I used this training data for training the model. The validation set helped determine if the model was over or under fitting.
I studyed the results, first with 10 epechos ![][image4] and selected that the best configuration is with 5 epecohs: ![][image5]

I used an adam optimizer so that manually training the learning rate wasn't necessary.

You can see in the [video2] that the system with the model trining can drive at 30 mph withou problems.
