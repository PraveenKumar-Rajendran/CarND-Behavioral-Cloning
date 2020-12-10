# **Behavioral Cloning** 

<!-- ### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.
-->
---

**summing-up**

The goal of this project is to clone the behaviour of driving with the use of deep neural network. With the use of Udacity self driving car [simulator](https://github.com/udacity/self-driving-car-sim) we can collect the data for training. Later the same is used for obtaining the results of the deep learning model which is then created as a video output for the better presentation.

Note that I have used [sample driving data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) available at udacity-gpu workspace for the training process.

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior ( sample driving data is also provided by udacity )
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./img/Architecture.png "Model Visualization"
[image2]: ./img/video.gif "Simulator Output for the trained model"
<!-- [image3]: ./examples/placeholder_small.png "Recovery Image"
[image4]: ./examples/placeholder_small.png "Recovery Image"
[image5]: ./examples/placeholder_small.png "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image" -->

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. Following model architecture has been employed
<pre>
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 70, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 33, 158, 24)       1824      
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 15, 77, 36)        21636     
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 6, 37, 48)         43248     
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 4, 35, 64)         27712     
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 2, 33, 64)         36928     
_________________________________________________________________
flatten_1 (Flatten)          (None, 4224)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               422500    
_________________________________________________________________
dropout_1 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dropout_2 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 559,419
Trainable params: 559,419
Non-trainable params: 0
_________________________________________________________________
</pre>

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 60-76) 

The model includes ELU activation function to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 62).

#### 2. Attempts to reduce overfitting in the model

The model contains `three` dropout layers in order to reduce overfitting (model.py lines 70, 72, 74).

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 78).

#### 4. Training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. Model is then experimented with different architecture so that the training loss and validation losss are in declining trend.

The final step was to run the simulator to see how well the car was driving around track one.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 61-76) consisted of a convolution neural network with before mentioned layers and layer sizes.

Here is a better visualization of the architecture created using [NN-SVG](http://alexlenail.me/NN-SVG/)

![alt text][image1]

To augment the data sat, I also flipped images and angles thinking that this would contribute for the model to generalize well.


I finally randomly shuffled the data set and put 2% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The optimum number of epochs was 5 as per the experimentation. I used an adam optimizer so that manually training the learning rate wasn't necessary.

## Glimpse of the output

![alt text][image2]
