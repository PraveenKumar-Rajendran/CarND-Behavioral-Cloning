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

[image1]: ./img/Architecture.PNG "Model Visualization"
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
* `train_final.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model_final.h5` containing a trained convolution neural network 
* `README.md` summarizing the results
* `video.mp4` simulator output created using `video.py` was also included

Model was completely trained in Udacity-GPU workspace. Local system only used for editing works.

#### 2. Submission includes functional code
Using the Udacity provided simulator and drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The train_final.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

It can be also noted that `model_architecture.py` contains the Model Architecture alone.

### Model Architecture and Training Strategy

#### 1. Following model architecture has been employed as [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) paper is proven to be a ideal choice for the project, same is implemented with the help of keras.
<pre>
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
lambda_1 (Lambda)            (None, 160, 320, 3)       0         
_________________________________________________________________
cropping2d_1 (Cropping2D)    (None, 65, 320, 3)        0         
_________________________________________________________________
dropout_1 (Dropout)          (None, 65, 320, 3)        0         
_________________________________________________________________
conv2d_1 (Conv2D)            (None, 31, 158, 24)       1824      
_________________________________________________________________
dropout_2 (Dropout)          (None, 31, 158, 24)       0         
_________________________________________________________________
conv2d_2 (Conv2D)            (None, 14, 77, 36)        21636     
_________________________________________________________________
dropout_3 (Dropout)          (None, 14, 77, 36)        0         
_________________________________________________________________
conv2d_3 (Conv2D)            (None, 5, 37, 48)         43248     
_________________________________________________________________
dropout_4 (Dropout)          (None, 5, 37, 48)         0         
_________________________________________________________________
conv2d_4 (Conv2D)            (None, 3, 35, 64)         27712     
_________________________________________________________________
dropout_5 (Dropout)          (None, 3, 35, 64)         0         
_________________________________________________________________
conv2d_5 (Conv2D)            (None, 1, 33, 64)         36928     
_________________________________________________________________
dropout_6 (Dropout)          (None, 1, 33, 64)         0         
_________________________________________________________________
flatten_1 (Flatten)          (None, 2112)              0         
_________________________________________________________________
dense_1 (Dense)              (None, 100)               211300    
_________________________________________________________________
dropout_7 (Dropout)          (None, 100)               0         
_________________________________________________________________
dense_2 (Dense)              (None, 50)                5050      
_________________________________________________________________
dropout_8 (Dropout)          (None, 50)                0         
_________________________________________________________________
dense_3 (Dense)              (None, 10)                510       
_________________________________________________________________
dropout_9 (Dropout)          (None, 10)                0         
_________________________________________________________________
dense_4 (Dense)              (None, 1)                 11        
=================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
_________________________________________________________________
</pre>

My model consists of a convolution neural network with 3x3 filter sizes and depths between 32 and 128 (model.py lines 60-76) 

The model includes `ReLU` activation function to introduce nonlinearity it is also a non-saturating activation.
Data is normalized in the before training using a Keras lambda layer between the `(converted to range of -1 to 1)`

#### 2. Attempts to reduce overfitting in the model

The model contains `Nine` dropout layers in order to reduce overfitting with different `keep_prob = [0.1, 0.2, 0.3, 0.4, 0.5]`

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the the first track.

However It is also noted that it's performance is poor in the second track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Training data

sample driving data is used.

#### 2. Final Model Architecture

The final model architecture is a convolutional neural network (CNN) developed by NVIDIA.

Here is the representation of the architecture utilized from the [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)

![alt text][image1]

augmenting data involves flipping images and angles thinking that this would contribute for the model to generalize well.


20% of the data utilized for a validation set. 

Totally model was trained for `2` epochs.

## Glimpse of the output

![alt text][image2]

You can also find the entire video output file (`video.mp4`) in this repository.

## Room for improvement

The model was not able to perform better in the second track. So for generalizing well, data may be collected for the second track extensively and trained again with more augmented image for generalizing well for the second track.