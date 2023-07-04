# Behavioral Cloning

---
[//]: # (Image References)

[image1]: ./img/Architecture.PNG "Model Visualization"
[image2]: ./img/video.gif "Simulator Output for the trained model"

## Summary

The objective of this project is to clone the behavior of driving using a deep neural network. The data for training is collected using the Udacity self-driving car [simulator](https://github.com/udacity/self-driving-car-sim), and the trained model's results are presented as a video output for better visualization.

Please note that for the training process, I have utilized [sample driving data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) available in the Udacity-GPU workspace.

The goals and steps of this project are as follows:
* Use the simulator to collect data of good driving behavior (Udacity also provides sample driving data)
* Build a convolutional neural network in Keras that predicts steering angles from images
* Train and validate the model using training and validation sets
* Test the model's ability to drive around track one without leaving the road
* Summarize the results in a written report


## Rubric Points
### In this section, I will address each of the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I have implemented them in my project.

---

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `train_final.py` - script to create and train the model
* `drive.py` - for driving the car in autonomous mode
* `model_final.h5` - trained convolutional neural network 
* `README.md` - summarizing the results
* `video.mp4` - simulator output video created using `video.py`

The model was entirely trained in the Udacity-GPU workspace. My local system was only used for editing purposes.

#### 2. Submission includes functional code
Using the provided Udacity simulator and the `drive.py` file, the car can be driven autonomously around the track by executing the following command:
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The `train_final.py` file contains the code for training and saving the convolutional neural network. It demonstrates the pipeline I used for training and validating the model, and it includes comments to explain how the code works.

Additionally, note that the `model_architecture.py` file contains only the Model Architecture.

### Model Architecture and Training Strategy

#### 1. Model architecture

I employed the model architecture described in the [End to End Learning for Self-Driving Cars](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) paper, as it is proven to be an ideal choice for this project. I implemented the architecture using Keras.

Here is a summary of the model architecture:

```plaintext
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
________________________________________________________________

_
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
dense_4 (Dense)              (None, 1)                  11        
=================================================================
Total params: 348,219
Trainable params: 348,219
Non-trainable params: 0
_________________________________________________________________
```

The model includes ReLU activation function to introduce nonlinearity, which is also a non-saturating activation function. The data is normalized before training using a Keras lambda layer to convert it to a range of -1 to 1.

#### 2. Reducing overfitting

To reduce overfitting, I have added nine dropout layers to the model, each with a different `keep_prob` value ranging from 0.1 to 0.5.

The model was trained and validated on different datasets to ensure that it was not overfitting. Testing was performed by running the model through the simulator and verifying that the vehicle stayed on track one.

However, it should be noted that the model's performance on track two was poor.

#### 3. Model parameter tuning

I used an Adam optimizer for training the model, so the learning rate was not manually tuned.

#### 4. Training data

I used the provided sample driving data for training.

#### 5. Final Model Architecture

The final model architecture is a convolutional neural network (CNN) developed by NVIDIA.

Here is a representation of the architecture used from the [paper](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf):

![alt text][image1]

To augment the data, I flipped the images and angles, assuming that this would contribute to the model's ability to generalize well.

I allocated 20% of the data for the validation set.

The model was trained for 2 epochs in total.

## Glimpse of the Output

![alt text][image2]

You can find the complete video output file (`video.mp4`) in this repository.

## Room for Improvement

The model did not perform well on track two. To improve its generalization ability, more data specific to track two could be collected and used for training. Additionally, increasing the augmentation of images may also help in achieving better performance on track two.