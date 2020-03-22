# Behavioral Cloning Project

## Keywords: Behavioral Cloning, Keras, Deep learning, autonomous

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

Overview
---
This repository contains starting files for the Behavioral Cloning Project.

### Files Submitted & Code Quality
- `model.py`
- `model.h5`
- `video.mp4`
- `README.md`
- `drive.py`

### Model Architecture and Training Strategy

#### 1. Model architecture

The model consists of a convolution neural network with 3x3 and 5x5 filter sizes and depths between 24 and 64 (model.py lines 91-117)
The model includes RELU layers to introduce nonlinearity and the data is normalized in the model using a Keras lambda layer.

```python

def model(loss='mse', optimizer='adam'):

    model = Sequential()
	
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))

    model.add(Conv2D(filters=24, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=36, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=48, kernel_size=5, strides=(2, 2), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))
    model.add(Conv2D(filters=64, kernel_size=3, strides=(1, 1), activation='relu'))

    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

    return model

```
#### 2. Attempts to reduce overfitting in the model
To prevent overfitting, several data augmentation techniques like flipping images horizontally
as well as using left and right images to help the model generalize were used.
The model was trained and validated on different data sets to ensure that the model was not overfitting.
The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

Another technique to reduce overfitting was to introduce dropout in the network, with a dropout rate of 0.5.

#### 3. Model parameter tuning

The model used an Adam optimizer, and various learning rates (LR) were iterated over to arrive at the final LR.
##### Hyperparameters
```python
batch_size = 32
number_of_epochs = 5
learning_rate = 0.0001
```
#### 4. Appropriate training data
Udacity sample data was used for training.  

To gauge how well the model was working, I split my image and steering angle data into a
training (80%) and validation (20%) set. A validation loss was calculated as the mean absolute
error between the true and predicted steering angles for the validation set and this was used to
monitor progress of training.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a good model was to use the [Nvidia](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf) architecture since it has been proven to be very successful
in self-driving car tasks. I would say that this was the easiest part since a lot of other students
have found it successful, the architecture was recommended in the lessons and it's adapted for this use case.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.
Since I was using data augmentation techniques, the mean squared error was low both on the training and validation steps.

The hardest part was getting the data augmentation to work. I had a working solution early on but because
there were a lot of syntax errors and minor omissions, it took a while to piece everything together. One of the
problems, for example, was that I was incorrectly applying the data augmentation techniques and until I created
a visualization of what the augmented images looked like, I was not able to train a model that drives successfully
around the track.

#### 2. Final Model Architecture
The final model architecture code is shown above.
Here is a visualization of the architecture:

![Net](https://devblogs.nvidia.com/parallelforall/wp-content/uploads/2016/08/cnn-architecture-624x890.png)

#### 3. Creation of the Training Set & Training Process
To create the training data, I used the Udacity sample data as a base. For each image, normalization
would be applied before the image was fed into the network. The training sample consisted
of three images:
1. Center camera image
2. Left camera image
3. Right camera image

I found that this was sufficient to meet the project requirements for the first track. To make the car
run on the same track, additional data augmenation techniques like adding random brightness, shearing and horizontal
shifting could be applied.

Instead of storing the preprocessed data in memory all at once, using a generator pieces of the data were pulled and processed on the fly as and when needed, which was much more memory-efficient.

The network was then trained for 5 epochs.

The model was then tested on the first track to ensure that the model was performing as expected.

#### Challenges and areas of improvement
The following challenges were faced and overcome during this project:
1. Network predicting constant steering angles:
During the initial phase of training, the network would produce a constant prediction for steering
angle regardless of the input. This was found out to be due to an unbalanced dataset with more
entries for zero steering angle than others.
Resizing the images.
Train the model on second track so that car can navigate on both tracks.
Play around with more data augumentation techniques to increase the accuracy.

The current code can steer the car such that it can drive autonomously over the track. However, there are
several improvements that can be made:
1. Keep the car from going over either lane markings. 
This may be accomplished by more recovery data and/or a larger network
2. Teaching the vehicle to steer and accelerate/brake autonomously. 
Since there are no obstacles/traffic on this test track, it can be used to investigate how the throttle may also be
autonomously controlled.
