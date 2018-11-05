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

[image1]: ./examples/Nvidia_network_architecture.png "Model Visualization"
[image2]: ./examples/histogram_raw.png "Original histogram"
[image3]: ./examples/histogram_balanced.png "Balanced histogram"
[image4]: ./examples/pre-processed_img.png "Augmentation of Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model. The model_project-3.ipynb file contains the overall work flow  of this project. Major parts include loading the images, preprocessing the images(cropping and color channel conversion), data augmentation (flipping of images, combining 3 cameras center/left/right) and addressing the imbalance of training data.

* drive.py for driving the car in autonomous mode. I did not modify much on this file except playing with the speed parameter; and the final result still used 9 as the speed.

* model.h5 containing a trained convolution neural network
* writeup_report.md summarizing the results
* video.mp4 as the recorded video of my model running on the simulator

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model_project-3.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network based on the Nvidia autonomous driving model. The model takes images with RGB channels as input, and has 5 convolutional layers (3 using 5*5 convolutional filters and 2 using 3*3 convolutional filters) followed by 4 fully connected layers. Please refer to the python notebook file for further references. The model used "RELU" (rectified linear unit) as the activation function so as to introduce nonlinearity, and the data is normalized and re-centered in the model using a Keras lambda layer like:
```sh
model.add(Lambda(lambda X_train: (X_train - 128)/255.0, input_shape = (160, 320, 3)))
```

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting. During the tuning and testing of the network architecture, both dropout layers and max pooling layers are implemented after each convolution layer, and dropout layers are also inserted after the fully connected layers. In practice, it is found that the model appears to work fine without too much overfitting (but could suffer underfitting) if the run of training is limited to 3-6 epochs. Therefore I have decided to turn off the dropouts and MaxPooling for the time being, and just use the model's parameters after 5 epochs before the model started to suffer from overfitting.

The model was trained and validated on different data sets to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

#### 4. Appropriate training data

I started with the data from Udacity, and then later on also tried to collect data from my end especially regarding the recovery data. But apparently my own data does not help at all in terms of addressing the imbalanced training data(over concentrated at steer angle == 0 ). After all, my control of the car using keyboard is far from smooth.  So after about struggling on the collection of training data for almost 3 weeks, I have decided to return to the Udacity data and focused on data augmentation and tweak the distribution of input data.

For details about how I obtained an appropriate training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach
In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (80/20 ratio).

My first step was to use a one layer neural network model similar to the multi-variable linear regression. The idea is that steering angle should be a number between (-max_steerAngle, +max_SteerAngle), so the last layer right before the output could be like linear regression. My car did not survive long on the track ...

But this is pretty much expected. Evidently, the correlation between input image and output steering angle must be way more non-linear. So my next try is to use LeNet. I thought this model might be appropriate because it has proven to be able to handle image recognition such as the traffic sign classifier challenge. I just slightly modified the last fully connected layer's dimension so that the model will spit out a number rather than a classification index (and the input layer resize). But I found it questionable when LeNet needs to grayscale the input image from the beginning. It seems like losing too much information...
When running LeNet, I found that the loss function of the model did not keep decreasing within a few epochs. It looks like there is currently a high bias issue (underfitting), both training and validation loss function being high.


#### 2. Final Model Architecture
So I decided to use a more complexed network, with more convolution layers and also more fully connected layers. So I learned from the Nvidia autonomous driving's network according to their published paper, and turn off all the dropout and MaxPooling to see if this model is capable of overfitting. The final model architecture consisted of a convolution neural network with the following layers and layer sizes:

Here is a outline of the architecture (picture from the Nvidia paper)

![alt text][image1]

Note that Nvidia has an input image size of (66, 200), but mine is (160, 320). I decided to keep it as it is at the input, and use cropping within the Kera model. So I will end up having a slightly larger number of parameters compared to the original Nvidia model, but it is of the same order of magnitude (based on that my cropped image is about (85, 320)). My final model has 5 convolutional layers (the first 3 layers using 5x5 convolutional filters and the later 2 layers using 3x3 convolutional filters) followed by 4 fully connected layers with sizes decreasing from 100 to 50, 10 before yielding the output value.



#### 3. Preprocessing of the Training Set & Training Process

The most critical issue to address for this project, based on my experience, is the imbalanced data with more than 70% data points having steering angle == 0.0. With such a concentrated distribution, the model will naturally predict well on straight roads and lack accuracy on sharp turns.
Therefore preprocessing and re-arranging the training data is crucial. A straightforward way of reducing the imbalance is to use only a subset/fraction of all the zero steering angle data points. After tuning the subset size, I ended up using 25% of all zero steering points via the mod (%) operator together with all the nonzero data points:
'''
  if float(sample[3]) != 0.0 or (i%4 == 0):
'''

And in the following are the histogram of original (left) and the subset (right) training data.

![alt text][image2]
![alt text][image3]

But more non-zero steering angle training data points are still needed. Since the simulator recorded center cameras and left/right side cameras at each timestamp, I also included the two side cameras, and apply an adjustment of steering angle (angle_adjust = 0.25). Note that this inclusion itself can help but cannot fully address the problem of imbalanced data.

To augment the data set, I also flipped images and angles, thinking that this would make my training data more evenly distributed about turning left and turning right. So the car will not tend to always steer to the left all along. In addition to flipping horizontally, I also tried to shifted the image horizontally so that the lane lines after shifting will be similar to what the side cameras observed (as is shown below) and can use similar adjustment steering values to offset. Note that the shift of horizontal images is to address the imbalanced data at zero steering and thus only need to be applied to zero steering angle data point. It roughly aims to mimic the recovery data (from off the center to at the center). Here is an image and the augmentation result (including flipping, angle cameras and their flipping, and the horizontally shifted images):

![alt text][image4]

Another problem is about color channels. imread() from cv2 uses BGR, while the later part include im.show() and simulator's intake of images uses RGB. It is found that without the alignment of color channel, a model can not work even the loss function drops to ~ 0.001. Initially I was thinking that the car will be most sensitive to line shapes in order to identify the lanes, but maybe the reality is more complex and thus color also matters (and grayscale may not work easily).

I also recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to steer itself back. But Even this recovery will have majority of data point at 0. I think this is because I used a keyboard and the control is not smooth. So I decided to not use the recovery data at present.

After the data pre-processing and data augmentation (mostly done in the generator), I had 22900 data points.
After feeding the data into the model, each image will be normalized and re-centered around 0 and also have top portion (mostly just the sky, trees, birds ...) and bottom portion (car hood) chopped out.

I finally randomly shuffled the data set and put 20% of the data into a validation set.
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 3-6 as evidenced by the plot of training loss and validation loss versus training epochs. I used an adam optimizer so that manually training the learning rate wasn't necessary.
