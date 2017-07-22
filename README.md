# Vehicle Detection

*Self-driving Car Nanodegree at Udacity.*

------------


**Goals:**

- Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a - classifier Linear SVM classifier
- Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your - HOG feature vector.
- Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
- Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
- Estimate a bounding box for vehicles detected.
------------

**Tools:**

- [CarND-Vehicle-Detection](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md)
- [CarND-Term1-Starter-Kit](https://github.com/udacity/CarND-Term1-Starter-Kit "CarND-Term1-Starter-Kit")
- [Conda](https://conda.io/docs/using/envs.htmlhttp:// "Conda")
- [Python](https://www.python.org "Python")

The Solution.ipybn file contains the code implementation to run this project. Apart from the dependencies listed in the notebook cell # 1, this project also uses a dataset of [vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicle](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) and a [video](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/project_video.mp4) in the root folder. **Note** These image/video assets are note included in the repository.

There is also a Playground.ipynb file used for testing and debuging.

Bellow are the steps to achieve this [result](https://www.youtube.com/watch?v=Aej0myN8Ksc)

### 1. Feature extraction from training images.

To extract features of vehicle and non-vehicle I used the KITTI vision benchmark suite and GTI vehicle image database. The dataset contains a good amount to images (8792 for vehicles and 8968 for non-vehicles) that can train a model to achieve good detection results. 

I used `skimage.feature.hog` function with the following parameters to extract features from all the RGB channels:

| Settings        | Value  |        
| ------------- |:-------------:
| color_space     | RGB | 
| orient    | 9 | 
| pix_per_cell     | 8 | 
| cell_per_block     | 2 | 

After some experiementation these parameters were chosen based on performace and accuracy while training the Linear SVC model.

I also used color histogram and spatial binning features and stacked them together with HOG features in the featured extraction process. Bellow are the settings I used for those:


| Settings        | Value  |        
| ------------- |:-------------:
| Spatial size     | (32, 32) | 
| Histogram bins     | 32 | 
| Histogram range     | (0, 256) | 

The code for feature extraction can be found in cell number 7.

###### HOG image example
![HOG image example](https://github.com/ismalakazel/vehicle-detection/blob/master/output_images/HOG.png)

###### Color histogram example
![Color histogram](https://github.com/ismalakazel/vehicle-detection/blob/master/output_images/color_histogram.png)

###### Spatial binning example
![Spatial binning](https://github.com/ismalakazel/vehicle-detection/blob/master/output_images/spatial_binning.png)

### 2. Linear SVC Classification

To detect predict car features in an image I used sklearn.svm.LinearSVC and trained a model with the features extracted in the process described above. The images were initially split in two **cars** and **notcars** lists of 1000 samples and shuffled to garanted randomness and generalization, then features were stacked and normalized using `sklearn.preprocessing.StandardScaler`. Before training the classifier I made a second split, this time to separate training and testing data, with sklearn.model_selection.train_test_split using test size of `0.2` and random state of `(0, 100)`. The model was trained in a MacBook Pro, achieving a test accuracy of **0.9825**.

The code for training the Linear SVC can be found in cell number 10.

### 2. Sliding Window

For the sliding window process I used the Hog Sub-sampling Window Search approach to extract features only once in a region of interest with a search window overlap of 75%. This approach proved more performance in comparison to extracting features from each window at a time. 

After experimenting a window search of multiple scales I settled with a single scale of 1.5, which proved to be fast and reliable in predicting cars features in the whole test video. I also restricted the feature extraction to a region of interest, which basically included all pixels in the x axis and pixels y axis range of 400 - 656. This ROI excluded unnecessary features such as the sky of treess

For feature extraction I used the same parameters in the pre-training feature extraction process.

Bellow are the settings used:

| Settings        | Value  |        
| ------------- |:-------------:
| color_space    | 'RGB' | 
| orient    | 9 | 
| pix_per_cell     | 8 | 
| pix_per_cell     | 2 | 
| hog_channel     | 'ALL' | 
| spatial_size    | (32, 32) | 
| hist_bins     | 32 | 
| hist_range     | (0, 256) | 
| hist_range     | (0, 256) | 
| ystart     | 400 | 
| ystop     | 656 | 
| scale     | 1.5 | 

The code for training the Linear SVC can be found in cell number 11.

###### Sliding window example
![Spatial binning](https://github.com/ismalakazel/vehicle-detection/blob/master/output_images/sliding_window.png)

### 3.False Positives

After performing the sliding window algorithm I combined the overlapping window boxes and added a hitmap in order to elimated features that weren't of cars, also refered as false positives. Every pixels of the detect window was added to an array, then a threshold was applied to eliminate the false positives. Finally, I used the `scipy.ndimage.measurements.label` function to build a bounding box around the heat map.

###### False positives image example
![Spatial binning](https://github.com/ismalakazel/vehicle-detection/blob/master/output_images/labeled_boxes.png)

### 4. Video Implementation and Observations

The result of this project can be seen (here)[https://www.youtube.com/watch?v=Aej0myN8Ksc]

For the purposes of this project I believe the pipeline perform well. Some optimizations, such as removing duplicate images in the dataset and augmenting the existing dataset could be made for a more generalized training data. Algorithm optimization for feature extraction and false positives elimination is an area of my interest which I want to further explore in the future. 
