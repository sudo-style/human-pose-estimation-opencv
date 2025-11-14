# Gesture Recogntion

1. Install all dependencies

### Place your folder of videos into Images
should look like this below:

* Root Folder
  * target variable 1
    * video_1_target_variable1.mp4
    * video_2_target_variable1.mp4
    * ...
    * video_m_target_variable1.mp4
  * target variable 2
  * ...
  * target variable N

### Extracting keypoints
On Line 57 of extract_keypoints_yolo.py
set the folder path to your root folder

Run extract_keypoints_yolo.py -> this should output pose_data.jsonl

This process may take a while, depending on how many videos, and the number of frames in the video.
If you ever need to stop the extraction, you can hit control+c on mac to end the program (or whatever equlivant). 
It goes in batches of 50, it is not recommended to stop the program as it is writing, so only do so after it says that it is done with a batch. 
Rerunning the extraction will skip the videos you have already completed. 
Note the model skips videos that have multiple people to avoid having to guess which one is the target variable.

### Training the model
Run gestures_training.ipynb

this will split the data into a 70-15-15 split for training, validation, and testing sets respectively.
It will then train on the data, using a bi-lstm, and output how well the model did using accuracy, f1-scores, percision and recall scores. 
It will also show the confusion matrix, and loss/accuracy per epoch graphs.
With all of this data you will be informed how how well the model preformed on the data provided.