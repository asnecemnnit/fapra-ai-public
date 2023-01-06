

Folder structure:

data:
miscellaneous images used during development

dataset:
The dataset folder contains the preprocessed video data in form of csv files containing the ball positions at each frame of a video.


model:
This folder contains the code for the lstm model in base.py and lstm.py as well as the training code in train.py with its dataset implementation in dataset.py and the script that is executed for the actual predictions, run.py. The predictions are saved as a video in the videos subfolder.

preprocessing:
Contains the code to preprocess the video clips into csv files that can easily be fed into the model during training or inference.

rendering:
contains code to display ball positions in a frame and output a prediction video

utils:
several shared constants and utilities


