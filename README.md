# LHL_Final_Project - Sign Language Interpreter

## Dependencies
Tensorflow, Keras, OpenCV, Mediapipe

## Data Resource
* Video data was self-recorded, with 100 mini-videos for each sign with 40 frames in each video. In this case, stored [data](data/recorded_keypoints) are real-world coordinates of pose and hand keypoints in meters extracted by using [Google Mediapipe Hands and Pose models](https://google.github.io/mediapipe/solutions/hands.html), collecting code is [here](src/collect_keypoints_pose_hands.py).
* There are some other options, such as collecting frame images (code is [here](collect_frames.py)) and loading images to extract keypoints, or collecting keypoints with another model, Google Mediapipe Holistic, which is capable of extracting face, pose and hands (code is [here](collect_keypoints_holistic.py)).

## Data Augmentation
Dataset size was boosted by adding mirror flipped and rotated data (both operated along z axis). [Preprocessed data](data/preprocessed_data)

## Model

## Live Capture and ASL Recogonization
