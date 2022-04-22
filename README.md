# LHL_Final_Project - Sign Language Interpreter

## Dependencies
Tensorflow, Keras, OpenCV, Mediapipe

## Data Resourse
* Videos were self-recorded, with 50 mini-videos for each sign with 30 frames in each video at different angles and distance. 
* While recording frames, pose and hand keypoints were also extracted by using [Google Mediapipe Hands and Pose models](https://google.github.io/mediapipe/solutions/hands.html).
* Keypoints are composed of x, y and z, x and y are normalized to [0.0, 1.0] by the image width and height respectively. z represents the landmark depth, and the smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x. The stored keypoints data is [here](data/keypoints), and the collecting code is [here](src/collect_keypoints_frames.py). 
* There are some other options, such as to collect frame images only (code is [here](collect_frames.py)) and then load images to extract keypoints; or collecting keypoints with another model, Google Mediapipe Holistic, which is capable of extracting keypoints of face, pose and hands (code is [here](collect_keypoints_holistic.py)). However, the z value in holistic hands module has not been well-trained yet according to their page.

## Data Augmentation
* Dataset size was boosted by adding mirror flipped, translated and scaled data. 
* The augmented data is too large to upload. Simply run [this code](src/augment_keypoints_pose_hands.py) to get your own. 

## Model
Bidirectional LSTM was the core for building this model. The categorical accuracy of validation set achieved 100%.
- [The jupyter notebook](src/train_model.ipynb)
- [The trained model](src/trained_model)

## Live Capture and ASL Recogonization

