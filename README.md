# LHL_Final_Project - Sign Language Interpreter
This repo contains data and codes of my capstone project at [Lighthousr Labs](https://www.lighthouselabs.ca/) data science bootcamp. <br>
Real-time American Sign Language (ASL) recognition using live video feed from webcam. Some ideas were inspired by [Nicholas Renotte](https://github.com/nicknochnack/ActionDetectionforSignLanguage).

## Demo
![alt text][logo]

[logo]: https://github.com/DDDDDDaisyS/LHL_Final_Project/blob/main/demo.gif "Logo Title Text 2"

## Dependencies
Tensorflow, Keras, OpenCV, Mediapipe

## Data Resourse
- **Chosen signs (10 classes)**: 'busy', 'finish', 'Hello', 'How', 'love', 'nothing', 'sign', 'Take Care', 'Thank you', 'you'. 'Nothing' is not an actual sign, just refers to 'no movements'.
- Videos were self-recorded, including 50 mini-videos with 30 frames for each sign at different angles and distance. 
   - While recording frames, pose and hand keypoints were also extracted by using [Google Mediapipe Hands and Pose models](https://google.github.io/mediapipe/solutions/hands.html).
   - Keypoints are composed of x, y and z, x and y are normalized to [0.0, 1.0] by the image width and height respectively. z represents the landmark depth, and the smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x. 
   - There are some other options, such as to collect frame images only (code is [here](collect_frames.py)) and then load images to extract keypoints; or extracting keypoints with another model, Google Mediapipe Holistic, which is capable of extracting landmarks of face, pose and hands (code is [here](collect_keypoints_holistic.py)). However, the z value in holistic hands module has not been well-trained yet according to their page.
- The collecting code is [here](src/collect_keypoints_frames.py). 
- The stored keypoints data is [here](data/keypoints).

## Data Augmentation
- Dataset size was boosted by adding mirror flipped, translated and scaled data. 
- The augmented data is too large to upload. Simply run [this code](src/augment_keypoints_pose_hands.py) to get your own. 

## Model
Bidirectional LSTM was the core for building this model. The categorical accuracy of validation set achieved 100%.
- [The jupyter notebook](src/train_model.ipynb)
- [The trained model](src/trained_model)

## Live Capture and ASL Recognization
#### Python File
Real-time interpreting can be realized by running [this code](interpreter.py).

#### Flask App Deployment
A flask app was also created. Run [this code](flask_app.py), copy and paste the address to the browser to realize real-time sign language recognition. 
