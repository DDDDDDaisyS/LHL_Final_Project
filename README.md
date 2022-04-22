# LHL_Final_Project - Sign Language Interpreter
This repo contains data and codes of my capstone project at [Lighthousr Labs data science bootcamp](https://www.lighthouselabs.ca/).
Some ideas were inspired by [Nicholas Renotte](https://github.com/nicknochnack/ActionDetectionforSignLanguage).

## Dependencies
Tensorflow, Keras, OpenCV, Mediapipe

## Data Resourse
- **Chosen signs**: 'busy', 'cat', 'finish', 'Hello', 'How', 'love', 'nothing', 'sign', 'Thank you', 'you'. Nothing refers to 'no movements'.
- Videos were self-recorded, including 50 mini-videos with 30 frames for each sign at different angles and distance. 
   - While recording frames, pose and hand keypoints were also extracted by using [Google Mediapipe Hands and Pose models](https://google.github.io/mediapipe/solutions/hands.html).
   - Keypoints are composed of x, y and z, x and y are normalized to [0.0, 1.0] by the image width and height respectively. z represents the landmark depth, and the smaller the value the closer the landmark is to the camera. The magnitude of z uses roughly the same scale as x. The stored keypoints data is [here](data/keypoints), and the collecting code is [here](src/collect_keypoints_frames.py). 
   - There are some other options, such as to collect frame images only (code is [here](collect_frames.py)) and then load images to extract keypoints; or collecting keypoints with another model, Google Mediapipe Holistic, which is capable of extracting keypoints of face, pose and hands (code is [here](collect_keypoints_holistic.py)). However, the z value in holistic hands module has not been well-trained yet according to their page.

## Data Augmentation
- Dataset size was boosted by adding mirror flipped, translated and scaled data. 
- The augmented data is too large to upload. Simply run [this code](src/augment_keypoints_pose_hands.py) to get your own. 

## Model
Bidirectional LSTM was the core for building this model. The categorical accuracy of validation set achieved 100%.
- [The jupyter notebook](src/train_model.ipynb)
- [The trained model](src/trained_model)

## Live Capture and ASL Recogonization
### Python file
Real-time interpreting can be realized through running [this code](interpreter.py).

### Flask App Deploy
A flask app was also created. Run [this code](flask_app.py), copy and paste the address to the browser to realze real-time recognition. 
