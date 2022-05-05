from tensorflow.keras.models import load_model
import cv2
import numpy as np
import mediapipe as mp
from modules.feed_connections import detect_video, draw_landmarks
from modules.keypoints import pose_hand_keypoints
from pickle import load

# load model
model = load_model('trained_model_10')
# remain same order as in augmentation code
signs = np.array(['busy', 'finish', 'Hello', 'How', 'love', 'nothing', 'sign', 'Take care', 'Thank you', 'you'])

# for write in new detection variables
frames = [] # keypoints in each frame
sentence = [] # predited signs
threshold = 0.99 # predicted probability threshold

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

mp_pose = mp.solutions.pose # pose model
mp_hands = mp.solutions.hands # Hand model

# Set mediapipe model 
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
hands = mp_hands.Hands(model_complexity=0,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

while cap.isOpened():
    # Read feed
    ret, frame = cap.read()

    # detect camera feed
    image, res_pose = detect_video(frame, pose)
    image, res_hands = detect_video(image, hands)

    # Draw landmarks
    draw_landmarks(image, res_pose, res_hands)

    # 2. Prediction logic
    keypoints = pose_hand_keypoints(res_pose, res_hands)
    frames.append(keypoints)
    frames = frames[-30:] # extraxt keypoints in the last 30 frames (model input size)

    if len(frames) == 30:
        res = model.predict(np.expand_dims(frames, axis=0))[0]

    #3. Viz logic
        if (res[np.argmax(res)] > threshold) and (signs[np.argmax(res)] != 'nothing'):
            if len(sentence) > 0: 
                if signs[np.argmax(res)] != sentence[-1]:
                    sentence.append(signs[np.argmax(res)])
            else:
                sentence.append(signs[np.argmax(res)])
            
            frames=[]

        if len(sentence) > 5: 
            sentence = sentence[-5:]

    cv2.rectangle(image, (0,0), (640, 40), (128, 128, 128), -1)
    cv2.putText(image, ' '.join(sentence), (10,25), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show to screen
    cv2.imshow('OpenCV Feed', image)

    # Break gracefully
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()