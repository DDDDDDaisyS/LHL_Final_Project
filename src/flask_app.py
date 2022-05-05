"""

"""

from flask import Flask, render_template, Response, send_from_directory   
import cv2
from tensorflow.keras.models import load_model
import numpy as np
import mediapipe as mp
from modules.feed_connections import detect_video, draw_landmarks
from modules.keypoints import pose_hand_keypoints
import os 

app=Flask(__name__)

cap = cv2.VideoCapture(0) # , cv2.CAP_DSHOW
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

# load trained model
model = load_model('trained_model_10')
# remain same order as in augmentation code
signs = np.array(['busy', 'finish', 'Hello', 'How', 'love', 'nothing', 'sign', 'Take care', 'Thank you', 'you'])

threshold = 0.999 # predicted probability threshold
mp_pose = mp.solutions.pose # pose model
mp_hands = mp.solutions.hands # Hand model

# Set mediapipe model 
pose = mp_pose.Pose(min_detection_confidence=0.5,
                    min_tracking_confidence=0.5)
hands = mp_hands.Hands(model_complexity=0,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)

def gen_frames(): 
    # for write in new detection variables
    frames = []
    sentence = []
    while True:
        # Read feed
        success, frame = cap.read()
        if not success:
            break
        else:
            # detect camera feed
            image, res_pose = detect_video(frame, pose)
            image, res_hands = detect_video(image, hands)

            # Draw landmarks
            draw_landmarks(image, res_pose, res_hands)

            # extract keypoints
            keypoints = pose_hand_keypoints(res_pose, res_hands)
            frames.append(keypoints)
            frames = frames[-30:] # extraxt keypoints in the last 40 frames (input size)

            if len(frames) == 30:
                # predict
                res = model.predict(np.expand_dims(frames, axis=0))[0]

                # print prediction only when there's movement and prob is over threshold
                if (res[np.argmax(res)] > threshold) and (signs[np.argmax(res)] != 'nothing'):
                    if len(sentence) > 0: 
                        if signs[np.argmax(res)] != sentence[-1]:
                            sentence.append(signs[np.argmax(res)])
                    else:
                        sentence.append(signs[np.argmax(res)])
                    # whenever there's a successful prediction, restart collecting frames
                    frames=[]

                # display no more than 5 predictions on screen
                if len(sentence) > 5: 
                    sentence = sentence[-5:]
            
            # print predictions
            cv2.rectangle(image, (0,0), (640, 40), (128, 128, 128), -1)
            cv2.putText(image, ' '.join(sentence), (10,25), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', image)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/favicon.ico') 
def favicon(): 
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

if __name__=='__main__':
    app.run(debug=True)