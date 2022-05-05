"""
record 50 mini videos for each sign and extract frames
"""
import cv2
import numpy as np
import os


# Base path to store keypoints coordinates/frame images
base_path = os.path.join('Data', 'frames') 
       
# signs to detect
signs = np.array(['How', 'love', 'No', 'Take care', 'Thank you', 'what', 'Yes', 'you']) # change, decrease/increase as needed

# number of videos for one sign, number of frames in one video
n_videos, n_frames = 50, 25

# make empty folders to store data
for sign in signs: 
    for video_i in range(n_videos):
        try: 
            os.makedirs(os.path.join(base_path, sign, str(video_i)))
        except:
            pass

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # capture video feed from camera '0', webcam on my machine
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)    # set up resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)

# Loop through signs
for sign in signs:
    # Loop through videos in each sign
    for video_i in range(n_videos):
        # Loop through video frames in each video
        for frame_i in range(n_frames):

            # Read feed
            ret, frame = cap.read() # ret is a boolean variable that returns true if the frame is available.
                                    # frame is an image array vector captured

            # add break in between recordings
            if frame_i == 0: # beginning of each video
                cv2.putText(image, 'Collecting frames for {} Video Number {}'.format(sign, video_i), (15,12), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
                # Show to screen
                cv2.imshow('OpenCV Feed', image)
                
                # export keypoints
                keypoints = pose_hand_keypoints(res_pose, res_hands)
                keypoints_path = os.path.join(base_path_keypoints, sign, str(start_video + video_i), str(frame_i))
                frames_path = os.path.join(base_path_frames, sign, str(start_video + video_i), f'{frame_i}.jpg')
                
                np.save(keypoints_path, keypoints)
                cv2.imwrite(frames_path, frame)

            # Break out live video with key 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # add break before record next sign
    if video_i == 49:
        cv2.waitKey(10000) # break for 10 sec

cap.release()
cv2.destroyAllWindows()