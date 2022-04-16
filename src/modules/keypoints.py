import cv2
import mediapipe as mp
import numpy as np


# extract keypoints coordinates from image
def extract_keypoints(results):
    # 468 keypoints in facemesh
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    # select keypoints 11-23 (shoulders and arms) in pose
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark[11:23]]).flatten() if results.pose_landmarks else np.zeros(12*4)
    # 21 keypoints in left and right hands each
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([face, pose, lh, rh])