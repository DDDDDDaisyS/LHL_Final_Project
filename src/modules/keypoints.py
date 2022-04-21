"""
extract keypoints coordinates from Mediapipe models
There are different models and attributes, second one was chosen in this project eventually
"""

import cv2
import mediapipe as mp
import numpy as np


# extract keypoints coordinates from image using holistic model
def holistic_keypoints(results):
    # 468 keypoints in facemesh
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468*3)
    # select keypoints 11-23 (shoulders and arms) in pose
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark[11:23]]).flatten() if results.pose_landmarks else np.zeros(12*4)
    # 21 keypoints in left and right hands each
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)

    return np.concatenate([pose, rh, lh])


# extract hands keypoints coordinates only from image using hand model
def pose_hand_keypoints(res_pose, res_hands):
    # select keypoints 11-16 (shoulders and arms) in pose
    pose = np.array([[res.x, res.y, res.z] for res in res_pose.pose_landmarks.landmark[11:17]]).flatten() if res_pose.pose_landmarks else np.zeros(6*3)
    
    # 42 keypoints in two hands
    if res_hands.multi_hand_landmarks:
        if len(res_hands.multi_hand_landmarks)==1:
            rh = np.array([[res.x, res.y, res.z] for res in res_hands.multi_hand_landmarks[0].landmark]).flatten() 
            lh = np.zeros(21*3)
        else:
            rh = np.array([[res.x, res.y, res.z] for res in res_hands.multi_hand_landmarks[1].landmark]).flatten() # when both hands are detected, index 1 is right hand
            lh = np.array([[res.x, res.y, res.z] for res in res_hands.multi_hand_landmarks[0].landmark]).flatten() # 0 is left hand
    else:
        rh = np.zeros(21*3)
        lh = np.zeros(21*3)
    
    return np.concatenate([pose, rh, lh])


# extract real-world hands keypoints coordinates only from image using hand model
def pose_hand_world_keypoints(res_pose, res_hands):
    # select keypoints 11-16 (shoulders and arms) in pose
    pose = np.array([[res.x, res.y, res.z] for res in res_pose.pose_world_landmarks.landmark[11:17]]).flatten() if res_pose.pose_world_landmarks else np.zeros(6*3)
    
    # 42 keypoints in two hands
    if res_hands.multi_hand_world_landmarks:
        if len(res_hands.multi_hand_world_landmarks)==1:
            rh = np.array([[res.x, res.y, res.z] for res in res_hands.multi_hand_world_landmarks[0].landmark]).flatten() 
            lh = np.zeros(21*3)
        else:
            rh = np.array([[res.x, res.y, res.z] for res in res_hands.multi_hand_world_landmarks[1].landmark]).flatten() # when both hands are detected, index 1 is right hand
            lh = np.array([[res.x, res.y, res.z] for res in res_hands.multi_hand_world_landmarks[0].landmark]).flatten() # 0 is left hand
    else:
        rh = np.zeros(21*3)
        lh = np.zeros(21*3)
    
    return np.concatenate([pose, rh, lh])
