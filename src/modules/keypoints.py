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
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, lh, rh])


# extract hands keypoints coordinates only from image using hand model
def pose_hand_keypoints(res_pose, res_hands):
    # select keypoints 11-23 (shoulders and arms) in pose
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in res_pose.pose_world_landmarks.landmark[11:23]]).flatten() if res_pose.pose_world_landmarks else np.zeros(12*4)
    
    # 21 keypoints in left and right hands each
    if res_hands.multi_hand_world_landmarks:
        h1 = np.array([[res.x, res.y, res.z] for res in res_hands.multi_hand_world_landmarks[0].landmark]).flatten() if len(res_hands.multi_hand_world_landmarks)==1 else np.zeros(21*3)
        h2 = np.array([[res.x, res.y, res.z] for res in res_hands.multi_hand_world_landmarks[1].landmark]).flatten() if len(res_hands.multi_hand_world_landmarks)==2 else np.zeros(21*3)
    else:
        h1 = np.zeros(21*3)
        h2 = np.zeros(21*3)
    
    return np.concatenate([pose, h1, h2])
