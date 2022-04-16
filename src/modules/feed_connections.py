import cv2
import mediapipe as mp

# detect video feed and make it writable to further draw keypoints and connections in video
def detect_video(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False                  # Image is no longer writeable
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True                   # Image is now writeable 
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR COVERSION RGB 2 BGR
    return image, results


mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

def draw_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION, 
                             mp_drawing.DrawingSpec(color=(229,225,98), thickness=1, circle_radius=1), 
                             mp_drawing.DrawingSpec(color=(229,225,98), thickness=1, circle_radius=1)
                             ) 
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(51,153,255), thickness=6, circle_radius=3), 
                             mp_drawing.DrawingSpec(color=(255,255,255), thickness=3, circle_radius=6)
                             ) 
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(153,153,255), thickness=4, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(255,255,255), thickness=3, circle_radius=6)
                             ) 
    # Draw right hand connections  
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                             mp_drawing.DrawingSpec(color=(204,153,153), thickness=4, circle_radius=4), 
                             mp_drawing.DrawingSpec(color=(255,255,255), thickness=3, circle_radius=6)
                             ) 