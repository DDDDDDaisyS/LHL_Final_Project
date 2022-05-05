"""
Original dataset size: (500, 30, 144)
Augmented dataset size: (16000, 30, 144)
    flipped keypoints (mirror x, represent people who have different dominant hands)
    scale (represent people who are taller or shorter)
    translation

    21 signs with 50 mini-videos for each sign (20 signs + 1 nothing(no movement))
    30 frames for each video
    144 keypoints for each frame
Return a stored dataset (.npy) with size of (33600, 40, 174)
    21 signs with 1600 mini-videos for each sign
"""

import numpy as np
import os


# zoom in or out the keypoints coordinates by given scale
def keypoints_scale(x, scale):
    x *= scale
    return x

# translate point x with delta_x of random values in [-0.2, 0.2)  
def keypoints_translation(x):
    delta_x = np.random.uniform(-0.2, 0.2)
    x += delta_x
    return x

# rotate point (x, y) about (0.5, 0.5) with an angle  
def keypoints_rotate(x, y, angle):
    coordinates = np.stack((x-0.5, y-0.5), axis=1)
    theta = angle * np.pi / 180
    coordinates = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]),
                         coordinates.T).T
    x = coordinates[:,0]
    y = coordinates[:,1]
    return x, y 

signs = np.array(['busy', 'finish', 'Hello', 'How', 'love', 'nothing', 'sign', 'Take care', 'Thank you', 'you'])
labels_num = {label:num for num, label in enumerate(signs)}

data = [] # X
labels = [] # y
# load data
# Loop through signs, videos, frames
for sign in signs:
    for video in range(50):
        frames = []
        for frame_i in range(1, 31):
            frame = np.load(os.path.join('data/keypoints/pose_hands', sign, str(video), f'{frame_i}.npy'))
            frames.append(frame)
        data.append(frames)
        labels.append(labels_num[sign])
        
X = np.array(data)
y = np.array(labels)


# extract x, y indice of keypoints in face, pose and hands
# order: pose 6*3, two hand 42*3, [x1, y1, z1, x2, y2, z2, ...]
idx = list(range(144))
x_idx = idx[0::3]
y_idx = idx[1::3]

# add flipped keypoints
# x, y were normalized in [0, 1] by image width and height, flip around x=0.5
flipped_X = X.copy()
flipped_X[:,:,x_idx] = 1-flipped_X[:,:,x_idx] # flip
rarm = np.copy(flipped_X[:,:,idx[0:6:2]])
larm = np.copy(flipped_X[:,:,idx[1:6:2]])
rh = np.copy(flipped_X[:,:,idx[81:]])
lh = np.copy(flipped_X[:,:,idx[18:81]])

flipped_X[:,:,idx[0:6:2]] = rarm # swap columns of right arm [0,2,4] and left arm [1,3,5]
flipped_X[:,:,idx[1:6:2]] = larm
flipped_X[:,:,idx[18:81]] = rh # swap columns of rh [18:81] and lh [81:]
flipped_X[:,:,idx[81:]] = lh

X = np.append(X, flipped_X, axis=0)
y = np.append(y, y, axis=0)

# make copies to make sure augmentation is operated on original data 
# since everytime when new keypoints are appended to X, X size increases
X_copy = X.copy()
y_copy = y.copy() 

# add translated keypoints
for i in range(5):
    X_translated = X_copy.copy()
    X_translated[:,:,x_idx] = keypoints_translation(X_translated[:,:,x_idx])
    X_translated[:,:,y_idx] = keypoints_translation(X_translated[:,:,y_idx])
    X = np.append(X, X_translated, axis=0)
    y = np.append(y, y_copy, axis=0)

# add scaled keypoints for people who is shorter or taller
scales = [0.8, 0.85, 0.9, 0.95, 1.05, 1.1, 1.15, 1.2, 1.25, 1.3]
for scale in scales:
    scaled_X = keypoints_scale(X_copy, scale)
    X = np.append(X, scaled_X, axis=0)
    y = np.append(y, y_copy, axis=0)
    
# # add rotated videos
# angles = [-15, -10, -5, 5, 10, 15]
# for angle in angles:
#     # data will be updated in place
#     # so make a new copy for each loop to make sure that the rotation is made on the original unrotated data 
#     X_rotated = X_copy.copy()
#     for video in X_rotated:
#         for frame in video:
#             frame[x_idx], frame[y_idx] = keypoints_rotate(frame[x_idx], frame[y_idx], angle)
#     X = np.append(X, X_rotated, axis=0)
#     y = np.append(y, y_copy, axis=0)


try:
    os.makedirs(os.path.join('data', 'augmented_data'))
except:
    pass
np.save(os.path.join('data', 'augmented_data', 'X'), X)
np.save(os.path.join('data', 'augmented_data', 'y'), y)