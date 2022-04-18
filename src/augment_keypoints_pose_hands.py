"""
Boost dataset size to 12000 with 100 flipped keypoints (mirror x, for people who have different dominant hands), 
and randomly rotated keypoints with angle between (-45, 45) around axis z

Original dataset size: (750, 40, 174)
    15 signs with 50 mini-videos for each sign
    40 frames for each video
    174 keypoints for each frame
Return a stored dataset (.npy) with size of (45000, 40, 174)
    15 signs with 1200 mini-videos for each sign
"""

import numpy as np
import os

# rotate coordinates (x, y) of keypoints with an angle of a random value in [-30, 30)  
def keypoints_rotate(x, y):
    coordinates = np.stack((x, y), axis=1)
    theta = np.random.uniform(-30, 30) * np.pi / 180
    coordinates = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]),
                         coordinates.T).T
    x = coordinates[:,0]
    y = coordinates[:,1]
    return x, y 


signs = np.array(['A', 'cat', 'D', 'excited', 'Hello', 'How', 'I', 'I_me', 'love', 'my', 'name', 'S', 'Thank you', 'Y', 'you'])
labels_num = {label:num for num, label in enumerate(signs)}

data = [] # X
labels = [] # y
# load data
# Loop through signs, videos, frames
for sign in signs:
    for video in range(50):
        frames = []
        for frame_i in range(40):
            frame = np.load(os.path.join('data/keypoints/pose_hands', sign, str(video), f'{frame_i}.npy'))
            frames.append(frame)
        data.append(frames)
        labels.append(labels_num[sign])
        
X = np.array(data)
y = np.array(labels)


# extract x, y indice of keypoints in face, pose and hands
# order: pose 12*4, two hand 42*3, [x1, y1, z1, x2, y2, z2, ...]
idx = list(range(174))
x_idx = idx[0::3]
y_idx = idx[1::3]

# add flipped videos
# flip coordinates x of keypoints   
flipped_X = X.copy()
flipped_X[:,:,x_idx] = -flipped_X[:,:,x_idx]
X = np.append(X, flipped_X, axis=0)

# add rotated videos
X_copy = X.copy()
y_copy = y.copy() 
for i in range(5):
    # data will be updated in place
    # so make new copy for each loop to make sure that the rotation is made on original unrotated data 
    X_rotated = X_copy.copy()
    for video in X_rotated:
        for frame in video:
            frame[x_idx], frame[y_idx] = keypoints_rotate(frame[x_idx], frame[y_idx])
    X = np.append(X, X_rotated, axis=0)
    y = np.append(y, y_copy, axis=0)

os.makedirs(os.path.join('data', 'augmented_data'))
path = os.path.join('data', 'augmented_data')

np.save(path, X)
np.save(path, y)