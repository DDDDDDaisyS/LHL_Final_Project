import numpy as np
import os

# translate coordinates x of keypoints with delta_x of random values in [-0.2, 0.2)  
def keypoints_translation(x):
    delta_x = np.random.uniform(-0.2, 0.2)
    x += delta_x
    return x

# rotate coordinates (x, y) of keypoints with an angle of a random value in [-30, 30)  
def keypoints_rotate(x, y):
    coordinates = np.stack((x, y), axis=1)
    theta = np.random.uniform(-30, 30) * np.pi / 180
    coordinates = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]),
                         coordinates.T).T
    x = coordinates[:,0]
    y = coordinates[:,1]
    return x, y 


signs = np.array(['cat', 'excited', 'How', 'love', 'No', 'Take care', 'Thank you', 'what', 'Yes', 'you'])
labels_num = {label:num for num, label in enumerate(signs)}

data = [] # X
labels = [] # y
for sign in signs:
    for video in range(50):
        frames = []
        for frame_i in range(25):
            frame = np.load(os.path.join('data', sign, str(video), f'{frame_i}.npy'))
            frames.append(frame)
        data.append(frames)
        labels.append(labels_num[sign])
        
X = np.array(data)
y = np.array(labels)


# extract x, y indice of keypoints in face, pose and hands
# order: face 468*3, pose 12*4, left hand 21*3, right hand 21*3
idx = list(range(1578))
face_x = idx[:1404:3] # index ends at 468*3
face_y = idx[1:1404:3]
pose_x = idx[1404:1452:4] # index ends at 468*3+12*4
pose_y = idx[1405:1452:4]
lh_x = idx[1452:1515:3] # index ends at 468*3+12*4+21*3
lh_y = idx[1453:1515:3]
rh_x = idx[1515::3]
rh_y = idx[1516::3]

x_idx = face_x + pose_x + lh_x + rh_x 
y_idx = face_y + pose_y + lh_y + rh_y
x_idx.sort()
y_idx.sort()

# add flipped videos
flipped_X = X.copy()
flipped_X[:,:,x_idx] = -flipped_X[:,:,x_idx]
X = np.append(X, flipped_X, axis=0)

# add rotated and translated videos
X_copy = X.copy()
y_copy = y.copy()
for i in range(5):
    X_translated = X_copy.copy()
    X_translated[:,:,x_idx] = keypoints_translation(X_translated[:,:,x_idx])
    X_translated[:,:,y_idx] = keypoints_translation(X_translated[:,:,y_idx])
    X = np.append(X, X_translated, axis=0)
    y = np.append(y, y_copy, axis=0)

    X_rotated = X_copy.copy()
    for video in X_rotated:
        for frame in video:
            frame[x_idx], frame[y_idx] = keypoints_rotate(frame[x_idx], frame[y_idx])
    X = np.append(X, X_rotated, axis=0)
    y = np.append(y, y_copy, axis=0)

os.makedirs(os.path.join('data', 'augmented_data'))
X_path = os.path.join('data', 'augmented_data', 'X')
y_path = os.path.join('data', 'augmented_data', 'y')
np.save(X_path, X)
np.save(y_path, y)