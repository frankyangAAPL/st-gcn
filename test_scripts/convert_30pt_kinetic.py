import os 
import numpy as np
import pickle
import matplotlib.pyplot as plt 

"""
    0 'base_of_neck',
    1     /'mid_chest',
    2     /'belly',
    3     /'center_of_pelvis',  # 0-3
    4     /'right_collarbone',
    5 'right_shoulder',
    6 'right_elbow',
    7 'right_wrist',  # 4-7
    8   /'left_collarbone',
    9 'left_shoulder',
    10 'left_elbow',
    11 'left_wrist',  # 8-11
    12 'right_hip',
    13 'right_knee',
    14 'right_ankle',  # 12-14
    15  'left_hip',
    16 'left_knee',
    17 'left_ankle',  # 15-17
    18     /'right_toe',
    19     /'left_toe',  # 18-19
    20     /'left_thumb',
    21   /'left_littlefinger',
    22   /'right_thumb',
    23   /'right_littlefinger',  # 20-23
    24 'nose',
    25  /'top_of_neck',
    26 'right_eye',
    27 'right_ear',
    28 'left_eye',
    29 'left_ear'  # 24-29
"""

convert_30_to_18_idx = {
    0: 24,
    1: 0,
    2: 5,
    3: 6,
    4: 7,
    5: 9,
    6: 10,
    7: 11,
    8: 12,
    9: 13,
    10: 14,
    11: 15,
    12: 16,
    13: 17,
    14: 26,
    15: 28,
    16: 27,
    17: 29
}

train_data = "/mnt/task_runtime/st-gcn/data/train/1005005010020062623"
video_name = os.path.basename(train_data)


max_num_person_in_frame = 5
max_num_person_out = 2
C, T, V, M = 3, 300, 18, max_num_person_out

st_gcn_input = np.zeros((C, T, V, max_num_person_in_frame))

for idx in range(200, 500):
    frame_idx = idx % 300
    with open(os.path.join(train_data, f"{frame_idx:09d}.pkl"), "rb") as f:
        p = pickle.load(f) 
        
    all_skeletons = p['body_keypoints_30pts']
    for skeleton_idx, skeleton in enumerate(all_skeletons):
        
        # Process only the first 5 skeletons 
        if skeleton_idx > max_num_person_in_frame:
            break 
        
        skeleton_coords = skeleton['keypoints']
        for j in range(18):
            skeleton_pt = skeleton_coords[convert_30_to_18_idx[j]]
            confidence = skeleton_pt['confidence']
            x = skeleton_pt['x']
            y = skeleton_pt['y']
            x_max = skeleton_pt['x_max']
            y_max = skeleton_pt['y_max']    
            x_normalized = x / x_max
            y_normalized = y / y_max
            
            st_gcn_input[0, frame_idx, j, 0] = x_normalized
            st_gcn_input[1, frame_idx, j, 0] = y_normalized
            st_gcn_input[2, frame_idx, j, 0] = confidence
        
# data centralization
st_gcn_input[0:2] = st_gcn_input[0:2] - 0.5

# sort by score
sort_index = (-st_gcn_input[2, :, :, :].sum(axis=1)).argsort(axis=1)
for t, s in enumerate(sort_index):
    st_gcn_input[:, t, :, :] = st_gcn_input[:, t, :, s].transpose((1, 2,
                                                                0))
st_gcn_input = st_gcn_input[:, :, :, 0:max_num_person_out]
st_gcn_input = st_gcn_input[np.newaxis, ...]

st_gcn_label = (
    [video_name],
    [325]
)

np.save("/mnt/task_runtime/st-gcn/data/train_reformat/test.npy", st_gcn_input)
with open("/mnt/task_runtime/st-gcn/data/train_reformat/test.pkl", "wb") as f:
    pickle.dump(st_gcn_label, f)



