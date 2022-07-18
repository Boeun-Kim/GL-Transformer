'''
This file is for data augmentation
'''

import numpy as np
import random
import cv2


def shear(data_numpy, r=0.5):

    center_trans = data_numpy[:,:, 20, :].copy()
    data_numpy[:,:,20,:] = 0.0

    s1_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]
    s2_list = [random.uniform(-r, r), random.uniform(-r, r), random.uniform(-r, r)]

    R = np.array([[1,          s1_list[0], s2_list[0]],
                  [s1_list[1], 1,          s2_list[1]],
                  [s1_list[2], s2_list[2], 1        ]])

    R = R.transpose()
    data_numpy = np.dot(data_numpy.transpose([1, 2, 3, 0]), R)
    data_numpy = data_numpy.transpose(3, 0, 1, 2)

    data_numpy[:,:,20,:] = center_trans
    
    return data_numpy


def interpolate(data_numpy, interpolate_ratio=0.0):
    C, T, V, M = data_numpy.shape

    if np.random.rand() > 0.5:
        sign = -1
    else:
        sign = 1
    rand_ratio = np.random.rand()
    interpolate_size = int(T * (1 + sign*rand_ratio*interpolate_ratio))
    interpolate_size = max(1, interpolate_size)
    new_data = np.zeros((C, interpolate_size, V, M))

    for i in range(M):
        tmp = cv2.resize(data_numpy[:, :, :, i].transpose(
            [1, 2, 0]), (V, interpolate_size), interpolation=cv2.INTER_LINEAR)

        tmp = tmp.transpose([2, 0, 1])

        new_data[:, :, :, i] = tmp

    max_frame = 300
    if new_data.shape[1] > max_frame:
        new_data = new_data[:, :max_frame, :, :]

    return new_data
