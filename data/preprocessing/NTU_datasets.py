'''
NTU_datasets.py is modified from https://github.com/LinguoLi/CrosSCLR/blob/main/feeder/NTUDatasets.py
'''

import numpy as np
import random
import pickle
import cv2
import torch


def _y_transmat(thetas):
    tms = np.zeros((0, 3, 3))
    thetas = thetas * np.pi / 180
    for theta in thetas:
        tm = np.zeros((3, 3))
        tm[0, 0] = np.cos(theta)
        tm[0, 2] = -np.sin(theta)
        tm[1, 1] = 1
        tm[2, 0] = np.sin(theta)
        tm[2, 2] = np.cos(theta)
        tm = tm[np.newaxis, :, :]
        tms = np.concatenate((tms, tm), axis=0)
    return tms

def parallel_skeleton(ins_data):
    right_shoulder = ins_data[:, :, 8]  # 9th joint
    left_shoulder = ins_data[:, :, 4]  # 5tf joint
    vec = right_shoulder - left_shoulder
    vec[1, :] = 0
    l2_norm = np.sqrt(np.sum(np.square(vec), axis=0))

    theta = vec[0, :] / (l2_norm + 0.0001)
    
    thetas = np.arccos(theta) * (180 / np.pi)
    isv = np.sum(vec[2, :])
    if isv >= 0:
        thetas = -thetas
    y_tms = _y_transmat(thetas)
    new_skel = np.zeros(shape=(0, 25, 3))
    ins_data = ins_data.transpose(1, 2, 0)
    for ind, each_s in enumerate(ins_data):
        r = np.reshape(each_s, newshape=(25, 3))
        r = np.transpose(r)
        r = np.dot(y_tms[ind], r)
        r_t = np.transpose(r)
        r_t = np.reshape(r_t, newshape=(1, -1, 3))
        new_skel = np.concatenate((new_skel, r_t), axis=0)
    return new_skel, ins_data
    

class SimpleLoader(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 label_path,
                 debug=False,
                 mmap=True,
                 data_type='relative',
                 displacement=False,
                 t_length=200,
                 y_rotation=True,
                 sampling='resize'):
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.data_type = data_type
        if displacement is not False:
            assert isinstance(displacement, int)
        self.displacement = displacement
        self.max_length = 300
        self.t_length = t_length if t_length < self.max_length else self.max_length
        self.sampling = sampling
        self.y_rotation = y_rotation
        self.mmap = mmap
        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M
        # load data
        position_path = self.data_path.replace('_data.', '_position.')
        motion_path = self.data_path.replace('_data.', '_motion.')
        print(position_path)
        print(motion_path)
        print(self.label_path)

        if mmap:
            self.data = np.load(position_path, mmap_mode='r')
            self.motion = np.load(motion_path, mmap_mode='r') if self.displacement > 0 else None
            self.label = np.load(self.label_path).reshape(-1)
        else:
            self.data = np.load(position_path)
            self.motion = np.load(motion_path) if self.displacement > 0 else None
            self.label = np.load(self.label_path).reshape(-1)
        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        position = np.array(self.data[index])
        motion = np.array(self.motion[index]) if self.displacement > 0 else None
        label = np.array(self.label[index])

        # return motion_data, label
        if motion is not None:
            return position, motion, label
        else:
            return position, label


class NTUMotionProcessor(torch.utils.data.Dataset):
    def __init__(self,
                 data_path,
                 label_path,
                 mmap=True,
                 data_type='normal',
                 t_length=300,
                 y_rotation=False,
                 sampling='force_crop'):
        self.data_path = data_path
        self.label_path = label_path
        self.data_type = data_type
        self.max_length = 300
        self.sampling = sampling
        self.y_rotation = y_rotation
        self.t_length = t_length if t_length < self.max_length else self.max_length
        neighbor_1base = [(21, 2), (21, 3), (21, 5), (21, 9), (3, 4),
                          (5, 6), (6, 7), (7, 8), (8, 23), (23, 22),
                          (9, 10), (10, 11), (11, 12), (12, 25), (25, 24),
                          (2, 1), (1, 13), (1, 17), (13, 14), (14, 15),
                          (15, 16), (17, 18), (18, 19), (19, 20)]
        self.neighbor_1base = [(i - 1, j - 1) for (i, j) in neighbor_1base]
        self.load_data(mmap)

    def load_data(self, mmap):
        # data: N C V T M
        # load label
        with open(self.label_path, 'rb') as f:
            print(self.label_path)
            self.sample_name, self.label = pickle.load(f)
            self.label = np.array(self.label)
            self.label = self.label - self.label.min()

        # load data
        print(self.data_path)
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # get skeleton sequence length, person number
        length = self.get_length(data_numpy)
        num = self.get_person_num(data_numpy)

        # relative coordinate
        if self.data_type == 'relative':
            # center = 21
            C, T, V, M = data_numpy.shape

            center = data_numpy[:, :, 20, :].reshape([C, T, 1, M])
            dist_2_center = data_numpy - center

            initial_center = data_numpy[:, 0, 20, :].reshape([C, 1, M])
            
            # add global translation (switch 21th joint from (0,0,0) to global translation)
            center_translation = data_numpy[:, :, 20, :] - initial_center
            dist_2_center[:,:,20,:] = center_translation
            data_numpy = dist_2_center

        elif self.data_type == 'normal':
            pass
        else:
            raise TypeError('Invalid data type: %s' % self.data_type)

        # crop length
        if self.sampling == 'force_crop':
            data_numpy = data_numpy[:, :self.t_length, :, :]
   
        elif self.sampling == 'resize':
            data_numpy = self.real_resize(data_numpy, length, self.t_length)
        else:
            raise TypeError('Invalid sampling type: ' % self.sampling)

        # y rotation
        if self.y_rotation:
            for n in range(2):
                tmp_new, tmp_old = parallel_skeleton(data_numpy[:, :, :, n])
                data_numpy[:, :, :, n] = tmp_new.transpose(2, 0, 1)
        
        # get actual length
        length = self.t_length if length > self.t_length else length

        C, T, V, M = data_numpy.shape
        temp = np.ones((C, self.t_length, V, M))*99.9
        temp[:, :length, :, :] = data_numpy[:, :length, :, :]
        data_numpy = temp

        # return data, label
        return data_numpy, label

    @staticmethod
    def get_length(data):
        length = (abs(data[:, :, 0, 0]).sum(axis=0) != 0).sum()
        return length

    @staticmethod
    def get_person_num(data):
        num = (abs(data[:, :, 0, :]).sum(axis=0).sum(axis=0) != 0) * 1.0
        return num

    @staticmethod
    def real_resize(data_numpy, length, crop_size):
        C, T, V, M = data_numpy.shape
        new_data = np.zeros([C, crop_size, V, M])
        for i in range(M):
            tmp = cv2.resize(data_numpy[:, :length, :, i].transpose(
                [1, 2, 0]), (V, crop_size), interpolation=cv2.INTER_LINEAR)
            tmp = tmp.transpose([2, 0, 1])
            new_data[:, :, :, i] = tmp
        return new_data.astype(np.float32)
