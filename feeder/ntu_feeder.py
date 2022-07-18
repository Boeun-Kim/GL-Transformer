'''
This file is for data feeder
'''

import numpy as np
import pickle, torch
from . import tools
import random


class FeederPretrain(torch.utils.data.Dataset):

    def __init__(self, data_path, label_path, shear_amplitude=0.3, interpolate_ratio=0.1, intervals=[1,5,10], mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.shear_amplitude = shear_amplitude
        self.interpolate_ratio = interpolate_ratio
        self.intervals = intervals

        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])

        # augmentation except [pad] token
        data_m_idx = data_numpy[0,:,0,0] != 99.9
        real_data_length = sum(data_m_idx)

        real_data = data_numpy[:, :real_data_length,:, :]

        real_data_aug = self._aug(real_data)
        
        data = np.ones(data_numpy.shape)*99.9
        data[:, :real_data_aug.shape[1],:,:] = real_data_aug

        c, num_frames, _, _ = data.shape  #frames, joints, people
        
        # calculate gt displacement
        rolls = self.intervals
        dis_list = []
        for roll in rolls:
            displacement = np.zeros_like(data)
            data_roll = np.roll(data, roll, axis=1)
            if roll > 0:
                data_roll[:,:roll, :, :] = np.expand_dims(data[:, 0,:,:], axis=1)
            else:
                m_idx = data[0,:,0,0] != 99.9
                real_length = int(sum(m_idx))
                data_roll[:,real_length+roll:real_length, :, :] = np.expand_dims(data[:, real_length-1,:,:], axis=1)
            
            displacement = data_roll - data
            dis_list.append(displacement)

        dis_concat = np.concatenate(dis_list, axis=2)

        # calculate gt displacement magnitude & decide class
        roll_mags = self.intervals
        mag_list = []
        for roll_mag in roll_mags:
            data_roll = np.roll(data, roll_mag, axis=1)
            if roll_mag > 0:
                    data_roll[:,:roll_mag, :, :] = np.expand_dims(data[:, 0,:,:], axis=1)
            else:
                m_idx = data[0,:,0,0] != 99.9
                real_length = int(sum(m_idx))
                data_roll[:,real_length+roll_mag:real_length, :, :] = np.expand_dims(data[:, real_length-1,:,:], axis=1)
            
            displacement = data_roll - data
            displacement = np.power(displacement[0], 2)+np.power(displacement[1], 2)+np.power(displacement[2], 2)
            mag = np.sqrt(displacement)
            
            mag_quant = mag//0.01 +1
            mag_quant[mag_quant > 14] = 14
            mag_quant[mag==0.0] = 0
            if np.sum(data_numpy[:,0,:,1]) == 0:
                mag_quant[:,:,1] = 15

            mag_quant = mag_quant.reshape(num_frames, -1)
            mag_quant = mag_quant.astype(int)
            mag_list.append(mag_quant)

        mag_gt = np.concatenate(mag_list, axis=1)

        # reshape
        data = data.reshape(c, num_frames, -1)
        dis_concat = dis_concat.reshape(c, dis_concat.shape[1], -1)
        c, num_frames, num_joints = data.shape

        ## decide class of displacement direction
        xyz_direction = np.zeros((c, dis_concat.shape[1], dis_concat.shape[2]), dtype=int)
        xyz_direction[dis_concat == 0] = 1
        xyz_direction[dis_concat > 0] = 2
        dir_gt = xyz_direction[0] + xyz_direction[1]*3 + xyz_direction[2]*9

        masked_node = np.zeros((num_frames, num_joints), dtype=int)
        return data, dir_gt, mag_gt, masked_node

    def _aug(self, data_numpy):
        if self.interpolate_ratio > 0:
            data_numpy = tools.interpolate(data_numpy, self.interpolate_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
      
        return data_numpy
        


class Feeder_actionrecog(torch.utils.data.Dataset):
    """ Feeder for single inputs """

    def __init__(self, data_path, label_path, shear_amplitude=-1, interpolate_ratio=0.1, mmap=True):
        self.data_path = data_path
        self.label_path = label_path

        self.bone_link = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6), (8, 7), (9, 21),
                     (10, 9), (11, 10), (12, 11), (13, 1), (14, 13), (15, 14), (16, 15), (17, 1),
                     (18, 17), (19, 18), (20, 19), (21, 21), (22, 23), (23, 8), (24, 25), (25, 12)]

        self.shear_amplitude = shear_amplitude
        self.interpolate_ratio = interpolate_ratio
       
        self.load_data(mmap)

    def load_data(self, mmap):
        # load label
        with open(self.label_path, 'rb') as f:
            self.sample_name, self.label = pickle.load(f)

        # load data
        if mmap:
            self.data = np.load(self.data_path, mmap_mode='r')
        else:
            self.data = np.load(self.data_path)
        
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        # get data
        data_numpy = np.array(self.data[index])
        label = self.label[index]
        
        # augmnetation
        data_m_idx = data_numpy[0,:,0,0] != 99.9
        real_data_length = sum(data_m_idx)
        real_data = data_numpy[:, :real_data_length,:, :]

        real_data_aug = self._aug(real_data)
        
        data = np.ones(data_numpy.shape)*99.9
        data[:, :real_data_aug.shape[1],:,:] = real_data_aug
        
        c, num_frames, j, p = data.shape  # batch, channels, frames, joints, people
        data = data.reshape(c, num_frames, -1)

        return data, label

    def _aug(self, data_numpy):
        if self.interpolate_ratio > 0:
            data_numpy = tools.interpolate(data_numpy, self.interpolate_ratio)

        if self.shear_amplitude > 0:
            data_numpy = tools.shear(data_numpy, self.shear_amplitude)
        
        return data_numpy
