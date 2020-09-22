# -*- coding: utf-8 -*-
"""
Created on Tue May 26 01:30:06 2020

@author: cstef
"""
from math import sqrt, pi, ceil, floor, exp
import numpy as np
# from numba import cuda
# from numba import *
# import cupy as cp

class VisualOdometry(SLAM_basics):
    
    def __init__(self):
        
        super().__init__()       
        self.vrot_previous = 0    
        self.old_vrot_template = np.zeros(self.y_size)
        self.odometry = np.pi/2

    def compare_profiles(self,img_prof1,img_prof2, slen):
        return super().compare_profiles(img_prof1,img_prof2, slen)
        
    def create_image_profile(self, image):
        return super().create_image_profile(image)
    
    def calculate_velocities(self, image):
        
        img_prof1 = self.create_image_profile(image)
        img_prof2 = self.old_vrot_template
        shift_len = self.SHIFT_LEN      
        min_offset, min_dif  = self.compare_profiles(img_prof1,img_prof2, shift_len)
        self.old_vrot_template = img_prof1       
        vrot = min_offset*(155./image.shape[1])*np.pi/180
        self.odometry += vrot
        
        return self.odometry, vrot 
