# -*- coding: utf-8 -*-
"""
Created on Tue May 26 01:30:06 2020

@author: cstef
"""
from math import sqrt, pi, ceil, floor, exp
import numpy as np
from ratslam_functions import SLAM_basics

class VisualOdometry(SLAM_basics):
    
    def __init__(self):
        super().__init__()       
        self.vrot_previous = 0    
        self.old_vrot_template = np.zeros(self.y_size)
        self.odometry = np.pi/2

    def compare_profiles(self,current_img_prof,previous_img_prof, slen):
        return super().compare_profiles(current_img_prof,previous_img_prof, slen)
        
    def create_image_profile(self, image):
        return super().create_image_profile(image)
    
    def calculate_velocities(self, image):
        current_img_prof = self.create_image_profile(image)
        previous_img_prof = self.old_vrot_template
        shift_len = self.SHIFT_LEN
        min_offset, min_dif  = self.compare_profiles(current_img_prof,previous_img_prof, shift_len)
        self.old_vrot_template = current_img_prof        
        vrot = -min_offset*(23./image.shape[1])*np.pi/180
        self.odometry += vrot
        return self.odometry, vrot 