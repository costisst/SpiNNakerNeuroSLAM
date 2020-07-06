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

    def compare_profiles(self,img_prof1,img_prof2, slen):
        return super().compare_profiles(img_prof1,img_prof2, slen)
        
    def create_image_profile(self, image):
        return super().create_image_profile(image)
    
    def calculate_velocities(self, image):
        
        img_prof1 = self.create_image_profile(image)
        img_prof2 = self.old_vrot_template
        shift_len = self.SHIFT_LEN
        
        min_offset, min_dif  = self.compare_profiles(img_prof1,img_prof2, shift_len)
        # print(min_dif,min_offset)
        # print(min_offset)
        self.old_vrot_template = img_prof1
        
        # vtrans = min_dif * self.VTRANS_SCALE
        # # print(vtrans)
        # # if vtrans > self.MAX_TRANS_V_THRESHOLD :
        # #     vtrans = self.vtrans_previous
        # if vtrans > 10 :
        #     vtrans = 0
        # else:
        #     self.vrot_previous = vtrans
        # vrot = min_offset*(185./image.shape[1])*np.pi/180
        vrot = min_offset*(135./image.shape[1])*np.pi/180
        print(vrot)
        # self.old_vrot_template = template
        
        
        self.odometry += vrot
        
        return self.odometry, vrot 