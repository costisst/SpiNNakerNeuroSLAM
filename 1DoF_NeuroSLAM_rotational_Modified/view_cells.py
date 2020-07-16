# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:30:49 2020

@author: cstef
"""

from ratslam_functions import SLAM_basics
import numpy as np
from numba import cuda
from numba import *
import cupy as cp



class ViewCell(SLAM_basics):
    
    def __init__(self, gc_th, profile, vc_id):
        
        super().__init__()
        self.id = vc_id
        self.th = gc_th
        self.profile = profile
        self.decay = self.VC_DECAY_VALUE
        self.is_new = True
        self.exps = []
    
class ViewCells(SLAM_basics):
    
    def __init__(self):
        
        super().__init__()
        self.local_view_cells = []
        self.vc_weight_matrix = np.zeros(self.gc_th_dim)
        self.vc_id = 0
    
    def compare_profiles(self,img_prof1,img_prof2, slen):
        return super().compare_profiles(img_prof1,img_prof2, slen)
    
    def create_image_profile(self, image):
        return super().create_image_profile(image)
               
    def create_view_cell(self,gc_th, img_profile, vc_id):
        
        cell = ViewCell(gc_th, img_profile, vc_id)
        self.local_view_cells.append(cell)
        return cell
    
    def calculate_score(self, img_profile):
    
        scores_array = []
        img_prof1 = img_profile
        
        for view_cell in self.local_view_cells:
            
            img_prof2 = view_cell.profile
            # view_cell.decay -= self.GLOBAL_VC_DECAY
            # if view_cell.decay < 0:
            #     view_cell.decay = 0
                
            _ , min_diff = self.compare_profiles(img_prof1,img_prof2, self.VT_SHIFT_MATCH)
            scores_array.append(min_diff) 
        
        return scores_array
    
    # active_view_cell
    def active_cell(self, img, gc_th):
        
        img_profile = self.create_image_profile(img)
        scores = self.calculate_score(img_profile)
        scores = np.array(scores)
        # print(scores)
        if scores.size > 0:
            
            matches = scores[scores < self.VT_MATCH_THRESHOLD]
            # print(matches)
            if matches.size > 0:
                
                # print(gc_th)
                
                index = np.where(scores == np.min(matches))
                # print(index[0][0])
                view_cell = self.local_view_cells[int(index[0][0])]
                dif = abs(view_cell.th - gc_th)
                if dif > 18:
                    dif = abs(36 - dif)
                
                rate = dif/18
                if dif == 0:
                    dif = 1                   
                    self.vc_weight_matrix[view_cell.id, gc_th] += self.lr * dif
                   
                else:
                    dif == 1
                    self.vc_weight_matrix[view_cell.id, view_cell.th ] -= self.lr * dif * (1 +  rate)
                    self.vc_weight_matrix[view_cell.id, gc_th] += self.lr * dif * (1 +  rate)
                
                
                max_val_index = np.where(self.vc_weight_matrix[view_cell.id] == np.max(self.vc_weight_matrix[view_cell.id]))
                gc_th_max = max_val_index[0][0]
                view_cell.decay = self.vc_weight_matrix[view_cell.id, gc_th_max]
                view_cell.th = gc_th_max
                # print(view_cell.decay)
                print(gc_th_max)
                # np.savetxt("foo1.csv", np.array(self.vc_weight_matrix), delimiter=",")
                # if view_cell.decay < 0:
                #     view_cell.decay = 0
                # view_cell.decay += self.VC_DECAY_VALUE    
                view_cell.is_new = False
                # print(view_cell.th)
                # print(self.vc_weight_matrix)
                
            else:
                # New view cell if the threshold is not triggered
                self.vc_id += 1
                view_cell = self.create_view_cell(gc_th, img_profile, self.vc_id)
                new_array = np.full((1, self.gc_th_dim), -4.5)
                self.vc_weight_matrix = np.vstack([self.vc_weight_matrix, new_array])
                # self.vc_weight_matrix = np.vstack([self.vc_weight_matrix, np.zeros(self.gc_th_dim)])
                self.vc_weight_matrix[self.vc_id, gc_th] = -1
        else:
            # Create the first view cell
            self.vc_id += 1
            view_cell = self.create_view_cell(gc_th, img_profile, self.vc_id)
            new_array = np.full((1, self.gc_th_dim), -4.5)
            self.vc_weight_matrix = np.vstack([self.vc_weight_matrix, new_array])
            # self.vc_weight_matrix = np.vstack([self.vc_weight_matrix, np.zeros(self.gc_th_dim)])
            self.vc_weight_matrix[self.vc_id, gc_th] = -1
            
        return view_cell