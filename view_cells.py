# -*- coding: utf-8 -*-
"""
Created on Thu May 28 13:30:49 2020

@author: cstef
"""

from ratslam_functions import SLAM_basics
import numpy as np


class ViewCell(SLAM_basics):
    
    def __init__(self, gc_x, profile, vc_id):
        
        super().__init__()
        self.id = vc_id
        self.x = gc_x
        self.profile = profile
        self.decay = self.VC_DECAY_VALUE
        self.is_new = True
        self.exps = []
    
class ViewCells(SLAM_basics):
    
    def __init__(self):
        
        super().__init__()
        self.local_view_cells = []
        self.vc_id = 0
    
    def compare_profiles(self,img_prof1,img_prof2, slen):
        return super().compare_profiles(img_prof1,img_prof2, slen)
    
    def create_image_profile(self, image):
        return super().create_image_profile(image)
               
    def create_view_cell(self,gc_x, img_profile, vc_id):
        
        cell = ViewCell(gc_x, img_profile, vc_id)
        self.local_view_cells.append(cell)
        return cell
    
    def calculate_score(self, img_profile):
    
        scores_array = []
        img_prof1 = img_profile
        
        for view_cell in self.local_view_cells:
            
            img_prof2 = view_cell.profile
            view_cell.decay -= self.GLOBAL_VC_DECAY
            if view_cell.decay < 0:
                view_cell.decay = 0
                
            _ , mindiff = self.compare_profiles(img_prof1,img_prof2, self.VT_SHIFT_MATCH)
            scores_array.append(mindiff) 
        
        return scores_array
    
    # active_view_cell
    def active_cell(self, img, gc_x):
        
        img_profile = self.create_image_profile(img)
        scores = self.calculate_score(img_profile)        
        scores = np.array(scores)
        # print(scores)
        if scores.size > 0:
            
            matches = scores[scores < self.VT_MATCH_THRESHOLD]
            # print(matches)
            if matches.size > 0:
                
                if matches.size == 1:
                    index = np.where(scores == matches[0])

                else:
                    index = np.where(scores == matches[-1])

                view_cell = self.local_view_cells[index[0][-1]]
                view_cell.decay += self.VC_DECAY_VALUE
                view_cell.is_new = False
                
            else:
                # New view cell if the threshold is not triggered
                self.vc_id += 1
                view_cell = self.create_view_cell(gc_x, img_profile, self.vc_id)
            
        else:
            # Create the first view cell
            self.vc_id += 1
            view_cell = self.create_view_cell(gc_x, img_profile, self.vc_id)
            
        return view_cell
            
            
    
                
            
            