# -*- coding: utf-8 -*-
"""
Created on Sun May 24 18:02:30 2020

@author: cstef
"""
from ratslam_functions import SLAM_basics
import numpy as np

class Experience(SLAM_basics):
    
    def __init__(self, gc_x, exp_x, view_cell):
        
        super().__init__()
        self.gc_x = gc_x #x_pc
        self.exp_x = exp_x #x_m
        # self.facing_rad = facing_rad
        self.view_cell = view_cell
        self.links = []
        
    def signed_delta_rad(self, angle1, angle2):
        return super().signed_delta_rad(angle1, angle2)

    def link_to(self, target, accum_delta_x):

        distance = np.sqrt(accum_delta_x**2)
        # heading_rad = signed_delta_rad(self.facing_rad, np.arctan2(accum_delta_y, accum_delta_x))
        # facing_rad = self.signed_delta_rad(self.facing_rad, accum_delta_facing)
        link = ExperienceLink( target, distance)
        self.links.append(link)    

class ExperienceLink():
    '''A representation of connection between experiences.'''

    def __init__(self, target, distance):

        self.target = target
        # self.facing_rad = facing_rad
        self.distance = distance
        # self.heading_rad = heading_rad            
        
class ExperienceMap(SLAM_basics):   
    
    def __init__(self):

          super().__init__()
          self.exps = []
          
          self.current_exp = None
          self.current_view_cell = None
  
          self.accum_delta_x = 0
          # self.accum_delta_y = 0
          # self.accum_delta_facing = np.pi/2
  
          self.history = []

    def create_experience(self, gc_x, view_cell):

        exp_x = self.accum_delta_x
        # facing_rad = clip_rad_180(self.accum_delta_facing)

        if self.current_exp is not None:
            exp_x += self.current_exp.exp_x
            # y_m += self.current_exp.y_m
            
        exp = Experience(gc_x, exp_x, view_cell)

        if self.current_exp is not None:
            self.current_exp.link_to(exp, self.accum_delta_x)

        self.exps.append(exp)
        view_cell.exps.append(exp)

        return exp

    def min_delta(self, d1, d2, max_val):
        return super().min_delta(d1, d2, max_val)

    def signed_delta_rad(self, angle1, angle2):
        return super().signed_delta_rad(angle1, angle2)
    
    def clip_rad_180(self, angle): 
       return super().clip_rad_180(angle)
    
    def exp_map_iter(self, view_cell, vtrans, gc_x):
     
        self.accum_delta_x += vtrans

        if self.current_exp is None:
            delta_pc = 0
        else:
            delta_pc = self.min_delta(self.current_exp.gc_x, gc_x, self.gc_x_dim)
            # print(delta_pc)
        # print(delta_pc)
        adjust_map = False
        if len(view_cell.exps) == 0 or delta_pc > self.EXP_DELTA_PC_THRESHOLD:
            exp = self.create_experience(gc_x, view_cell)

            self.current_exp = exp
            self.accum_delta_x = 0

        # if the vt has changed (but isn't new) search for the matching exp
        elif view_cell != self.current_exp.view_cell:

            # find the exp associated with the current vt and that is under the
            # threshold distance to the centre of pose cell activity
            # if multiple exps are under the threshold then don't match (to reduce
            # hash collisions)
            adjust_map = True
            matched_exp = None

            delta_pcs = []
            n_candidate_matches = 0
            for exp in view_cell.exps:
                delta_pc = self.min_delta(exp.gc_x, gc_x, self.gc_x_dim)
                delta_pcs.append(delta_pc)

                if delta_pc < self.EXP_DELTA_PC_THRESHOLD:
                    n_candidate_matches += 1

            if n_candidate_matches > 1:
                pass

            else:
                min_delta_id = np.argmin(delta_pcs)
                min_delta_val = delta_pcs[min_delta_id]

                if min_delta_val < self.EXP_DELTA_PC_THRESHOLD:
                    matched_exp = view_cell.exps[min_delta_id]

                    # see if the prev exp already has a link to the current exp
                    link_exists = False
                    for linked_exp in [l.target for l in self.current_exp.links]:
                        if linked_exp == matched_exp:
                            link_exists = True

                    if not link_exists:
                        self.current_exp.link_to(matched_exp, self.accum_delta_x)

                if matched_exp is None:
                    matched_exp = self.create_experience(gc_x, view_cell)

                self.current_exp = matched_exp
                self.accum_delta_x = 0

        self.history.append(self.current_exp)

        if not adjust_map:
            return


        # Iteratively update the experience map with the new information     
        for i in range(0, self.EXP_LOOPS):
            for e0 in self.exps:
                for link in e0.links:
                    # e0 is the experience under consideration
                    # e1 is an experience linked from e0
                    # l is the link object which contains additoinal heading

                    e1 = link.target
                    
                    # correction factor
                    cf = self.EXP_CORRECTION
                    
                    # work out where exp0 thinks exp1 (x,y) should be based on 
                    # the stored link information
                    lx = e0.exp_x

                    # correct e0 and e1 (x,y) by equal but opposite amounts
                    # a 0.5 correction parameter means that e0 and e1 will be 
                    # fully corrected based on e0's link information
                    e0.exp_x = e0.exp_x + (e1.exp_x - lx) * cf
                    e1.exp_x = e1.exp_x - (e1.exp_x - lx) * cf


    
        return

        