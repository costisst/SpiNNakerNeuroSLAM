# -*- coding: utf-8 -*-
"""
Created on Sun May 24 16:31:41 2020

@author: cstef
"""

from math import sqrt, pi, ceil, floor, exp
import numpy as np
from ratslam_functions import SLAM_basics


class GridCells(SLAM_basics):
    
    def __init__(self):
        
        super().__init__()
           
        # initialize grid cells network
        self.can_network = np.zeros(self.gc_x_dim)
        gc_x_center = floor(self.gc_x_dim/2)
        self.can_network[gc_x_center] = 1
        self.inhib_x_center = self.find_center(self.inhib_x_dim)
        self.excit_x_center = self.find_center(self.excit_x_dim)
        self.excit_weights = self.weight_init(self.excit_x_dim, self.excit_x_center, self.sigma)
        self.inhib_weights = self.weight_init(self.inhib_x_dim, self.inhib_x_center, self.sigma)
        
    def find_center(self, vector_len):
        
        vec_center = floor(vector_len/2)
        return vec_center
        
        
    def weight_init(self, x_dim, x_center, sigma):
        
        # create weights
        weights = np.zeros(x_dim)
    
        # initialize weights using 1D Gaussian Distribution
        for x in range(0,x_dim):
            weights[x] = (1/(sigma*sqrt(2*pi)))*exp((-1/2)*(x-x_center/sigma)**2)   
        
        # ensure that is normalized
        weights = weights / sum(weights)
        
        return weights 
    
    def update_attractor_dynamics(self, ad_weight, ad_dim):
        
        # create empty temp gc network
        gc_temp = np.zeros(self.gc_x_dim)
        half = floor(ad_dim/2)        
        wrapped = list(range(self.gc_x_dim-half, self.gc_x_dim)) + list(range(self.gc_x_dim)) + list(range(half))

        #retrieve non empty indices 
        indices = np.nonzero(self.can_network)

        for i in indices[0]:
            
            gc_temp[wrapped[i:i+ad_dim]] +=  self.can_network[i] * ad_weight
                       
        return gc_temp   
    
    
    
    def find_activated(self):
        
        x = np.argmax(self.can_network)
        return(x)
      
    def network_iteration(self, excit_weights, excit_x_dim, inhib_weights, inhib_x_dim, view_cell, vtrans):

        # If it is a new view_cell do not inject energy 
        if view_cell.is_new == False:
            vc_x = view_cell.x
            energy_injected = self.PC_VT_INJECT_ENERGY*(1./30.)*(30 - np.exp(1.2 * view_cell.decay))
            # print(view_cell.decay)
            if energy_injected > 0:
                
                self.can_network[vc_x] += energy_injected
            
        # Local excitation            
        temp_gc_net = self.update_attractor_dynamics(excit_weights, excit_x_dim)
        self.can_network = temp_gc_net

        # Local inhibition
        temp_gc_net = self.update_attractor_dynamics(inhib_weights, inhib_x_dim)
        self.can_network -= temp_gc_net    

        
        # Global inhibition
        self.can_network[self.can_network < self.GLOBAL_INHIB] = 0
        self.can_network[self.can_network >= self.GLOBAL_INHIB] -= self.GLOBAL_INHIB

        # Normalization
        total = np.sum(self.can_network)      
        self.can_network = self.can_network/total
        self.can_network = self.can_network * (1.0 - vtrans) + np.roll(self.can_network, 1)*vtrans

        return self.can_network 
