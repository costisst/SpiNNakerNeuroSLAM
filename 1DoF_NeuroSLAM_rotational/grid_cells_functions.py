# -*- coding: utf-8 -*-
"""
Created on Sun May 24 16:31:41 2020

@author: cstef
"""

import math
from math import sqrt, pi, ceil, floor, exp
import numpy as np
from ratslam_functions import SLAM_basics


class GridCells(SLAM_basics):
    
    def __init__(self):
        
        super().__init__()
           
        self.decayor = []
        self.kkkk = 0
        # initialize grid cells network
        self.can_network = np.zeros(self.gc_th_dim)
        gc_th_center = floor(self.gc_th_dim/2)
        self.can_network[gc_th_center] = 1
        self.inhib_th_center = self.find_center(self.inhib_th_dim)
        self.excit_th_center = self.find_center(self.excit_th_dim)
        self.excit_weights = self.weight_init(self.excit_th_dim, self.excit_th_center, self.sigma)
        self.inhib_weights = self.weight_init(self.inhib_th_dim, self.inhib_th_center, self.sigma)
        
    def find_center(self, vector_len):
        
        vec_center = floor(vector_len/2)
        return vec_center
        
        
    def weight_init(self, th_dim, th_center, sigma):
        
        # create weights
        weights = np.zeros(th_dim)
    
        # initialize weights using 1D Gaussian Distribution
        for th in range(0,th_dim):
            weights[th] = (1/(sigma*sqrt(2*pi)))*exp((-1/2)*(th-th_center/sigma)**2)   
        
        # ensure that is normalized
        weights = weights / sum(weights)
        
        return weights 
    
    def update_attractor_dynamics(self, ad_weight, ad_dim):
        
        # create empty temp gc network
        gc_temp = np.zeros(self.gc_th_dim)
        half = floor(ad_dim/2)        
        wrapped = list(range(self.gc_th_dim-half, self.gc_th_dim)) + list(range(self.gc_th_dim)) + list(range(half))

        #retrieve non empty indices 
        indices = np.nonzero(self.can_network)

        for i in indices[0]:
            
            gc_temp[wrapped[i:i+ad_dim]] +=  self.can_network[i] * ad_weight
                       
        return gc_temp   
    
    
    
    def find_activated(self):
        
        th = np.argmax(self.can_network)
        return(th)
      
    def network_iteration(self, view_cell, vrot):
        
        # If it is a new view_cell do not inject energy 
        if view_cell.is_new == False:
            self.kkkk += 1
            # print(self.can_network) 
            vc_th = view_cell.th
            # energy_injected = self.PC_VT_INJECT_ENERGY*(1./30.)*(30 - np.exp(1.2 * view_cell.decay))
            # energy_injected = self.PC_VT_INJECT_ENERGY * view_cell.decay
            energy_injected = self.PC_VT_INJECT_ENERGY *(1 / (1 + math.exp(-view_cell.decay)))
            # print(view_cell.id)
            # print(view_cell.decay)
            # print(energy_injected)
            # print(self.kkkk)
            # print(self.can_network)
            # self.decayor.append([view_cell.decay,energy_injected])
            # np.savetxt("decayor1.csv", self.decayor, delimiter=",")
            if energy_injected > 0:
                # print(energy_injected)
                self.can_network[vc_th] += energy_injected
                # print(self.can_network)
            # print(self.can_network)    
            
        # print(self.can_network)
        # Local excitation            
        temp_gc_net = self.update_attractor_dynamics(self.excit_weights, self.excit_th_dim)
        self.can_network = temp_gc_net

        # Local inhibition
        temp_gc_net = self.update_attractor_dynamics(self.inhib_weights, self.inhib_th_dim)
        self.can_network -= temp_gc_net    

        
        # Global inhibition
        self.can_network[self.can_network < self.GLOBAL_INHIB] = 0
        self.can_network[self.can_network >= self.GLOBAL_INHIB] -= self.GLOBAL_INHIB

        # Normalization
        total = np.sum(self.can_network)      
        self.can_network = self.can_network/total
        # print(self.can_network)
        
        # Path Integration - Theta
        # Shift the pose cells +/- theta given by vrot
        if vrot != 0: 
            weight = (np.abs(vrot)/self.PC_C_SIZE_TH)%1
            # print(weight)
            if weight == 0:
                weight = 1.0

            shift1 = int(np.sign(vrot) * int(np.floor(abs(vrot)/self.PC_C_SIZE_TH)))
            shift2 = int(np.sign(vrot) * int(np.ceil(abs(vrot)/self.PC_C_SIZE_TH)))
            self.can_network = np.roll(self.can_network, shift1) * (1.0 - weight) + np.roll(self.can_network, shift2) * (weight)
            # print(np.roll(self.can_network, shift1) * (1.0 - weight))
            # print(np.roll(self.can_network, shift2) * (weight))
        # self.can_network = self.can_network * (1.0 - vtrans) + np.roll(self.can_network, 1)*vtrans          
        # print(self.can_network)
            # print(self.can_network)       


            
        return self.can_network 