# -*- coding: utf-8 -*-
"""
Created on Sat May 30 23:52:54 2020

@author: cstef
"""
from ratslam_functions import SLAM_basics
from grid_cells_functions import GridCells
from experience_map_functions import ExperienceMap
from visual_odometry_functions import VisualOdometry
from view_cells import ViewCells
import numpy as np
import cv2 
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from math import sqrt, pi, ceil, floor, exp

class NeuroSLAM(SLAM_basics):
    
    def __init__(self):
        
        
        super().__init__()
        self.grid_cells = GridCells()
        self.visual_odometry = VisualOdometry()
        self.view_cells = ViewCells()
        self.experience_map = ExperienceMap()
        
    def test(self):
        
        data_path = 'D:/Msc_AI_UoM/Msc Project/NeuroSLAM/101215_153851_MultiCamera0'
        
        data = 'D:/Msc_AI_UoM/Msc Project/NeuroSLAM/gold_coast_compressed.mp4'
        video = cv2.VideoCapture(data)
        
        # data = [f for f in listdir(data_path) if isfile(join(data_path, f))]
        odo = 0   
        raw = []
        # for i in range(len(data)-1):
        #     image_path = str.join('',[data_path,'/',data[i]])   
        #     image = cv2.imread(image_path,-1)
        #     img = np.array(image)
            
        #     # print(data[i])
            # x = self.grid_cells.find_activated()
            # # print(x)
            # view_cell = self.view_cells.active_cell(img, x)
            # # print(view_cell.id)
            # # print(view_cell.decay)
            # vtrans = self.visual_odometry.calculate_velocities(img)
            # # print(vtrans)
            # self.grid_cells.can_network = self.grid_cells.network_iteration(self.grid_cells.excit_weights, self.grid_cells.excit_x_dim, self.grid_cells.inhib_weights, self.grid_cells.inhib_x_dim, view_cell, vtrans)
            # # print(self.grid_cells.can_network)
            # x = self.grid_cells.find_activated()
            # self.experience_map.exp_map_iter(view_cell, vtrans, x)
        i = 0
        while True:
            
            _, frame = video.read()
            if frame is None: break
            # if i == 50 : break    
        
            i += 1
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            x = self.grid_cells.find_activated()
            # print(x)
            view_cell = self.view_cells.active_cell(img, x)
            # print(view_cell.id)
            # print(view_cell.decay)
            vtrans = self.visual_odometry.calculate_velocities(img)
            # print(vtrans)
            self.grid_cells.can_network = self.grid_cells.network_iteration(self.grid_cells.excit_weights, self.grid_cells.excit_x_dim, self.grid_cells.inhib_weights, self.grid_cells.inhib_x_dim, view_cell, vtrans)
            # print(self.grid_cells.can_network)
            x = self.grid_cells.find_activated()
            self.experience_map.exp_map_iter(view_cell, vtrans, x)
            
            odo += vtrans
            raw.append(odo)
            # plt each 50 frames
            if i%50 != 0:
                continue
            # plt.clf()
            print(i)
            # RAW IMAGE -------------------
            ax = plt.subplot(2, 2, 1)
            plt.title('RAW IMAGE')
            plt.imshow(img, cmap= 'gray')
            ax.get_xaxis().set_ticks([])
            ax.get_yaxis().set_ticks([])
            # -----------------------------
            
            # RAW ODOMETRY ----------------
            
            # print(odo)
            plt.subplot(2, 2, 2)
            plt.title('RAW ODOMETRY')
            plt.plot(raw)
            #------------------------------
            
            # POSE CELL ACTIVATION --------
            ax = plt.subplot(2, 2, 3)
            plt.title('POSE CELL ACTIVATION')
            # ax.plot(x, self.gc_x_dim ,'x')
            ax.hist(x,self.gc_x_dim,[0,self.gc_x_dim])
            
            # EXPERIENCE MAP --------------
            plt.subplot(2, 2, 4)
            plt.title('EXPERIENCE MAP')
            xs = []
            for experience in self.experience_map.exps:
                xs.append(experience.exp_x/50)
            

            plt.plot(xs, 'bo')

            plt.tight_layout()
            plt.pause(0.1)

        
