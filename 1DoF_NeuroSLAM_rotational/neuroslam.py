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
        
        data = 'D:/Msc_AI_UoM/Msc Project/NeuroSLAM/rotate_around_self.mp4'
        video = cv2.VideoCapture(data)
        
        # data = [f for f in listdir(data_path) if isfile(join(data_path, f))]
        odometry = 0   
        raw = []
        rawxs = []
        rawys = []
        xs = []
        ys = []
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
        the_frames = []
        while True:
            i += 1      
            
            # if i > 90:
            _, frame = video.read()
            the_frames.append(frame)
            
            if i == 1000: break
        
            if i%50 != 0:
                continue
            print(i)
            
        i = 0
        # the_frames.reverse()
        for frame in the_frames:
            # if i == 50 : break    
            
            i += 1
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            th = self.grid_cells.find_activated()
            # print(th)
            view_cell = self.view_cells.active_cell(img, th)
            # print(view_cell.id)
            # print(view_cell.decay)
            odometry, vrot = self.visual_odometry.calculate_velocities(img)
            if i == 1:               
                vrot = 0
            # print(vrot)
            self.grid_cells.can_network = self.grid_cells.network_iteration(view_cell, vrot)
            # print(self.grid_cells.can_network)
            th = self.grid_cells.find_activated()
            self.experience_map.exp_map_iter(view_cell, vrot, th)

            print(th)
            # print('\\\\\\\\\\\\\\\\')
            # print(len(self.experience_map.exps))
            # sam = 0
            # for kk in range(len(self.view_cells.local_view_cells)):
            #     # print(self.view_cells)
            #     sam +=  len(self.view_cells.local_view_cells[kk].exps)
            # print(sam)
            # print('\\\\\\\\\\\\\\\\')
            # odo += vtrans
            raw.append(odometry)
            rawxs.append([np.cos(odometry)])
            rawys.append([np.sin(odometry)])
            # plt each 50 frames
            # if i%50 != 0:
            #     continue
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
            # plt.plot(raw)
            plt.scatter(rawxs, rawys)
            #------------------------------
            
            # POSE CELL ACTIVATION --------
            ax = plt.subplot(2, 2, 3)
            plt.title('POSE CELL ACTIVATION')
            # ax.plot(x, self.gc_x_dim ,'x')
            ax.hist(th,self.gc_th_dim,[0,self.gc_th_dim])
            
            # EXPERIENCE MAP --------------
            plt.subplot(2, 2, 4)
            plt.title('EXPERIENCE MAP')

            for experience in self.experience_map.exps:
                xs.append([np.cos(experience.facing_rad)])
                ys.append([np.sin(experience.facing_rad)])
                # xs.append(np.degrees(experience.facing_rad))
            # print(xs)
            print('------')
            # plt.plot(xs, 'bo')
            plt.scatter(xs, ys)

            plt.tight_layout()
            plt.pause(0.1)

            # x = input()
            # if x == '':
            #     continue
            # else:
            #     break
