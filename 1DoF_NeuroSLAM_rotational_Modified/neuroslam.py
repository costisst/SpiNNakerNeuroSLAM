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
from timeit import default_timer as timer
from numba import cuda
from numba import *
import cupy as cp



class NeuroSLAM(SLAM_basics):

    def __init__(self):
        
        
        super().__init__()
        self.grid_cells = GridCells()
        self.visual_odometry = VisualOdometry()
        self.view_cells = ViewCells()
        self.experience_map = ExperienceMap()
         
    def test(self):
        start1 = timer()
        
        colors = ['blue','red','green','black','purple']
        j = 0
        exp_index = []
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
        xs_final = []
        ys_final = []
        trigger = False
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
            
            # start2 = timer()
            view_cell = self.view_cells.active_cell(img, th)
            # dt = timer() - start2
            # print("active_cell %f" % dt)
            
            # print(view_cell.id)
            # print(view_cell.decay)
            
            # start3 = timer()
            odometry, vrot = self.visual_odometry.calculate_velocities(img)
            # dt = timer() - start3
            # print("calculate_velocities %f" % dt)
            
            if i == 1:               
                vrot = 0
            # print(vrot)
            self.grid_cells.can_network = self.grid_cells.network_iteration(view_cell, vrot)
            # print(self.grid_cells.can_network)
            th = self.grid_cells.find_activated()
            _ = self.experience_map.exp_map_iter(view_cell, vrot, th)
            
            # start = timer()
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

            # print(adjust_map)
            # if adjust_map:
            for experience in self.experience_map.exps:
                xs.append([np.cos(experience.facing_rad)])
                ys.append([np.sin(experience.facing_rad)])
                    # xs.append(np.degrees(experience.facing_rad))
                # self.xs_saved = xs
                # self.ys_saved = ys
            #     xs_final = xs
            #     ys_final = ys
            #     # del xs
            #     # del ys
            #     j += 1
            #     if j>4:
            #         j = 0
                
            # else:
                # self.xs_saved.append([np.cos(self.experience_map.exps[-1].facing_rad)])
                # self.ys_saved.append([np.sin(self.experience_map.exps[-1].facing_rad)])
            #     xs_final = self.xs_saved
            #     ys_final = self.ys_saved
            #     # j += 1
            #     # if j>4:
            #     #     j = 0

            # if i == 1:
            #     self.starting_x = np.cos(self.experience_map.current_exp.facing_rad)
            #     self.starting_y = np.sin(self.experience_map.current_exp.facing_rad)
            # else:
            #     print(np.cos(self.experience_map.current_exp.facing_rad) - self.starting_x)
            #     print(np.sin(self.experience_map.current_exp.facing_rad) - self.starting_y)
            #     if abs(np.cos(self.experience_map.current_exp.facing_rad) - self.starting_x) < 0.1  and abs(np.sin(self.experience_map.current_exp.facing_rad) - self.starting_y) < 0.1:
            #         print('LMAO')
            #         # trigger = True
            #         self.xs_saved = []
            #         self.ys_saved = []
            #         j += 1
            #         if j>4:
            #             j = 0

            # self.xs_saved.append([np.cos(self.experience_map.current_exp.facing_rad)])
            # self.ys_saved.append([np.sin(self.experience_map.current_exp.facing_rad)])
                
            # xs_final = self.xs_saved
            # ys_final = self.ys_saved
            
            # plt.plot(xs, 'bo')
            # plt.scatter(xs_final, ys_final, color =  colors[j])
            # plt.plot(xs, 'bo')
            plt.scatter(xs, ys)
            plt.tight_layout()
            plt.pause(0.1)
            # del xs_final
            # del ys_final

            # x = input()
            # if x == '':
            #     continue
            # else:
            #     break
            print('------')
        # dt = timer() - start
        # print("graphics %f" % dt)
        dt = timer() - start1
        print("total took %f" % dt)
            