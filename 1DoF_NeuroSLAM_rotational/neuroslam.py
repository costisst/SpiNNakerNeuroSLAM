# -*- coding: utf-8 -*-
"""
Created on Sat May 30 23:52:54 2020

@author: cstef
"""
#%matplotlib inline
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
import glob
import csv
import pandas as pd
import re


def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]



class NeuroSLAM(SLAM_basics):

    def __init__(self):
        super().__init__()
        self.grid_cells = GridCells()
        self.visual_odometry = VisualOdometry()
        self.view_cells = ViewCells()
        self.experience_map = ExperienceMap()
         
    def execute(self):
        start1 = timer()
        
        odometry = 0   
        raw = []
        rawxs = []
        rawys = []
        xs = []
        ys = []
        xs_final = []
        ys_final = []
        trigger = False
        all_info = []

        degrees = {}
        deg = [0,-10,-20,-30,-40,-50,-60,-70,-80,-90,-100,-110,-120,-130,-140,-150,-160,-170,180,170,160,150,140,130,120,110,100,90,80,70,60,50,40,30,20,10]
        degrees = {}
        for i,d in enumerate(deg):
            degrees[i] = d


        i = 0
        asdf = 0
        all_errors = []
        couples = []
        couples1 = []
        total_error = 0
        piou = []
        ta_panta.sort(key=natural_keys)
        the_dif = []
        vrot_orient = []
        previous_index = 0
        starting_orientation = temp_csv['orientation.z'].loc[0]
        circles_found = 0
        starting_i = 0
        
        co = []
        collect_error = []
        collect_rate_error = []
        total_collect_error = [] 
        total_collect_rate_error = []
        collect_positions_th = []
        total_collect_positions_th =  []
        for frame in the_list_final:
            i += 1
    
            # if circles_found == 5:
            #      print(i)
            #      break
             
            the_input = frame
            frame = cv2.imread('3o_peirama/images/' + str(frame) + '.png')
            frame = frame.astype('uint8')
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            th = self.grid_cells.find_activated()
            view_cell,view_cell.id,view_cell.th,decay = self.view_cells.active_cell(img, th)
            odometry, vrot = self.visual_odometry.calculate_velocities(img)
            if i == 1:               
                vrot = 0
            self.grid_cells.can_network,weight, is_new, energy_injected, vc_th = self.grid_cells.network_iteration(view_cell, vrot,th1)
            th = self.grid_cells.find_activated()
            _ = self.experience_map.exp_map_iter(view_cell, vrot, th)
            vc_matrix = self.view_cells.STDP(view_cell, th)
            
            the_index = my_csv[my_csv['time'] == the_input].index.values
            if len(the_index) > 1:
                if abs(the_index[0] - previous_index) <= abs(the_index[1] - previous_index):
                    sel_idx = the_index[0] 
                else:
                    sel_idx = the_index[1] 
            else:
                sel_idx = the_index[0]
                
            n1 = int(degrees[th])
            n2 = int(my_csv['orientation.z'].loc[sel_idx])
            
            if np.sign(n1) == np.sign(n2):
                result = abs(n1 - n2)
            else:
                result = min(180 - abs(n1), abs(n1) - 0) + min(180 - abs(n2), abs(n2) - 0)

            all_errors.append(result)
            sel_th = th
            previous_index = sel_idx
            total_error += result


            collect_error.append(result)
            collect_rate_error.append(total_error/i)
            collect_positions_th.append(th)
            current_orientation = temp_csv['orientation.z'].loc[sel_idx]
            co.append([current_orientation, starting_orientation])
            if abs(current_orientation - starting_orientation) < 0.06 and abs(starting_i - i) > 70:
                circles_found += 1 
                starting_orientation = current_orientation
                starting_i = i
                total_collect_error.append(collect_error)
                total_collect_positions_th.append(collect_positions_th)                
                collect_error = []
                collect_positions_th = []

            raw.append(odometry)
            rawxs.append([np.cos(odometry)])
            rawys.append([np.sin(odometry)])
            
            # if i%50 != 0:
            #     continue
                
            # plt.clf()
            # plt.imshow(img)
            # plt.plot(img)
            #RAW IMAGE -------------------
            # ax = plt.subplot(2, 2, 1)
            # plt.title('RAW IMAGE')
            # plt.imshow(img, cmap= 'gray')
            # plt.get_xaxis().set_ticks([])
            # plt.get_yaxis().set_ticks([])
            # -----------------------------
            
            # # RAW ODOMETRY ----------------
            
            # # print(odo)
            # plt.subplot(2, 2, 2)
            # plt.title('RAW ODOMETRY')
            # # plt.plot(raw)
            # plt.scatter(rawxs, rawys)
            # #------------------------------
            
            # # POSE CELL ACTIVATION --------
            # ax = plt.subplot(2, 2, 3)
            # plt.title('POSE CELL ACTIVATION')
            # # ax.plot(x, self.gc_x_dim ,'x')
            # ax.hist(th,self.gc_th_dim,[0,self.gc_th_dim])
            
            # # EXPERIENCE MAP --------------
            # plt.subplot(2, 2, 4)
            # plt.title('EXPERIENCE MAP')

            # # print(adjust_map)
            # # if adjust_map:
            # for experience in self.experience_map.exps:
            #     xs.append([np.cos(experience.facing_rad)])
            #     ys.append([np.sin(experience.facing_rad)])

            # plt.scatter(xs, ys)
            # plt.tight_layout()
            # plt.pause(0.1)
            # plt.show();
            # del xs_final
            # del ys_final
            
        dt = timer() - start1
        print("total took %f" % dt)
        return total_error/i, all_errors