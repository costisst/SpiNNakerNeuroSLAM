# -*- coding: utf-8 -*-
"""
Created on Sat May 30 23:52:54 2020

@author: cstef
"""
import contextlib
import os

devnull = open(os.devnull, 'w')
contextlib.redirect_stderr(devnull)

import numpy as np
import cv2 
import os
from os import listdir
from os.path import isfile, join
from matplotlib import pyplot as plt
from math import sqrt, pi, ceil, floor, exp
from timeit import default_timer as timer
try:
    import pyNN.spiNNaker as sim
except Exception:
    import spynnaker8 as sim
import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt
import glob
import pandas as pd
# from numba import cuda
# from numba import *
# import cupy as cp

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

#print(the_frames)


class NeuroSLAM(SLAM_basics):

    def __init__(self):
        
        
        super().__init__()
        self.grid_cells = GridCells()
        self.visual_odometry = VisualOdometry()
        self.view_cells = ViewCells()
        self.experience_map = ExperienceMap()
        
         
    def test(self):

        self.index  = 0
        self.injector_label = "injector"
        self.injector_label_vc = "injector_vc"
        self.injector_label_vcgc = "injector_vc_gc"
        self.pop_label = "pop"
        self.vc_label = "vc"
        # SPINNAKER
        sim.setup(timestep=0.1)
        
        # Keep track of the label of the injector as this needs to match up in several places
        
        self.gc_network_snn = sim.Population(self.gc_th_dim, sim.IF_curr_exp(), label=self.pop_label)
        the_input = sim.Population(1, sim.SpikeSourceArray(spike_times=[0]), label="input")
        sim.Projection(the_input, self.gc_network_snn, sim.FromListConnector([[0,5]]),synapse_type=sim.StaticSynapse(weight=5, delay=1))
        # Create the connection, noting that the label will be a "sender".
        self.connection = sim.external_devices.SpynnakerLiveSpikesConnection(local_port=None, send_labels=[self.injector_label, self.injector_label_vc, self.injector_label_vcgc],receive_labels=[self.pop_label, self.vc_label])
        #self.connection_vc = sim.external_devices.SpynnakerLiveSpikesConnection(local_port=None, send_labels=[self.injector_label_vc],receive_labels=[self.vc_label])
    #asdasd
        
        #the networks
        self.view_cell_net = sim.Population(36, sim.IF_curr_exp(),label=self.vc_label)
        # stdp
        timing_rule = sim.SpikePairRule(tau_plus=50.0, tau_minus=50.0, A_plus=1, A_minus=0.5)
        weight_rule = sim.AdditiveWeightDependence(w_max=5.0, w_min=0.0)
        stdp_model = sim.STDPMechanism(timing_dependence=timing_rule, weight_dependence=weight_rule, weight=0.0, delay=10.0)
        stdp_projection = sim.Projection(self.view_cell_net, self.gc_network_snn, sim.OneToOneConnector(), synapse_type=stdp_model)


        # Add a callback to be called at the start of the simulation
        self.connection.add_start_resume_callback(self.injector_label, self.send_spikes)
        #self.connection.add_start_resume_callback(self.injector_label_vc, self.send_spikes1)
        #self.connection.add_start_resume_callback(self.injector_label_vcgc, self.send_spikes2)
        #self.connection.add_receive_callback(self.pop_label, self.receive_spikes)
        #self.connection_vc.add_start_resume_callback(self.injector_label_vc, self.send_spikes1)
        self.connection.add_receive_callback(self.vc_label, self.receive_spikes1)
        self.connection.add_receive_callback(self.pop_label, self.receive_spikes)
        # Set up the injector population with 5 neurons, 
        # simultaneously registering the connection as a listener
        self.injector = sim.Population(self.gc_th_dim, sim.external_devices.SpikeInjector(database_notify_port_num = self.connection.local_port),label=self.injector_label)
        self.injector_vc = sim.Population(self.gc_th_dim, sim.external_devices.SpikeInjector(database_notify_port_num = self.connection.local_port),label=self.injector_label_vc)
        self.injector_vcgc = sim.Population(self.gc_th_dim, sim.external_devices.SpikeInjector(database_notify_port_num = self.connection.local_port),label=self.injector_label_vcgc)
       
   
        self.can_prev = np.zeros(self.gc_th_dim)
        # initialize grid cells network
        self.can_network_saved = np.zeros(self.gc_th_dim)


              
        con_list = []
        for i in range(0,36):
            
            b = i
            if b == 0:
                a = 35
                c = 1
            elif b==35:
                a = 34
                c = 0
            else:
                a = b - 1
                c = b + 1
            con_list.append((a,b))
            con_list.append((b,c))
            con_list.append((b,a))
            con_list.append((c,b))
        
        # con_list = set(con_list)
        sim.Projection(self.gc_network_snn, self.gc_network_snn, sim.FromListConnector(con_list), sim.StaticSynapse(weight=0.5, delay=5), receptor_type="excitatory")
        sim.Projection(self.gc_network_snn, self.gc_network_snn, sim.FromListConnector(con_list), sim.StaticSynapse(weight=-1.2, delay=5), receptor_type="inhibitory")
        
        sim.external_devices.activate_live_output_for( self.gc_network_snn, database_notify_port_num=self.connection.local_port)
        sim.external_devices.activate_live_output_for( self.view_cell_net, database_notify_port_num=self.connection.local_port)

        self.gc_network_snn.record(["spikes", "v"])
        self.view_cell_net.record(["spikes", "v"])
        # Connect the injector to the population
        sim.Projection(self.injector, self.gc_network_snn, sim.OneToOneConnector(), sim.StaticSynapse(weight = 1))
        sim.Projection(self.injector_vcgc, self.gc_network_snn, sim.OneToOneConnector(), sim.StaticSynapse(weight = 1))
        sim.Projection(self.injector_vc, self.view_cell_net, sim.OneToOneConnector(), sim.StaticSynapse(weight = 5))
       # sim.Projection(self.injector_vc, self.gc_network_snn, sim.OneToOneConnector(), sim.StaticSynapse(weight = 5))
        weights = stdp_projection.get(["weight"], "list")
        neo = self.gc_network_snn.get_data(variables=["spikes", "v"])
        spikes = neo.segments[0].spiketrains
        v = neo.segments[0].filter(name='v')[0]
        sim.run(100)
        simtime = 0
        # Run the simulation and get the spikes out
        

        start1 = timer()
        
        colors = ['blue','red','green','black','purple']
        j = 0
        exp_index = []
        # data_path = 'D:/Msc_AI_UoM/Msc Project/NeuroSLAM/101215_153851_MultiCamera0'
        the_frames = []
        ta_panta = []
        for j in range(0,3):
            print("03.NeuroroboticsPlatform/dataset_images" + str(j) +str(j)+ "/*.png")
            the_frames = [f for f in glob.glob("03.NeuroroboticsPlatform/dataset_images" + str(j)+str(j) + "/*.png")]
            ta_panta += the_frames
        #ta_panta.sort(key=natural_keys)
        
        asdf = []
        with open('03.NeuroroboticsPlatform/ta_panta_ola1.txt') as f:
            lines = [line.rstrip() for line in f]
            asdf.append(lines)
        with open('03.NeuroroboticsPlatform/ta_panta_ola2.txt') as f:
            lines = [line.rstrip() for line in f]
            asdf.append(lines)
       # with open('03.NeuroroboticsPlatform/ta_panta_ola_1o_snn2.txt') as f:
        #    lines = [line.rstrip() for line in f]
         #   asdf.append(lines)
         
        ff = []
        for line in asdf[0]:
            k1,k2 = line.split(',')    
            ff.append([k1,float(k2)])
            
        for line in asdf[1]:
            k1,k2 = line.split(',')    
            ff.append([k1,float(k2)])   

        #for line in asdf[2]:
        #    k1,k2 = line.split(',')    
        #    ff.append([k1,float(k2)])   

        my_csv = pd.DataFrame(ff, columns =["time", "orientation.z"])
        all_stamps = []
        for frame in ta_panta:
            temp = frame.split('/')[2]
            all_stamps.append(temp.split('.')[0])
        #my_csv = my_csv[my_csv['time'].isin(all_stamps)]
        
        my_csv = my_csv.loc[my_csv['time'].isin(all_stamps)]
        #my_csv = my_csv.loc[my_csv['time'].isin(all_stamps)].reset_index()
        kk = my_csv['time'].to_list()
        #kk = [ int(x) for x in kk ]
        #all_stamps = [ int(x) for x in all_stamps ]
        asdff = list(set(all_stamps) - set(kk))
        rrrr = []
        for item in asdff:
            t1 = str(item) + ".png"
            rrrr.append(t1)
        ta_panta = [x for x in ta_panta if x.split('/')[2] not in rrrr]
        all_stamps = []
        for frame in ta_panta:
            temp = frame.split('/')[2]
            all_stamps.append(temp.split('.')[0])
        #my_csv = my_csv[my_csv['time'].isin(all_stamps)]
        
        my_csv = my_csv.loc[my_csv['time'].isin(all_stamps)]
        #print(len(ta_panta))
       #print(len(all_stamps))
        my_csv = my_csv.reset_index()
        temp_csv = my_csv.copy()
        #print(len(my_csv))
        #my_csv['orientation.z'] = my_csv['orientation.z']*180
        my_csv['orientation.z'] = np.degrees(my_csv['orientation.z'])
        my_csv['orientation.z'] = np.ceil(my_csv['orientation.z'])
        my_csv['orientation.z'] = my_csv['orientation.z']/10
        my_csv['orientation.z'] = np.ceil(my_csv['orientation.z'])
        my_csv['orientation.z'] = (my_csv['orientation.z'])*10
        del my_csv['index']
        angles = []
         
        #all_info_2o_peirama_50.xlsx
        the_csv = pd.read_csv('03.NeuroroboticsPlatform/correct_times.csv', header = None )
        the_csv = the_csv.values.tolist()
        the_list = []
        fff = []
        
        for item in the_csv:
            item = item[0]
            #print(item)
            k = item.split("/")[2]
            #temp = "03.NeuroroboticsPlatform/dataset_images" + str(0) + str(0) + "/"  + k
            #temp1 = "03.NeuroroboticsPlatform/dataset_images" + str(1) + str(1) + "/"  + k
            #temp2 = "03.NeuroroboticsPlatform/dataset_images" + str(2) + str(2) + "/"  + k
            temp = "03.NeuroroboticsPlatform/dataset_images" + str(0) + str(0) + "/"   + k
            temp1 = "03.NeuroroboticsPlatform/dataset_images" + str(1) + str(1) + "/"  + k
            temp2 = "03.NeuroroboticsPlatform/dataset_images" + str(2) + str(2) + "/"  + k

            if os.path.isfile(temp):
                string = temp

            if os.path.isfile(temp1):
                string = temp1

            if os.path.isfile(temp2):
                string = temp2
            fff.append(string)
            
        the_csv = fff    
        for name in the_csv:
            f = name.split('/')[2]
            the_list.append(f.split('.')[0])
        
        the_list_final = []
        the_list_final_comp = []
        for item in the_list:
            g = 'dataset_images_50/' + str(item) + '.png'
            if item in all_stamps and item not in the_list_final:
                #the_list_final.append('dataset_images_50/' + str(item) + '.png')
                the_list_final.append(str(item))
                the_list_final_comp.append(str(item) + '.png')
                
        the_list_final = [x for x in the_list_final if x in all_stamps]
        degrees = {}

            
        deg = [0,-10,-20,-30,-40,-50,-60,-70,-80,-90,-100,-110,-120,-130,-140,-150,-160,-170,180,170,160,150,140,130,120,110,100,90,80,70,60,50,40,30,20,10]
        degrees = {}
        for i,d in enumerate(deg):
            degrees[i] = d
        #print(qwer)
        #the_frames.reverse()
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
        collect_error = []
        collect_rate_error = []
        total_collect_error = [] 
        total_collect_rate_error = []
        collect_positions_th = []
        total_collect_positions_th =  []
        for frame in the_list_final:
            i += 1
        # for i in range(10000):     
            #if i == 10 : break    
            if circles_found == 5:
                 print(i)
                 break
            the_input = frame
            # print(the_input)
            # _, frame = video.read()
            # frame = frame.astype('uint8')
            #temp = "03.NeuroroboticsPlatform/dataset_images" + str(0) + str(0) + "/"  + str(frame) + ".png"
            #temp1 = "03.NeuroroboticsPlatform/dataset_images" + str(1) + str(1) + "/"  + str(frame) + ".png"
            #temp2 = "03.NeuroroboticsPlatform/dataset_images" + str(2) + str(2) + "/"  + str(frame) + ".png
            temp = "03.NeuroroboticsPlatform/dataset_images" + str(0) +  str(0) +"/"  + str(frame) + ".png"
            temp1 = "03.NeuroroboticsPlatform/dataset_images" + str(1) + str(1) + "/"  + str(frame) + ".png"
            #temp2 = "03.NeuroroboticsPlatform/2o_peirama" + str(2) +  "/"  + str(frame) + ".png"
            if os.path.isfile(temp):
                frame = cv2.imread(temp)
                frame = frame.astype('uint8')
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                print(temp)
            if os.path.isfile(temp1):
                frame = cv2.imread(temp1)
                frame = frame.astype('uint8')
                img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                print(temp1)
           # if os.path.isfile(temp2):
                #frame = cv2.imread(temp2)
                #frame = frame.astype('uint8')
                #img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                #print(temp2)
                
            
            self.index  += 1
            # print(self.gc_spike_id)
            th = self.gc_spike_id
            # print(th)
            
            # start2 = timer()
            view_cell,view_cell.id,view_cell.th,decay = self.view_cells.active_cell(img, th)
            #print(view_cell.id)
            # dt = timer() - start2
            # print("active_cell %f" % dt)
            
            # print(view_cell.id)
            # print(view_cell.decay)
            
            # start3 = timer()
            odometry, vrot = self.visual_odometry.calculate_velocities(img)
            # print(odometry, vrot)
            # dt = timer() - start3
            # print("calculate_velocities %f" % dt)
            
            if i == 1:               
                vrot = 0
            # print(vrot)
            self.vc_weight, self.pi_weight, self.vc_selected_gc, self.trigger_vc, energy_injected = self.grid_cells.network_iteration(view_cell, vrot, th)
            # print(self.pi_weight, self.vc_weight)
            # self.gc_spike_id = th
            start2 = timer()
            sim.run(100)
            dt = timer() - start2
            print("total took %f" % dt)
            simtime += dt
            # print(self.grid_cells.can_network)
            # th = self.grid_cells.find_activated()
            # th = self.gc_spike_id
            if self.all_spikes:
                self.gc_spike_id = mean(self.all_spikes)
                if self.gc_spike_id%1 < 0.5:
                    self.gc_spike_id = floor(self.gc_spike_id)
                else:
                    self.gc_spike_id = floor(self.gc_spike_id)
            th = self.gc_spike_id
        
            _ = self.experience_map.exp_map_iter(view_cell, vrot, th)
            
            vc_matrix = self.view_cells.STDP(view_cell, th)
            #print('Rotational: ' + str(vrot))
            #print('Decay: ' + str(view_cell.decay))
            #print('ID: ' + str(view_cell.id))
            #print('Is new: ' + str(view_cell.is_new))
            
            # # start = timer()
            #print('Theta: ' + str(th))
            #print('NS Z: ' + str(degrees[th]))
            #print('GT Z: ' + str(my_csv['orientation.z'].loc[i]))
            #print('SPIKE ID: ' + str(self.gc_spike_id))
            #asdf += angles[i]
            the_index = my_csv[my_csv['time'] == the_input].index.values
            #print(the_input)
            #print(the_index)
            yy = 0
            if len(the_index) > 1:
                if abs(the_index[0] - previous_index) <= abs(the_index[1] - previous_index):
                    
                    sel_idx = the_index[0] 
                else:
                    sel_idx = the_index[1] 
            else:
                sel_idx = the_index[0]
                
            #print("Index: " + str(sel_idx))
            result = None
            n1 = int(degrees[th])
            n2 = int(my_csv['orientation.z'].loc[sel_idx])
            
            if np.sign(n1) == np.sign(n2):
                result = abs(n1 - n2)
            else:
                result = min(180 - abs(n1), abs(n1) - 0) + min(180 - abs(n2), abs(n2) - 0)
            # print('Error: ' + str(result))
            all_errors.append(result)
            sel_th = th
            previous_index = sel_idx
            # if abs(degrees[str(th)] - my_csv['orientation.z'].loc[i]) > abs(inv_degrees[str(th)] - my_csv['orientation.z'].loc[i]):
            #     sel_th = inv_degrees[str(th)]
            # else:
            #     sel_th = degrees[str(th)]
            couples1.append([vrot, self.pi_weight, view_cell.decay, energy_injected ])
            couples.append([result,th,degrees[th],my_csv['orientation.z'].loc[sel_idx]])

            total_error += float(str(result))
            the_dif.append([result,int(degrees[th]),int(my_csv['orientation.z'].loc[sel_idx])])
            piou.append(sel_idx)
            vrot_orient.append([vrot, self.pi_weight, int(degrees[th]), int(my_csv['orientation.z'].loc[sel_idx])])
            # th = self.gc_spike_id    
            # self.gc_th_dim = th
            # start = timer()
            # print('&&&&&&&&&&&&&&&&&&&&')
            # print('&&&&&&&&&&&&&&&&&&&&')
            # print('&&&&&&&&&&&&&&&&&&&&')
            #print('&&&&&&&&&&&&&&&&&&&&')
            #print(vrot)
            #print(self.PC_C_SIZE_TH)
            #print(self.pi_weight, self.vc_weight, self.trigger_vc)
            #print(self.pi_weight_index, self.vc_weight_index)

            #print(self.gc_spike_id)
            collect_error.append(result)
            collect_rate_error.append(total_error/i)
            collect_positions_th.append(th)
            current_orientation = temp_csv['orientation.z'].loc[sel_idx]
            #co.append([current_orientation, starting_orientation])
            # print(current_orientation, starting_orientation)
            if abs(current_orientation - starting_orientation) < 0.06 and abs(starting_i - i) > 70:
                circles_found +=1 
                starting_orientation = current_orientation
                starting_i = i
                total_collect_error.append(collect_error)
                # total_collect_rate_error.append(collect_rate_error)
                total_collect_positions_th.append(collect_positions_th)
                
                collect_error = []
                collect_positions_th = []
            #all_pos.append(self.gc_spike_id)
            self.all_spikes = []
            # print('\\\\\\\\\\\\\\\\')
            # print(len(self.experience_map.exps))
            # sam = 0
            # for kk in range(len(self.view_cells.local_view_cells)):
            #     # print(self.view_cells)
            #     sam +=  len(self.view_cells.local_view_cells[kk].exps)
            # print(sam)
            # print('\\\\\\\\\\\\\\\\')
            # odo += vtrans
            #raw.append(odometry)
            #rawxs.append([np.cos(odometry)])
            #rawys.append([np.sin(odometry)])
            # plt each 50 frames
            # if i%50 != 0:
            #     continue
            # plt.clf()
            print(i)
            
            #print('&&&&&&&&&&&&&&&&&&&&')

           
            print('------')

            # plot.

            
        # dt = timer() - start
        # print("graphics %f" % dt)
        
        sim.end()
        print(weights)
        print(simtime)

        dt = timer() - start1
        print("total took %f" % dt)
        return couples,couples1, total_error/i, all_errors,vrot_orient, circles_found ,total_collect_error,collect_rate_error,total_collect_positions_th
