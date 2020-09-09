import numpy as np
from math import sqrt, pi, ceil, floor, exp


def comparison(img_prof1, img_prof2, slen, y_size):
    min_offset = 0
    min_dif = 10e6
    end = y_size
    for i in range(slen):
        
        dif = np.abs(np.array(img_prof1[i: end]) - np.array(img_prof2[:(end-i)]))
        dif_norm = np.sum(dif)/(end-i)
        if dif_norm < min_dif:
            min_dif = dif_norm
            min_offset = i
        
        dif = np.abs(np.array(img_prof1[:(end-i)]) - np.array(img_prof2[i: end]))
        dif_norm = np.sum(dif)/(end-i)
        if dif_norm < min_dif:
            min_dif = dif_norm
            min_offset = -i
            
    return min_offset, min_dif
    

class SLAM_basics():
    
    def __init__(self):
        
        
        # Grid Cells 
        self.gc_th_dim = 36
        self.inhib_th_dim = 5
        self.excit_th_dim = 7
        self.PC_C_SIZE_TH = (2.*np.pi)/self.gc_th_dim
        self.sigma = 1
        self.y_size = 320
        self.PC_TH_SUM_SIN_LOOKUP = np.sin(np.multiply(np.arange(1, self.gc_th_dim+1), (2*np.pi)/self.gc_th_dim))
        self.PC_TH_SUM_COS_LOOKUP = np.cos(np.multiply(np.arange(1, self.gc_th_dim+1), (2*np.pi)/self.gc_th_dim))
        self.GC_PACKET_SIZE = 3  
        self.GLOBAL_INHIB = 0.00002
        self.VROT_SCALING_FACTOR = 12
        
        # View Cells
        self.PC_VT_INJECT_ENERGY = 0.1
        self.VC_DECAY_VALUE = 1
        self.GLOBAL_VC_DECAY= 0.1
        self.VT_SHIFT_MATCH = 20
        self.VTRANS_SCALE = 100
        self.VT_MATCH_THRESHOLD = 0.05
        self.SHIFT_LEN = 140
        self.lr = 0.1
        
        # Visual Odometry
        self.ODO_ROT_SCALING = np.pi/180./7.
        self.POSECELL_VTRANS_SCALING = 1./10. 
        self.MAX_TRANS_V_THRESHOLD = 1
        
        # Experience Map
        self.EXP_DELTA_PC_THRESHOLD = 3
        self.EXP_LOOPS = 100
        self.EXP_CORRECTION = 0.5
        
        # General
        self.starting_x = 0
        self.starting_y = 0
        self.xs_saved = []
        self.ys_saved = [] 

        
    def create_image_profile(self, image):
        x_sums = np.sum(image, 0)
        avint = np.sum(x_sums)/x_sums.size
        return x_sums/avint

    def compare_profiles(self, img_prof1, img_prof2, slen):
        y_size = self.y_size        
        return comparison(img_prof1, img_prof2, slen, y_size)

    def min_delta(self, d1, d2, max_val):
        delta = np.min(np.array([np.abs(d1-d2), max_val - np.abs(d1-d2)]))
        return delta

    def clip_rad_180(self, angle): 
        while angle > np.pi:
            angle -= 2*np.pi 
        while angle <= -np.pi:
            angle += 2*np.pi
            
        return angle
    
    def clip_rad_360(self, angle):      
        while angle < 0:
            angle += 2*np.pi  
        while angle >= 2*np.pi:
            angle -= 2*np.pi
            
        return angle
    
    def signed_delta_rad(self, angle1, angle2):
        direction = self.clip_rad_180(angle2 - angle1)        
        delta_angle = abs(self.clip_rad_360(angle1) - self.clip_rad_360(angle2))
        
        if (delta_angle < (2*np.pi-delta_angle)):
            if direction > 0:
                angle = delta_angle
            else:
                angle = -delta_angle       
        else: 
            if direction > 0:
                angle = 2*np.pi - delta_angle  
            else:
                angle = -(2*np.pi-delta_angle)
                
        return angle