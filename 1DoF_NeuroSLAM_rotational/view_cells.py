from ratslam_functions import SLAM_basics
import numpy as np


class ViewCell(SLAM_basics):
    """
    A class used to represent a view cell

    ...

    Attributes
    ----------
    id : int
        the id number of the view cell
    th : int
        the theta value associated with the view cell
    profile : str
        a pixel intensity vector of the input image frame (visual template)
    certainty : float
        represents how certain is the view cell about the visual template it stores
    is_new : boolean
        true if the view cell was created in this iteration
    exps : list
        a list that contains all the associated experiences with this view cell
        
    """
    def __init__(self, gc_th, profile, vc_id):
        super().__init__()
        self.id = vc_id
        self.th = gc_th
        self.profile = profile
        self.certainty = self.VC_CERTAINTY_VALUE
        self.is_new = True
        self.exps = []
    
class ViewCells(SLAM_basics):

    """
    A class used to represent all view cells and their methods

    ...

    Attributes
    ----------
    local_view_cells : list
        a list that stores all the created view cells
    vc_weight_matrix : numpy array
        an array that stores all the certainty of each view cells and their possible theta positions
    vc_id : int
        a counter for the number of view cells


    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes
    """
    
    def __init__(self):
        super().__init__()
        self.local_view_cells = []
        self.vc_weight_matrix = np.zeros(self.gc_th_dim)
        self.vc_id = 0
    
    def compare_profiles(self,img_prof1,img_prof2, slen):
        return super().compare_profiles(img_prof1,img_prof2, slen)
    
    def create_image_profile(self, image):
        return super().create_image_profile(image)
               
    def create_view_cell(self,gc_th, img_profile, vc_id):
        cell = ViewCell(gc_th, img_profile, vc_id)
        self.local_view_cells.append(cell)
        return cell
    
    def calculate_score(self, img_profile):
        scores_array = []
        img_prof1 = img_profile
        for view_cell in self.local_view_cells:
            img_prof2 = view_cell.profile
            # view_cell.certainty -= self.GLOBAL_VC_certainty
            # if view_cell.certainty < 0:
            #     view_cell.certainty = 0
            _ , min_diff = self.compare_profiles(img_prof1,img_prof2, self.VT_SHIFT_MATCH)
            scores_array.append(min_diff) 
        return scores_array
    
    # active_view_cell
    def active_cell(self, img, gc_th):
        img_profile = self.create_image_profile(img)
        scores = self.calculate_score(img_profile)
        scores = np.array(scores)
        
        if scores.size > 0:
            matches = scores[scores < self.VT_MATCH_THRESHOLD]
            
            if matches.size > 0:
                index = np.where(scores == np.min(matches))
                view_cell = self.local_view_cells[int(index[0][0])]
                view_cell.is_new = False
            else:
                # New view cell if the threshold is not triggered
                self.vc_id += 1
                view_cell = self.create_view_cell(gc_th, img_profile, self.vc_id)
                new_array = np.full((1, self.gc_th_dim), -4.5)
                self.vc_weight_matrix = np.vstack([self.vc_weight_matrix, new_array])
                self.vc_weight_matrix[self.vc_id, gc_th] = 0
        else:
            # Create the first view cell
            self.vc_id += 1
            view_cell = self.create_view_cell(gc_th, img_profile, self.vc_id)
            new_array = np.full((1, self.gc_th_dim), -4.5)
            self.vc_weight_matrix = np.vstack([self.vc_weight_matrix, new_array])
            self.vc_weight_matrix[self.vc_id, gc_th] = 0      
        return view_cell,view_cell.id,view_cell.th,view_cell.certainty
    
    def STDP(self, view_cell, gc_th):    
        dif = abs(view_cell.th - gc_th)
        if dif > 18:
            dif = abs(36 - dif)
        
        rate = dif/18
        
        if dif == 0:
            dif = 1                   
            self.vc_weight_matrix[view_cell.id, gc_th] += self.lr * dif
        else:
            dif == 1
            self.vc_weight_matrix[view_cell.id, view_cell.th ] -= self.lr * dif * (1 +  rate)
            self.vc_weight_matrix[view_cell.id, gc_th] += self.lr * dif * (1 +  rate)
        
        max_val_index = np.where(self.vc_weight_matrix[view_cell.id] == np.max(self.vc_weight_matrix[view_cell.id]))
        gc_th_max = max_val_index[0][0]
        view_cell.certainty = self.vc_weight_matrix[view_cell.id, gc_th_max]
        view_cell.th = gc_th_max
        view_cell.is_new = False
        return np.array(self.vc_weight_matrix)
        