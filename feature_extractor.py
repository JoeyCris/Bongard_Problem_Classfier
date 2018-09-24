# coding: utf-8

# In[1]:


'''
author: Joey, Xinyu Yun

Des: create a class to load feature maps and vectorize all layers feature maps
'''
import os
import numpy as np
import pickle

#set the location where the feature map files are
FEATURE_MAP_PKL_PATH = "/notebooks/xinyu_testing/"

class FeatureMapExtractor:
    def __init__(self, path=FEATURE_MAP_PKL_PATH, layers = ['conv9', 'conv8', 'conv7', 'conv6', 'conv5'], debug=False):
        self.path = path
        self.real_bongard_problems = dict()
        self.manual_bongard_problems = dict()
        self.layer_name_list = layers
     
        self._debug = debug
        self._extract_bongard_feature_vec() #(self, type)
    
    def get_encodings(self, problem_num, side, need_manual=True):
        returned_encodings = []
        assert side in ["left", "right"], "ERROR: must specify left or right side"
        assert problem_num in self.real_bongard_problems, "ERROR: problem number {} not found".format(problem_num)
        if side == 'left':
            returned_encodings.extend(list(self.real_bongard_problems[problem_num][0:6]))
        else:
            returned_encodings.extend(list(self.real_bongard_problems[problem_num][6:12]))
        if need_manual:
            assert problem_num in self.manual_bongard_problems, "ERROR: problem number {} not found in manual data".format(problem_num)
            returned_encodings.append(np.array(self.manual_bongard_problems[problem_num][side]))
            
        return returned_encodings
            
        
            
        
    '''
    compute mean values from middle conv filters and return all problems feature map(231*12*N) N is the filters count  
    '''
    def get_vector_feature_map(self, feature_map_conv):
        features_all_problems = []
        for problem_count in range(len(feature_map_conv)):
            features_per_problem = [] #12*filter count

            for feature_map in feature_map_conv[problem_count]['features']:
                channels = np.array(feature_map).shape[2]   
                mean_feature_vec = []
                for channel in range(channels):
                    mean_feature_vec.append(np.mean(feature_map[:,:,channel]))

                features_per_problem.append(mean_feature_vec)
            features_all_problems.append(features_per_problem)
        return features_all_problems
    '''
    @input layers: array with layer names ['conv9', 'conv8', 'conv7', 'conv6', 'conv5']
            type: 'real' - load the feature map from real bongard problems
                    'manual' - load manually generated bongard image feature maps
    '''
    def _extract_bongard_feature_vec(self):
        layers = self.layer_name_list
        
        #if type == 'real':
        con_feature_maps = []
        if self._debug:
            print("LOADING ORIGINAL BONGARD PROBLEM DATA...")
        for i_l, layer_name in enumerate(layers):
            if self._debug:
                print('layer {} ({}/{})'.format(layer_name, i_l, len(layers)))
            feature_map_path =  self.path + 'bongard_problems_feature_maps_'+layer_name+'.pkl'
            curr_pkl_file = open(feature_map_path, 'rb')
            curr_feature_map = pickle.load(curr_pkl_file)
            curr_feature_vec = self.get_vector_feature_map(curr_feature_map) #231*12*filtersize
            #concatenate all features from 3rd dim
            if len(con_feature_maps) != 0:
                con_feature_maps = np.concatenate((np.array(curr_feature_vec)[:,:,:], con_feature_maps[:,:,:]), axis=2)
            else:
                con_feature_maps = np.array(curr_feature_vec)
            #print(np.shape(con_feature_maps))
        #add all problems feature maps(12*1050) to self.problems
        for index in range(con_feature_maps.shape[0]):
            self.real_bongard_problems[index] = con_feature_maps[index,:,:]
        #return con_feature_maps
        #elif type == 'manual':
        con_manual_features = []
        if self._debug:
            print("LOADING MANUAL BONGARD PROBLEM DATA...")
        
        for i_l, layer_name in enumerate(layers):
            
            if self._debug:
                print('layer {} ({}/{})'.format(layer_name, i_l, len(layers)))
            
            feature_map_path = self.path + 'manual_problems_feature_maps_'+layer_name+'.pkl'
            curr_pkl_file = open(feature_map_path, 'rb')
            curr_feature_map = pickle.load(curr_pkl_file)
            
            if self._debug:
                print('total problems ', len(curr_feature_map))
            
            for problem in range(len(curr_feature_map)):
                if self._debug:
                    print('problem #', problem)
                
                feature_dic = {}
                
                for side in ['left', 'right']:
                    if layer_name != layers[0]:
                        mean_feature_vec = con_manual_features[problem][side]
                    else:
                        mean_feature_vec = []
                    for filter in range(np.array(curr_feature_map[problem][side]).shape[2]):
                        mean_feature_vec.append(np.mean(curr_feature_map[problem][side][:,:,filter]))
                    
                    feature_dic[side] = mean_feature_vec
                
                if layer_name != layers[0]:
                    con_manual_features[problem] = feature_dic
                else:
                    con_manual_features.append(feature_dic)
        
        for index in range(len(con_manual_features)):
            self.manual_bongard_problems[index] = con_manual_features[index]
            #return con_manual_features


