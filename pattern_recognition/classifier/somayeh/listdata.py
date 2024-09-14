# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 15:30:59 2019

@author: sakbar0
"""
#import sys
#sys.modules[__name__].__dict__.clear()
# Importing the required libraries
import os
import numpy as np
#import nrrd
import skimage as sk
import scipy
import scipy.io as sio
import matplotlib.pyplot as plt
from scipy.io import loadmat
import pandas as pd
from skimage import io
from skimage.transform import resize
from numpy import empty

#tmp_dir = 'C:/Somayeh/Deformation data & codes/Subjects with 18 Segment_Doppler/KUL/'
#sample_list = os.listdir(tmp_dir)
tmp_dir='F:/Strain Analysis/Data-strain/Subjects with 18 Segment_Doppler/'
file_list = os.listdir(tmp_dir)
#Raw_Curves of all subjects in Datan  
datan=[]
for j in range(len(file_list)) :
    sample_list = os.listdir(tmp_dir+file_list[j])
    for i in range(len(sample_list)):
        data0= loadmat(tmp_dir+file_list[j]+'/'+sample_list[i])
        datan.append(data0['Raw_Curves'])
      
labeln=[]
ev_lengthn=[]
strainn=[] 
srn=[]
ev_timen=[]
for j in range(len(datan)) :
    data1=datan[j]
    label0=[]
    ev_length0=[]
    strain0=[]
    sr0=[]
    ev_time0=[]
    label=data1['Label'][0,0]
    ev_length=data1['ev_length'][0,0]
    strain=data1['strain'][0,0]
    sr=data1['sr'][0,0]
    ev_time=data1['event_time'][0,0]
    #Labels of all subjects in Labeln 
    for i in range(len(label)) :
        label0.append(label[i,0])
    labeln.append(label0)
    #ev_lenghts of all subjects in ev_lengthn      
    for i in range(len(ev_length)) :
        ev_length0.append(ev_length[i,0])
    ev_lengthn.append(ev_length0)
   #strains of all subjects in strainn
    for i in range(len(strain)) :
        strain0.append(strain[i,0])
    strainn.append(strain0)
    #strain rates of all subjects in srn
    for i in range(len(sr)) :
        sr0.append(sr[i,0])
    srn.append(sr0)
    for i in range(len(ev_time)) :
        ev_time0.append(ev_time[i,0])
    ev_timen.append(ev_time0)
timref=[]  
strainref=[]
reflabel=[1,2,3,4,5,6]
for j in range(len(label0)):
    
    for i in range(len(label0)):
        
        if label0[i][0]==reflabel[j]:
            
            timref.append(ev_time0[i])
            strainref.append(strain0[i])
        
    
    