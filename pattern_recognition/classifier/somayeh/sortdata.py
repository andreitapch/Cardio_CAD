# -*- coding: utf-8 -*-
"""
Created on Fri Oct  4 12:28:40 2019

@author: sakbar0
"""

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
from scipy import interpolate

#tmp_dir = 'C:/Somayeh/Deformation data & codes/Subjects with 18 Segment_Doppler/KUL/'
#sample_list = os.listdir(tmp_dir)
######################Normal#################################################################################
tmp_dir='F:/Strain Analysis/Data-strain/Subjects with 18 Segment_Doppler/'
file_list = os.listdir(tmp_dir)

datan=[]  #Raw_Curves of all Normal subejcts in datan
for j in range(len(file_list)) :
    sample_list = os.listdir(tmp_dir+file_list[j])
    
    for i in range(len(sample_list)):
        
        data0= loadmat(tmp_dir+file_list[j]+'/'+sample_list[i])
        datan.append(data0['Raw_Curves'])

reflabel=[1,2,3,4,5,6] #Reference label is based on reflabel
#sort all data
labelsort=[]
ev_lengthsort=[]
strainsort=[]
srsort=[]
ev_timesort=[]
        
for k in range(len(datan)):  #sorf of all data
    
    data1=datan[k]  #subject k
    label=data1['Label'][0,0]
    ev_length=data1['ev_length'][0,0]
    strainbase=data1['strain'][0,0]
    srbase=data1['sr'][0,0]
    ev_time=data1['event_time'][0,0]
    
    label0=[]
    ev_length0=[]
    strain0=[]
    sr0=[]
    ev_time0=[]
    #sort datas
    timref=[]
    lenghtref=[]
    strainref=[]
    srref=[]
    labelref=[]
    
    for p in range(len(label)) :
        
        label0.append(label[p,0])
        ev_length0.append(ev_length[p,0])
        strain0.append(strainbase[p,0])
        ev_time0.append(ev_time[p,0])
        sr0.append(srbase[p,0])
        
    for j in range(len(label0)):
        
        for i in range(len(label0)):
            
            if label0[i][0]==reflabel[j]:
                
               timref.append(ev_time0[i])       #sorf of ev_times
               strainref.append(strain0[i])     #sorf of strain data
               srref.append(sr0[i])             #sorf of strain_rate data
               lenghtref.append(ev_length0[i])  #sorf of lengths
               labelref.append(label0[i])
               
    labelsort.append(labelref)    
    ev_lengthsort.append(lenghtref)
    strainsort.append(strainref)
    srsort.append(srref)
    ev_timesort.append(timref)
#################################strain###########################
strain_phase1=[]   
strain_phase2=[]  
strain_phase3=[]  
strain_phase4=[]  
strain_phase5=[]  
strain_phase6=[] 
lphase1=[]
lphase2=[]   
lphase3=[]  
lphase4=[]  
lphase5=[]  
lphase6=[]  
#############seprate six phases of strain data for 18 segments#######
for k in range(len(datan)):  
    
    for i in range(len(strainsort[k])):
        
        for j in range(len(strainsort[k][i])):
            ###phase1   
            strain_phase11=strainsort[k][i][j][0:ev_timesort[k][i][0][0]]
            l1=len(strain_phase11)
            lphase1.append(l1)
            strain_phase1.append(strain_phase11)
            ###phase2  
            strain_phase21=strainsort[k][i][j][ev_timesort[k][i][0][0]:ev_timesort[k][i][1][0]]
            l2=len(strain_phase21)
            lphase2.append(l2)
            strain_phase2.append(strain_phase21)
            ###phase3  
            strain_phase31=strainsort[k][i][j][ev_timesort[k][i][1][0]:ev_timesort[k][i][2][0]]
            l3=len(strain_phase31)
            lphase3.append(l3)
            strain_phase3.append(strain_phase31)
            ###phase4  
            strain_phase41=strainsort[k][i][j][ev_timesort[k][i][2][0]:ev_timesort[k][i][3][0]]
            l4=len(strain_phase41)
            lphase4.append(l4)
            strain_phase4.append(strain_phase41)
            ###phase5
            strain_phase51=strainsort[k][i][j][ev_timesort[k][i][3][0]:ev_timesort[k][i][4][0]]
            l5=len(strain_phase51)
            lphase5.append(l5)
            strain_phase5.append(strain_phase51)
            ###phase6
            strain_phase61=strainsort[k][i][j][ev_timesort[k][i][4][0]:ev_timesort[k][i][5][0]]
            l6=len(strain_phase61)
            lphase6.append(l6)
            strain_phase6.append(strain_phase61)
        
maxphase1=max(lphase1)            
indxmax1=lphase1.index(maxphase1) 

maxphase2=max(lphase2)            
indxmax2=lphase2.index(maxphase2)     
        
maxphase3=max(lphase3)            
indxmax3=lphase3.index(maxphase3)        
   
maxphase4=max(lphase4)            
indxmax4=lphase4.index(maxphase4)

maxphase5=max(lphase5)            
indxmax5=lphase5.index(maxphase5)

maxphase6=max(lphase6)            
indxmax6=lphase6.index(maxphase6)
#########Strain_interpolation########
int_strain_phase1=[]
int_strain_phase2=[]
int_strain_phase3=[]
int_strain_phase4=[]
int_strain_phase5=[]
int_strain_phase6=[]
         
for n in range(len(lphase1)):
    ###interpolation of phase1 
    x1=np.linspace(0,(lphase1[n]-1),maxphase1)
    xp1=np.arange(0,lphase1[n])
    int_strain_phase11=np.interp(x1,xp1,strain_phase1[n])
    int_strain_phase1.append(int_strain_phase11)
    ###interpolation of phase2 
    x2=np.linspace(0,(lphase2[n]-1),maxphase2)
    xp2=np.arange(0,lphase2[n])
    int_strain_phase21=np.interp(x2,xp2,strain_phase2[n])
    int_strain_phase2.append(int_strain_phase21)
    ###interpolation of phase3 
    x3=np.linspace(0,(lphase3[n]-1),maxphase3)
    xp3=np.arange(0,lphase3[n])
    int_strain_phase31=np.interp(x3,xp3,strain_phase3[n])
    int_strain_phase3.append(int_strain_phase31)
    ###interpolation of phase4 
    x4=np.linspace(0,(lphase4[n]-1),maxphase4)
    xp4=np.arange(0,lphase4[n])
    int_strain_phase41=np.interp(x4,xp4,strain_phase4[n])
    int_strain_phase4.append(int_strain_phase41)
    ###interpolation of phase5
    x5=np.linspace(0,(lphase5[n]-1),maxphase5)
    xp5=np.arange(0,lphase5[n])
    int_strain_phase51=np.interp(x5,xp5,strain_phase5[n])
    int_strain_phase5.append(int_strain_phase51)
     ###interpolation of phase6
    x6=np.linspace(0,(lphase6[n]-1),maxphase6)
    xp6=np.arange(0,lphase6[n])
    int_strain_phase61=np.interp(x6,xp6,strain_phase6[n])
    int_strain_phase6.append(int_strain_phase61)
    
##########concatenat of all strian interpolated data 
int_strain=[] 
strain=[]  
for m in range(len(lphase1)):
    ####int_strain is interpolated of all strain data
    int_strain1=np.concatenate((int_strain_phase1[m],int_strain_phase2[m],int_strain_phase3[m],int_strain_phase4[m],int_strain_phase5[m],int_strain_phase6[m]), axis=None)  
    int_strain.append(int_strain1)
    ####strain is  all strain data
    strain1=np.concatenate((strain_phase1[m],strain_phase2[m],strain_phase3[m],strain_phase4[m],strain_phase5[m],strain_phase6[m]), axis=None)
    strain.append(strain1) 
    
#################################strain_rate=sr###########################
sr_phase1=[]   
sr_phase2=[]  
sr_phase3=[]  
sr_phase4=[]  
sr_phase5=[]  
sr_phase6=[]    
    
for k in range(len(datan)):  
    
    for i in range(len(srsort[k])):
        
        for j in range(len(srsort[k][i])):
            ###phase1   
            sr_phase11=srsort[k][i][j][0:ev_timesort[k][i][0][0]]
            sr_phase1.append(sr_phase11)
            ###phase2  
            sr_phase21=srsort[k][i][j][ev_timesort[k][i][0][0]:ev_timesort[k][i][1][0]]
            sr_phase2.append(sr_phase21)
            ###phase3  
            sr_phase31=srsort[k][i][j][ev_timesort[k][i][1][0]:ev_timesort[k][i][2][0]]
            sr_phase3.append(sr_phase31)
            ###phase4  
            sr_phase41=srsort[k][i][j][ev_timesort[k][i][2][0]:ev_timesort[k][i][3][0]]
            sr_phase4.append(sr_phase41)
            ###phase5
            sr_phase51=srsort[k][i][j][ev_timesort[k][i][3][0]:ev_timesort[k][i][4][0]]
            sr_phase5.append(sr_phase51)
            ###phase6
            sr_phase61=srsort[k][i][j][ev_timesort[k][i][4][0]:ev_timesort[k][i][5][0]]
            sr_phase6.append(sr_phase61)   

#########Strain_rate_interpolation########
int_sr_phase1=[]
int_sr_phase2=[]
int_sr_phase3=[]
int_sr_phase4=[]
int_sr_phase5=[]
int_sr_phase6=[]    
for q in range(len(lphase1)):
    ###interpolation of phase1 
    x1=np.linspace(0,(lphase1[q]-1),maxphase1)
    xp1=np.arange(0,lphase1[q])
    int_sr_phase11=np.interp(x1,xp1,sr_phase1[q])
    int_sr_phase1.append(int_sr_phase11)
    ###interpolation of phase2 
    x2=np.linspace(0,(lphase2[q]-1),maxphase2)
    xp2=np.arange(0,lphase2[q])
    int_sr_phase21=np.interp(x2,xp2,sr_phase2[q])
    int_sr_phase2.append(int_sr_phase21)
    ###interpolation of phase3
    x3=np.linspace(0,(lphase3[q]-1),maxphase3)
    xp3=np.arange(0,lphase3[q])
    int_sr_phase31=np.interp(x3,xp3,sr_phase3[q])
    int_sr_phase3.append(int_sr_phase31)
    ###interpolation of phase4 
    x4=np.linspace(0,(lphase4[q]-1),maxphase4)
    xp4=np.arange(0,lphase4[q])
    int_sr_phase41=np.interp(x4,xp4,sr_phase4[q])
    int_sr_phase4.append(int_sr_phase41)
    ###interpolation of phase5
    x5=np.linspace(0,(lphase5[q]-1),maxphase5)
    xp5=np.arange(0,lphase5[q])
    int_sr_phase51=np.interp(x5,xp5,sr_phase5[q])
    int_sr_phase5.append(int_sr_phase51)
    ###interpolation of phase6
    x6=np.linspace(0,(lphase6[q]-1),maxphase6)
    xp6=np.arange(0,lphase6[q])
    int_sr_phase61=np.interp(x6,xp6,sr_phase6[q])
    int_sr_phase6.append(int_sr_phase61)    

##########concatenat of all strian_rate interpolated data 
int_sr=[] 
sr=[]  
for r in range(len(lphase1)):
    ####int_strain is interpolated of all strain data
    int_sr1=np.concatenate((int_sr_phase1[r],int_sr_phase2[r],int_sr_phase3[r],int_sr_phase4[r],int_sr_phase5[r],int_sr_phase6[r]), axis=None)  
    int_sr.append(int_sr1)
    ####sr is  all strain_rate data
    sr1=np.concatenate((sr_phase1[r],sr_phase2[r],sr_phase3[r],sr_phase4[r],sr_phase5[r],sr_phase6[r]), axis=None)
    sr.append(sr1)     
    
    
    
    
    
           
           
            
           
        
            
              
            
        
       
       
       
 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    