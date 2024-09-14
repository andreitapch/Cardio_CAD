# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 10:31:50 2019

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

dataabn=[]  #Raw_Curves of all Abnormal subejcts in dataabn
for j in range(len(file_list)) :
    sample_list = os.listdir(tmp_dir+file_list[j])
    
    for i in range(len(sample_list)):
        
        dataabn0= loadmat(tmp_dir+file_list[j]+'/'+sample_list[i])
        dataabn.append(dataabn0['Raw_Curves'])
#####       
reflabel=[1,2,3,4,5,6] #Reference label is based on reflabel
#sort all data
labelsortabn=[]
ev_lengthsortabn=[]
strainsortabn=[]
srsortabn=[]
ev_timesortabn=[]
        
for k in range(len(dataabn)):  #sorf of all data
    
    dataabn1=dataabn[k]  #subject k
    labelabn=dataabn1['Label'][0,0]
    ev_lengthabn=dataabn1['ev_length'][0,0]
    strainbaseabn=dataabn1['strain'][0,0]
    srbaseabn=dataabn1['sr'][0,0]
    ev_timeabn=dataabn1['event_time'][0,0]
    
    labelabn0=[]
    ev_lengthabn0=[]
    strainabn0=[]
    srabn0=[]
    ev_timeabn0=[]
    #sort datas
    timrefabn=[]
    lenghtrefabn=[]
    strainrefabn=[]
    srrefabn=[]
    labelrefabn=[]
    
    for p in range(len(labelabn)) :
        
        labelabn0.append(labelabn[p,0])
        ev_lengthabn0.append(ev_lengthabn[p,0])
        strainabn0.append(strainbaseabn[p,0])
        ev_timeabn0.append(ev_timeabn[p,0])
        srabn0.append(srbaseabn[p,0])
        
    for j in range(len(labelabn0)):
        
        for i in range(len(labelabn0)):
            
            if labelabn0[i][0]==reflabel[j]:
                
               timrefabn.append(ev_timeabn0[i])       #sorf of ev_times
               strainrefabn.append(strainabn0[i])     #sorf of strain data
               srrefabn.append(srabn0[i])             #sorf of strain_rate data
               lenghtrefabn.append(ev_lengthabn0[i])  #sorf of lengths
               labelrefabn.append(labelabn0[i])
               
    labelsortabn.append(labelrefabn)    
    ev_lengthsortabn.append(lenghtrefabn)
    strainsortabn.append(strainrefabn)
    srsortabn.append(srrefabn)
    ev_timesortabn.append(timrefabn)
#################################strain###########################
strain_phaseabn1=[]   
strain_phaseabn2=[]  
strain_phaseabn3=[]  
strain_phaseabn4=[]  
strain_phaseabn5=[]  
strain_phaseabn6=[] 
lphaseabn1=[]
lphaseabn2=[]   
lphaseabn3=[]  
lphaseabn4=[]  
lphaseabn5=[]  
lphaseabn6=[]  
#############seprate six phases of strain data for 18 segments#######
for k in range(len(dataabn)):  
    
    for i in range(len(strainsortabn[k])):
        
        for j in range(len(strainsortabn[k][i])):
            ###phase1   
            strain_phaseabn11=strainsortabn[k][i][j][0:ev_timesortabn[k][i][0][0]]
            l1=len(strain_phaseabn11)
            lphaseabn1.append(l1)
            strain_phaseabn1.append(strain_phaseabn11)
            ###phase2  
            strain_phaseabn21=strainsortabn[k][i][j][ev_timesortabn[k][i][0][0]:ev_timesortabn[k][i][1][0]]
            l2=len(strain_phaseabn21)
            lphaseabn2.append(l2)
            strain_phaseabn2.append(strain_phaseabn21)
            ###phase3  
            strain_phaseabn31=strainsortabn[k][i][j][ev_timesortabn[k][i][1][0]:ev_timesortabn[k][i][2][0]]
            l3=len(strain_phaseabn31)
            lphaseabn3.append(l3)
            strain_phaseabn3.append(strain_phaseabn31)
            ###phase4  
            strain_phaseabn41=strainsortabn[k][i][j][ev_timesortabn[k][i][2][0]:ev_timesortabn[k][i][3][0]]
            l4=len(strain_phaseabn41)
            lphaseabn4.append(l4)
            strain_phaseabn4.append(strain_phaseabn41)
            ###phase5
            strain_phaseabn51=strainsortabn[k][i][j][ev_timesortabn[k][i][3][0]:ev_timesortabn[k][i][4][0]]
            l5=len(strain_phaseabn51)
            lphaseabn5.append(l5)
            strain_phaseabn5.append(strain_phaseabn51)
            ###phase6
            strain_phaseabn61=strainsortabn[k][i][j][ev_timesortabn[k][i][4][0]:ev_timesortabn[k][i][5][0]]
            l6=len(strain_phaseabn61)
            lphaseabn6.append(l6)
            strain_phaseabn6.append(strain_phaseabn61)
        
maxphaseabn1=max(lphaseabn1)            
indxmaxabn1=lphaseabn1.index(maxphaseabn1) 

maxphaseabn2=max(lphaseabn2)            
indxmaxabn2=lphaseabn2.index(maxphaseabn2)     
        
maxphaseabn3=max(lphaseabn3)            
indxmaxabn3=lphaseabn3.index(maxphaseabn3)        
   
maxphaseabn4=max(lphaseabn4)            
indxmaxabn4=lphaseabn4.index(maxphaseabn4)

maxphaseabn5=max(lphaseabn5)            
indxmaxabn5=lphaseabn5.index(maxphaseabn5)

maxphaseabn6=max(lphaseabn6)            
indxmaxabn6=lphaseabn6.index(maxphaseabn6)
##############maximum lengths of Normal#####################
maxphase1=26
maxphase2=35
maxphase3=128
maxphase4=44
maxphase5=209
maxphase6=73
#########Strain_interpolation########
int_strain_phaseabn1=[]
int_strain_phaseabn2=[]
int_strain_phaseabn3=[]
int_strain_phaseabn4=[]
int_strain_phaseabn5=[]
int_strain_phaseabn6=[]
         
for n in range(len(lphaseabn1)):
    ###interpolation of phase1 
    x1=np.linspace(0,(lphaseabn1[n]-1),maxphase1)
    xp1=np.arange(0,lphaseabn1[n])
    int_strain_phaseabn11=np.interp(x1,xp1,strain_phaseabn1[n])
    int_strain_phaseabn1.append(int_strain_phaseabn11)
    ###interpolation of phase2 
    x2=np.linspace(0,(lphaseabn2[n]-1),maxphase2)
    xp2=np.arange(0,lphaseabn2[n])
    int_strain_phaseabn21=np.interp(x2,xp2,strain_phaseabn2[n])
    int_strain_phaseabn2.append(int_strain_phaseabn21)
    ###interpolation of phase3 
    x3=np.linspace(0,(lphaseabn3[n]-1),maxphase3)
    xp3=np.arange(0,lphaseabn3[n])
    int_strain_phaseabn31=np.interp(x3,xp3,strain_phaseabn3[n])
    int_strain_phaseabn3.append(int_strain_phaseabn31)
    ###interpolation of phase4 
    x4=np.linspace(0,(lphaseabn4[n]-1),maxphase4)
    xp4=np.arange(0,lphaseabn4[n])
    int_strain_phaseabn41=np.interp(x4,xp4,strain_phaseabn4[n])
    int_strain_phaseabn4.append(int_strain_phaseabn41)
    ###interpolation of phase5
    x5=np.linspace(0,(lphaseabn5[n]-1),maxphase5)
    xp5=np.arange(0,lphaseabn5[n])
    int_strain_phaseabn51=np.interp(x5,xp5,strain_phaseabn5[n])
    int_strain_phaseabn5.append(int_strain_phaseabn51)
     ###interpolation of phase6
    x6=np.linspace(0,(lphaseabn6[n]-1),maxphase6)
    xp6=np.arange(0,lphaseabn6[n])
    int_strain_phaseabn61=np.interp(x6,xp6,strain_phaseabn6[n])
    int_strain_phaseabn6.append(int_strain_phaseabn61)
    
##########concatenat of all strian interpolated data 
int_strainabn=[] 
strainabn=[]  
for m in range(len(lphaseabn1)):
    ####int_strain is interpolated of all strain data
    int_strainabn1=np.concatenate((int_strain_phaseabn1[m],int_strain_phaseabn2[m],int_strain_phaseabn3[m],int_strain_phaseabn4[m],int_strain_phaseabn5[m],int_strain_phaseabn6[m]), axis=None)  
    int_strainabn.append(int_strainabn1)
    ####strain is  all strain data
    strainabn1=np.concatenate((strain_phaseabn1[m],strain_phaseabn2[m],strain_phaseabn3[m],strain_phaseabn4[m],strain_phaseabn5[m],strain_phaseabn6[m]), axis=None)
    strainabn.append(strainabn1) 
    
#################################strain_rate=sr###########################
sr_phaseabn1=[]   
sr_phaseabn2=[]  
sr_phaseabn3=[]  
sr_phaseabn4=[]  
sr_phaseabn5=[]  
sr_phaseabn6=[]    
    
for k in range(len(dataabn)):  
    
    for i in range(len(srsortabn[k])):
        
        for j in range(len(srsortabn[k][i])):
            ###phase1   
            sr_phaseabn11=srsortabn[k][i][j][0:ev_timesortabn[k][i][0][0]]
            sr_phaseabn1.append(sr_phaseabn11)
            ###phase2  
            sr_phaseabn21=srsortabn[k][i][j][ev_timesortabn[k][i][0][0]:ev_timesortabn[k][i][1][0]]
            sr_phaseabn2.append(sr_phaseabn21)
            ###phase3  
            sr_phaseabn31=srsortabn[k][i][j][ev_timesortabn[k][i][1][0]:ev_timesortabn[k][i][2][0]]
            sr_phaseabn3.append(sr_phaseabn31)
            ###phase4  
            sr_phaseabn41=srsortabn[k][i][j][ev_timesortabn[k][i][2][0]:ev_timesortabn[k][i][3][0]]
            sr_phaseabn4.append(sr_phaseabn41)
            ###phase5
            sr_phaseabn51=srsortabn[k][i][j][ev_timesortabn[k][i][3][0]:ev_timesortabn[k][i][4][0]]
            sr_phaseabn5.append(sr_phaseabn51)
            ###phase6
            sr_phaseabn61=srsortabn[k][i][j][ev_timesortabn[k][i][4][0]:ev_timesortabn[k][i][5][0]]
            sr_phaseabn6.append(sr_phaseabn61)   

#########Strain_rate_interpolation########
int_sr_phaseabn1=[]
int_sr_phaseabn2=[]
int_sr_phaseabn3=[]
int_sr_phaseabn4=[]
int_sr_phaseabn5=[]
int_sr_phaseabn6=[]    
for q in range(len(lphaseabn1)):
    ###interpolation of phase1 
    x1=np.linspace(0,(lphaseabn1[q]-1),maxphase1)
    xp1=np.arange(0,lphaseabn1[q])
    int_sr_phaseabn11=np.interp(x1,xp1,sr_phaseabn1[q])
    int_sr_phaseabn1.append(int_sr_phaseabn11)
    ###interpolation of phase2 
    x2=np.linspace(0,(lphaseabn2[q]-1),maxphase2)
    xp2=np.arange(0,lphaseabn2[q])
    int_sr_phaseabn21=np.interp(x2,xp2,sr_phaseabn2[q])
    int_sr_phaseabn2.append(int_sr_phaseabn21)
    ###interpolation of phase3
    x3=np.linspace(0,(lphaseabn3[q]-1),maxphase3)
    xp3=np.arange(0,lphaseabn3[q])
    int_sr_phaseabn31=np.interp(x3,xp3,sr_phaseabn3[q])
    int_sr_phaseabn3.append(int_sr_phaseabn31)
    ###interpolation of phase4 
    x4=np.linspace(0,(lphaseabn4[q]-1),maxphase4)
    xp4=np.arange(0,lphaseabn4[q])
    int_sr_phaseabn41=np.interp(x4,xp4,sr_phaseabn4[q])
    int_sr_phaseabn4.append(int_sr_phaseabn41)
    ###interpolation of phase5
    x5=np.linspace(0,(lphaseabn5[q]-1),maxphase5)
    xp5=np.arange(0,lphaseabn5[q])
    int_sr_phaseabn51=np.interp(x5,xp5,sr_phaseabn5[q])
    int_sr_phaseabn5.append(int_sr_phaseabn51)
    ###interpolation of phase6
    x6=np.linspace(0,(lphaseabn6[q]-1),maxphase6)
    xp6=np.arange(0,lphaseabn6[q])
    int_sr_phaseabn61=np.interp(x6,xp6,sr_phaseabn6[q])
    int_sr_phaseabn6.append(int_sr_phaseabn61)    

##########concatenat of all strian_rate interpolated data 
int_srabn=[] 
srabn=[]  
for r in range(len(lphaseabn1)):
    ####int_strain is interpolated of all strain data
    int_srabn1=np.concatenate((int_sr_phaseabn1[r],int_sr_phaseabn2[r],int_sr_phaseabn3[r],int_sr_phaseabn4[r],int_sr_phaseabn5[r],int_sr_phaseabn6[r]), axis=None)  
    int_srabn.append(int_srabn1)
    ####sr is  all strain_rate data
    srabn1=np.concatenate((sr_phaseabn1[r],sr_phaseabn2[r],sr_phaseabn3[r],sr_phaseabn4[r],sr_phaseabn5[r],sr_phaseabn6[r]), axis=None)
    srabn.append(srabn1) 
    
 ###############Figures
#figure,
plt.plot(srabn[0])
plt.plot(srabn[1])
plt.plot(srabn[2])
plt.title('title')
plt.ylabel('ylabel')
plt.xlabel('xlabel')
#plt.legend()
plt.show()


    
    
    
    
    
    
    