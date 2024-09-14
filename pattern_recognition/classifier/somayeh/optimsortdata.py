# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 10:36:16 2019

@author: sakbar0
"""

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
import pandas as pd
from sklearn.decomposition import PCA


######################Normal#################################################################################
data=[]
tmp_dir= 'D:/PHD/Experiments/CAD/Data-strain/Normal/'
# tmp_dir='C:/Somayeh/Deformation data & codes/Subjects with 18 Segment_Doppler/'
file_list = os.listdir(tmp_dir)

data_n = []  #Raw_Curves of all Normal subejcts in datan
for fList in file_list:
    if fList=='Doppler':
        u_sample_list = os.listdir(os.path.join(tmp_dir, fList))
        for i_u_SampleList in u_sample_list:
            sampleListName = os.listdir(os.path.join(tmp_dir, fList, i_u_SampleList))
            for i_u_SampleName in sampleListName:
                data0 = loadmat(os.path.join(tmp_dir, fList, i_u_SampleList,i_u_SampleName))
                data_n.append(data0['Raw_Curves'])  # Extract Raw curves file
    else:
        sample_list = os.listdir(os.path.join(tmp_dir, fList))
        for iSampleList in sample_list:
            data0 = loadmat(os.path.join(tmp_dir,fList,iSampleList))
            data_n.append(data0['Raw_Curves'])  #Extract Raw curves file
data.append(data_n)
lengthn = len(data_n)
##############Abnormal##############################################
# tmp_dir='F:/Strain Analysis/Data-strain/Abnormals/'## TODO > search for abnormal
tmp_dir='D:/PHD/Experiments/CAD/Data-strain/Abnormal/'

file_list = os.listdir(tmp_dir)        
data_abn = []  #Raw_Curves of all Abnormal subejcts in data_abn
for fList in file_list:
    sample_list = os.listdir(os.path.join(tmp_dir, fList))
    for sList in sample_list:
        dataabn0 = loadmat(os.path.join(tmp_dir,fList,sList))
        data_abn.append(dataabn0['Raw_Curves'])  #Extract Raw curves file
lengthabn=len(data_abn)         
data.append(data_abn)          
############Sort all data    
reflabel=[1,2,3,4,5,6] #Reference label is based on reflabel
labelsort=[]
ev_lengthsort=[]
strainsort=[]
srsort=[]
ev_timesort=[]

for r in range(len(data)):
    if r==1:
        stop=1
    for k in range(len(data[r])):  #sorf of all data
        
        label0=[]
        ev_length0=[]
        strain0=[]
        sr0=[]
        ev_time0=[]
        #######Sort datas
        timref=[]
        lengthref=[]
        strainref=[]
        srref=[]
        labelref=[]
        
        for p in range(len(data[r][k]['Label'][0,0])) :
            
            label0.append(data[r][k]['Label'][0,0][p,0])
            ev_length0.append(data[r][k]['ev_length'][0,0][p,0])
            strain0.append(data[r][k]['strain'][0,0][p,0])
            ev_time0.append(data[r][k]['event_time'][0,0][p,0])
            sr0.append(data[r][k]['sr'][0,0][p,0])
            
        for j in range(len(label0)):  
            for i in range(len(label0)):
                if label0[i][0]==reflabel[j]:
                    timref.append(ev_time0[i])       #sorf of ev_times
                    strainref.append(strain0[i])     #sorf of strain data
                    srref.append(sr0[i])             #sorf of strain_rate data
                    lengthref.append(ev_length0[i])  #sorf of lengths
                    labelref.append(label0[i])
                    
        labelsort.append(labelref)
        ev_lengthsort.append(lengthref)
        strainsort.append(strainref)
        srsort.append(srref)
        ev_timesort.append(timref)

#import pdb; pdb.set_trace()  
##seprate six phases of strain and strain_rate=sr data for 18 segments##
strian_phases=[[], [], [], [], [], []] 
sr_phases=[[], [], [], [], [], []] 
lphase= [[], [], [], [], [], []]  
counter=0
for k in range(len(strainsort)):
    stop=1
    for i in range(len(strainsort[k])):
        counter += 1
        print(counter)

        for j in range(len(strainsort[k][i])):
            
            strain_phase11=strainsort[k][i][j][0:ev_timesort[k][i][0][0]]
            strian_phases[0].append(strain_phase11)
            lphase[0].append(len(strain_phase11))
            sr_phase11=srsort[k][i][j][0:ev_timesort[k][i][0][0]]
            sr_phases[0].append(sr_phase11)
            
            for p in range((len(ev_timesort[k][i]))-1):
                strain_phase=strainsort[k][i][j][ev_timesort[k][i][p][0]:ev_timesort[k][i][p+1][0]]
                strian_phases[p+1].append(strain_phase)
                lphase[p+1].append(len(strain_phase))
                sr_phase=srsort[k][i][j][ev_timesort[k][i][p][0]:ev_timesort[k][i][p+1][0]]
                sr_phases[p+1].append(sr_phase)
                
# maximum length in normal data (length of normal data=lenghtn*18)    
maxpahses=[]
indxmax=[]
for i in range(len(lphase)):
    
    maxlength=max(lphase[i][0:(lengthn*18)])
    maxpahses.append(maxlength) 
    indxmax.append(lphase[i].index(maxlength)) 
    
#########Strain and strain_rate interpolation########
int_strian_phases=[[], [], [], [], [], []]
int_sr_phases=[[], [], [], [], [], []]
         
for n in range(len(strian_phases)): #n indicates the number of phase
    
    for m in range(len(strian_phases[n])):
        
        x1=np.linspace(0,(lphase[n][m]-1),maxpahses[n])
        xp1=np.arange(0,lphase[n][m])
        int_strain_phase1=np.interp(x1,xp1,strian_phases[n][m])
        int_strian_phases[n].append(int_strain_phase1)
        int_sr_phase1=np.interp(x1,xp1,sr_phases[n][m])
        int_sr_phases[n].append(int_sr_phase1)
    
##########concatenat of all strian and strain_rate interpolated data 
int_strain=[] 
strain=[]  
int_sr=[] 
sr=[] 
for i in range(len(strian_phases[0])):
    
    
    ####int_strain is interpolated of all strain data
    #int_strain1=np.concatenate((int_strian_phases[0][i],int_strian_phases[1][i],int_strian_phases[2][i],int_strian_phases[3][i],int_strian_phases[4][i],int_strian_phases[5][i]), axis=None)  
   #int_strain.append(int_strain1)
    ####strain is  all strain data
    strain1=np.concatenate((strian_phases[0][i],strian_phases[1][i],strian_phases[2][i],strian_phases[3][i],strian_phases[4][i],strian_phases[5][i]), axis=None)
    strain.append(strain1) 
    ####int_strain is interpolated of all strain data
    #int_sr1=np.concatenate((int_sr_phases[0][i],int_sr_phases[1][i],int_sr_phases[2][i],int_sr_phases[3][i],int_sr_phases[4][i],int_sr_phases[5][i]), axis=None)  
    #int_sr.append(int_sr1)
    ####sr is  all strain_rate data
    sr1=np.concatenate((sr_phases[0][i],sr_phases[1][i],sr_phases[2][i],sr_phases[3][i],sr_phases[4][i],sr_phases[5][i]), axis=None)
    sr.append(sr1)   
#### concatenate int_strain and int_sr with other command  
int_strain=np.concatenate((int_strian_phases[:][:]), axis=1)
#int_strain211=np.hstack((int_strian_phases[:][:]))

int_sr=np.concatenate((int_sr_phases[:][:]), axis=1)
#int_sr211=np.hstack((int_sr_phases[:][:]))

#####################Seprate Normal and Abnormal data 
# int_strainn=np.asarray(int_strain[0:18*lengthn])  # convert list to array format
# int_strainabn=np.asarray(int_strain[18*lengthn:])
#
# int_srn=np.asarray(int_sr[0:18*lengthn])  # convert list to array format
# int_srabn=np.asarray(int_sr[18*lengthn:])
#############################concatenate all segments of each subject 
subject_int_strain=[]
subject_int_sr=[]

#for s in range(lengthn+lengthabn):
    #subject_int_strain1=np.concatenate((int_strain[s*18:(s+1)*18][:]), axis=None)
    #subject_int_strain.append(subject_int_strain1)
    #subject_int_sr1=np.concatenate((int_sr[s*18:(s+1)*18][:]), axis=None)
    #subject_int_sr.append(subject_int_sr1)
    
#####################Seprate Normal and Abnormal data######
#for s in range(lengthn+lengthabn):
    
norm_int_strain=int_strain[0:18*lengthn]   
abnorm_int_strain=int_strain[18*lengthn:]

norm_int_sr=int_sr[0:18*lengthn]
abnorm_int_sr=int_sr[18*lengthn:]    
##############PCA###########
nsi=norm_int_strain.transpose((1,0))
pca = PCA()
principalComponents = pca.fit_transform(int_strain[0:18][:])
# principalComponents = pca.fit_transform(int_strain[:][:])
pcas=principalComponents.transpose((1,0))
fig=plt.figure
plt.plot(pcas)
plt.show()

nsi=norm_int_strain.T
























       
fig1 = plt.figure()
#plt.legend()
#plt.show()
#n=0  £ indicate numeber of subject
#num=18*n
ax1=plt.subplot(2,1,1)
plt.plot(strain[969])
plt.plot(strain[970])
plt.plot(strain[971])
plt.title('Strain of segments  (6,12,18)')
plt.ylabel('Strain')
plt.xlabel('sample')
plt.axvline(x=ev_timesort[53][5][0][0],linewidth=1, color='r', linestyle='--')
plt.axvline(x=ev_timesort[53][5][1][0],linewidth=1, color='r', linestyle='--')
plt.axvline(x=ev_timesort[53][5][2][0],linewidth=1, color='r', linestyle='--')
plt.axvline(x=ev_timesort[53][5][3][0],linewidth=1, color='r', linestyle='--')
plt.axvline(x=ev_timesort[53][5][4][0],linewidth=1, color='r', linestyle='--')
plt.axvline(x=ev_timesort[53][5][5][0],linewidth=1, color='r', linestyle='--')
#plt.legend()
#plt.show()
ax2=plt.subplot(2,1,2)
plt.plot(int_strain[969])
plt.plot(int_strain[970])
plt.plot(int_strain[971])
plt.title('Interpolated Strain')
plt.ylabel('Strain')
plt.xlabel('sample')
plt.axvline(x=maxpahses[0],linewidth=1, color='r', linestyle='--')
plt.axvline(x=maxpahses[0]+maxpahses[1],linewidth=1, color='r', linestyle='--')
plt.axvline(x=maxpahses[0]+maxpahses[1]+maxpahses[2],linewidth=1, color='r', linestyle='--')
plt.axvline(x=maxpahses[0]+maxpahses[1]+maxpahses[2]+maxpahses[3],linewidth=1, color='r', linestyle='--')
plt.axvline(x=maxpahses[0]+maxpahses[1]+maxpahses[2]+maxpahses[3]+maxpahses[4],linewidth=1, color='r', linestyle='--')
plt.axvline(x=maxpahses[0]+maxpahses[1]+maxpahses[2]+maxpahses[3]+maxpahses[4]+maxpahses[5],linewidth=1, color='r', linestyle='--')
fig1.subplots_adjust(hspace=0.75)
plt.show()
        
fig1 = plt.figure()
ax1=plt.subplot(2,1,1)
plt.plot(sr[969])
plt.plot(sr[970])
plt.plot(sr[971])
plt.title('Strain_rate of segments  (6,12,18)')
plt.ylabel('Strain_Rate')
plt.xlabel('sample')
plt.axvline(x=ev_timesort[53][5][0][0],linewidth=1, color='r', linestyle='--')
plt.axvline(x=ev_timesort[53][5][1][0],linewidth=1, color='r', linestyle='--')
plt.axvline(x=ev_timesort[53][5][2][0],linewidth=1, color='r', linestyle='--')
plt.axvline(x=ev_timesort[53][5][3][0],linewidth=1, color='r', linestyle='--')
plt.axvline(x=ev_timesort[53][5][4][0],linewidth=1, color='r', linestyle='--')
plt.axvline(x=ev_timesort[53][5][5][0],linewidth=1, color='r', linestyle='--')
#plt.legend()
#plt.show()
ax2=plt.subplot(2,1,2)
plt.plot(int_sr[969])
plt.plot(int_sr[970])
plt.plot(int_sr[971])
plt.title('Interpolated Strain_Rate')
plt.ylabel('Strain_Rate')
plt.xlabel('sample')
plt.axvline(x=maxpahses[0],linewidth=1, color='r', linestyle='--')
plt.axvline(x=maxpahses[0]+maxpahses[1],linewidth=1, color='r', linestyle='--')
plt.axvline(x=maxpahses[0]+maxpahses[1]+maxpahses[2],linewidth=1, color='r', linestyle='--')
plt.axvline(x=maxpahses[0]+maxpahses[1]+maxpahses[2]+maxpahses[3],linewidth=1, color='r', linestyle='--')
plt.axvline(x=maxpahses[0]+maxpahses[1]+maxpahses[2]+maxpahses[3]+maxpahses[4],linewidth=1, color='r', linestyle='--')
plt.axvline(x=maxpahses[0]+maxpahses[1]+maxpahses[2]+maxpahses[3]+maxpahses[4]+maxpahses[5],linewidth=1, color='r', linestyle='--')
fig1.subplots_adjust(hspace=0.75)
plt.show()   
    
    
    
           
           
            
           
        
            
              
            
        
       
       
       
 
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    