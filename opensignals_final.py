#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from statistics import variance
from scipy.stats import pearsonr

def kokyu(Xk, sampling_duration,offset):

    X = X = np.copy(Xk)
    X[:,0] = X[:,0]/1000
    X_mean = X[:,3].mean()
    Gs = np.zeros(int(1200/sampling_duration))
    peaks = np.zeros(int(1200/sampling_duration))

    for i in range(0,int(1200/sampling_duration)):
        N = sampling_duration*1000

        Freq = np.arange(0,1,1/sampling_duration)
        X_mean = X[offset+ i*N:offset + (i+1)*N,3].mean()

        Y = X[offset + i*N: offset + (i+1)*N,3] - X_mean

        F = np.fft.fft(Y)
        Amp = np.abs(F)
        #pick up frequency smaller than 1.0 Hz
        h_Amp = Amp[0:sampling_duration]
        Amp_sum = np.sum(h_Amp)
        Amp_nor = h_Amp/Amp_sum
        G = np.dot(Amp_nor,Freq)
        peak = np.argmax(h_Amp)/sampling_duration
        #peak_frequency
        Gs[i] = G
        #peak frequency
        peaks[i] = peak
        



    return Gs, np.abs(Gs -peaks)



def shinpaku(Xk,sampling_duration,offset):

    X = np.copy(Xk)
    X[:,0] = X[:,0]/1000

    sampling_number = int(1200/sampling_duration)

   
    #rri
    rri_ave = np.zeros(sampling_number)
    #rriの分散 
    rri_var = np.zeros(sampling_number)
    flag = 0  
    peek_value = 0
    pre_peek_time = 0
    peek_time = 0
    count = np.zeros(sampling_number)
    #rriを収容
    rris = [[0 for col in range(0)] for row in range(sampling_number)]

    for i in range(0,sampling_number):


        first_flag = 0
      

        if i == 0:
            first_flag = 1
    
        #print(i) 
        a =  int(i*1000*sampling_duration+offset)
        b =  int((i+1)*1000*sampling_duration+offset)

        Xe = X[a:b,2]
        Xt = X[a:b,0]

        threshold = (max(Xe)+min(Xe))/2
        buff = 5000
        for j in range(0,b-a):

            
            if(Xe[j]>threshold):
                if(flag == 0):
                    flag = 1
                    peek_time = Xt[j]
                    peek_value = Xe[j]
                    if first_flag == 0:
                        count[i] = count[i] +1
                else:

                    if peek_value > Xe[j]: 
                        peek_time = Xt[j]
                            
            else:

                if(Xe[j]<(threshold -buff)):
                    if(flag == 1):
                        flag = 0
                        dura = float(peek_time) - float(pre_peek_time)

                        
                        if first_flag == 0:
                            if(count[i]<0.5):
                                rris[i-1].append(dura)
                            else:    
                                
                                rris[i].append(dura) 
                                
                        else:
                            first_flag = 0
                        pre_peek_time = peek_time
    
    
    for k in range(0,sampling_number):
        rri_ave[k] = sum(rris[k])/len(rris[k])
        rri_var[k] = variance(rris[k])

    return rri_ave, rri_var


def n_back(Xk,sampling_duration):
    X = Xk[:,3]
    N = int(1200/sampling_duration)
    M = sampling_duration/2
    A = np.zeros(N)
    for i in range(0,N):
        a = int(M*i)
        b = int(min(M*(i+1),Xk.shape[0]))
        X_m = X[a:b]
        A[i] = np.average(X_m)
    
    return A

def eyes(Xk,sampling_duration):
    Xr = Xk[:,3]
    Xl = Xk[:,2]
    Xt = Xk[:,0]
    Xg = np.diff(Xt)
    freq = np.average(Xg)
    N = int(1200/sampling_duration)
    M = int(sampling_duration/freq)
    Ar = np.zeros(N)
    Al = np.zeros(N)
    for i in range(0,N):
        a = int(M*i)
        b = int(min(M*(i+1),Xk.shape[0]))
        X_rm = Xr[a:b]
        X_lm = Xl[a:b]
        Ar[i] = np.average(X_rm)
        Al[i] = np.average(X_lm)
    return Ar, Al


def visualize(signal_name,eye_name, n_back_name, sampling_duration,out_put_file_name,corref_file_name):

    signal_data = np.loadtxt(signal_name)
    eye_data = np.loadtxt(eye_name,delimiter=',')
    n_back_data = np.loadtxt(n_back_name,delimiter=',')

    #A[0]:time A[1]:fg  A[2]:|fp-fg| A[3]: HR A[4]: RRV  A[5]: right_eye A[6]: left_eye A[7] : 2back

    A = np.zeros((8,int(1200/sampling_duration)))

    offset = signal_data.shape[0]-1200000
    A[0] = np.arange(sampling_duration/2,1200 ,sampling_duration)
    A[3],A[4] = shinpaku(signal_data,sampling_duration,offset)
    A[3] = 60/A[3]
    A[1],A[2] = kokyu(signal_data,sampling_duration,offset)

    A[5],A[6] = eyes(eye_data,sampling_duration)
    A[7] = n_back(n_back_data,sampling_duration)
    tA = A.T
    print(tA)
    np.savetxt(out_put_file_name, tA,delimiter=',')

    fig1, ax = plt.subplots(nrows=5, sharex=True, figsize=(6,6))
    ax[0].plot(A[0],A[1], label="Freq_g")
    ax[0].set_ylabel("Freq_g[Hz]")
    #ax[1].plot(A[0],A[2], label="|Freq_g-Freq_peak|")
    #ax[1].set_ylabel("|Freq_g-Freq_peak|[Hz]")
    ax[1].plot(A[0],A[3], label="HR")
    ax[1].set_ylabel("HR[bpm]")
    #ax[3].plot(A[0],A[4], label="RRV")
    #ax[3].set_ylabel("RRV[s^2]")
    ax[2].plot(A[0],A[5], label="right_eye")
    ax[2].set_ylabel("Pupil\n Diameter(R)[m]")
    ax[3].plot(A[0],A[6], label="left_eye")
    ax[3].set_ylabel("Pupil\n Diameter(L)[m]")
    ax[4].plot(A[0],A[7],label = "2_back")
    ax[4].set_ylabel("2_back correct\n answer rate")
    ax[4].set_xlabel("Time[s]")
    
    

    #相関係数
    Corrcoef = np.empty((6,3))
    for i in range (1,7): 

        #Corrcoef[i][0] = np.corrcoef(A[7], A[i])[0, 1]
        Corrcoef[i-1][0], Corrcoef[i-1][1] = pearsonr(A[7], A[i])
        if Corrcoef[i-1][1]<0.05:
            Corrcoef[i-1][2] = 1
        else:
            Corrcoef[i-1][2] = 0
            

    
    print(Corrcoef)
    np.savetxt( corref_file_name,Corrcoef,delimiter=',')


    fig2, ax2 = plt.subplots(2,2, sharex=True, figsize=(6,6))
    ax2[0,0].scatter(A[7],A[1], label="Freq_g")
    ax2[0,0].set_ylabel("Freq_g[Hz]")
    ax2[0,0].set_xlabel("2_back correct\n answer rate")
    #ax2[1,0].scatter(A[7],A[2], label="|Freq_g-Freq_peak|")
    #ax2[1,0].set_ylabel("|Freq_g-Freq_peak|[Hz]")
    #ax2[1,0].set_xlabel("n_back_score")
    ax2[1,0].scatter(A[7],A[3], label="HR")
    ax2[1,0].set_ylabel("HR[bpm]")
    ax2[1,0].set_xlabel("2_back correct\n answer rate")
    #ax2[0,1].scatter(A[7],A[4], label="RRV")
    #ax2[0,1].set_ylabel("RRV[s^2]")
    #ax2[0,1].set_xlabel("n_back_score")
    ax2[0,1].scatter(A[7],A[5], label="right_eye")
    ax2[0,1].set_ylabel("Pupil\n Diameter(R)[m]")
    ax2[0,1].set_xlabel("2_back correct\n answer rate")
    ax2[1,1].scatter(A[7],A[6], label="left_eye")
    ax2[1,1].set_ylabel("Pupil\n Diameter(L)[m]")
    ax2[1,1].set_xlabel("2_back correct\n answer rate")
    plt.show()
 
if __name__ == '__main__':
    visualize('opensignals.csv','eye.csv','2back.csv',60,'result.csv','corrcoef.csv')


