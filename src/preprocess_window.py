# -*- coding: utf-8 -*-
"""
Created on Mon Apr 24 11:43:29 2017

@author: Jian Yang local
"""

import scipy.io,os
import numpy as np
import scipy.stats as stat
import scipy.fftpack as fft


class PreProcessor():

    def __init__(self):

        self.actions = ["walking-forward","walking-left","walking-right","walking-upstairs","walking-downstairs","running","jumping","sitting","standing","sleeping","elevator-up","elevator-down"]
        if os.path.isfile("../data/processed_data_window.csv"):
            os.remove("../data/processed_data_window.csv")
            print("old file removed ! ")

        self.count = 0
        self.windows = 4
        
        

#        with open("../data/processed_data.csv", 'a') as file:
        filedata = []
        for f in sorted(os.listdir("../data/raw")):
            mat = scipy.io.loadmat("../data/raw/"+f)
            rdata  =  mat["sensor_readings"]
            data = np.asarray(rdata)
            label = mat['activity'][0].encode("utf-8")
            
            #convert string to number, e.g. walking-forward -->0
            label_code = self.is_action_valid(label)
            if  -1 == label_code:
                print(label)
                continue
            #get 30 features in a numpy 1D array
            rs  = self.feature_extraction(data)
            #append the label to the last position in the array
            row_data = np.append(rs,label_code)
            row_data.astype(float)
            filedata.append(row_data)
        filecsv = np.array(filedata)
        print filecsv.shape
        np.savetxt("../data/processed_data_window.csv", filecsv, delimiter=",")
        
                
                
#                file.write(" ".join(map(str, row_data))+"\n")
#                self.count+=1
#                print(str(f) + " processed")
#                print(self.count)

                # np.savetxt(filename,row_data,delimiter=" , ")

    def mad(self,a,axis=None):
        '''

        :param a:array-like
        Input array or object that can be converted to an array.
        :param axis:int, optional
        Axis along which the MADs are computed.  The default (`None`) is
        to compute the MAD of the flattened array.
        :return: float or `~numpy.ndarray`
        The median absolute deviation of the input array.  If ``axis``
        is `None` then a scalar will be returned, otherwise a
        `~numpy.ndarray` will be returned.
        '''


        if isinstance(a, np.ma.MaskedArray):
            func = np.ma.median
        else:
            func = np.median

        a = np.asanyarray(a)
        a_median = func(a, axis=axis)

        # broadcast the median array before subtraction
        if axis is not None:
            a_median = np.expand_dims(a_median, axis=axis)

        return func(np.abs(a - a_median), axis=axis)

    def feature_extraction(self,np_2d_arr):
        #first 5 elements in a row would be min values of 6 attributes
        mins = self.extract(np_2d_arr,np.min)
        maxs = self.extract(np_2d_arr,np.max)
        means = self.extract(np_2d_arr,np.mean)
        stds = self.extract(np_2d_arr,np.std)
        mads = self.extract(np_2d_arr,self.mad)
        
        skew = self.extract(np_2d_arr,stat.skew)
        kurtosis= self.extract(np_2d_arr, stat.kurtosis)
        maxfreq = self.extract(np_2d_arr, self.maxFreq)
        meanfreq = self.extract(np_2d_arr, self.meanFreq)
        
        corr = self.extract_pearson(np_2d_arr)
        rs = np.concatenate((mins,maxs,means,stds,mads,skew, kurtosis, maxfreq, meanfreq, corr),axis=0)
#        rs = np.concatenate((mins,maxs,means,stds,mads, maxfreq),axis=0)
        return rs

    def extract(self,np_2d_arr,function):
        row_record = []
        n_col = np_2d_arr.shape[1]
        for i in range(n_col):
            col = np_2d_arr[:,i]
            n_row = col.size
            half_window = int(np.floor(n_row/(self.windows +1)))
            window = 2*half_window
            
            for j in range(self.windows):
                value = function(col[j*half_window:j*half_window+window])    
                row_record.append(value)
                
            
        return np.asarray(row_record)
        
    def extract_pearson(self, np_2d_arr):
        Corr = []
        Corr_XY_value = stat.pearsonr(np_2d_arr[:,0], np_2d_arr[:,1])[0]
        Corr_YZ_value = stat.pearsonr(np_2d_arr[:,1], np_2d_arr[:,2])[0] 
        Corr_XZ_value = stat.pearsonr(np_2d_arr[:,0], np_2d_arr[:,2])[0] 
        Corr_ab_value = stat.pearsonr(np_2d_arr[:,3], np_2d_arr[:,4])[0]
        Corr_bc_value = stat.pearsonr(np_2d_arr[:,4], np_2d_arr[:,5])[0] 
        Corr_ac_value = stat.pearsonr(np_2d_arr[:,3], np_2d_arr[:,5])[0] 
        Corr.append (Corr_XY_value)
        Corr.append (Corr_YZ_value)
        Corr.append (Corr_XZ_value)
        Corr.append (Corr_ab_value)
        Corr.append (Corr_bc_value)
        Corr.append (Corr_ac_value)
        
        return np.asarray(Corr)
        
        
#    def maxFreq(self, y):
#        
#        Fs = 100  # sampleing frequency 100 hz
#        n =len(y)
#        T = n/Fs
#        k = 0.5*np.arange(n)
#        
#        freq = k/(T) # all possible frequency
#    
#        Y = fft.rfft(y) # fft computing and normalizatin
#              
#        return freq[np.abs(Y)==np.max(np.abs(Y))][0] 
                    
    def maxFreq(self, y):
        """
        This function transform y into frequency space through fourier transformation
        :param y: array-like
        Input array 
        :return: the frequency component with strongest intensity        
        """
        freq, Y = self.fft_freq(y)
        
        return freq[np.abs(Y)==np.max(np.abs(Y))][0] 
                    
    def fft_freq(self, y):
        
        """
        This function transform y into frequency space through fourier transformation
        :param y: array-like
        Input array 
        :return: freq:the frequency components with intensity >1e10-6, Y: corresponding intensity       
        """
        Fs = 100  # sampleing frequency 100 hz
        n =len(y)
        T = n/Fs
        k = 0.5*np.arange(n)
        
        freq = k/(T) # all possible frequency
    
        Y = abs(fft.rfft(y)) # fft computing and normalizatin
        tot = 1e-10
        return freq[Y>tot], Y[Y>tot]
                    
    def meanFreq(self, y):
        freq, Y = self.fft_freq(y)
        return  np.dot(freq,Y)/np.sum(Y)

    def is_action_valid(self,action):
        # in actions
        if action in self.actions:
            return self.actions.index(action)
        for act in self.actions:
            # is a subtring of some action
            if action in act:
                return self.actions.index(act)
            # parts of label are subtrings of some action
            if action.split("-")[0] in act.split("-")[0] and action.split("-")[0] in act.split("-")[0] :
                return self.actions.index(act)
        return -1
        
a = PreProcessor()