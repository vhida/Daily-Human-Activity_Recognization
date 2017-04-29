
import scipy.io,os
import numpy as np
import scipy.stats as stat
import scipy.fftpack as fft


class PreProcessor():

    def __init__(self):

        self.actions = ["walking-forward","walking-left","walking-right","walking-upstairs","walking-downstairs","running forward","jumping Up","sitting","standing","sleeping","elevator-up","elevator-down"]

    def condense(self):
        return self.process(self.condense_time_series,"../data/condensed_time_series_data")

    def extract_stats(self):
        return self.process(self.feature_extraction,"../data/stats_data")

    def condense_3d(self):
        return self.process_3d(self.condense_3d_time_series, "../data/condensed_3d_time_series_data")

    def condense_time_series(self,np_2d_arr):
        n_rows = np_2d_arr.shape[0]
        indice = []
        factor = n_rows/float(600)
        limit = int(n_rows/factor)+1
        for i in range(limit+1):
            if 600 == len(indice):
                break
            index = i*factor
            if index-int(index) < int(index)+1-index:
                indice.append(int(index))
            else:
                indice.append(int(index)+1)
        print len(indice)
        np_600rows_2d_arr = np_2d_arr[indice]

        return np_600rows_2d_arr.flatten()

    def condense_3d_time_series(self,np_2d_arr):
            n_rows = np_2d_arr.shape[0]
            indice = []
            factor = n_rows/float(600)
            limit = int(n_rows/factor)+1
            for i in range(limit+1):
                if 600 == len(indice):
                    break
                index = i*factor
                if index-int(index) < int(index)+1-index:
                    indice.append(int(index))
                else:
                    indice.append(int(index)+1)
            print len(indice)
            np_600rows_2d_arr = np_2d_arr[indice]
            print np_600rows_2d_arr.shape
            return np_600rows_2d_arr

    def process_3d(self,func,npy_file):

        if os.path.isfile(npy_file):
            os.remove(npy_file+"_x")
            os.remove(npy_file+"_y")
            print("binary file removed ! ")

        self.count = 0

        temp_x = []
        temp_y = []
        for f in sorted(os.listdir("../data/raw")):
                mat = scipy.io.loadmat("../data/raw/" + f)
                rdata = mat["sensor_readings"]
                data = np.asarray(rdata)
                label = mat["activity"][0]
                # convert string to number, e.g. walking-forward -->0
                label_code = self.is_action_valid(label)
                cat = label_code
                if -1 == label_code:
                    print(label)
                    continue
                if label_code in range(4):
                    cat = 0
                if 11 == label_code or 10 == label_code:
                    cat = 1
                # get 30 features in a numpy 1D array
                rs = func(data)
                # append the label to the last position in the array
                cat = float(cat)
                label_code = float(label_code)
                temp_y.append(np.asarray([cat,label_code]))
                rs.astype(float)
                temp_x.append(rs)
                self.count += 1
                # print(str(f) + " processed")

        print(str(self.count) + " rows processed")
        # save to binary npy file
        np.save(npy_file+"_x", np.asarray(temp_x))
        np.save(npy_file+"_y", np.asarray(temp_y))



    def process(self,func,npy_file):

        if os.path.isfile(npy_file):
            os.remove(npy_file)
            print("binary file removed ! ")

        self.count = 0

        temp = []
        for f in sorted(os.listdir("../data/raw")):
                mat = scipy.io.loadmat("../data/raw/" + f)
                rdata = mat["sensor_readings"]
                data = np.asarray(rdata)
                label = mat["activity"][0]
                # convert string to number, e.g. walking-forward -->0
                label_code = self.is_action_valid(label)
                cat = label_code
                if -1 == label_code:
                    print(label)
                    continue
                if label_code in range(4):
                    cat = 0
                if 11 == label_code or 10 == label_code:
                    cat = 1
                # get 30 features in a numpy 1D array
                rs = func(data)
                # append the label to the last position in the array
                row_data = np.append(rs, cat)
                row_data = np.append(row_data, label_code)
                row_data.astype(float)
                temp.append(row_data)
                self.count += 1
                # print(str(f) + " processed")


        print(str(self.count) + " rows processed" )
        #save to binary npy file
        np.save(npy_file,np.asarray(temp))
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
    
    def mean_freq(self, y):
        freq, Y = self.fft_freq(y)
        return  np.dot(freq,Y)/np.sum(Y)

    def feature_extraction(self,np_2d_arr):
        #first 5 elements in a row would be min values of 6 attributes
        mins = self.extract(np_2d_arr,np.min)
        maxs = self.extract(np_2d_arr,np.max)
        means = self.extract(np_2d_arr,np.mean)
        stds = self.extract(np_2d_arr,np.std)
        mads = self.extract(np_2d_arr,self.mad)
        
       
        meanfreq = self.extract(np_2d_arr,self.mean_freq)
        
#        skew = self.extract(np_2d_arr,stat.skew)
#        kurtosis= self.extract(np_2d_arr, stat.kurtosis)
        maxfreq = self.extract(np_2d_arr, self.maxFreq)
        rs = np.concatenate((mins,maxs,means,stds,mads, maxfreq, meanfreq),axis=0)
        
        return rs

    def extract(self,np_2d_arr,function):
        row_record = []
        n_col = np_2d_arr.shape[1]
        for i in range(n_col):
            value = function(np_2d_arr[:, i])
            row_record.append(value)
        return np.asarray(row_record)

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


