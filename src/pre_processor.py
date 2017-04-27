import scipy.io,os
import numpy as np


class PreProcessor():

    def __init__(self):

        self.actions = ["walking-forward","walking-left","walking-right","walking-upstairs","walking-downstairs","running forward","jumping Up","sitting","standing","sleeping","elevator-up","elevator-down"]

        if os.path.isfile("../data/csv_data.csv"):
            os.remove("../data/csv_data.csv")
            print("old csv file removed ! ")

        if os.path.isfile("../data/binary_data.npy"):
            os.remove("../data/binary_data.npy")
            print("old binary file removed ! ")

        self.count = 0

        temp = []
        with open("../data/csv_data.csv", 'a') as file:
            for f in sorted(os.listdir("../data/raw")):
                mat = scipy.io.loadmat("../data/raw/"+f)
                rdata  =  mat["sensor_readings"]
                data = np.asarray(rdata)
                label = mat["activity"][0]
                #convert string to number, e.g. walking-forward -->0
                label_code = self.is_action_valid(label)
                cat = label_code
                if  -1 == label_code:
                    print(label)
                    continue
                if label_code in range(4):
                    cat = 0
                if 11==label_code or 10 == label_code:
                    cat = 1
                #get 30 features in a numpy 1D array
                rs  = self.feature_extraction(data)
                #append the label to the last position in the array
                row_data = np.append(rs,cat)
                row_data = np.append(row_data,label_code)
                row_data.astype(float)
                temp.append(row_data)
                file.write(",".join(map(str, row_data))+"\n")
                self.count+=1
                #print(str(f) + " processed")

        print(str(self.count) + " rows processed" )
        #save to binary npy file
        np.save("../data/binary_data.npy",np.asarray(temp))
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
        rs = np.concatenate((mins,maxs,means,stds,mads),axis=0)
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
