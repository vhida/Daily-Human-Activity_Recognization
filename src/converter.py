import scipy.io,os
import numpy as np


shape = []
for f in sorted(os.listdir("../data/raw")):
    try:
        mat = scipy.io.loadmat("../data/raw/"+f)
        rdata  =  mat["sensor_readings"]
        data = np.asarray(rdata)
        label = mat["activity"]
        shape.append(data.shape)
    except ValueError:
        print f
print len(shape)