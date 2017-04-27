import scipy.io,os
import numpy as np


least_time_series = 2000000
series = []
for f in sorted(os.listdir("../data/raw")):
    mat = scipy.io.loadmat("../data/raw/" + f)
    rdata = mat["sensor_readings"]
    data = np.asarray(rdata)
    series.append(data.shape[0])
    if data.shape[0]<least_time_series:
        least_time_series=data.shape[0]

print least_time_series
print np.mean(series), np.std(series), np. unique(series)

