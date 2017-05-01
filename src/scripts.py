import numpy as np
import matplotlib.pyplot as plt
import scipy.io,os


X = np.load('../data/condensed_3d_time_series_data_x.npy')
y = np.load('../data/condensed_3d_time_series_data_y.npy')[:,-1]
# print X.shape
# for i in y:
#    print i





def plotLine(x, lines,filename):
         # Create x values and labels for line graphs
         inds = x
         # labels = ["Training_Errors", "Scores"]
         plt.clf()
         plt.close()
         # Plot a line graph
         plt.figure(2, figsize=(6, 4))  # 6x4 is the aspect ratio for the plot
         for line in lines:
            plt.plot(inds, line)  # Plot the first series in red with circle marker

         # This plots the data
         plt.grid(True)  # Turn the grid on
         # plt.ylabel("Error") #Y-axis label

         # Make sure labels and titles are inside plot area
         plt.tight_layout()

         # Save the chart
         plt.savefig("../stats/_{}.png".format(filename))

         plt.clf()
         plt.close()

elevator_up=[]
elevator_down=[]
walking_forward = []
walking_left = []
walking_upstairs = []
runing = []
sitting = []

for i in range(X.shape[0]):
   if y[i] ==10:
      elevator_up.append(X[i])
   if y[i] ==11:
      elevator_down.append(X[i])
   if y[i] ==7:
      sitting.append(X[i])
   if y[i] ==6:
      runing.append(X[i])
   if y[i] ==0:
       walking_forward.append(X[i])
   if y[i] ==1:
       walking_left.append(X[i])
   if y[i] ==3:
       walking_upstairs.append(X[i])


elevator_up = np.asarray(elevator_up)
elevator_down = np.asarray(elevator_down)
walking_forward= np.asarray(walking_forward)
walking_left= np.asarray(walking_left)
walking_upstairs= np.asarray(walking_upstairs)
runing= np.asarray(runing)
sitting= np.asarray(sitting)


for k in range(6):
   e_u = elevator_up[k]
   e_d = elevator_down[k]

   e_u = e_u.T
   e_d = e_d.T

   wf =walking_forward[k]
   wf = wf.T

   wl= walking_left[k]
   wl = wl.T

   wu = walking_upstairs[k]
   wu = wu.T

   run = runing[k]
   run = run.T

   sit = sitting[k]
   sit = sit.T

   plotLine(range(20),run,"sleeping_{}".format(k))
   plotLine(range(20),sit,"sitting_{}".format(k))
   plotLine(range(20),wu,"walking_upstairs_{}".format(k))
   plotLine(range(20),wl,"walking_left_{}".format(k))
   plotLine(range(20),wf,"waling_forward_{}".format(k))
   plotLine(range(20),e_u,"elevator_up_{}".format(k))
   plotLine(range(20),e_d,"elevator_down{}".format(k))

# for f in sorted(os.listdir("../data/raw")):
#       mat = scipy.io.loadmat("../data/raw/" + f)
#       rdata = mat["sensor_orientation"]
#       rdata1 = mat["sensor_location"]
#
#       print rdata1

# from sklearn.model_selection import KFold
# X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]]*4)
# y = np.array([1, 2, 3, 4]*4)
# kf = KFold(n_splits=3)
# kf.get_n_splits(X)
#
# print(kf)
# print 16 %3 , 16//4
# for train_index, test_index in kf.split(X):
#    print("TRAIN:", train_index, "TEST:", test_index)
#    X_train, X_test = X[train_index], X[test_index]
#    y_train, y_test = y[train_index], y[test_index]