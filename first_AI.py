import numpy as np

x_enter = np.array(([3,1.5],[2,1],[4,1.5],[3,1],[3.5,0.5],[5.5,1],[1,1],[4,1.5]),dtype=float)
#entry of numbers
y = np.array(([1], [0], [1],[0],[1],[0],[1],[0]), dtype=float)
#1 is red and 0 is blue

x_enter = x_enter/np.amax(x_enter,axis=0)#devide every entry with the highest entry
