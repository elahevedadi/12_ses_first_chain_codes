import math
import matplotlib.pyplot as plt
import math
import os
import numpy

x1 = (numpy.reshape(numpy.loadtxt("/Users/Apple/Documents/terme8/project2/Results/Inferred_Graphs/set2_theta/my_theta_F_rr_data27_rr_data35_M_L_X_20_Y_10_R_1_A_1.0_J_0.9_N_"+str(0)+"_mean.txt"),(1,4332)))


for i in range(4331):
    
 
       x2 = (numpy.reshape(numpy.loadtxt("/Users/Apple/Documents/terme8/project2/Results/Inferred_Graphs/set2_theta/my_theta_F_rr_data27_rr_data35_M_L_X_20_Y_10_R_1_A_1.0_J_0.9_N_"+str(i+1)+"_mean.txt"),(1,4332)))
       x1 = numpy.concatenate((x1,x2),axis=0)


print(x1.shape)

for i in range(4332):
             
        x1[i,:] = (x1[i,:])/(0.0001 + numpy.linalg.norm(x1[i,:]))
