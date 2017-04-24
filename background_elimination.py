import math
import matplotlib.pyplot as plt
import math
import os
import numpy

#x1 = (numpy.reshape(numpy.loadtxt("/Users/Apple/Documents/terme8/project2/Results/Inferred_Graphs/set3_theta/my_theta_F_rr_data39_rr_data46_M_L_X_20_Y_10_R_1_A_1.0_J_0.9_N_"+str(0)+"_mean.txt"),(1,4332)))


#for i in range(4331):
    
 
 #      x2 = (numpy.reshape(numpy.loadtxt("/Users/Apple/Documents/terme8/project2/Results/Inferred_Graphs/set3_theta/my_theta_F_rr_data39_rr_data46_M_L_X_20_Y_10_R_1_A_1.0_J_0.9_N_"+str(i+1)+"_mean.txt"),(1,4332)))
 #      x1 = numpy.concatenate((x1,x2),axis=0)


#print(x1.shape)

#for i in range(4332):
             
 #       x1[i,:] = (x1[i,:])/(0.0001 + numpy.linalg.norm(x1[i,:]))


#

back_index = numpy.zeros((1,4000))
m =0

theta_mean = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/9_ses_set2_10run/zero_initialize-random_train_examp/my_theta_mean.txt")
     

for i in range(theta_mean.shape[0]):
    if -0.002<theta_mean[i]<0.002:
        back_index[0,m]=i
        m=m+1

x1 = numpy.loadtxt("/Users/Apple/Desktop/set2_new.txt")
x3 = x1[:,0]
x3 = numpy.reshape(x3,(4332,1))



          

for i in range(4332):
    
      if i not in back_index: 
    
         x5 = numpy.reshape(x1[:,i],(4332,1))

         x3 = numpy.concatenate((x3,x5),axis=1)

print(x3.shape)



x10 = x3[0,:]
x10 = numpy.reshape(x10,(1,x3.shape[1]))

for i in range(4332):

    if i not in back_index:  

       x11 = numpy.reshape(x3[i,:],(1,x3.shape[1]))

       x10 = numpy.concatenate((x10,x11),axis=0)

print(x10.shape)

    
      
