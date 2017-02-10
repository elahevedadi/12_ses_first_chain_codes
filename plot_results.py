#===============Import the Necessary Libraries================
import os
try:
    import nibabel as nib
except:
    print("nibabel is not installed")
    
try:
    import matplotlib.pyplot as plt
except:
    print("matplotlib is not installed")

try:
    import numpy
except:
    print("numpy is not installed")
    
import math
import pdb
import sys,getopt
#=============================================================

#==============Read the Command Line Arguments================
reduce_alpha_coef = 0.2
alpha = 1.2
target_voxel_ind = 3342
inference_method = 'L'
num_iter = 20
num_storing_sets_of_theta = 10
training_at_random = 1
num_train_examp = 0.85
file_name_base_results = "../Results"
#=============================================================

#================Create the List of Sessions==================
Session_Names = ['rr_data36','rr_data41']
#=============================================================

#===================File Name Initialization==================
file_name_base = file_name_base_results  + "/Inferred_Graphs/my_theta"
file_name_base = file_name_base + '_F_' + str(Session_Names[0]) + '_'+ str(Session_Names[-1])
file_name_base = file_name_base + '_M_' + str(inference_method)
file_name_base = file_name_base + '_X_' + str(num_iter)
file_name_base = file_name_base + '_Y_' + str(num_storing_sets_of_theta)
file_name_base = file_name_base + '_R_' + str(training_at_random)
file_name_base = file_name_base + '_A_' + str(alpha)
file_name_base = file_name_base + '_J_' + str(num_train_examp)

#=============================================================

#===================Load the Saved Weights====================
if inference_method == 'L':
    file_name = file_name_base + '_N_' + str(target_voxel_ind) + "_mean.txt"
    my_theta_mean = numpy.loadtxt(file_name)
         
    file_name = file_name_base + '_N_' + str(target_voxel_ind) + "_variance.txt"
    my_theta_variance = numpy.loadtxt(file_name)
elif inference_method == 'S':
    file_name = file_name_base + '_N_' + str(target_voxel_ind) + "_sparse.txt"
    my_theta_mean = numpy.loadtxt(file_name)

elif inference_method == 'I':
    file_name = file_name_base + '_N_' + str(target_voxel_ind) + "_pinv.txt"
    my_theta_mean = numpy.loadtxt(file_name)

plt.figure(1)
plt.plot(my_theta_mean) 
plt.title("my_theta_mean")

plt.show()
#=============================================================





