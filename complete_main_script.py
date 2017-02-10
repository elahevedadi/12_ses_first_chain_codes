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

from complete_first_chain_function import make_train_and_test_concat_data,find_critical_times,concatenate_func_new
from complete_first_chain_function import test_train_check_func_concat_data
from complete_first_chain_function import find_test_cost,find_mean_and_variance_of_theta
from complete_first_chain_function import parse_commandline_args,find_theta_by_solving_matrix_equation,Lasso_linear_regression
#=============================================================

#==============Read the Command Line Arguments================
input_opts, args = getopt.getopt(sys.argv[1:],"hA:F:M:X:Y:R:J:o:")
alpha,target_voxel_ind_range,session_inds_range,inference_method,num_iter,num_storing_sets_of_theta,training_at_random,num_train_examp,file_name_base_results = parse_commandline_args(input_opts)
reduce_alpha_coef = 0.2
#=============================================================


#================Create the List of Sessions==================
#Session_Names = ['rr_data36','rr_data37','rr_data38','rr_data39','rr_data40','rr_data41','rr_data42','rr_data43','rr_data44','rr_data45','rr_data46']
Session_Names = []
for session_ind in range(session_inds_range[0],session_inds_range[1]):
    session_name = 'rr_data' + str(session_ind)
    Session_Names.append(session_name)
#=============================================================


#===================Concatenate the Data=====================
concat_data,critical_times_set = concatenate_func_new(Session_Names)

print(concat_data.shape)
file = open('concat_data.txt', "w")
        
numpy.savetxt('../Data/Concatenated/concat_data.txt'  , concat_data , fmt = '%.18e')
#=============================================================

#============Create the Training and the Test Sets============
x_train , x_test = make_train_and_test_concat_data(concat_data , num_train_examp )
n5 = x_train.shape[1]
t1 = x_train.shape[0]
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

#=============Optimization for Linear Regression==============
for target_voxel_ind in range(target_voxel_ind_range[0],target_voxel_ind_range[1]):

    if inference_method == 'L':
    
        my_theta = numpy.zeros(shape=(n5,num_storing_sets_of_theta))
        my_train_cost = numpy.zeros(shape =(num_storing_sets_of_theta,1))
        my_before_train_cost = numpy.zeros(shape =(num_storing_sets_of_theta,1))
        my_train_cost_per_iter = numpy.zeros(shape =(num_iter,num_storing_sets_of_theta))
        my_test_cost = numpy.zeros(shape =(num_storing_sets_of_theta,1))
        my_before_test_cost = numpy.zeros(shape =(num_storing_sets_of_theta,1))
        my_test_cost_per_iter = numpy.zeros(shape =(num_iter,num_storing_sets_of_theta))
    
        for i in range(num_storing_sets_of_theta):
            theta_transpose,  test_cost , test_cost_per_iter , train_cost , train_cost_per_iter , before_test_cost , before_train_cost = test_train_check_func_concat_data( x_train , x_test ,
                                                                                                                             target_voxel_ind , alpha ,
                                                                                                                             num_iter ,reduce_alpha_coef,
                                                                                                                             critical_times_set,1,training_at_random)
        
            my_theta[:,i] = theta_transpose[:,0]
            my_train_cost[i] = train_cost
            my_train_cost_per_iter[:,i] = train_cost_per_iter
            my_test_cost[i] = test_cost
            my_test_cost_per_iter[:,i] = test_cost_per_iter
            my_before_train_cost[i] = before_train_cost
            my_before_test_cost[i] = before_test_cost

        
            file_name = file_name_base + '_N_' + str(target_voxel_ind) + "_" + str(i) + ".txt"
            numpy.savetxt(file_name, my_theta[:,i] , fmt = '%.18e')
     
            print("before_test_cost_"+str(i)+" for voxel "+ str(target_voxel_ind) + " is "+str(my_before_test_cost[i]) + '\n')
            print("before_train_cost_"+str(i)+" for voxel "+ str(target_voxel_ind) + " is "+str(my_before_train_cost[i]) + '\n')
     
            print("train_cost_"+str(i)+ " for voxel "+ str(target_voxel_ind) + " is "+str(my_train_cost[i]) + '\n')
            print("test_cost_"+str(i)+" for voxel "+ str(target_voxel_ind) + " is "+str(my_test_cost[i]) + '\n')
 
        my_theta_mean , my_theta_variance = find_mean_and_variance_of_theta(my_theta)

        file_name = file_name_base  + '_N_' + str(target_voxel_ind) + "_mean.txt"
        numpy.savetxt(file_name , my_theta_mean , fmt = '%.18e')
     
        file_name = file_name_base + '_N_' + str(target_voxel_ind) + "_variance.txt"
        numpy.savetxt(file_name, my_theta_variance , fmt = '%.18e')

        test_cost = find_test_cost(x_test ,my_theta_mean , target_voxel_ind )
        print("test_cost_theta_mean "+" for voxel "+ str(target_voxel_ind) + " is " +str(test_cost) + '\n')
#=============================================================

#================Optimization for Sparsity====================
    if inference_method == 'S':
    
        sparse_theta =  Lasso_linear_regression(x_train,target_voxel_ind)
    
        file_name = file_name_base + '_N_' + str(target_voxel_ind) + "_sparse.txt"
        numpy.savetxt(file_name, sparse_theta, fmt = '%.18e')
        if 0:
            plt.figure(9)
            plt.plot(sparse_theta)
            plt.title("plot of sparse_theta")
            plt.xlabel("number of voxels")
            plt.ylabel("sparse_theta")
            plt.show()
#=============================================================

#==============Optimization for Pseudo-Inverse================
    if inference_method == 'I':
        pinv_theta = find_theta_by_solving_matrix_equation(x_train , target_voxel_ind)
    
        file_name = file_name_base + '_N_' + str(target_voxel_ind) + "_pinv.txt"
        numpy.savetxt(file_name, pinv_theta , fmt = '%.18e')
#=============================================================

#my_theta_mean = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/12ses_10run/my_theta_mean.txt")

#pinv_theta = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/12ses_10run/pinv_theta.txt")

#train_cost = find_train_cost(x_train , random_theta , target_voxel_ind )
#train_cost_random_theta = train_cost
#print("train_cost_random_theta = "+str(train_cost_random_theta))

#train_cost = find_train_cost(x_train , pinv_theta , target_voxel_ind )
#train_cost_pinv_theta = train_cost
#print("train_cost_pinv_theta = "+str(train_cost_pinv_theta))

#train_cost = find_train_cost(x_train , my_theta_mean , target_voxel_ind )
#train_cost_my_theta_mean = train_cost
#print("train_cost_my_theta_mean= "+str(train_cost_my_theta_mean))




#test_cost = find_test_cost(x_test , random_theta , target_voxel_ind )
#test_cost_random_theta = test_cost
#print("test_cost_random_theta = "+str(test_cost_random_theta))

#test_cost = find_test_cost(x_test , pinv_theta , target_voxel_ind )
#test_cost_pinv_theta = test_cost
#print("test_cost_pinv_theta = "+str(test_cost_pinv_theta))

#test_cost = find_test_cost(x_train , my_theta_mean , target_voxel_ind )
#test_cost_my_theta_mean = test_cost
#print("test_cost_my_theta_mean= "+str(test_cost_my_theta_mean))


#plt.figure(1)
#plt.plot(my_theta_mean) 
#plt.title("my_theta_mean")

#plt.figure(2)
#plt.plot(pinv_theta)
#plt.title(" pinv_theta ")

#plt.show()
###################






