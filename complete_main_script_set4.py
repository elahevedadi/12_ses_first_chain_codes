#python libraries
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy
import math
import pdb
import random
import time
import sklearn
from sklearn import linear_model

#my functions

# 1
from complete_first_chain_function import prepare_12_ses_data
# 2
from complete_first_chain_function import reduced_number_of_voxels
# 3
from complete_first_chain_function import concatenate_func
# 4
from complete_first_chain_function import make_train_and_test_concat_data
# 5
from complete_first_chain_function import find_critical_times
# 6
from complete_first_chain_function import shift
# 7
from complete_first_chain_function import test_train_check_func_concat_data
# 8
from complete_first_chain_function import find_mean_and_variance_of_theta
# 9
from complete_first_chain_function import plotting_results
# 10
from complete_first_chain_function import find_test_cost
# 11
from complete_first_chain_function import find_corresponding_voxel_after_reshape
# 12
from complete_first_chain_function import find_train_cost
# 13
from complete_first_chain_function import find_theta_by_solving_matrix_equation
# 14
from complete_first_chain_function import Lasso_linear_regression





######################################################## 1

#ses36 = "/Users/Apple/Desktop/bold data_set4/sub-01_ses-036_task-rest_run-001_bold.nii.gz"
#ses37 = "/Users/Apple/Desktop/bold data_set4/sub-01_ses-037_task-rest_run-001_bold.nii.gz"
#ses38 = "/Users/Apple/Desktop/bold data_set4/sub-01_ses-038_task-rest_run-001_bold.nii.gz"
#ses39 = "/Users/Apple/Desktop/bold data_set4/sub-01_ses-050_task-rest_run-001_bold.nii.gz"
#ses40 = "/Users/Apple/Desktop/bold data_set4/sub-01_ses-051_task-rest_run-001_bold.nii.gz"
#ses41 = "/Users/Apple/Desktop/bold data_set4/sub-01_ses-053_task-rest_run-001_bold.nii.gz"
#ses42 = "/Users/Apple/Desktop/bold data_set4/sub-01_ses-054_task-rest_run-001_bold.nii.gz"
#ses43 = "/Users/Apple/Desktop/bold data_set4/sub-01_ses-055_task-rest_run-001_bold.nii.gz"
#ses44 = "/Users/Apple/Desktop/bold data_set4/sub-01_ses-056_task-rest_run-001_bold.nii.gz"
#ses45 = "/Users/Apple/Desktop/bold data_set4/sub-01_ses-057_task-rest_run-001_bold.nii.gz"
#ses46 = "/Users/Apple/Desktop/bold data_set4/sub-01_ses-058_task-rest_run-001_bold.nii.gz"
#ses47 = "/Users/Apple/Desktop/bold data_set4/sub-01_ses-059_task-rest_run-001_bold.nii.gz"

#data39 , data40 ,  data41 , data42 , data43 , data44 , data45 , data46 , data47 = prepare_12_ses_data( ses39 ,
#                                                                                                                                             ses40 , ses41 , ses42 , ses43 ,
#                                                                                                                                             ses44 , ses45 , ses46 , ses47)
############################################################## 2
#reduced_data , rr_data  =  reduced_number_of_voxels(data36)

#rr_data36 = rr_data

#file = open('rr_data36.txt' , "w")
        
#numpy.savetxt('rr_data36.txt' , rr_data36, fmt = '%.18e')





#reduced_data , rr_data  =  reduced_number_of_voxels(data37)

#rr_data37 = rr_data

#file = open('rr_data37.txt', "w")
        
#numpy.savetxt('rr_data37.txt'  , rr_data37 , fmt = '%.18e')



#reduced_data , rr_data  =  reduced_number_of_voxels(data38)

#rr_data38 = rr_data

#file = open('rr_data38.txt', "w")
        
#numpy.savetxt('rr_data38.txt'  , rr_data38 , fmt = '%.18e')




#reduced_data , rr_data  =  reduced_number_of_voxels(data39)

#rr_data39 = rr_data

#file = open('rr_data50.txt' , "w")
        
#numpy.savetxt('rr_data50.txt' , rr_data39 , fmt = '%.18e')




#reduced_data , rr_data  =  reduced_number_of_voxels(data40)

#rr_data40 = rr_data

#file = open('rr_data51.txt' , "w")
        
#numpy.savetxt('rr_data51.txt' , rr_data40 , fmt = '%.18e')



#reduced_data , rr_data  =  reduced_number_of_voxels(data41)

#rr_data41 = rr_data

#file = open('rr_data53.txt' , "w")
        
#numpy.savetxt('rr_data53.txt' , rr_data41 , fmt = '%.18e')



#reduced_data , rr_data  =  reduced_number_of_voxels(data42)

#rr_data42 = rr_data

#file = open('rr_data54.txt' , "w")
        
#numpy.savetxt('rr_data54.txt' , rr_data42 , fmt = '%.18e')



#reduced_data , rr_data  =  reduced_number_of_voxels(data43)

#rr_data43 = rr_data

#file = open('rr_data55.txt' , "w")
        
#numpy.savetxt('rr_data55.txt' , rr_data43 , fmt = '%.18e')



#reduced_data , rr_data  =  reduced_number_of_voxels(data44)

#rr_data44 = rr_data

#file = open('rr_data56.txt' , "w")
        
#numpy.savetxt('rr_data56.txt' , rr_data44 , fmt = '%.18e')



#reduced_data , rr_data  =  reduced_number_of_voxels(data45)

#rr_data45 = rr_data

#file = open('rr_data57.txt' , "w")
        
#numpy.savetxt('rr_data57.txt' , rr_data45 , fmt = '%.18e')


#reduced_data , rr_data  =  reduced_number_of_voxels(data46)

#rr_data46 = rr_data

#file = open('rr_data58.txt' , "w")
        
#numpy.savetxt('rr_data58.txt' , rr_data46 , fmt = '%.18e')



#reduced_data , rr_data  =  reduced_number_of_voxels(data47)

#rr_data47 = rr_data

#file = open('rr_data59.txt' , "w")
        
#numpy.savetxt('rr_data59.txt' , rr_data47 , fmt = '%.18e')

#rr_data36 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set3_10run/rr_data36.txt')
#rr_data37 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set3_10run/rr_data37.txt')
#rr_data38= numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set3_10run/rr_data38.txt')
rr_data39 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set4_10run/rr_data50.txt')
rr_data40 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set4_10run/rr_data51.txt')
rr_data41 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set4_10run/rr_data53.txt')
rr_data42 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set4_10run/rr_data54.txt')
rr_data43 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set4_10run/rr_data55.txt')
rr_data44 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set4_10run/rr_data56.txt')
rr_data45 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set4_10run/rr_data57.txt')
rr_data46 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set4_10run/rr_data58.txt')
rr_data47 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set4_10run/rr_data59.txt')



######################################### 3

concat_data = concatenate_func(rr_data39 ,
                     rr_data40 , rr_data41 ,
                     rr_data42 , rr_data43 ,
                     rr_data44 , rr_data45 ,
                     rr_data46 , rr_data47 )

print(concat_data.shape)
file = open('concat_data.txt', "w")
        
numpy.savetxt('concat_data.txt'  , concat_data , fmt = '%.18e')

 








##3temp

target_voxel_ind = 3122
my_theta_mean = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/9_ses_set4_10run/zero_initialize-random_train_examp/my_theta_mean.txt")
c_back0 = numpy.zeros(concat_data.shape)
c_back0 = concat_data

for i in range(my_theta_mean.shape[0]):
    if i != target_voxel_ind:
       if -0.002<my_theta_mean[i]<0.002:
	         c_back0[i,:] = 0 

 


################################### 4
num_train_examp = 0.9

x_train , x_test = make_train_and_test_concat_data(c_back0  , num_train_examp )

################################### 5

critical_times_set = find_critical_times(rr_data39 ,
                        rr_data40 , rr_data41 ,  rr_data42 ,
                        rr_data43 , rr_data44 , rr_data45 , rr_data46,
                        rr_data47)

###################################### 6



num_iter = 100

alpha = 12

reduce_alpha_coef = 1.5

target_voxel_ind = 3122

n5 = x_train.shape[1]

t1 = x_train.shape[0]

num_storing_sets_of_theta = 10

##
training_at_random = 1
##
Lasso= 0
##
simple_linear_regression = 1



if simple_linear_regression == 1:

         
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




   for i in range(num_storing_sets_of_theta):
       file1 = open("my_theta_"+str(i)+".txt" , "w")
       numpy.savetxt("my_theta_"+str(i)+".txt" , my_theta[:,i] , fmt = '%.18e')
    
       print("before_test_cost_"+str(i)+" "+"is "+str(my_before_test_cost[i]) , end='\n')
       print("before_train_cost_"+str(i)+" "+"is "+str(my_before_train_cost[i]) , end='\n')
    
       print("train_cost_"+str(i)+" "+"is "+str(my_train_cost[i]) , end='\n')
       print("test_cost_"+str(i)+" "+"is "+str(my_test_cost[i]) , end='\n')
 

    
    

######################################################################### 7

   my_theta_mean , my_theta_variance = find_mean_and_variance_of_theta(my_theta)

   file_mean = open("my_theta_mean.txt" , "w")
   numpy.savetxt("my_theta_mean.txt" , my_theta_mean , fmt = '%.18e')

   file_variance = open("my_theta_variance.txt" , "w")
   numpy.savetxt("my_theta_variance.txt" , my_theta_variance , fmt = '%.18e')





####################################################################################333

   test_cost = find_test_cost(x_test ,my_theta_mean , target_voxel_ind )
   print("test_cost_theta_mean is "+str(test_cost) , end='\n')


######################################################################


   plotting_results(my_train_cost_per_iter , my_test_cost_per_iter,
                     my_theta, 
                     my_theta_mean ,my_theta_variance,
                     num_storing_sets_of_theta)
###############################################

if Lasso == 1:

   sparse_theta =  Lasso_linear_regression(x_train,target_voxel_ind)

   file = open("sparse_theta.txt" , "w")
   numpy.savetxt("sparse_theta.txt" , sparse_theta, fmt = '%.18e')
   
   plt.figure(9)
   plt.plot(sparse_theta)
   plt.title("plot of sparse_theta")
   plt.xlabel("number of voxels")
   plt.ylabel("sparse_theta")
   plt.show()

    


######################################################################




################################################################## related to function 13
#concat_data = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/12_ses_results/concat_data.txt")

#x_train , x_test = make_train_and_test_concat_data(concat_data , num_train_examp)

######

#pinv_theta = find_theta_by_solving_matrix_equation(x_train , target_voxel_ind)

#file = open("pinv_theta.txt" , "w")
#numpy.savetxt("pinv_theta.txt" , pinv_theta , fmt = '%.18e')

########################

#my_theta_mean = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/12ses_10run/my_theta_mean.txt")

#pinv_theta = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/12ses_10run/pinv_theta.txt")

#random_theta = numpy.random.random((4332))
#random_theta = (random_theta)/(0.0001 + numpy.linalg.norm(random_theta))

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





