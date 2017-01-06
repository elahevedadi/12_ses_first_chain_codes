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

#ses24 = "/Users/Apple/Desktop/bold data_set2/sub-01_ses-024_task-rest_run-001_bold.nii.gz"
#ses25 = "/Users/Apple/Desktop/bold data_set2/sub-01_ses-025_task-rest_run-001_bold.nii.gz"
#ses26 = "/Users/Apple/Desktop/bold data_set2/sub-01_ses-026_task-rest_run-001_bold.nii.gz"
#ses27 = "/Users/Apple/Desktop/bold data_set2/sub-01_ses-027_task-rest_run-001_bold.nii.gz"
#ses28 = "/Users/Apple/Desktop/bold data_set2/sub-01_ses-028_task-rest_run-001_bold.nii.gz"
#ses29 = "/Users/Apple/Desktop/bold data_set2/sub-01_ses-029_task-rest_run-001_bold.nii.gz"
#ses30 = "/Users/Apple/Desktop/bold data_set2/sub-01_ses-030_task-rest_run-001_bold.nii.gz"
#ses31 = "/Users/Apple/Desktop/bold data_set2/sub-01_ses-031_task-rest_run-001_bold.nii.gz"
#ses32 = "/Users/Apple/Desktop/bold data_set2/sub-01_ses-032_task-rest_run-001_bold.nii.gz"
#ses33 = "/Users/Apple/Desktop/bold data_set2/sub-01_ses-033_task-rest_run-001_bold.nii.gz"
#ses34 = "/Users/Apple/Desktop/bold data_set2/sub-01_ses-034_task-rest_run-001_bold.nii.gz"
#ses35 = "/Users/Apple/Desktop/bold data_set2/sub-01_ses-035_task-rest_run-001_bold.nii.gz"

#data24 , data25 , data26 , data27 , data28 ,  data29 , data30 , data31 , data32 , data33 , data34 , data35 = prepare_12_ses_data(ses24 , ses25 , ses26 , ses27 ,
#                                                                                                                                             ses28 , ses29 , ses30 , ses31 ,
#                                                                                                                                             ses32 , ses33 , ses34 , ses35)
############################################################## 2
#reduced_data , rr_data  =  reduced_number_of_voxels(data24)

#rr_data24 = rr_data

#file = open('rr_data24.txt' , "w")
        
#numpy.savetxt('rr_data24.txt' , rr_data24, fmt = '%.18e')





#reduced_data , rr_data  =  reduced_number_of_voxels(data25)

#rr_data25 = rr_data

#file = open('rr_data25.txt', "w")
        
#numpy.savetxt('rr_data25.txt'  , rr_data25 , fmt = '%.18e')



#reduced_data , rr_data  =  reduced_number_of_voxels(data26)

#rr_data26 = rr_data

#file = open('rr_data26.txt', "w")
        
#numpy.savetxt('rr_data26.txt'  , rr_data26 , fmt = '%.18e')




#reduced_data , rr_data  =  reduced_number_of_voxels(data27)

#rr_data27 = rr_data

#file = open('rr_data27.txt' , "w")
        
#numpy.savetxt('rr_data27.txt' , rr_data27 , fmt = '%.18e')




#reduced_data , rr_data  =  reduced_number_of_voxels(data28)

#rr_data28 = rr_data

#file = open('rr_data28.txt' , "w")
        
#numpy.savetxt('rr_data28.txt' , rr_data28 , fmt = '%.18e')



#reduced_data , rr_data  =  reduced_number_of_voxels(data29)

#rr_data29 = rr_data

#file = open('rr_data29.txt' , "w")
        
#numpy.savetxt('rr_data29.txt' , rr_data29 , fmt = '%.18e')



#reduced_data , rr_data  =  reduced_number_of_voxels(data30)

#rr_data30 = rr_data

#file = open('rr_data30.txt' , "w")
        
#numpy.savetxt('rr_data30.txt' , rr_data30 , fmt = '%.18e')



#reduced_data , rr_data  =  reduced_number_of_voxels(data31)

#rr_data31 = rr_data

#file = open('rr_data31.txt' , "w")
        
#numpy.savetxt('rr_data31.txt' , rr_data31 , fmt = '%.18e')



#reduced_data , rr_data  =  reduced_number_of_voxels(data32)

#rr_data32 = rr_data

#file = open('rr_data32.txt' , "w")
        
#numpy.savetxt('rr_data32.txt' , rr_data32 , fmt = '%.18e')



#reduced_data , rr_data  =  reduced_number_of_voxels(data33)

#rr_data33 = rr_data

#file = open('rr_data33.txt' , "w")
        
#numpy.savetxt('rr_data33.txt' , rr_data33 , fmt = '%.18e')


#reduced_data , rr_data  =  reduced_number_of_voxels(data34)

#rr_data34 = rr_data

#file = open('rr_data34.txt' , "w")
        
#numpy.savetxt('rr_data34.txt' , rr_data34 , fmt = '%.18e')



#reduced_data , rr_data  =  reduced_number_of_voxels(data35)

#rr_data35 = rr_data

#file = open('rr_data35.txt' , "w")
        
#numpy.savetxt('rr_data35.txt' , rr_data35 , fmt = '%.18e')

#rr_data24 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set2_10run/rr_data24.txt')
#rr_data25 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set2_10run/rr_data25.txt')
#rr_data26 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set2_10run/rr_data26.txt')
rr_data27 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set2_10run/rr_data27.txt')
rr_data28 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set2_10run/rr_data28.txt')
rr_data29 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set2_10run/rr_data29.txt')
rr_data30 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set2_10run/rr_data30.txt')
rr_data31 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set2_10run/rr_data31.txt')
rr_data32 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set2_10run/rr_data32.txt')
rr_data33 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set2_10run/rr_data33.txt')
rr_data34 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set2_10run/rr_data34.txt')
rr_data35 = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/9_ses_set2_10run/rr_data35.txt')



######################################### 3

concat_data = concatenate_func(rr_data27 ,
                     rr_data28 , rr_data29 ,
                     rr_data30 , rr_data31 ,
                     rr_data32 , rr_data33 ,
                     rr_data34 , rr_data35 )

print(concat_data.shape)
file = open('concat_data.txt', "w")
        
numpy.savetxt('concat_data.txt'  , concat_data , fmt = '%.18e')

##3temp

target_voxel_ind = 3122
my_theta_mean = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/9_ses_set2_10run/zero_initialize-random_train_examp/my_theta_mean.txt")
c_back0 = numpy.zeros(concat_data.shape)
c_back0 = concat_data

for i in range(my_theta_mean.shape[0]):
    if i != target_voxel_ind:
       if -0.005<my_theta_mean[i]<0.005:
	         c_back0[i,:] = 0 

 


################################### 4
num_train_examp = 0.9

x_train , x_test = make_train_and_test_concat_data(c_back0 , num_train_examp )

################################### 5

critical_times_set = find_critical_times(rr_data27 ,
                        rr_data28 , rr_data29 ,  rr_data30 ,
                        rr_data31 , rr_data32 , rr_data33 , rr_data34,
                        rr_data35)

###################################### 6


num_iter = 200

alpha = 12

reduce_alpha_coef = 1.5

target_voxel_ind = 3122

n5 = x_train.shape[1]

t1 = x_train.shape[0]

num_storing_sets_of_theta = 1

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
 #      file1 = open("my_theta_"+str(i)+".txt" , "w")
#       numpy.savetxt("my_theta_"+str(i)+".txt" , my_theta[:,i] , fmt = '%.18e')
    
       print("before_test_cost_"+str(i)+" "+"is "+str(my_before_test_cost[i]) , end='\n')
       print("before_train_cost_"+str(i)+" "+"is "+str(my_before_train_cost[i]) , end='\n')
    
       print("train_cost_"+str(i)+" "+"is "+str(my_train_cost[i]) , end='\n')
       print("test_cost_"+str(i)+" "+"is "+str(my_test_cost[i]) , end='\n')
 

    
    

######################################################################### 7

   my_theta_mean , my_theta_variance = find_mean_and_variance_of_theta(my_theta)

 #  file_mean = open("my_theta_mean.txt" , "w")
#   numpy.savetxt("my_theta_mean.txt" , my_theta_mean , fmt = '%.18e')

#   file_variance = open("my_theta_variance.txt" , "w")
#   numpy.savetxt("my_theta_variance.txt" , my_theta_variance , fmt = '%.18e')





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






