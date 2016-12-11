#python libraries
import os
import nibabel as nib
import matplotlib.pyplot as plt
import numpy
import math
import pdb
import random
import time

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




######################################################## 1

ses12 = "/Users/Apple/Desktop/bold data/sub-01_ses-012_task-rest_run-001_bold.nii.gz"
ses13 = "/Users/Apple/Desktop/bold data/sub-01_ses-013_task-rest_run-001_bold.nii.gz"
ses14 = "/Users/Apple/Desktop/bold data/sub-01_ses-014_task-rest_run-001_bold.nii.gz"
ses15 = "/Users/Apple/Desktop/bold data/sub-01_ses-015_task-rest_run-001_bold.nii.gz"
ses16 = "/Users/Apple/Desktop/bold data/sub-01_ses-016_task-rest_run-001_bold.nii.gz"
ses17 = "/Users/Apple/Desktop/bold data/sub-01_ses-017_task-rest_run-001_bold.nii.gz"
ses18 = "/Users/Apple/Desktop/bold data/sub-01_ses-016_task-rest_run-001_bold.nii.gz"
ses19 = "/Users/Apple/Desktop/bold data/sub-01_ses-019_task-rest_run-001_bold.nii.gz"
ses20 = "/Users/Apple/Desktop/bold data/sub-01_ses-020_task-rest_run-001_bold.nii.gz"
ses21 = "/Users/Apple/Desktop/bold data/sub-01_ses-021_task-rest_run-001_bold.nii.gz"
ses22 = "/Users/Apple/Desktop/bold data/sub-01_ses-022_task-rest_run-001_bold.nii.gz"
ses23 = "/Users/Apple/Desktop/bold data/sub-01_ses-023_task-rest_run-001_bold.nii.gz"

data12 , data13 , data14 , data15 , data16 ,  data17 , data18 , data19 , data20 , data21 , data22 , data23 = prepare_12_ses_data(ses12 , ses13 , ses14 , ses15 ,
                                                                                                                                             ses16 , ses17 , ses18 , ses19 ,
                                                                                                                                             ses20 , ses21 , ses22 , ses23)
############################################################## 2
#reduced_data , rr_data  =  reduced_number_of_voxels(data12)

#rr_data12 = rr_data

#file = open('rr_data12.txt' , "w")
        
#numpy.savetxt('rr_data12.txt' , rr_data12, fmt = '%.18e')





#reduced_data , rr_data  =  reduced_number_of_voxels(data13)

#rr_data13 = rr_data

#file = open('rr_data13.txt', "w")
        
#numpy.savetxt('rr_data13.txt'  , rr_data13 , fmt = '%.18e')



#reduced_data , rr_data  =  reduced_number_of_voxels(data14)

#rr_data14 = rr_data

#file = open('rr_data14.txt', "w")
        
#numpy.savetxt('rr_data14.txt'  , rr_data14 , fmt = '%.18e')




#reduced_data , rr_data  =  reduced_number_of_voxels(data15)

#rr_data15 = rr_data

#file = open('rr_data15.txt' , "w")
        
#numpy.savetxt('rr_data15.txt' , rr_data15 , fmt = '%.18e')




#reduced_data , rr_data  =  reduced_number_of_voxels(data16)

#rr_data16 = rr_data

#file = open('rr_data16.txt' , "w")
        
#numpy.savetxt('rr_data16.txt' , rr_data16 , fmt = '%.18e')



#reduced_data , rr_data  =  reduced_number_of_voxels(data17)

#rr_data17 = rr_data

#file = open('rr_data17.txt' , "w")
        
#numpy.savetxt('rr_data17.txt' , rr_data17 , fmt = '%.18e')



#reduced_data , rr_data  =  reduced_number_of_voxels(data18)

#rr_data18 = rr_data

#file = open('rr_data18.txt' , "w")
        
#numpy.savetxt('rr_data18.txt' , rr_data18 , fmt = '%.18e')



#reduced_data , rr_data  =  reduced_number_of_voxels(data19)

#rr_data19 = rr_data

#file = open('rr_data19.txt' , "w")
        
#numpy.savetxt('rr_data19.txt' , rr_data19 , fmt = '%.18e')



#reduced_data , rr_data  =  reduced_number_of_voxels(data20)

#rr_data20 = rr_data

#file = open('rr_data20.txt' , "w")
        
#numpy.savetxt('rr_data20.txt' , rr_data20 , fmt = '%.18e')



#reduced_data , rr_data  =  reduced_number_of_voxels(data21)

#rr_data21 = rr_data

#file = open('rr_data21.txt' , "w")
        
#numpy.savetxt('rr_data21.txt' , rr_data21 , fmt = '%.18e')


#reduced_data , rr_data  =  reduced_number_of_voxels(data22)

#rr_data22 = rr_data

#file = open('rr_data22.txt' , "w")
        
#numpy.savetxt('rr_data22.txt' , rr_data22 , fmt = '%.18e')



#reduced_data , rr_data  =  reduced_number_of_voxels(data23)

#rr_data23 = rr_data

#file = open('rr_data23.txt' , "w")
        
#numpy.savetxt('rr_data23.txt' , rr_data23 , fmt = '%.18e')

rr_data12 = numpy.loadtxt('/Users/Apple/Desktop/rr_data12.txt')
rr_data13 = numpy.loadtxt('/Users/Apple/Desktop/rr_data13.txt')
rr_data14 = numpy.loadtxt('/Users/Apple/Desktop/rr_data14.txt')
rr_data15 = numpy.loadtxt('/Users/Apple/Desktop/rr_data15.txt')
rr_data16 = numpy.loadtxt('/Users/Apple/Desktop/rr_data16.txt')
rr_data17 = numpy.loadtxt('/Users/Apple/Desktop/rr_data17.txt')
rr_data18 = numpy.loadtxt('/Users/Apple/Desktop/rr_data18.txt')
rr_data19 = numpy.loadtxt('/Users/Apple/Desktop/rr_data19.txt')
rr_data20 = numpy.loadtxt('/Users/Apple/Desktop/rr_data20.txt')
rr_data21 = numpy.loadtxt('/Users/Apple/Desktop/rr_data21.txt')
rr_data22 = numpy.loadtxt('/Users/Apple/Desktop/rr_data22.txt')
rr_data23 = numpy.loadtxt('/Users/Apple/Desktop/rr_data23.txt')



######################################### 3

concat_data = concatenate_func(rr_data12 , rr_data13 ,
                     rr_data14 , rr_data15 ,
                     rr_data16 , rr_data17 ,
                     rr_data18 , rr_data19 ,
                     rr_data20 , rr_data21 ,
                     rr_data22 , rr_data23 )

print(concat_data.shape)


#file = open('concat_data.txt', "w")
        
#numpy.savetxt('concat_data.txt'  , concat_data , fmt = '%.18e')


#concat_data = numpy.loadtxt('/Users/Apple/Desktop/concat_data.txt')
 


################################### 4
num_train_examp = 0.7

x_train , x_test = make_train_and_test_concat_data(concat_data , num_train_examp )

################################### 5

critical_times_set = find_critical_times(rr_data12 , rr_data13 , rr_data14 , rr_data15 ,
                        rr_data16 , rr_data17 ,  rr_data18 ,
                        rr_data19 , rr_data20 , rr_data21 , rr_data22,
                        rr_data23)

###################################### 6


num_iter = 100

alpha = 1.2

reduce_alpha_coef = 0.2

target_voxel_ind = 3382

n5 = x_train.shape[1]

num_storing_sets_of_theta = 10


         
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
                                                                                                                         critical_times_set)
    
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


    
    

#########################################################################

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







