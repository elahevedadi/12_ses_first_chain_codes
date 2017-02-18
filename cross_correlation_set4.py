import math
import numpy
import os
import pdb
import matplotlib.pyplot as plt

from complete_first_chain_function import shift

concat_data = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/9_ses_set4_10run/concat_data.txt")



my_theta_mean = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/9_ses_set3_10run/zero_initialize-random_train_examp/my_theta_mean.txt")

target_voxel_ind = 3122

for i in range(my_theta_mean.shape[0]):
    if i != target_voxel_ind:
       if -0.002<my_theta_mean[i]<0.002:
	         concat_data[i,:] = 0
	         
concat_transpose = numpy.transpose(concat_data)


for i in range(concat_transpose.shape[0]):
             
    concat_transpose[i,:] = (concat_transpose[i,:])/(0.0001 + numpy.linalg.norm(concat_transpose[i,:]))



target_voxel_data = concat_transpose[:,target_voxel_ind]
shifted_target_voxel_data = shift(target_voxel_data , -1)

cross_correlation_result_of_target_voxel = numpy.zeros((concat_transpose.shape[1]))

for i in range(concat_transpose.shape[1]):

    cross_correlation_result_of_target_voxel[i] = numpy.dot(shifted_target_voxel_data,concat_transpose[:,i])

    
cross_correlation_result_of_target_voxel = (cross_correlation_result_of_target_voxel)/(0.0001 + numpy.linalg.norm(cross_correlation_result_of_target_voxel))

file = open('cross_correlation_result_of_set_4.txt' , "w")
        
numpy.savetxt('cross_correlation_result_of_set_4.txt' , cross_correlation_result_of_target_voxel, fmt = '%.18e')     
    
plt.plot(cross_correlation_result_of_target_voxel),plt.show()
    
    
