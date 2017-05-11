import math
import numpy
import os
import pdb
import matplotlib.pyplot as plt

from complete_first_chain_function import shift

concat_data = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/9_ses_set2_10run/concat_data.txt")

brain_index = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/complete_weight_matrix_set2-3_examining/brain_index.txt")

mj = -1
mi = -1
	         
concat_transpose = numpy.transpose(concat_data)

t1 =concat_transpose.shape[0]
#n5 = concat_transpose.shape[1]
num_train_examp = 0.9


#for i in range(t1):
             
#    concat_transpose[i,:] = (concat_transpose[i,:])/(0.0001 + numpy.linalg.norm(concat_transpose[i,:]))

for i in brain_index:
    concat_transpose[:,i] = (concat_transpose[:,i])/(0.0001 + numpy.linalg.norm(concat_transpose[:,i]))
    

x_train = concat_transpose[0:int((num_train_examp)*t1) , :]


cross_correlation_result_of_target_voxel = numpy.zeros((brain_index.shape[0],brain_index.shape[0]))

for j in brain_index:
    mi = -1
    target_voxel_data = x_train[:,j]
    shifted_target_voxel_data = shift(target_voxel_data , -1)
    mj = mj+1
    for i in brain_index:
        mi = mi+1
        cross_correlation_result_of_target_voxel[mi,mj] = numpy.dot(shifted_target_voxel_data,x_train[:,i])######
        
    
#cross_correlation_result_of_target_voxel = (cross_correlation_result_of_target_voxel)/(0.0001 + numpy.linalg.norm(cross_correlation_result_of_target_voxel))

file = open('complete_cross_correlation_matrix_set2.txt' , "w")
        
numpy.savetxt('complete_cross_correlation_matrix_set2.txt' , cross_correlation_result_of_target_voxel, fmt = '%.18e')     
    
#plt.plot(cross_correlation_result_of_target_voxel),plt.show()
    
    
