import numpy
import os
import math

vox_num = 4332
set_num = 7

pinv_matrix = numpy.zeros((vox_num , set_num))
sparse_matrix = numpy.zeros((vox_num , set_num))
theta_mean_matrix = numpy.zeros((vox_num , set_num))

for i in range(2 , set_num+2):
    
    pinv_matrix[:,i-2] = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/9_ses_set"+str(i)+"_10run/set"+str(i)+"_pinvtheta.txt")

    pinv_matrix[:,i-2] = (pinv_matrix[:,i-2])/(0.0001 + numpy.linalg.norm(pinv_matrix[:,i-2]))
    
    

    sparse_matrix[:,i-2] = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/9_ses_set"+str(i)+"_10run/set"+str(i)+"_sparse_alpha10-7.txt")

    sparse_matrix[:,i-2] = (sparse_matrix[:,i-2])/(0.0001 + numpy.linalg.norm(sparse_matrix[:,i-2]))
    
    

    theta_mean_matrix[:,i-2] = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/9_ses_set"+str(i)+"_10run/random_th_back0/my_theta_mean.txt")




def similarity_function(input_pinv , input_sparse , input_theta_mean):

    pinv_var = numpy.var(input_pinv , axis = 1)

    sparse_var = numpy.var(input_sparse , axis = 1)

    theta_mean_var = numpy.var(input_theta_mean , axis = 1)


    pinv_sim = numpy.mean(pinv_var , axis=0)

    sparse_sim = numpy.mean(sparse_var , axis=0)

    theta_mean_sim = numpy.mean(theta_mean_var , axis=0)

    return pinv_var, sparse_var, theta_mean_var, pinv_sim, sparse_sim, theta_mean_sim
