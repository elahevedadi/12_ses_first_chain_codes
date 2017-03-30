import math
import os
import numpy
from sklearn.cluster import KMeans
import scipy.cluster.vq
import matplotlib.pyplot as plt



my_theta_mean_linear = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/9_ses_set2_10run/linear_regression/my_theta_mean.txt")
my_theta_mean_linear = my_theta_mean_linear.reshape(-1,1)

my_theta_mean_logistic = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/9_ses_set2_10run/logistic_regression/my_theta_mean.txt")
my_theta_mean_logistic = my_theta_mean_logistic.reshape(-1,1)

my_theta_mean_exp = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/9_ses_set2_10run/exp_regression/my_theta_mean.txt")
my_theta_mean_exp = my_theta_mean_exp.reshape(-1,1)

pinv_theta = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/9_ses_set2_10run/my_pinv_theta.txt")
pinv_theta = pinv_theta.reshape(-1,1)

cross_correlation_x_train = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/9_ses_set2_10run/cross_correlation3122/cross_correlation_3122.txt")
cross_correlation_x_train = cross_correlation_x_train.reshape(-1,1)






def kmeans_function(theta,w_min,w_max):
    

           kmeans = KMeans(n_clusters=3, random_state=1).fit(theta)

 #   print((kmeans.cluster_centers_))#chch

                              ######### weighted_Kmeans ##############



           kmeans.cluster_centers_ = numpy.ndarray.tolist(kmeans.cluster_centers_)

           kmeans.cluster_centers_.sort()
  
           kmeans.cluster_centers_ = numpy.array(kmeans.cluster_centers_)

           print(kmeans.cluster_centers_)

           kmeans.cluster_centers_[0] = kmeans.cluster_centers_[0]*w_min
			
           kmeans.cluster_centers_[2] =kmeans.cluster_centers_[2]*w_max


 #   print((kmeans.cluster_centers_)) ##chchch


           w_l = scipy.cluster.vq.vq(theta,kmeans.cluster_centers_)

           kmeans_label = w_l[0]-1
           kmeans_centers  = kmeans.cluster_centers_


           return kmeans_label, kmeans_centers



#######################linear########

w_min = 1
w_max = 1        


kmeans_label, kmeans_centers = kmeans_function(my_theta_mean_linear,w_min,w_max)

linear_label = kmeans_label
linear_centers = kmeans_centers

file = open('linear_label.txt' , "w")       
numpy.savetxt('linear_label.txt' , linear_label, fmt = '%.18e')

########################logistic##########

w_min = 1
w_max = 1

kmeans_label, kmeans_centers = kmeans_function(my_theta_mean_logistic,w_min,w_max)


logistic_label = kmeans_label
logistic_centers = kmeans_centers

file = open('logistic_label.txt' , "w")       
numpy.savetxt('logistic_label.txt' , logistic_label, fmt = '%.18e')


################################exponential#######

w_min = 1
w_max = 1 


kmeans_label, kmeans_centers = kmeans_function(my_theta_mean_exp ,w_min,w_max)


exp_label = kmeans_label
exp_centers = kmeans_centers

file = open('exp_label.txt' , "w")       
numpy.savetxt('exp_label.txt' , exp_label, fmt = '%.18e')

########################sodo inverse#####################

w_min = 1
w_max = 1 

kmeans_label, kmeans_centers = kmeans_function(pinv_theta,w_min,w_max)

pinv_label = kmeans_label
pinv_centers = kmeans_centers

file = open('pinv_label.txt' , "w")       
numpy.savetxt('pinv_label.txt' , pinv_label, fmt = '%.18e')


#########################cross correlation##############
w_min = 1
w_max = 1 


kmeans_label, kmeans_centers = kmeans_function(cross_correlation_x_train,w_min,w_max)


cross_correlation_label = kmeans_label
cross_correlation_centers = kmeans_centers

file = open('cross_correlation_label.txt' , "w")       
numpy.savetxt('cross_correlation_label.txt' , cross_correlation_label, fmt = '%.18e')
    
########################

xx1 = numpy.reshape(linear_label,(19,19,12))
xx2 = numpy.reshape(logistic_label,(19,19,12))
xx3 = numpy.reshape(exp_label,(19,19,12))
xx4 = numpy.reshape(pinv_label,(19,19,12))
xx5 = numpy.reshape(cross_correlation_label,(19,19,12))

plt.figure(1)
plt.imshow(xx1[:,:,2]),plt.colorbar()
plt.figure(2)
plt.imshow(xx2[:,:,2]),plt.colorbar()
plt.figure(3)
plt.imshow(xx3[:,:,2]),plt.colorbar()
plt.figure(4)
plt.imshow(xx4[:,:,2]),plt.colorbar()
plt.figure(5)
plt.imshow(xx5[:,:,2]),plt.colorbar()

plt.show()
