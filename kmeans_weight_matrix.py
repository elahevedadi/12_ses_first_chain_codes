import math
import os
import numpy
from sklearn.cluster import KMeans
import scipy.cluster.vq
import matplotlib.pyplot as plt



weight_matrix = numpy.loadtxt("/Users/Apple/Desktop/set2_no_background.txt")







def kmeans_function(theta,w_min,w_max):
    

           kmeans = KMeans(n_clusters=3, random_state=1).fit(theta)

 #   print((kmeans.cluster_centers_))#chch

                              ######### weighted_Kmeans ##############



           kmeans.cluster_centers_ = numpy.ndarray.tolist(kmeans.cluster_centers_)

           kmeans.cluster_centers_.sort()
  
           kmeans.cluster_centers_ = numpy.array(kmeans.cluster_centers_)

 #          print(kmeans.cluster_centers_)

           kmeans.cluster_centers_[0] = kmeans.cluster_centers_[0]*w_min
			
           kmeans.cluster_centers_[2] =kmeans.cluster_centers_[2]*w_max


 #   print((kmeans.cluster_centers_)) ##chchch


           w_l = scipy.cluster.vq.vq(theta,kmeans.cluster_centers_)

           kmeans_label = w_l[0]-1
           kmeans_centers  = kmeans.cluster_centers_


           return kmeans_label, kmeans_centers



#######################linear########

w_min = 4
w_max = 2

set2_label = numpy.zeros((1817,1817)) 

for i in range(1817):
    a = weight_matrix[i,:].reshape(1817,1)
    kmeans_label, kmeans_centers = kmeans_function(a,w_min,w_max)
    set2_label[i,:] = kmeans_label

#linear_label = kmeans_label


#file = open('linear_label.txt' , "w")       
#numpy.savetxt('linear_label.txt' , linear_label, fmt = '%.18e')

########################logistic##########




