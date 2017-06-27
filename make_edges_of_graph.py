import matplotlib.pyplot as plt
import networkx as nx
import numpy
import os
import math
import scipy

weight_matrix = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/complete_weight_matrix_set2-3_examining/true_results/true_set2_kmeans_ww11.txt")

# discrete inhibitory and excitetory

#excitatory_weights = []
#inhibitory_weights = []

#for i in range(weight_matrix.shape[0]):
#    for j in range(weight_matrix.shape[0]):
#        if weight_matrix[i,j] == 1:

#           excitatory_weights = excitatory_weights +[[str(i),str(j)]]

#        elif weight_matrix[i,j] == -1:
#             inhibitory_weights = inhibitory_weights +[[str(i),str(j)]]


# don't discrete them

weights = []
#range = weight_matrix.shape[0]
for i in range(weight_matrix.shape[0]):
    for j in range(weight_matrix.shape[0]):
        if weight_matrix[i,j] == 1 or weight_matrix[i,j] == -1:
            weights =weights+[[str(i),str(j)]]
            
            
open("string edge matrix of lR set2 graph.txt","w")
numpy.savetxt("string edge matrix of lR set2 graph.txt",weights,'%s')

graph = nx.Graph(weights)
degree=list(graph.degree().values())
degree=numpy.array(degree)
degree1 = -numpy.sort(-degree)

plt.figure(1)
plt.plot(degree1),plt.title("degree distribution of the linear_regression graph")

plt.figure(2)
plt.plot(degree)

plt.show()




#g1 = nx.Graph(excitatory_weights) 

#g2 = nx.Graph(inhibitory_weights)

#p1 = nx.draw(g1 , node_size = 20   , node_color='r')

#plt.figure(1)
#plt.show(p1)
         
#p2 = nx.draw(g2 , node_size = 20   , node_color='b')

           
#plt.figure(2)
#plt.show(p2)            
#weight_matrix.shape[0]

#a = numpy.array(excitatory_weights)
#b = numpy.array(inhibitory_weights)
#a.shape
#b.shape
#c = numpy.concatenate((a,b),axis=0)
#g1 = nx.Graph(list(c))
#p1 = nx.draw(g1 , node_size = 20   , node_color='g')
#plt.show(p1)


####numpy.savetxt("a.txt",weights,'%s') -->format of saving  is "%s"
# list(graph.degree().values()) --> calculate degre of each node            
