import numpy
import math
import matplotlib.pyplot as plt

set2_label = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/complete_weight_matrix_set2-3_examining/kmeans_set3_test_train_notnormalw+2_w-4.txt")
brain_index = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/complete_weight_matrix_set2-3_examining/brain_index.txt')

labels_with_background = numpy.zeros((1817,4332))



k = 0
for i in range(4332):
    if i in brain_index:
        labels_with_background[:,i] = set2_label[:,k]
        k=k+1

    else:

        labels_with_background[:,i] = -2
        
        
            

        
        



#plt.imshow( labels_with_background),plt.colorbar(),plt.show()       

open("back-2_set3_new_w+2_w-4.txt","w")
numpy.savetxt("back-2_set3_new_w+2_w-4.txt",labels_with_background,"%.18e")
