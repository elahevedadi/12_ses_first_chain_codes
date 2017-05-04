import numpy
import math
import matplotlib.pyplot as plt

set2_label = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/complete_weight_matrix_set2-3_examining/set3_no_back_traintest_not_normal_weight.txt")
brain_index = numpy.loadtxt('/Users/Apple/Desktop/first_chain_code_results/complete_weight_matrix_set2-3_examining/brain_index.txt')

labels_with_background = numpy.zeros((1817,4332))



k = 0
for i in range(4332):
    if i in brain_index:
        labels_with_background[:,i] = set2_label[:,k]
        k=k+1

        
        



plt.imshow( labels_with_background),plt.colorbar(),plt.show()       

open("original_set3_new_w+1_w-1.txt","w")
numpy.savetxt("original_set3_new_w+1_w-1.txt",labels_with_background,"%.18e")
