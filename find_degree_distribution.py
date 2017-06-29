import numpy
import seaborn as sns
import matplotlib.pyplot as plt
binary_LR_weight_matrix = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/complete_weight_matrix_set2-3_examining/true_results/true_set2_kmeans_ww11.txt")

columns = (binary_LR_weight_matrix != 0).sum(0)
rows    = (binary_LR_weight_matrix != 0).sum(1)
total = columns+rows

sns.distplot(total)
plt.show()
