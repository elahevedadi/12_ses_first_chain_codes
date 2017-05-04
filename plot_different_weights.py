import numpy
import matplotlib.pyplot as plt

w_original = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/complete_weight_matrix_set2-3_examining/14_ordibehesht/original_set3_new_w+1_w-1.txt")

w_1_1 = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/complete_weight_matrix_set2-3_examining/14_ordibehesht/back_set3_new_w+1_w-1.txt")

w_15_5 = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/complete_weight_matrix_set2-3_examining/14_ordibehesht/back_set3_new_w+1.5_w-5.txt")

w_2_4 = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/complete_weight_matrix_set2-3_examining/14_ordibehesht/back_set3_new_w+2_w-4.txt")


w1 = numpy.reshape(w_original[1402,:],(19,19,12))
w2 = numpy.reshape(w_1_1[1402,:],(19,19,12))
w3 = numpy.reshape(w_15_5[1402,:],(19,19,12))
w4 = numpy.reshape(w_2_4[1402,:],(19,19,12))

for i in range(12):

    plt.figure(i)
    plt.imshow(w1[:,:,i]),plt.colorbar(),plt.title("original weights voxel 2855 z = "+str(i))


    plt.figure(i+20)
    plt.imshow(w2[:,:,i]),plt.colorbar(),plt.title("kmeans weights(wmax=1 , wmin=1) weights voxel 2855 z = "+str(i))
        

    plt.figure(i+40)
    plt.imshow(w3[:,:,i]),plt.colorbar(),plt.title("kmeans weights(wmax=1.5 , wmin=5) weights voxel 2855 z = "+str(i))
    
    
    plt.figure(i+70)
    plt.imshow(w4[:,:,i]),plt.colorbar(),plt.title("kmeans weights(wmax=2 , wmin=4) weights voxel 2855 z = "+str(i))                     
