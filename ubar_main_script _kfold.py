import numpy
import matplotlib.pyplot as plt
import csv
from ubar_functions import learning_algorithm
from ubar_functions import find_test_cost
from ubar_functions import find_train_cost
from ubar_functions import plotting_results
from ubar_functions import min_max_algorithm


#data = numpy.loadtxt("../ubar_price/ubar_data.txt")

with open('../ubar_price/new_update_orders_data.csv', 'r') as f:
    reader = csv.reader(f)
    data = list(reader)


with open('../ubar_price/update_orders_data.csv', 'r') as f:
    reader = csv.reader(f)
    old_data = list(reader)

data = numpy.array(data)
old_data = numpy.array(old_data)
old_data[:,0] = 1 # in update dataset sum of the elements of first culomn is blank and this element (' ') can't convert to float


old_data = old_data.astype(numpy.float)


# new fearure = features_name + ["name of the new feature"]
features_name = []

features_name = features_name + ['weight']
features_name = features_name + ['distance']
features_name = features_name + ['vehicle_type']
##features_name = features_name + ['vehicle_options']
features_name = features_name + ['unload_option']
##features_name = features_name + ['dispatch_day']

##features_name = ['price','surplus_costs','load_value','weight','height','length','width','distance','packaging_type','dispatch_month','dispatch_weekday','source_lat','source_lng','destination_lat','destination_lng','unload_option','baarnameh_type','vehicle_type','vehicle_options','dispatch_day','dispatch_date_ms']

features_name = numpy.array(features_name)

f = dict()
x = numpy.zeros((data.shape[0]-1,1))


for i in features_name:
   for j in range(22):
       if str(i) == str(data[0,j]):
          f[j] = numpy.reshape(data[1:data.shape[0],j],(data.shape[0]-1,1))
          x = numpy.concatenate((x,f[j]),1)


data = old_data

x = x.astype(float)
          

#a4 = numpy.reshape(data[:,4],(data.shape[0],1)) #feature_1 = 'weight':4
#a8 = numpy.reshape(data[:,8],(data.shape[0],1)) #feature_2='distance':8
#a18 = numpy.reshape(data[:,18],(data.shape[0],1)) #feature_3='vehicle_type':18
#a19 = numpy.reshape(data[:,19],(data.shape[0],1)) #feature_4='vehicle_options':19
#a11 = numpy.reshape(data[:,11],(data.shape[0],1)) #feature_5='dispatch_weekday':11
#a20 = numpy.reshape(data[:,20],(data.shape[0],1)) # feature_6='dispatch_day':20


#for i in range(a4.shape[0]):
#   a4[i] = numpy.log(a4[i])
#   a8[i] = numpy.log(a8[i])
#   a18[i] = numpy.log(a18[i])
#   a19[i] = numpy.log(a19[i])
 #  a11[i] = numpy.log(a11[i])
#   a20[i] = numpy.log(a20[i])

#b3 = numpy.reshape(data[:,3],(data.shape[0],1))
#b5 = numpy.reshape(data[:,5],(data.shape[0],1))
#b6 = numpy.reshape(data[:,6],(data.shape[0],1))
#b7 = numpy.reshape(data[:,7],(data.shape[0],1))
#b9 = numpy.reshape(data[:,9],(data.shape[0],1))
#b10 = numpy.reshape(data[:,10],(data.shape[0],1))
#b12 = numpy.reshape(data[:,12],(data.shape[0],1))
#b13 = numpy.reshape(data[:,13],(data.shape[0],1))
#b14 = numpy.reshape(data[:,14],(data.shape[0],1))
#b15 = numpy.reshape(data[:,15],(data.shape[0],1))
#b16 = numpy.reshape(data[:,16],(data.shape[0],1))
#b17 = numpy.reshape(data[:,17],(data.shape[0],1))
#b21 = numpy.reshape(data[:,21],(data.shape[0],1))

#b1 = numpy.power(a1,2)
#b2 = numpy.power(a2,2)
#b3 = numpy.power(a3,2)
#b4 = numpy.power(a4,2)
#b5 = numpy.power(a5,2)
#b6 = numpy.power(a6,2)


#c1 = numpy.power(a1,4)
#c2 = numpy.power(a2,4)
#c3 = numpy.power(a3,4)
#c4 = numpy.power(a4,4)
#c5 = numpy.power(a5,4)
#c6 = numpy.power(a6,4)

#x = numpy.concatenate((a4,a8,a11,a18,a19,a20),1)

#x = numpy.loadtxt("/Users/Apple/Documents/tabestane 96/ubar/ubar_codes/ubar_all_19_features_matrix.txt")

x = x[:,1:x.shape[1]]

num_iter = 150

training_at_random = 1

coef_train = 0.2

number_of_features=x.shape[1]

n5 = number_of_features

t1 = numpy.round(coef_train*data.shape[0])

# k fold_cross_validation  ********importatnt point:(coef_train * k = 1) & k is integer***********
k = 5

fold= dict()

fold[0] = x[0*t1:(0+1)*t1 , :]
k_fold_matrix_1 = fold[0]
for i in range(1,k):
   
   fold[i] = x[i*t1:(i+1)*t1 , :]
 #  fold[i] = numpy.reshape(fold[i],(fold[i].shape[0],n5))

   k_fold_matrix_1 = numpy.concatenate((k_fold_matrix_1,fold[i]) ,0)

print(k_fold_matrix_1.shape)

k_fold_matrix = numpy.concatenate((k_fold_matrix_1,k_fold_matrix_1),0)
k_fold_labels = numpy.concatenate((data[0:(k-1)*numpy.round(coef_train*data.shape[0])+numpy.round(coef_train*data.shape[0]),1],data[0:(k-1)*numpy.round(coef_train*data.shape[0])+numpy.round(coef_train*data.shape[0]),1]),0)

print(k_fold_labels.shape)
print(k_fold_matrix.shape)
   
##########################################################


#train_label = data[0:(k-1)*numpy.round(coef_train*data.shape[0]),1]

#test_label = data[(k-1)*numpy.round(coef_train*data.shape[0]):(k-1)*numpy.round(coef_train*data.shape[0])+numpy.round(coef_train*data.shape[0]),1]

num_storing_sets_of_theta = 1

simple_linear_regression = 1

#############################################################################

if simple_linear_regression == 1:

         
   my_theta = numpy.zeros(shape=(k,n5,num_storing_sets_of_theta))

   my_train_cost = numpy.zeros(shape =(k,num_storing_sets_of_theta,1))

   my_before_train_cost = numpy.zeros(shape =(k,num_storing_sets_of_theta,1))

   my_train_cost_per_iter = numpy.zeros(shape =(k,num_iter,num_storing_sets_of_theta))

   my_test_cost = numpy.zeros(shape =(k,num_storing_sets_of_theta,1))

   my_before_test_cost = numpy.zeros(shape =(k,num_storing_sets_of_theta,1))

   my_test_cost_per_iter = numpy.zeros(shape =(k,num_iter,num_storing_sets_of_theta))

   x_train = dict()
   x_test = dict()
   train_label = dict()
   test_label = dict()

   
   

   for j in range(k):
      x_train[j] = k_fold_matrix[j*t1:(j+4)*t1,:]
      x_test[j] = k_fold_matrix[(j+4)*t1:(j+5)*t1,:]
      train_label[j] = k_fold_labels[j*t1:(j+4)*t1]
      test_label[j] = k_fold_labels[(j+4)*t1:(j+5)*t1]

     
      
   
      for i in range(num_storing_sets_of_theta):
    
           theta_transpose,  test_cost , test_cost_per_iter , train_cost , train_cost_per_iter , before_test_cost , before_train_cost = learning_algorithm(x_train[j],x_test[j],train_label[j],test_label[j], num_iter ,training_at_random,coef_train)

           my_theta[j,:,i] = theta_transpose[:,0]
           my_train_cost[j,i] = train_cost
           my_train_cost_per_iter[j,:,i] = train_cost_per_iter
           my_test_cost[j,i] = test_cost
           my_test_cost_per_iter[j,:,i] = test_cost_per_iter
           my_before_train_cost[j,i] = before_train_cost
           my_before_test_cost[j,i] = before_test_cost


  # for i in range(num_storing_sets_of_theta):
#      for j in range(k):
#          file1 = open("my_theta_"+str(i,j)+".txt" , "w")
#          numpy.savetxt("my_theta_"+str(i,j)+".txt" , my_theta[:,i] , fmt = '%.18e')
    
#          print("before_test_cost_"+str(i,j)+" "+"is "+str(my_before_test_cost[j,i]) , end='\n')
#          print("before_train_cost_"+str(i,j)+" "+"is "+str(my_before_train_cost[j,i]) , end='\n')
    
#          print("train_cost_"+str(i,j)+" "+"is "+str(my_train_cost[j,i]) , end='\n')
#          print("test_cost_"+str(i,j)+" "+"is "+str(my_test_cost[j,i]) , end='\n')




##################################################################################################


   my_theta_mean_1 = numpy.mean(my_theta , axis=2)

   my_theta_variance_1 = numpy.var(my_theta , axis = 2)

   my_theta_mean = numpy.mean(my_theta_mean_1 , axis=0)

   my_theta_variance = numpy.var(my_theta_variance_1 , axis =0)

   
       
   file_mean = open("my_theta_mean.txt" , "w")
   numpy.savetxt("my_theta_mean.txt" , my_theta_mean , fmt = '%.18e')       

   file_variance = open("my_theta_variance.txt" , "w")
   numpy.savetxt("my_theta_variance.txt" , my_theta_variance , fmt = '%.18e')
   

#########################################################################################################       

   
   
   test_cost = find_test_cost(test_label[0],x_test[0] ,my_theta_mean ,number_of_features, coef_train )
   train_cost = find_train_cost(train_label[0],x_train[0] ,my_theta_mean ,number_of_features, coef_train  )
   print("test_cost_theta_mean is "+str(test_cost) , end='\n')
   print("train_cost_theta_mean is "+str(train_cost) , end='\n')


################################################################################
 
   mean_my_train_cost_per_iter = numpy.mean(my_train_cost_per_iter,0)
   mean_my_test_cost_per_iter = numpy.mean(my_test_cost_per_iter,0)
   mean_my_theta = numpy.mean(my_theta,0)

                                        


   plotting_results(mean_my_train_cost_per_iter , mean_my_test_cost_per_iter,
                     mean_my_theta, 
                     my_theta_mean ,my_theta_variance,
                     num_storing_sets_of_theta)
   
#######################################


 #  min_max_theta  = min_max_algorithm(train_label, x[:,1:x.shape[1]],coef_train)
