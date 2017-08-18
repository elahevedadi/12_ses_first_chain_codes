
# 1 #

def learning_algorithm(input_train_label,input_test_label, input_x, num_iter ,training_at_random,coef_train):
                                        
     # num_iter is number of gradient descend iteration
     
     
     import random
     import time
     import os
     import numpy
     import math



     input_x_train = input_x[0:numpy.round(coef_train*input_x.shape[0]),:]
     input_x_test = input_x[numpy.round(coef_train*input_x.shape[0])+1:input_x.shape[0]-1,:]


     
     t1_train = input_x_train.shape[0] 
     n5 = input_x_train.shape[1]
     
     numpy.seterr(divide='ignore', invalid='ignore')
     
     theta_transpose = numpy.random.seed(int(10000 * time.clock()))
     theta_transpose = numpy.random.random((n5 , 1 )) #initial theta whith random matrix

 #    theta_transpose = numpy.zeros((n5 , 1 )) #initial theta whith zero matrix
 #    theta_transpose = (theta_transpose)/(0.0001 + numpy.linalg.norm(theta_transpose))
     
    
     
     x_train_normalized = numpy.zeros(shape=(t1_train , n5))


         
     for i in range(n5):
         x_train_normalized[:,i] = (input_x_train[:,i])/(0.0001 + numpy.linalg.norm(input_x_train[:,i]))
################################################################################         
     
     
     train_label_normalized = (input_train_label)/(0.0001 + numpy.linalg.norm(input_train_label))

     test_label_normalized = (input_test_label)/(0.0001 + numpy.linalg.norm(input_test_label))
           

     t1_test = input_x_test.shape[0]
  
    
     test_cost = 0

     x_test_normalized = numpy.zeros(shape=(t1_test , n5))


     for i in range(n5):
         x_test_normalized[:,i] = (input_x_test[:,i])/(0.0001 + numpy.linalg.norm(input_x_test[:,i]))
###############################################################################         
         



     
     ### nonlinear regression

     exp_regression = 0
     logistic_regression = 0


     if exp_regression == 1:

             train_label_normalized = train_label_normalized + 0.00000001
             test_label_normalized = test_label_normalized + 0.00000001
             train_label_normalized = numpy.log(train_label_normalized)
             test_label_normalized = numpy.log(test_label_normalized)

 #            train_label_normalized = (train_label_normalized)/(0.0001 + numpy.linalg.norm(train_label_normalized))
#             test_label_normalized = (test_label_normalized)/(0.0001 + numpy.linalg.norm(test_label_normalized))

     if logistic_regression == 1:
             
             train_label_normalized = train_label_normalized + 0.00000001
             test_label_normalized = test_label_normalized + 0.00000001
             train_label_normalized = -numpy.log((1/(train_label_normalized))-1)
             test_label_normalized = -numpy.log((1/(test_label_normalized))-1)

 #            train_label_normalized = (train_label_normalized)/(0.0001 + numpy.linalg.norm(train_label_normalized))
#             test_label_normalized = (test_label_normalized)/(0.0001 + numpy.linalg.norm(test_label_normalized))





     
     
     #gradient descend algorithm
     cost_func_per_iter = numpy.zeros(shape=(num_iter))
     s = numpy.zeros(shape = (n5,1))
     test_cost_per_iter = numpy.zeros(shape=(num_iter))

     train_cost_per_iter = numpy.zeros(shape=(num_iter))
     before_train_cost_per_iter = numpy.zeros(shape=(num_iter))
     before_test_cost_per_iter=numpy.zeros(shape=(num_iter))

     before_hypo_func_train = numpy.dot((x_train_normalized) , (theta_transpose))
     before_train_cost = (1/t1_train) * math.pow((numpy.linalg.norm( before_hypo_func_train[0:(t1_train)] - (train_label_normalized)[0:(t1_train)])) , 2)

     before_hypo_func_test = numpy.dot((x_test_normalized) , (theta_transpose))
     before_test_cost = (1/t1_test) * math.pow((numpy.linalg.norm( before_hypo_func_test[0:(t1_test)] - (test_label_normalized)[0:(t1_test)])) , 2)



     #training at random:

     if training_at_random == 1:
        training_order =  numpy.random.seed(int(10000 * time.clock()))
        training_order = numpy.random.randint((t1_train - 1) , size = (t1_train - 1))

     elif training_at_random == 0:

        training_order = range(t1_train - 1)

     ##########################   
     
     
     

            
     xT = numpy.transpose(x_train_normalized)

     x_xT = numpy.dot(x_train_normalized , xT)

     y = (train_label_normalized)
     y = numpy.reshape(y,(t1_train,1))
 
     yT = numpy.transpose(y)

     yT = numpy.reshape(yT,(1,t1_train))

     

#########################
             
     
     for ite in range(num_iter):
             cost_func = 0
             test_cost = 0

             train_cost = 0


             itr_i = 0


             
             theta_transpose_T = numpy.transpose(theta_transpose)
             
             m1 = yT - (numpy.dot(theta_transpose_T, xT))

             M = 4*numpy.dot(m1, x_xT)

             h1 = y - numpy.dot(x_train_normalized, theta_transpose)

             pp = numpy.dot(M, h1)

             q1 = numpy.dot(M,x_xT)

             qq = numpy.dot(q1,h1)

             alp = (-1/2)*pp/qq 





               
             for i in training_order:

                 itr_i = itr_i + 1

  
                         
                     
                 hypo_func = numpy.dot((x_train_normalized[i,:]),(theta_transpose))
                     
                 temp= ((  train_label_normalized[i+1]-hypo_func ) * x_train_normalized[i,:])
                 s = s+numpy.reshape(temp,[n5,1])
#                 if  itr_i % (t1_train - 2) == 0:
     
                 if  itr_i % (1) == 0: ### online gradient decsent
 
                       theta_transpose = theta_transpose - (alp/(ite+1)) *s 

                       
                       #theta_transpose = (theta_transpose)/(0.0001 + numpy.linalg.norm(theta_transpose))

                       s = numpy.zeros(shape = (n5,1))
                    
                        
                                      
             
             hypo_func = numpy.dot((x_test_normalized) , (theta_transpose)) 


             hypo_func_train = numpy.dot((x_train_normalized) , (theta_transpose))
             train_cost = (1/t1_train) * math.pow((numpy.linalg.norm( hypo_func_train[0:(t1_train)] - (train_label_normalized)[0:(t1_train)])) , 2)
             train_cost_per_iter[ite] = train_cost

    
             

             test_cost =  (1/t1_test) * math.pow((numpy.linalg.norm( hypo_func[0:(t1_test)] - (test_label_normalized )[0:(t1_test)])) , 2)

             test_cost_per_iter[ite] = test_cost  
             
              
                   
     return theta_transpose,  test_cost , test_cost_per_iter , train_cost , train_cost_per_iter , before_test_cost , before_train_cost






###########################################################################################################################





# 2 #



def find_test_cost(input_test_label,input_x , input_theta_transpose,input_num_features,coef_train):

     import numpy
     import math

     n5  = input_num_features
     
     x_test = input_x[numpy.round(coef_train*input_x.shape[0])+1:input_x.shape[0]-1,:]
     
     t1 = x_test.shape[0]

     test_cost = 0

     x_test_normalized = numpy.zeros(shape=(t1 , n5))

     for i in range(n5):
             
          x_test_normalized[:,i] = (x_test[:,i])/(0.0001 + numpy.linalg.norm(x_test[:,i]))
     
     hypo_func = numpy.dot((x_test_normalized) , (input_theta_transpose))
  

     test_label_normalized = (input_test_label)/(0.0001 + numpy.linalg.norm(input_test_label))


     exp_regression = 0
     logistic_regression = 0


     if exp_regression == 1:


             test_label_normalized = test_label_normalized + 0.00000001

             test_label_normalized = numpy.log(test_label_normalized)

 #            test_label_normalized = (test_label_normalized)/(0.0001 + numpy.linalg.norm(test_label_normalized))

     if logistic_regression == 1:
             

             test_label_normalized = test_label_normalized + 0.00000001

             test_label_normalized = -numpy.log((1/(test_label_normalized))-1)

 #            test_label_normalized = (test_label_normalized)/(0.0001 + numpy.linalg.norm(test_label_normalized))


      
     test_cost =  (1/t1) * math.pow((numpy.linalg.norm( hypo_func[:] - (test_label_normalized)[:])) , 2)
    
    


     return test_cost 
     
####################################################################################


# 3 #



def find_train_cost(input_train_label,input_x , input_theta_transpose,input_num_features,coef_train):

     import numpy
     import math

     n5  = input_num_features
     
     x_train = input_x[0:numpy.round(coef_train*input_x.shape[0]),:]
     
     t1 = x_train.shape[0]

     train_cost = 0

     x_train_normalized = numpy.zeros(shape=(t1 , n5))

     for i in range(n5):
             
          x_train_normalized[:,i] = (x_train[:,i])/(0.0001 + numpy.linalg.norm(x_train[:,i]))
     
     hypo_func = numpy.dot((x_train_normalized) , (input_theta_transpose))
  

     train_label_normalized = (input_train_label)/(0.0001 + numpy.linalg.norm(input_train_label))


     exp_regression = 0
     logistic_regression = 0


     if exp_regression == 1:


             train_label_normalized = train_label_normalized + 0.00000001

             train_label_normalized = numpy.log(train_label_normalized)

#             train_label_normalized = (train_label_normalized)/(0.0001 + numpy.linalg.norm(train_label_normalized))


     if logistic_regression == 1:
             

             train_label_normalized = train_label_normalized + 0.00000001

             train_label_normalized = -numpy.log((1/(train_label_normalized))-1)

 #            train_label_normalized = (train_label_normalized)/(0.0001 + numpy.linalg.norm(train_label_normalized))



      
     train_cost =  (1/t1) * math.pow((numpy.linalg.norm( hypo_func[:] - (train_label_normalized)[:])) , 2)
    
    


     return train_cost




#################################################################################

# 4 #

def plotting_results(input_my_train_cost_per_iter , input_my_test_cost_per_iter,
                     input_my_theta, 
                     input_my_theta_mean ,input_my_theta_variance,
                     num_storing_sets_of_theta):

     import numpy
     import math
     import matplotlib.pyplot as plt

     for i in range(num_storing_sets_of_theta):
         plt.figure(1)
         plt.plot(numpy.log10(input_my_train_cost_per_iter[:,i]))
         plt.title("logaritm plot of "+str(num_storing_sets_of_theta)+
                          "cost_func_per_iter with the same parameters")
         plt.xlabel("number of iterations")
         plt.ylabel("train_cost_per_iter")          
     
     


         plt.figure(2)
         plt.plot(numpy.log10(input_my_test_cost_per_iter[:,i]))
         plt.title("logaritm plot of "+str(num_storing_sets_of_theta)+
                          "test_cost_per_iter with the same parameters")
         plt.xlabel("number of iterations")
         plt.ylabel("test_cost_per_iter")



         
         plt.figure(3)
         plt.plot(input_my_theta[:,i])
         plt.title(" plot of "+str(num_storing_sets_of_theta)+
                          "theta with the same parameters for all important features")
         plt.xlabel("number of features")
         plt.ylabel("theta")


         
         plt.figure(4)
         plt.plot(input_my_theta_mean)
         plt.title(" plot of theta_mean")
         plt.xlabel("number of features")
         plt.ylabel("theta_mean")


         plt.figure(5)
         plt.plot(input_my_theta_variance)
         plt.title(" plot of theta_variance")
         plt.xlabel("number of features")
         plt.ylabel("theta_variance")

     plt.show()



 ##################################################################


# 5 #

def min_max_algorithm(input_train_label, input_x,coef_train):

     import numpy
     import math
     from scipy.optimize import linprog
     import pdb


     
     x_train = input_x[0:numpy.round(coef_train*input_x.shape[0]),:]


     x_train_normalized = numpy.zeros(shape=(x_train.shape[0] , x_train.shape[1]))
        
     for i in range(x_train.shape[1]):
         x_train_normalized[:,i] = (x_train[:,i])/(0.0001 + numpy.linalg.norm(x_train[:,i]))

         

     train_label_normalized = (input_train_label)/(0.0001 + numpy.linalg.norm(input_train_label))    

     

     c = [-1]    # coefficient of r (the maximum residual)
 #    pdb.set_trace()
     A = []



     for i in range(x_train.shape[0]):
          eq_1 = [-1]
          
          for j in range(x_train.shape[1]):
               eq_1 = eq_1+[(-x_train_normalized[i,j])/(train_label_normalized[i])]
               
          A = A + [eq_1]

      



     for i in range(x_train.shape[0]):
          eq_2 = [-1]
          
          for j in range(x_train.shape[1]):
               eq_2 = eq_2+[(x_train_normalized[i,j])/(train_label_normalized[i])]

          A = A + [eq_2]     

                        
     for i in range(x_train.shape[1]):
          c = c +[0]
          
     
     

     b = numpy.zeros((1,2*x_train.shape[0]))

     for i in range(0,x_train.shape[0]):

          b[0,i] = -1
          
     for i in range(x_train.shape[0],2*x_train.shape[0]):

          b[0,i] = 1

 #    pdb.set_trace()          
 #    b = numpy.array(b)
#     c = numpy.array(c)
#     A = numpy.array(A)
######################
     x0_bounds = (0, 100)
     x1_bounds = (-10000, 10000)
     x2_bounds = (-10000, 10000)
     x3_bounds = (-10000, 10000)
     x4_bounds = (-10000, 10000)
     x5_bounds = (-10000, 10000)
     x6_bounds = (-10000, 10000)
     
 #    pdb.set_trace()


 

 #    kwargs = {
          
#           "tableau" : A ,
#           "nit" :100,          
          #"pivot" :The pivot (row, column) used for the next iteration,
#          "phase" : 1
#           }
     



 #    kwargs={'complete': True, 'basis': numpy.array([0, 3]), 'tableau': numpy.array([[ 1., -1.,  2.],
#       [ 0.,  0.,  0.],
#       [ 0., -1.,  2.]]), 'phase': 2, 'pivot': (, 1), 'nit': 1}



# callback = c, **kwargs

            
     res = linprog(c, A_ub=A, b_ub=b,method = "simplex", bounds =(x0_bounds,x1_bounds,x2_bounds,x3_bounds,x4_bounds,x5_bounds,x6_bounds) ,options={"bland": True, "disp":True , "maxiter":500})


     print(res)

     min_max_theta = res.x
    
     return min_max_theta 
     


###############################################


#6#

def price_calculator(input_theta, input_new_customer,input_train_label):

     import numpy

     norm_train_data_price = numpy.linalg.norm(input_train_label)

     print(norm_train_data_price)
    
     t1_train = input_new_customer.shape[0] 
     n5 = input_new_customer.shape[1]

     x_train_normalized = numpy.zeros(shape=(t1_train , n5))
        
     for i in range(n5):
         x_train_normalized[:,i] = (input_new_customer[:,i])/(0.0001 + numpy.linalg.norm(input_new_customer[:,i]))

     input_theta = numpy.reshape(input_theta , (input_new_customer.shape[1],1))

     price = numpy.dot(x_train_normalized,input_theta)

     price = price * norm_train_data_price

     return price

##################################################

#a2 = x[:,1:x.shape[1]]
#a1 = numpy.loadtxt("/Users/Apple/Documents/tabestane 96/ubar/ubar_price/my_theta_mean.txt")
#price = price_calculator(a1,a2,train_label)

#################################################


#7#

























                      
