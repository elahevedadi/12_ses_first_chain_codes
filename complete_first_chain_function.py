
import pdb
################ 1
       
def prepare_12_ses_data(input_ses12 , input_ses13 ,
                   input_ses14 , input_ses15 ,
                   input_ses16 , input_ses17 ,
                   input_ses18 , input_ses19 ,
                   input_ses20 , input_ses21 ,
                   input_ses22 , input_ses23):

  
        import os
        import numpy
        import math
        import nibabel as nib
        

        example12 = nib.load(input_ses12)
        example13 = nib.load(input_ses13)
        example14 = nib.load(input_ses14)
        example15 = nib.load(input_ses15)
        example16 = nib.load(input_ses16)
        example17 = nib.load(input_ses17)
        example18 = nib.load(input_ses18)
        example19 = nib.load(input_ses19)
        example20 = nib.load(input_ses20)
        example21 = nib.load(input_ses21)
        example22 = nib.load(input_ses22)
        example23 = nib.load(input_ses23)



        data12 = example12.get_data()
        data13 = example13.get_data()
        data14 = example14.get_data()
        data15 = example15.get_data()
        data16 = example16.get_data()
        data17 = example17.get_data()
        data18 = example18.get_data()
        data19 = example19.get_data()
        data20 = example20.get_data()
        data21 = example21.get_data()
        data22 = example22.get_data()
        data23 = example23.get_data()

        return data12,data13,data14,data15,data16,data17,data18,data19,data20,data21,data22,data23




################## 2
def reduced_number_of_voxels(input_data):
     
 
        import os
        import numpy
        import math
        
     
        x = input_data.shape[0] 
        y = input_data.shape[1]
        z = input_data.shape[2]
        t = input_data.shape[3]
        n = x*y*z
        reduced_data = numpy.zeros(shape=(x//5,y//5,z//5,t))
        for tt in range(t):
    
            for m in range(x//5):
                for n in range(y//5):
                    for p in range(z//5):
                        i = (5*(m+1))-3
                        j = (5*(n+1))-3
                        k = (5*(p+1))-3
                        totaldata = [input_data[i-2,j+2,k,tt],input_data[i-1,j+2,k,tt],input_data[i,j+2,k,tt],input_data[i+1,j+2,k,tt],input_data[i+2,j+2,k,tt],
                                     input_data[i-2,j+1,k,tt],input_data[i-1,j+1,k,tt],input_data[i,j+1,k,tt],input_data[i+1,j+1,k,tt],input_data[i+2,j+1,k,tt],
                                     input_data[i-2,j,k,tt],input_data[i-1,j,k,tt],input_data[i,j,k,tt],input_data[i+1,j,k,tt],input_data[i+2,j,k,tt],
                                     input_data[i-2,j-1,k,tt],input_data[i-1,j-1,k,tt],input_data[i,j-1,k,tt],input_data[i+1,j-1,k,tt],input_data[i+2,j-1,k,tt],
                                     input_data[i-2,j-2,k,tt],input_data[i-1,j-2,k,tt],input_data[i,j-2,k,tt],input_data[i+1,j-2,k,tt],input_data[i+2,j-2,k,tt],
                                     input_data[i-2,j+2,k-2,tt],input_data[i-1,j+2,k-2,tt],input_data[i,j+2,k-2,tt],input_data[i+1,j+2,k-2,tt],input_data[i+2,j+2,k-2,tt],
                                     input_data[i-2,j+1,k-2,tt],input_data[i-1,j+1,k-2,tt],input_data[i,j+1,k-2,tt],input_data[i+1,j+1,k-2,tt],input_data[i+2,j+1,k-2,tt],
                                     input_data[i-2,j,k-2,tt],input_data[i-1,j,k-2,tt],input_data[i,j,k-2,tt],input_data[i+1,j,k-2,tt],input_data[i+2,j,k-2,tt],
                                     input_data[i-2,j-1,k-2,tt],input_data[i-1,j-1,k-2,tt],input_data[i,j-1,k-2,tt],input_data[i+1,j-1,k-2,tt],input_data[i+2,j-1,k-2,tt],
                                     input_data[i-2,j-2,k-2,tt],input_data[i-1,j-2,k-2,tt],input_data[i,j-2,k-2,tt],input_data[i+1,j-2,k-2,tt],input_data[i+2,j-2,k-2,tt],
                                     input_data[i-2,j+2,k-1,tt],input_data[i-1,j+2,k-1,tt],input_data[i,j+2,k-1,tt],input_data[i+1,j+2,k-1,tt],input_data[i+2,j+2,k-1,tt],
                                     input_data[i-2,j+1,k-1,tt],input_data[i-1,j+1,k-1,tt],input_data[i,j+1,k-1,tt],input_data[i+1,j+1,k-1,tt],input_data[i+2,j+1,k-1,tt],
                                     input_data[i-2,j,k-1,tt],input_data[i-1,j,k-1,tt],input_data[i,j,k-1,tt],input_data[i+1,j,k-1,tt],input_data[i+1,j,k-1,tt],
                                     input_data[i-2,j-1,k-1,tt],input_data[i-1,j-1,k-1,tt],input_data[i,j-1,k-1,tt],input_data[i+1,j-1,k-1,tt],input_data[i+2,j-1,k-1,tt],
                                     input_data[i-2,j-2,k-1,tt],input_data[i-1,j-2,k-1,tt],input_data[i,j-2,k-1,tt],input_data[i+1,j-2,k-1,tt],input_data[i+2,j-2,k-1,tt],
                                     input_data[i-2,j+2,k+1,tt],input_data[i-1,j+2,k+1,tt],input_data[i,j+2,k+1,tt],input_data[i+1,j+2,k+1,tt],input_data[i+2,j+2,k+1,tt],
                                     input_data[i-2,j+1,k+1,tt],input_data[i-1,j+1,k+1,tt],input_data[i,j+1,k+1,tt],input_data[i+1,j+1,k+1,tt],input_data[i+2,j+1,k+1,tt],
                                     input_data[i-2,j,k+1,tt],input_data[i-1,j,k+1,tt],input_data[i,j,k+1,tt],input_data[i+1,j,k+1,tt],input_data[i+1,j,k+1,tt],
                                     input_data[i-2,j-1,k+1,tt],input_data[i-1,j-1,k+1,tt],input_data[i,j-1,k+1,tt],input_data[i+1,j-1,k+1,tt],input_data[i+2,j-1,k+1,tt],
                                     input_data[i-2,j-2,k+1,tt],input_data[i-1,j-2,k+1,tt],input_data[i,j-2,k+1,tt],input_data[i+1,j-2,k+1,tt],input_data[i+2,j-2,k+1,tt],
                                     input_data[i-2,j+2,k+2,tt],input_data[i-1,j+2,k+2,tt],input_data[i,j+2,k+2,tt],input_data[i+1,j+2,k+2,tt],input_data[i+2,j+2,k+2,tt],
                                     input_data[i-2,j+1,k+2,tt],input_data[i-1,j+1,k+2,tt],input_data[i,j+1,k+2,tt],input_data[i+1,j+1,k+2,tt],input_data[i+2,j+1,k+2,tt],
                                     input_data[i-2,j,k+2,tt],input_data[i-1,j,k+2,tt],input_data[i,j,k+2,tt],input_data[i+1,j,k+2,tt],input_data[i+1,j,k+2,tt],
                                     input_data[i-2,j-1,k+2,tt],input_data[i-1,j-1,k+2,tt],input_data[i,j-1,k+2,tt],input_data[i+1,j-1,k+2,tt],input_data[i+2,j-1,k+2,tt],
                                     input_data[i-2,j-2,k+2,tt],input_data[i-1,j-2,k+2,tt],input_data[i,j-2,k+2,tt],input_data[i+1,j-2,k+2,tt],input_data[i+2,j-2,k+2,tt]]

                                                          
                                                                                 
                        reduced_data[m,n,p,tt]= numpy.mean(totaldata)

        x1 = reduced_data.shape[0] 
        y1 = reduced_data.shape[1]
        z1 = reduced_data.shape[2]
        t1 = reduced_data.shape[3]
        n5 = x1*y1*z1

        rr_data = numpy.reshape(reduced_data , (n5,t1))
        return reduced_data , rr_data

################################################## 3

def concatenate_func( input_rr_data27 ,
                     input_rr_data28 , input_rr_data29 ,
                     input_rr_data30 , input_rr_data31 ,
                     input_rr_data32 , input_rr_data33 ,
                     input_rr_data34 , input_rr_data35 ):
    

    import os
    import numpy
    import math


    concat_data = numpy.concatenate(( input_rr_data27 ,
                     input_rr_data28 , input_rr_data29 ,
                     input_rr_data30 , input_rr_data31 ,
                     input_rr_data32 , input_rr_data33 ,
                     input_rr_data34 , input_rr_data35 ) , axis=1)


    return concat_data




###################################################### 4


def make_train_and_test_concat_data(input_concat_data , num_train_examp ):

    import os
    import numpy
    import math


    t1 = input_concat_data.shape[1]
    n5 = input_concat_data.shape[0]
    

    concat_data_transpose = numpy.transpose(input_concat_data)



    x_train = concat_data_transpose[0:int((num_train_examp)*t1) , :]

    x_test  = concat_data_transpose[int((num_train_examp*t1)) : t1 , :]

    return x_train , x_test

################################################## 5

def find_critical_times( input_rr_data27 ,
                     input_rr_data28 , input_rr_data29 ,
                     input_rr_data30 , input_rr_data31 ,
                     input_rr_data32 , input_rr_data33 ,
                     input_rr_data34 , input_rr_data35 ):

    import os
    import numpy
    import math

    critical_time_27 = input_rr_data27.shape[1] - 1
    
    critical_time_28 = input_rr_data28.shape[1] + critical_time_27

    critical_time_29 = input_rr_data29.shape[1] + critical_time_28

    critical_time_30 = input_rr_data30.shape[1] + critical_time_29

    critical_time_31 = input_rr_data31.shape[1] + critical_time_30

    critical_time_32 = input_rr_data32.shape[1] + critical_time_31

    critical_time_33 = input_rr_data33.shape[1] + critical_time_32

    critical_time_34 = input_rr_data34.shape[1] + critical_time_33

    critical_time_35 = input_rr_data35.shape[1] + critical_time_34



    
    critical_times_set = [ critical_time_27 , critical_time_28 , critical_time_29 , critical_time_30 ,
                           critical_time_31 , critical_time_32 , critical_time_33 , critical_time_34 ,
                           critical_time_35 ]

    return critical_times_set


####################################################### 6

def shift(xs, n):

        import numpy
        import math

        if n >= 0:
           return numpy.r_[np.full(n, 0), xs[:-n]]
        else:
           return numpy.r_[xs[-n:], numpy.full(-n, 0)]


################################################## 7


def test_train_check_func_concat_data( input_x_train , input_x_test , target_voxel_ind , num_iter ,reduce_alpha_coef,critical_times_set,forget_factor,training_at_random):
                                        
     # p is number of target voxel and it is in range[0:4693]
     # alpha is steps in gradient descend
     # num_iter is number of gradient descend iteration
     
     
     import random
     import time
     import os
     import numpy
     import math
     import scipy as sp
     import scipy.optimize
     import sympy
     from sympy.solvers import solve
     from sympy import Symbol
  
     t1_train = input_x_train.shape[0] 
     n5 = input_x_train.shape[1]
     
    
     
     theta_transpose = numpy.random.seed(int(10000 * time.clock()))
     theta_transpose = numpy.random.random((n5 , 1 )) #initial theta whith random matrix

 #    theta_transpose = numpy.zeros((n5 , 1 )) #initial theta whith zero matrix
 ###    theta_transpose = (theta_transpose)/(0.0001 + numpy.linalg.norm(theta_transpose))
     
     
 #removing background of random initial theta


     theta_mean = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/9_ses_set2_10run/zero_initialize-random_train_examp/my_theta_mean.txt")
     

     for i in range(theta_mean.shape[0]):
         if i != target_voxel_ind:
            if -0.002<theta_mean[i]<0.002:
	              theta_transpose[i,:]=0 

     
     
     x_train_normalized = numpy.zeros(shape=(t1_train , n5))


     for i in range(t1_train - 1):
             
         x_train_normalized[i,:] = (input_x_train[i,:])/(0.0001 + numpy.linalg.norm(input_x_train[i,:]))

    
     

     train_label_normalized = x_train_normalized[:,target_voxel_ind ]


             
             

     

     


     t1_test = input_x_test.shape[0]#########
  
     test_label = input_x_test[:,target_voxel_ind ]
     test_cost = 0

     x_test_normalized = numpy.zeros(shape=(t1_test , n5))

     

             
    

     for i in range(t1_test - 1):
             
         x_test_normalized[i,:] = (input_x_test[i,:])/(0.0001 + numpy.linalg.norm(input_x_test[i,:]))


         
     test_label_normalized =  x_test_normalized[:,target_voxel_ind ]





     
     ### nonlinear regression

     exp_regression = 0
     logistic_regression = 0


     if exp_regression == 1:

             train_label_normalized = train_label_normalized + 0.00000001
             test_label_normalized = test_label_normalized + 0.00000001
             train_label_normalized = numpy.log(train_label_normalized )
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
     #####new
     train_cost_per_iter = numpy.zeros(shape=(num_iter))
     before_train_cost_per_iter = numpy.zeros(shape=(num_iter))
     before_test_cost_per_iter=numpy.zeros(shape=(num_iter))
     #3####3


     ###new
     before_hypo_func_train = numpy.dot((x_train_normalized) , (theta_transpose))
     before_train_cost = (1/t1_train) * math.pow((numpy.linalg.norm( before_hypo_func_train[0:(t1_train)-2] - shift(train_label_normalized , -1)[0:(t1_train)-2])) , 2)
     #before_train_cost_per_iter[ite] = train_cost
     before_hypo_func_test = numpy.dot((x_test_normalized) , (theta_transpose))
     before_test_cost = (1/t1_test) * math.pow((numpy.linalg.norm( before_hypo_func_test[0:(t1_test)-2] - shift(test_label_normalized , -1)[0:(t1_test)-2])) , 2)
     #before_test_cost_per_iter[ite] = test_cost
     ###


     #training at random:

     if training_at_random == 1:
        training_order =  numpy.random.seed(int(10000 * time.clock()))
        training_order = numpy.random.randint((t1_train - 1) , size = (t1_train - 1))

     elif training_at_random == 0:

        training_order = range(t1_train - 1)

     ##########################   
     
     
     
 #    print("hello")
            
     xT = numpy.transpose(x_train_normalized)

     x_xT = numpy.dot(x_train_normalized , xT)
 #    pdb.set_trace()
     y = shift(train_label_normalized , -1)
     y = numpy.reshape(y,(t1_train,1))
 
     yT = numpy.transpose(y)

     yT = numpy.reshape(yT,(1,t1_train))

     
    


     
  #   pdb.set_trace()
            
             

#########################
             
     
     for ite in range(num_iter):
             cost_func = 0
             test_cost = 0
             ####new
             train_cost = 0

             ####
             itr_i = 0


             
             theta_transpose_T = numpy.transpose(theta_transpose)
             
             m1 = yT - (numpy.dot(theta_transpose_T, xT))

             M = 4*numpy.dot(m1, x_xT)

             h1 = y - numpy.dot(x_train_normalized, theta_transpose)

             pp = numpy.dot(M, h1)##3

             q1 = numpy.dot(M,x_xT)

             qq = numpy.dot(q1,h1)

             alp =1000* (-1/2)*pp/qq


            # print(alp)


 #            pdb.set_trace()
        



               
             for i in training_order:

                 itr_i = itr_i + 1

                 if i not in critical_times_set:
                         
                     
                    hypo_func = numpy.dot((x_train_normalized[i,:]),(theta_transpose))
                     
                    temp= ((  train_label_normalized[i+1]-hypo_func ) * x_train_normalized[i,:])# i and i+1 is because of causality
                    s = s+numpy.reshape(temp,[n5,1])
                    if  itr_i % (t1_train - 2) == 0:
                       #print (numpy.linalg.norm(theta_transpose))
                       #print( numpy.linalg.norm(temp))

 #                      s = s/(0.0001 + numpy.linalg.norm(s))
                       theta_transpose = (forget_factor) * theta_transpose - (alp/(reduce_alpha_coef*ite+1)) * (2/(t1_train)) *s


# numpy.max(theta_transpose) , s
                       
                       #theta_transpose = (theta_transpose)/(0.0001 + numpy.linalg.norm(theta_transpose))
                       #pdb.set_trace()
                       s = numpy.zeros(shape = (n5,1))
                    
 ##                   cost_func =  cost_func + (1/t1_train) * ( math.pow(( hypo_func - train_label_normalized[i+1] ) , 2))

                        
                                
             theta_transpose[target_voxel_ind ] = 0 # so we remove train_label from x_train      
 ##            cost_func_per_iter[ite] = cost_func       
             
             hypo_func = numpy.dot((x_test_normalized) , (theta_transpose)) # it is a m*1 or (t1/2 * 1) matrix

             #######new
             hypo_func_train = numpy.dot((x_train_normalized) , (theta_transpose))
             train_cost = (1/t1_train) * math.pow((numpy.linalg.norm( hypo_func_train[0:(t1_train)-2] - shift(train_label_normalized , -1)[0:(t1_train)-2])) , 2)
             train_cost_per_iter[ite] = train_cost
             #########
    
             

             test_cost =  (1/t1_test) * math.pow((numpy.linalg.norm( hypo_func[0:(t1_test)-2] - shift(test_label_normalized , -1)[0:(t1_test)-2])) , 2)

             test_cost_per_iter[ite] = test_cost  
             
              
                   
     return theta_transpose,  test_cost , test_cost_per_iter , train_cost , train_cost_per_iter , before_test_cost , before_train_cost
###################################################### 8
def find_test_cost(input_x_test , input_theta_transpose , target_voxel_ind ):

    import os
    import numpy
    import math
 

    
    n5  = input_x_test.shape[1]
    t1 = input_x_test.shape[0]
  
    test_cost = 0

    x_test_normalized = numpy.zeros(shape=(t1 , n5))

    for i in range(t1 - 1):
             
         x_test_normalized[i,:] = (input_x_test[i,:])/(0.0001 + numpy.linalg.norm(input_x_test[i,:]))

        
     
    hypo_func = numpy.dot((x_test_normalized) , (input_theta_transpose)) # it is a m*1 or (t1/2 * 1) matrix
    
    test_label_normalized = x_test_normalized[:,target_voxel_ind ]

    exp_regression = 0
    logistic_regression = 0


    if exp_regression == 1:

 #            train_label_normalized = train_label_normalized + 0.00000001
             test_label_normalized = test_label_normalized + 0.00000001
 #            train_label_normalized = numpy.log(train_label_normalized )
             test_label_normalized = numpy.log(test_label_normalized)

#             train_label_normalized = (train_label_normalized)/(0.0001 + numpy.linalg.norm(train_label_normalized))
 #            test_label_normalized = (test_label_normalized)/(0.0001 + numpy.linalg.norm(test_label_normalized))

    if logistic_regression == 1:
             
#             train_label_normalized = train_label_normalized + 0.00000001
             test_label_normalized = test_label_normalized + 0.00000001
 #            train_label_normalized = -numpy.log((1/(train_label_normalized))-1)
             test_label_normalized = -numpy.log((1/(test_label_normalized))-1)

 #            train_label_normalized = (train_label_normalized)/(0.0001 + numpy.linalg.norm(train_label_normalized))
 #            test_label_normalized = (test_label_normalized)/(0.0001 + numpy.linalg.norm(test_label_normalized))

   


     
    test_cost =  (1/t1) * math.pow((numpy.linalg.norm( hypo_func[0:(t1)-2] - shift(test_label_normalized , -1)[0:(t1)-2])) , 2)
    
    


    return test_cost 




###################################################### 9


def find_mean_and_variance_of_theta(input_my_theta):


    import os
    import numpy
    import math

    
    my_theta_mean = numpy.mean(input_my_theta , axis=1)

 #   my_theta_mean = (my_theta_mean)/(0.0001 + numpy.linalg.norm(my_theta_mean))
    
    my_theta_variance = numpy.var(input_my_theta , axis = 1)

    return my_theta_mean , my_theta_variance
    





####################################################### 10


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
                          "theta with the same parameters for all voxels")
                plt.xlabel("number of voxels")
                plt.ylabel("theta")



                plt.figure(4)
                plt.plot(input_my_theta[1000:1020,i])
                plt.title(" plot of "+str(num_storing_sets_of_theta)+
                          "theta with the same parameters for  voxel 1000 to 1020")
                plt.xlabel("voxel 1000 to 1020")
                plt.ylabel("theta")


                

        plt.figure(5)
        plt.plot(input_my_theta_mean)
        plt.title(" plot of theta_mean")
        plt.xlabel("number of voxels")
        plt.ylabel("theta_mean")




        plt.figure(6)
        plt.plot(input_my_theta_variance)
        plt.title(" plot of theta_variance")
        plt.xlabel("number of voxels")
        plt.ylabel("theta_variance")




        plt.figure(7)
        plt.plot(input_my_theta_mean[1000:1020])
        plt.title(" plot of theta_mean for  voxel 1000 to 1020 ")
        plt.xlabel("voxel 1000 to 1020")
        plt.ylabel("theta_mean")




                   
        plt.figure(8)
        plt.plot(input_my_theta_variance[1000:1020])
        plt.title(" plot of theta_variance for  voxel 1000 to 1020 ")
        plt.xlabel("voxel 1000 to 1020")
        plt.ylabel("theta_variance")

                   


        


        plt.show()

################################################
 #theta_transpose, cost_func , cost_func_per_iter, test_cost , test_cost_per_iter = test_train_check_func_concat_data( x_train , x_test ,3382 , 0.5,1000,0.005,critical_times_set)

 #theta_transpose,  test_cost , test_cost_per_iter , train_cost , train_cost_per_iter , before_test_cost , before_train_cost = test_train_check_func_concat_data( x_train , x_test ,3382,0.5 , 1 , 0.01 ,critical_times_set)
        
    
###################################### 11
def  find_corresponding_voxel_after_reshape(input_theta , r0 ,r1 , r2,target_voxel):
        import numpy
        import math

        reshape_theta = numpy.reshape(input_theta ,(r0,r1,r2))

        y1 = input_theta[target_voxel]

        x3 = (((target_voxel)+1) % r2) - 1

        q = ((target_voxel)+1) // r2

        x2 = q % r1

        x1 = q // r1

        if (((target_voxel)+1) % r2) != 0:

                y2 = reshape_theta[x1,x2,x3]

                return x1,x2,x3,y2-y1

        else:
                y2 = reshape_theta[x1,x2 - 1,r2 - 1]

                return x1,x2 - 1,r2 - 1 , y2-y1

#################################################  12


def find_train_cost(input_x_train , input_theta_transpose , target_voxel_ind ):

    import os
    import numpy
    import math
 

    
    n5  = input_x_train.shape[1]
    t1 = input_x_train.shape[0]
  
    train_cost = 0

    x_train_normalized = numpy.zeros(shape=(t1 , n5))

    for i in range(t1 - 1):
             
         x_train_normalized[i,:] = (input_x_train[i,:])/(0.0001 + numpy.linalg.norm(input_x_train[i,:]))

        
     
    hypo_func = numpy.dot((x_train_normalized) , (input_theta_transpose)) 
    
    train_label_normalized = x_train_normalized[:,target_voxel_ind ]




    exp_regression = 0
    logistic_regression = 0


    if exp_regression == 1:

             train_label_normalized = train_label_normalized + 0.00000001
#             test_label_normalized = test_label_normalized + 0.00000001
             train_label_normalized = numpy.log(train_label_normalized )
#             test_label_normalized = numpy.log(test_label_normalized)

 #            train_label_normalized = (train_label_normalized)/(0.0001 + numpy.linalg.norm(train_label_normalized))
 #            test_label_normalized = (test_label_normalized)/(0.0001 + numpy.linalg.norm(test_label_normalized))

    if logistic_regression == 1:
             
            train_label_normalized = train_label_normalized + 0.00000001
#             test_label_normalized = test_label_normalized + 0.00000001
            train_label_normalized = -numpy.log((1/(train_label_normalized))-1)
#             test_label_normalized = -numpy.log((1/(test_label_normalized))-1)

 #           train_label_normalized = (train_label_normalized)/(0.0001 + numpy.linalg.norm(train_label_normalized))
 #            test_label_normalized = (test_label_normalized)/(0.0001 + numpy.linalg.norm(test_label_normalized))



            

   


     
    train_cost =  (1/t1) * math.pow((numpy.linalg.norm( hypo_func[0:(t1)-2] - shift(train_label_normalized , -1)[0:(t1)-2])) , 2)
    
    


    return train_cost

################################################# 13


def find_theta_by_solving_matrix_equation(input_x_train , target_voxel_ind ):

    import numpy
    import math

    n5  = input_x_train.shape[1]
    t1 = input_x_train.shape[0]
    

    x_train_normalized = numpy.zeros(shape=(t1 , n5))

    for i in range(t1 - 1):
             
         x_train_normalized[i,:] = (input_x_train[i,:])/(0.0001 + numpy.linalg.norm(input_x_train[i,:]))

    

    train_label_normalized = x_train_normalized[:,target_voxel_ind ]

    x_train_inverse = numpy.linalg.pinv(input_x_train)

    pinv_theta = numpy.dot(x_train_inverse , shift(train_label_normalized , -1) )
   

 #   pinv_theta = (pinv_theta)/(0.0001 + numpy.linalg.norm(pinv_theta))

    return pinv_theta
######################################################### 14
def Lasso_linear_regression(input_x_train,target_voxel_ind):

    import numpy
    import math
    import sklearn
    from sklearn import linear_model
    import math

    t1_train = input_x_train.shape[0]
    n5  = input_x_train.shape[1]

    x_train_normalized = numpy.zeros(shape=(t1_train , n5))


    for i in range(t1_train - 1):
             
        x_train_normalized[i,:] = (input_x_train[i,:])/(0.0001 + numpy.linalg.norm(input_x_train[i,:]))
    

    train_label_normalized = x_train_normalized[:,target_voxel_ind ]

   


    clf = linear_model.Lasso(alpha=0 ,copy_X =True ,fit_intercept=True, max_iter=100,  normalize=False, positive=False, precompute=False, random_state=None,
    selection='cyclic', tol=0.0001, warm_start=False)
    clf.fit(x_train_normalized , shift(train_label_normalized , -1))

    sparse_theta = clf.coef_


  

    return sparse_theta
###################################################15
def linear_regression(input_x_train,target_voxel_ind):

    import numpy
    import math
    import sklearn
    from sklearn import linear_model
    import math

    t1_train = input_x_train.shape[0]
    n5  = input_x_train.shape[1]

    x_train_normalized = numpy.zeros(shape=(t1_train , n5))


    for i in range(t1_train - 1):
             
        x_train_normalized[i,:] = (input_x_train[i,:])/(0.0001 + numpy.linalg.norm(input_x_train[i,:]))
    

    train_label_normalized = x_train_normalized[:,target_voxel_ind ]

   


    clf = linear_model.LinearRegression()
    clf.fit(x_train_normalized , shift(train_label_normalized , -1))

    theta = clf.coef_


  

    return theta



        
 #######################       

 #for i in range(my_theta_mean.shape[0]):
#	if i != target_voxel_ind:
#	   if -0.005<my_theta_mean[i]<0.005:
#		     concat_data[i,:] = 0                       

                                      

                
        



 


    
        
        
        
    
