
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
################################################### 4' apply PCA version_1 get number of principal components as input


def dimensionality_reduction_version_1( input_x_train, input_x_test, number_of_principal_components):

        import numpy
        import math
        import os

        k = number_of_principal_components

        n5 = input_x_train.shape[1]
        t1_train = input_x_train.shape[0]
        t1_test = input_x_test.shape[0]


        x_train_normalized = numpy.zeros(shape=(t1_train , n5))
        x_test_normalized = numpy.zeros(shape=(t1_test , n5))


                        
        for i in range(t1_test - 1):
             
             x_test_normalized[i,:] = (input_x_test[i,:])/(0.0001 + numpy.linalg.norm(input_x_test[i,:]))




                     
        for i in range(t1_train - 1):
             
             x_train_normalized[i,:] = (input_x_train[i,:])/(0.0001 + numpy.linalg.norm(input_x_train[i,:]))



        

        sigma_train = (1/n5) * numpy.dot(numpy.transpose(x_train_normalized) , x_train_normalized)

        u_train, s_train, v_train = numpy.linalg.svd(sigma_train, full_matrices=1)

        u_reduce_train = u_train[:,0:k]

        z_train = numpy.dot(x_train_normalized , u_reduce_train )



        
        sigma_test = (1/n5) * numpy.dot(numpy.transpose(x_test_normalized) , x_test_normalized)

        u_test, s_test, v_test = numpy.linalg.svd(sigma_test, full_matrices=1)

        u_reduce_test = u_test[:,0:k]

        z_test = numpy.dot( x_test_normalized ,u_reduce_test )

 ##      pdb.set_trace()


        return z_train, z_test, u_reduce_train

        
################################################## 4'' apply PCA version_2 choosing number of principal components (k)


def dimensionality_reduction_version_2( input_x_train, input_x_test):

        import numpy
        import math
        import os


        n5 = input_x_train.shape[1]
        t1_train = input_x_train.shape[0]
        t1_test = input_x_test.shape[0]

        x_train_normalized = numpy.zeros(shape=(t1_train , n5))
        x_test_normalized = numpy.zeros(shape=(t1_test , n5))



                
        for i in range(t1_test - 1):
             
             x_test_normalized[i,:] = (input_x_test[i,:])/(0.0001 + numpy.linalg.norm(input_x_test[i,:]))




                     
        for i in range(t1_train - 1):
             
             x_train_normalized[i,:] = (input_x_train[i,:])/(0.0001 + numpy.linalg.norm(input_x_train[i,:]))


        

        sigma_train = (1/n5) * numpy.dot(numpy.transpose(x_train_normalized) , x_train_normalized)

        u_train, s_train, v_train = numpy.linalg.svd(sigma_train, full_matrices=1)

        s_train_tot_sum = numpy.sum(s_train)
        sii_train = 0

        # choosing the number of principal components
        for i in range(n5):

                sii_train = sii_train+s_train[i]

                k = i

                if ((sii_train)/(s_train_tot_sum)) >= 0.99:

                        break

        

        u_reduce_train = u_train[:,0:k]

        z_train = numpy.dot(x_train_normalized , u_reduce_train )





        
        sigma_test = (1/n5) * numpy.dot(numpy.transpose(x_test_normalized) , x_test_normalized)

        u_test, s_test, v_test = numpy.linalg.svd(sigma_test, full_matrices=1)

        u_reduce_test = u_test[:,0:k]

        z_test = numpy.dot( x_test_normalized ,u_reduce_test )

 ##       pdb.set_trace()

        print(k)


        return k, z_train, z_test,u_reduce_train


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

def test_train_check_func_concat_data( input_x_train, input_x_test, input_z_train , input_z_test , target_voxel_ind , alpha , num_iter ,reduce_alpha_coef,critical_times_set,forget_factor,training_at_random):
                                        
     # p is number of target voxel and it is in range[0:4693]
     # alpha is steps in gradient descend
     # num_iter is number of gradient descend iteration
     
     
     import random
     import time
     import os
     import numpy
     import math


     
  
     t1_train = input_z_train.shape[0]
     t1_test = input_z_test.shape[0]
     n5_z = input_z_train.shape[1]



     

     theta_1 = numpy.random.seed(int(10000 * time.clock()))
     theta_1 = numpy.random.random((1)) #initial theta whith random matrix
     theta_1_train = numpy.dot(numpy.ones((t1_train,1)) , theta_1)
     theta_1_test = numpy.dot(numpy.ones((t1_test,1)) , theta_1)

     
     theta_2 = numpy.random.seed(int(10000 * time.clock()))
     theta_2 = numpy.random.random((1)) #initial theta whith random matrix
     theta_2_train = numpy.dot(numpy.ones((t1_train,1)) , theta_2)
     theta_2_test = numpy.dot(numpy.ones((t1_test,1)) , theta_2)


     
     theta_3 = numpy.random.seed(int(10000 * time.clock()))
     theta_3 = numpy.random.random((n5_z, 1 )) #initial theta whith random matrix


     

 #    theta_transpose = numpy.zeros((n5 , 1 )) #initial theta whith zero matrix
     theta_3 = (theta_3)/(0.0001 + numpy.linalg.norm(theta_3))
 #    theta_2 = (theta_2)/(0.0001 + numpy.linalg.norm(theta_2))
#     theta_1 = (theta_1)/(0.0001 + numpy.linalg.norm(theta_1))
     
 #removing background of random initial theta


 #    theta_mean = numpy.loadtxt("/Users/Apple/Desktop/first_chain_code_results/9_ses_set2_10run/zero_initialize-random_train_examp/my_theta_mean.txt")
     

 #    for i in range(theta_mean.shape[0]):
 #        if i != target_voxel_ind:
 #           if -0.002<theta_mean[i]<0.002:
#	              theta_3[i,:]=0 

     
     
  #   z_train_normalized = numpy.zeros(shape=(t1_train , n5_z))
    

     train_label = input_x_train[:,target_voxel_ind ]
     train_label_normalized = (train_label)/(0.0001 + numpy.linalg.norm(train_label))
     


     #for i in range(t1_train - 1):
             
 #        z_train_normalized[i,:] = (input_z_train[i,:])/(0.0001 + numpy.linalg.norm(input_z_train[i,:]))



    
     

     


     t1_test = input_z_test.shape[0]#########
  
     test_label = input_x_test[:,target_voxel_ind ]
     test_label_normalized = (test_label)/(0.0001 + numpy.linalg.norm(test_label))
     
     test_cost = 0

     z_test_normalized = numpy.zeros(shape=(t1_test , n5_z))
     

     

             
    

 #    for i in range(t1_test - 1):
             
#         z_test_normalized[i,:] = (input_z_test[i,:])/(0.0001 + numpy.linalg.norm(input_z_test[i,:]))




          

     
     
     
     #gradient descend algorithm
     cost_func_per_iter = numpy.zeros(shape=(num_iter))
     
 #    s1 = numpy.zeros(shape = (1))
#     s2 = numpy.zeros(shape = (1))
     s3 = numpy.zeros(shape = (n5_z,1))
     
     test_cost_per_iter = numpy.zeros(shape=(num_iter))
     #####new
     train_cost_per_iter = numpy.zeros(shape=(num_iter))
 #    before_train_cost_per_iter = numpy.zeros(shape=(num_iter))
#     before_test_cost_per_iter=numpy.zeros(shape=(num_iter))
     #3####3

     before_hypo_func_train = numpy.zeros((t1_train))
     before_hypo_func_test = numpy.zeros((t1_test))



     ###new

     for i in range(int(t1_train)):
            
         before_hypo_func_train[i] = theta_1_train[i] + theta_2_train[i] * math.exp(numpy.dot((input_z_train[i,:]) , (theta_3)))

     before_train_cost = (1/t1_train) * math.pow((numpy.linalg.norm( before_hypo_func_train[0:(t1_train)-2] - shift(train_label_normalized , -1)[0:(t1_train)-2])) , 2)
 #   before_train_cost_per_iter[ite] = train_cost

     for i in range(int(t1_test)):
            
         before_hypo_func_test[i] = theta_1_test[i] + theta_2_test[i] * math.exp(numpy.dot((input_z_test[i,:]) , (theta_3)))
        
     before_test_cost = (1/t1_test) * math.pow((numpy.linalg.norm( before_hypo_func_test[0:(t1_test)-2] - shift(test_label_normalized , -1)[0:(t1_test)-2])) , 2)
     #before_test_cost_per_iter[ite] = test_cost
     


     #training at random:

     if training_at_random == 1:
        training_order =  numpy.random.seed(int(10000 * time.clock()))
        training_order = numpy.random.randint((t1_train - 1) , size = (t1_train - 1))

     elif training_at_random == 0:

        training_order = range(t1_train - 1)

     ##########################        
     
     
     
     
            
     
     for ite in range(num_iter):
             cost_func = 0
             test_cost = 0
             ####new
             train_cost = 0

             ####





               
             for i in training_order:

                 if i not in critical_times_set:

                       
                         
                     
                    hypo_func = theta_1_train[i] + theta_2_train[i] *math.exp(numpy.dot((input_z_train[i,:]),(theta_3)))
                     
 #                   temp_1= ( hypo_func- train_label_normalized[i+1] )# i and i+1 is because of causality
#                    s1 = s1 + temp_1



#                    temp_2= ( hypo_func- train_label_normalized[i+1] ) * math.exp(numpy.dot((z_train_normalized[i,:]),(theta_3)))# i and i+1 is because of causality
#                    s2 = s2 + temp_2



 ##                   pdb.set_trace()



                    temp_3= (( hypo_func- train_label_normalized[i+1] ) * theta_2 *math.exp(numpy.dot((input_z_train[i,:]),(theta_3)))) * input_z_train[i,:]# i and i+1 is because of causality
                    s3 = s3 + numpy.reshape(temp_3,[n5_z,1])

                

                    
                    if i% (t1_train - 2) == 0:
#                       theta_1_trai = (forget_factor) * theta_1 - (alpha/(reduce_alpha_coef*ite+1)) * (2/(t1_train)) *s1
 #                      theta_1 = (theta_1)/(0.0001 + numpy.linalg.norm(theta_1))


#                       theta_2 = (forget_factor) * theta_2 - (alpha/(reduce_alpha_coef*ite+1)) * (2/(t1_train)) *s2
#                       theta_2 = (theta_2)/(0.0001 + numpy.linalg.norm(theta_2))


                       theta_3 = (forget_factor) * theta_3 - (alpha/(reduce_alpha_coef*ite+1)) * (2/(t1_train)) *s3
                       theta_3 = (theta_3)/(0.0001 + numpy.linalg.norm(theta_3))
                       
 #                      s1 = numpy.zeros(shape = (1))
#                       s2 = numpy.zeros(shape = (1))
                       s3 = numpy.zeros(shape = (n5_z,1))


                      


                    
 ##                   cost_func =  cost_func + (1/t1_train) * ( math.pow(( hypo_func - train_label_normalized[i+1] ) , 2))

                        
                                
 #            theta_3[target_voxel_ind ] = 0 # so we remove train_label from x_train      
 ##            cost_func_per_iter[ite] = cost_func






             hypo_func = numpy.zeros(shape = (t1_test))
             hypo_func_train = numpy.zeros(shape = (t1_train))

             for i in range(int(t1_test)):
                                              
                 hypo_func[i] = theta_1_test[i] + theta_2_test[i] * math.exp(numpy.dot((input_z_test[i,:]) , (theta_3))) # it is a m*1 or (t1/2 * 1) matrix

                 

             for i in range(int(t1_train)):
                     
                 hypo_func_train[i] =theta_1_train[i] + theta_2_train[i] * math.exp(numpy.dot((input_z_train[i,:]) , (theta_3)))    
                     

             #######new
             
             train_cost = (1/t1_train) * math.pow((numpy.linalg.norm( hypo_func_train[0:(t1_train)-2] - shift(train_label_normalized , -1)[0:(t1_train)-2])) , 2)
             train_cost_per_iter[ite] = train_cost
             #########
    
             

             test_cost =  (1/t1_test) * math.pow((numpy.linalg.norm( hypo_func[0:(t1_test)-2] - shift(test_label_normalized , -1)[0:(t1_test)-2])) , 2)

             test_cost_per_iter[ite] = test_cost

#             pdb.set_trace()
             
              
                   
     return theta_3, test_cost , test_cost_per_iter , train_cost , train_cost_per_iter , before_test_cost , before_train_cost 
###################################################### 8
def find_test_cost(input_x_test ,input_z_test, input_theta_3 ,input_final_theta_mean, target_voxel_ind ):

    import os
    import numpy
    import math
 

    
    n5_z  = input_z_test.shape[1]

    n5  = input_x_test.shape[1]
    t1_test = input_x_test.shape[0]
  
    test_cost = 0

    x_test_normalized = numpy.zeros(shape=(t1_test , n5))
 
    test_label = input_x_test[:,target_voxel_ind ]

    test_label_normalized =  (test_label)/(0.0001 + numpy.linalg.norm(test_label))

    for i in range(t1_test - 1):
             
        x_test_normalized[i,:] = (input_x_test[i,:])/(0.0001 + numpy.linalg.norm(input_x_test[i,:]))


        
     
    hypo_func_z = numpy.dot((input_z_test) , (input_theta_3)) # it is a m*1 or (t1/2 * 1) matrix
    
    hypo_func_x = numpy.dot((x_test_normalized) , (input_final_theta_mean))

   


     
    z_test_cost =  (1/t1_test) * math.pow((numpy.linalg.norm( hypo_func_z[0:(t1_test)-2] - shift(test_label_normalized , -1)[0:(t1_test)-2])) , 2)
    x_test_cost =  (1/t1_test) * math.pow((numpy.linalg.norm( hypo_func_x[0:(t1_test)-2] - shift(test_label_normalized , -1)[0:(t1_test)-2])) , 2)
    


    return z_test_cost , x_test_cost




###################################################### 9


def find_mean_and_variance_of_theta(input_my_theta):


    import os
    import numpy
    import math

    
    my_theta_mean = numpy.mean(input_my_theta , axis=1)

    my_theta_mean = (my_theta_mean)/(0.0001 + numpy.linalg.norm(my_theta_mean))
    
    my_theta_variance = numpy.var(input_my_theta , axis = 1)

    return my_theta_mean , my_theta_variance
    





####################################################### 10


def plotting_results(input_my_train_cost_per_iter , input_my_test_cost_per_iter,
                     input_my_theta, 
                     input_my_theta_mean ,input_my_theta_variance,input_final_theta_mean,
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



 #               plt.figure(4)
#                plt.plot(input_my_theta[1000:1020,i])
#                plt.title(" plot of "+str(num_storing_sets_of_theta)+
#                          "theta with the same parameters for  voxel 1000 to 1020")
#                plt.xlabel("voxel 1000 to 1020")
#                plt.ylabel("theta")


                

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




 #       plt.figure(7)
#        plt.plot(input_my_theta_mean[1000:1020])
#        plt.title(" plot of theta_mean for  voxel 1000 to 1020 ")
#        plt.xlabel("voxel 1000 to 1020")
#        plt.ylabel("theta_mean")




                   
 #       plt.figure(8)
#        plt.plot(input_my_theta_variance[1000:1020])
#        plt.title(" plot of theta_variance for  voxel 1000 to 1020 ")
#        plt.xlabel("voxel 1000 to 1020")
#        plt.ylabel("theta_variance")


        plt.figure(9)
        plt.plot(input_final_theta_mean)
        plt.title(" plot of final theta mean")
        plt.xlabel("number of voxels")
        plt.ylabel("final_theta_mean")




                   


        


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

def find_train_cost(input_x_train ,input_z_train, input_theta_3 ,input_final_theta_mean, target_voxel_ind ):

    import os
    import numpy
    import math
 

    
    n5_z  = input_z_train.shape[1]

    n5  = input_x_train.shape[1]
    t1_train = input_x_train.shape[0]
  
    train_cost = 0

    x_train_normalized = numpy.zeros(shape=(t1_train , n5))
 
    train_label = input_x_train[:,target_voxel_ind ]

    train_label_normalized =  (train_label)/(0.0001 + numpy.linalg.norm(train_label))

    for i in range(t1_train - 1):
             
        x_train_normalized[i,:] = (input_x_train[i,:])/(0.0001 + numpy.linalg.norm(input_x_train[i,:]))


        
     
    hypo_func_z = numpy.dot((input_z_train) , (input_theta_3)) # it is a m*1 or (t1/2 * 1) matrix
    
    hypo_func_x = numpy.dot((x_train_normalized) , (input_final_theta_mean))

   


     
    z_train_cost =  (1/t1_train) * math.pow((numpy.linalg.norm( hypo_func_z[0:(t1_train)-2] - shift(train_label_normalized , -1)[0:(t1_train)-2])) , 2)
    x_train_cost =  (1/t1_train) * math.pow((numpy.linalg.norm( hypo_func_x[0:(t1_train)-2] - shift(train_label_normalized , -1)[0:(t1_train)-2])) , 2)
    


    return z_train_cost , x_train_cost

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
   

    pinv_theta = (pinv_theta)/(0.0001 + numpy.linalg.norm(pinv_theta))

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



        
 ####################### 16
def cross_correlation(input_concat_data):

        import math
        import numpy
        import os
        import pdb
        


        n5 = input_concat_data.shape[0]
        cross_correlation_matrix = numpy.zeros((n5,n5))


        for i in range(n5):
                for j in range(n5):

                        cross_correlation_matrix[i,j] = numpy.correlate(input_concat_data[i,:],input_concat_data[j,:])


        return cross_correlation_matrix                 
                        
        

                      

                
        



 


    
        
        
        
    
